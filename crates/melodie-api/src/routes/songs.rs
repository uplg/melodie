use std::convert::Infallible;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::routing::{get, post};
use chrono::{DateTime, Utc};
use melodie_core::authz::{self, Action, Resource};
use melodie_core::ids::SongId;
use melodie_core::model::{Song, SongMode, SongStatus};
use melodie_db::clips::UpsertClip;
use melodie_db::songs::NewSong;
use serde::{Deserialize, Serialize};
use suno_client::SunoError;
use suno_client::types::{ControlSliders, GenerateRequest};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::extract::AuthUser;
use crate::poll::SongEvent;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/songs", post(create).get(list))
        .route(
            "/songs/{id}",
            get(detail).delete(delete_song).patch(rename),
        )
        .route("/songs/{id}/events", get(events))
}

// --- daily quota ---

pub const DAILY_CAP: u32 = 4;

// --- field caps (mirror Suno's documented limits) ---

const TITLE_MAX: usize = 100;
const TAGS_MAX: usize = 1000;
const EXCLUDE_MAX: usize = 1000;
const LYRICS_MAX: usize = 5000;
const PROMPT_MAX: usize = 500;

// --- request / response views ---

/// Body shape for `POST /api/songs`. Tagged-union by `mode`:
/// - `{ "mode": "custom",   ... }` — full lyrics + tags + sliders.
/// - `{ "mode": "describe", ... }` — single free-text prompt; Suno authors lyrics.
#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum CreateSongRequest {
    Custom(CustomFields),
    Describe(DescribeFields),
}

#[derive(Debug, Deserialize)]
pub struct CustomFields {
    pub title: String,
    pub tags: String,
    #[serde(default)]
    pub exclude_tags: Option<String>,
    pub lyrics: String,
    #[serde(default)]
    pub vocal: Option<String>,
    #[serde(default)]
    pub weirdness: Option<i32>,
    #[serde(default)]
    pub style_influence: Option<i32>,
    #[serde(default)]
    pub variation: Option<String>,
    #[serde(default)]
    pub instrumental: bool,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DescribeFields {
    pub prompt: String,
    #[serde(default)]
    pub instrumental: bool,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SongView {
    pub id: String,
    pub mode: String,
    pub title: Option<String>,
    pub tags: Option<String>,
    pub exclude_tags: Option<String>,
    pub lyrics: Option<String>,
    pub prompt: Option<String>,
    pub model: String,
    pub status: String,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub clips: Vec<ClipView>,
}

#[derive(Debug, Serialize)]
pub struct ClipView {
    pub id: String,
    pub variant_index: i32,
    pub status: String,
    pub duration_s: Option<f64>,
    pub image_url: Option<String>,
}

impl From<&Song> for SongView {
    fn from(s: &Song) -> Self {
        Self {
            id: s.id.to_string(),
            mode: match s.mode {
                SongMode::Custom => "custom".into(),
                SongMode::Describe => "describe".into(),
            },
            title: s.title.clone(),
            tags: s.tags.clone(),
            exclude_tags: s.exclude_tags.clone(),
            lyrics: s.lyrics.clone(),
            prompt: s.prompt.clone(),
            model: s.model.clone(),
            status: match s.status {
                SongStatus::Pending => "pending".into(),
                SongStatus::Generating => "generating".into(),
                SongStatus::Complete => "complete".into(),
                SongStatus::Failed => "failed".into(),
            },
            error: s.error.clone(),
            created_at: s.created_at,
            updated_at: s.updated_at,
            clips: s
                .clips
                .iter()
                .map(|c| ClipView {
                    id: c.id.clone(),
                    variant_index: c.variant_index,
                    status: c.status.clone(),
                    duration_s: c.duration_s,
                    image_url: c.image_url.clone(),
                })
                .collect(),
        }
    }
}

// --- handlers ---

async fn create(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Json(req): Json<CreateSongRequest>,
) -> ApiResult<(StatusCode, Json<SongView>)> {
    validate_create(&req)?;

    // --- shared preamble (auth / quota / captcha) ---

    // Pull the active Suno client first — a 503 here shouldn't burn the
    // user's daily quota slot.
    let client = state
        .suno
        .current()
        .await
        .ok_or_else(|| ApiError::Suno(SunoError::AuthMissing))?;

    // Quota check (admin bypass). If Suno or captcha fails *after* this point
    // the slot is consumed; accepted — the friends-project scope doesn't
    // justify a compensating-rollback path.
    if user.role != melodie_core::model::Role::Admin {
        let new_count = melodie_db::quota::try_increment(&state.db, user.id, DAILY_CAP).await?;
        if new_count.is_none() {
            return Err(ApiError::TooManyRequests);
        }
    }

    // Solve hCaptcha. Boots Chrome on first call (~10s) and reuses it across
    // subsequent solves. Required for both modes — Suno's v2-web endpoint
    // gates everything.
    let auth = client.auth_snapshot();
    let captcha_token = suno_client::captcha::solve(&auth).await?;

    // --- mode-specific: build the upstream request + DB row ---

    let (sreq, song_id, model_key) = match &req {
        CreateSongRequest::Custom(c) => {
            let model_key = c
                .model
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or("chirp-fenix")
                .to_string();

            let mut sreq = GenerateRequest::new(&model_key, "custom");
            sreq.token = Some(captcha_token);
            sreq.title = Some(c.title.trim().to_string());
            sreq.tags = Some(c.tags.trim().to_string());
            sreq.negative_tags = c
                .exclude_tags
                .as_deref()
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            sreq.prompt = c.lyrics.trim().to_string();
            sreq.make_instrumental = c.instrumental;
            if c.weirdness.is_some() || c.style_influence.is_some() {
                sreq.metadata.control_sliders = Some(ControlSliders {
                    weirdness_constraint: c.weirdness.map(|v| f64::from(v) / 100.0),
                    style_weight: c.style_influence.map(|v| f64::from(v) / 100.0),
                });
            }

            let song_id = melodie_db::songs::create(
                &state.db,
                NewSong {
                    owner_id: user.id,
                    mode: SongMode::Custom,
                    title: Some(&c.title),
                    tags: Some(&c.tags),
                    exclude_tags: c.exclude_tags.as_deref(),
                    lyrics: Some(&c.lyrics),
                    prompt: None,
                    vocal: c.vocal.as_deref(),
                    weirdness: c.weirdness,
                    style_inf: c.style_influence,
                    variation: c.variation.as_deref(),
                    model: &model_key,
                },
            )
            .await?;

            (sreq, song_id, model_key)
        }
        CreateSongRequest::Describe(d) => {
            let model_key = d
                .model
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or("chirp-fenix")
                .to_string();

            // Suno's "Simple" / description mode wire shape (per upstream
            // `API_INTELLIGENCE.md`):
            //   - `gpt_description_prompt`: the user's free-text description
            //   - `prompt`: empty string (NOT the description)
            //   - `title` / `tags`: empty strings (Suno authors them)
            //   - `metadata.create_mode`: "inspiration"
            //
            // Earlier we put the user text into `prompt` and Suno dutifully
            // sang it back as lyrics. The fix is `gpt_description_prompt`.
            let mut sreq = GenerateRequest::new(&model_key, "inspiration");
            sreq.token = Some(captcha_token);
            sreq.title = Some(String::new());
            sreq.tags = Some(String::new());
            sreq.prompt = String::new();
            sreq.gpt_description_prompt = Some(d.prompt.trim().to_string());
            sreq.make_instrumental = d.instrumental;

            let song_id = melodie_db::songs::create(
                &state.db,
                NewSong {
                    owner_id: user.id,
                    mode: SongMode::Describe,
                    title: None,
                    tags: None,
                    exclude_tags: None,
                    lyrics: None,
                    prompt: Some(&d.prompt),
                    vocal: None,
                    weirdness: None,
                    style_inf: None,
                    variation: None,
                    model: &model_key,
                },
            )
            .await?;

            (sreq, song_id, model_key)
        }
    };

    // --- shared tail: submit, persist clips, spawn poller, return view ---

    let _ = model_key; // model_key is captured into the DB row above; silence unused-binding lint

    let suno_clips = match client.generate(&sreq).await {
        Ok(clips) => clips,
        Err(e) => {
            // Generation failed; mark the song as failed so the user sees it
            // in the list rather than a phantom "generating" stuck forever.
            let _ = melodie_db::songs::set_status(
                &state.db,
                song_id,
                SongStatus::Failed,
                Some(&e.to_string()),
            )
            .await;
            return Err(ApiError::from(e));
        }
    };

    if suno_clips.is_empty() {
        let _ = melodie_db::songs::set_status(
            &state.db,
            song_id,
            SongStatus::Failed,
            Some("Suno returned 0 clips"),
        )
        .await;
        return Err(ApiError::Internal(
            "Suno returned 0 clips for a successful generate".into(),
        ));
    }

    let upserts: Vec<UpsertClip> = suno_clips
        .iter()
        .enumerate()
        .map(|(i, c)| UpsertClip {
            id: c.id.clone(),
            song_id,
            variant_index: i as i32,
            status: c.status.clone(),
            duration_s: c.metadata.duration,
            image_url: c.image_url.clone(),
        })
        .collect();
    melodie_db::clips::upsert_many(&state.db, &upserts).await?;

    let clip_ids: Vec<String> = suno_clips.iter().map(|c| c.id.clone()).collect();
    crate::poll::spawn(state.clone(), song_id, clip_ids);

    let song = melodie_db::songs::find_with_clips(&state.db, song_id)
        .await?
        .ok_or(ApiError::Internal("song vanished after insert".into()))?;
    Ok((StatusCode::CREATED, Json(SongView::from(&song))))
}

async fn events(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let song_id = parse_song_id(&id)?;
    let song = melodie_db::songs::find_with_clips(&state.db, song_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    if !authz::can(
        user.role,
        user.id,
        Action::Read,
        Resource::Song {
            owner_id: song.owner_id,
            song_id,
        },
    ) {
        return Err(ApiError::Forbidden);
    }

    // Snapshot the current state from DB so the client gets a frame
    // immediately on connect — no waiting for the next poll tick.
    let initial = SongEvent::from_song(&song);
    let initial_terminal = matches!(song.status, SongStatus::Complete | SongStatus::Failed);
    let song_id_str = song_id.to_string();
    let rx = state.events.subscribe();

    let stream = async_stream::stream! {
        // Initial frame.
        match SseEvent::default().event("update").json_data(&initial) {
            Ok(ev) => yield Ok::<_, Infallible>(ev),
            Err(e) => {
                tracing::warn!(error = %e, "failed to encode initial SSE event");
            }
        }
        if initial_terminal {
            return;
        }

        let mut rx = rx;
        loop {
            match rx.recv().await {
                Ok(ev) if ev.song_id == song_id_str => {
                    let terminal = matches!(ev.status.as_str(), "complete" | "failed");
                    match SseEvent::default().event("update").json_data(&ev) {
                        Ok(sse) => yield Ok(sse),
                        Err(e) => {
                            tracing::warn!(error = %e, "failed to encode SSE event");
                        }
                    }
                    if terminal {
                        return;
                    }
                }
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::debug!(lagged = n, "SSE subscriber lagged");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    };

    let sse = Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));

    // Anti-buffering hints. `X-Accel-Buffering: no` tells nginx (and most
    // reverse proxies that cargo-culted nginx semantics) to not collect the
    // response. Cache-Control mirrors what axum already sets but explicit
    // here makes it harder to forget on some intermediaries.
    let mut headers = HeaderMap::new();
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    headers.insert("cache-control", HeaderValue::from_static("no-cache, no-transform"));

    Ok((headers, sse))
}

async fn list(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
) -> ApiResult<Json<Vec<SongView>>> {
    let songs = melodie_db::songs::list_by_owner(&state.db, user.id, 50).await?;
    Ok(Json(songs.iter().map(SongView::from).collect()))
}

async fn detail(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(id): Path<String>,
) -> ApiResult<Json<SongView>> {
    let song_id = parse_song_id(&id)?;
    let song = melodie_db::songs::find_with_clips(&state.db, song_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    if !authz::can(
        user.role,
        user.id,
        Action::Read,
        Resource::Song {
            owner_id: song.owner_id,
            song_id,
        },
    ) {
        return Err(ApiError::Forbidden);
    }
    Ok(Json(SongView::from(&song)))
}

#[derive(Debug, Deserialize)]
struct RenameRequest {
    title: String,
}

async fn rename(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(id): Path<String>,
    Json(req): Json<RenameRequest>,
) -> ApiResult<Json<SongView>> {
    let song_id = parse_song_id(&id)?;
    let title = req.title.trim();
    if title.is_empty() || title.len() > TITLE_MAX {
        return Err(ApiError::BadRequest(format!(
            "title must be 1-{TITLE_MAX} characters"
        )));
    }
    let song = melodie_db::songs::find_with_clips(&state.db, song_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    if !authz::can(
        user.role,
        user.id,
        Action::Write,
        Resource::Song {
            owner_id: song.owner_id,
            song_id,
        },
    ) {
        return Err(ApiError::Forbidden);
    }
    melodie_db::songs::set_title(&state.db, song_id, title).await?;
    let song = melodie_db::songs::find_with_clips(&state.db, song_id)
        .await?
        .ok_or(ApiError::Internal("song vanished after rename".into()))?;
    Ok(Json(SongView::from(&song)))
}

async fn delete_song(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let song_id = parse_song_id(&id)?;
    let song = melodie_db::songs::find_with_clips(&state.db, song_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    if !authz::can(
        user.role,
        user.id,
        Action::Delete,
        Resource::Song {
            owner_id: song.owner_id,
            song_id,
        },
    ) {
        return Err(ApiError::Forbidden);
    }

    // Best-effort upstream trash. We log on failure and still delete locally —
    // the user's intent is to remove the song from their view, not to keep
    // garbage in our DB if Suno is down.
    if let Some(client) = state.suno.current().await {
        let clip_ids: Vec<String> = song.clips.iter().map(|c| c.id.clone()).collect();
        if !clip_ids.is_empty()
            && let Err(e) = client.delete_clips(&clip_ids).await
        {
            tracing::warn!(error = %e, song_id = %song_id, "suno delete_clips failed, deleting locally anyway");
        }
    }

    melodie_db::songs::delete(&state.db, song_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

// --- helpers ---

fn parse_song_id(s: &str) -> ApiResult<SongId> {
    Uuid::parse_str(s)
        .map(SongId)
        .map_err(|_| ApiError::BadRequest("invalid song id".into()))
}

fn validate_create(req: &CreateSongRequest) -> ApiResult<()> {
    match req {
        CreateSongRequest::Custom(c) => validate_custom(c),
        CreateSongRequest::Describe(d) => validate_describe(d),
    }
}

fn validate_custom(c: &CustomFields) -> ApiResult<()> {
    let title = c.title.trim();
    if title.is_empty() || title.len() > TITLE_MAX {
        return Err(ApiError::BadRequest(format!(
            "title must be 1-{TITLE_MAX} characters"
        )));
    }
    let tags = c.tags.trim();
    if tags.is_empty() || tags.len() > TAGS_MAX {
        return Err(ApiError::BadRequest(format!(
            "tags must be 1-{TAGS_MAX} characters"
        )));
    }
    if let Some(excl) = c.exclude_tags.as_deref()
        && excl.len() > EXCLUDE_MAX
    {
        return Err(ApiError::BadRequest(format!(
            "exclude_tags must be at most {EXCLUDE_MAX} characters"
        )));
    }
    let lyrics = c.lyrics.trim();
    if lyrics.is_empty() || lyrics.len() > LYRICS_MAX {
        return Err(ApiError::BadRequest(format!(
            "lyrics must be 1-{LYRICS_MAX} characters"
        )));
    }
    if let Some(v) = c.weirdness
        && !(0..=100).contains(&v)
    {
        return Err(ApiError::BadRequest("weirdness must be 0-100".into()));
    }
    if let Some(v) = c.style_influence
        && !(0..=100).contains(&v)
    {
        return Err(ApiError::BadRequest(
            "style_influence must be 0-100".into(),
        ));
    }
    if let Some(v) = c.vocal.as_deref()
        && !matches!(v, "male" | "female")
    {
        return Err(ApiError::BadRequest("vocal must be 'male' or 'female'".into()));
    }
    if let Some(v) = c.variation.as_deref()
        && !matches!(v, "high" | "normal" | "subtle")
    {
        return Err(ApiError::BadRequest(
            "variation must be 'high', 'normal' or 'subtle'".into(),
        ));
    }
    Ok(())
}

fn validate_describe(d: &DescribeFields) -> ApiResult<()> {
    let prompt = d.prompt.trim();
    if prompt.is_empty() || prompt.len() > PROMPT_MAX {
        return Err(ApiError::BadRequest(format!(
            "prompt must be 1-{PROMPT_MAX} characters"
        )));
    }
    Ok(())
}
