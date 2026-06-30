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
use melodie_core::model::{Song, SongStatus};
use melodie_db::clips::UpsertClip;
use melodie_db::songs::NewSong;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::events::SongEvent;
use crate::extract::AuthUser;
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

// --- field caps ---

const TITLE_MAX: usize = 100;
const TAGS_MAX: usize = 1000;
const LYRICS_MAX: usize = 5000;

// --- request / response views ---

/// Body shape for `POST /api/songs`. The local HeartMuLa engine's only
/// generation inputs are `lyrics`, `styles` (short comma-separated genre/mood
/// tags the user calls "tags") and `language`; `model` records the generator
/// on the row.
#[derive(Debug, Deserialize)]
pub struct CreateSongRequest {
    pub lyrics: String,
    pub styles: String,
    pub language: String,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct SongView {
    pub id: String,
    pub title: Option<String>,
    pub tags: Option<String>,
    pub lyrics: Option<String>,
    pub prompt: Option<String>,
    pub language: String,
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
            title: s.title.clone(),
            tags: s.tags.clone(),
            lyrics: s.lyrics.clone(),
            prompt: s.prompt.clone(),
            language: s.language.clone(),
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

    // Quota check (admin bypass).
    if user.role != melodie_core::model::Role::Admin {
        let new_count = melodie_db::quota::try_increment(&state.db, user.id, DAILY_CAP).await?;
        if new_count.is_none() {
            return Err(ApiError::TooManyRequests);
        }
    }

    // Build the engine tags exactly like the HeartMuLa reference
    // (`server.py:297-298`): a lowercase, comma-joined, space-free list with the
    // language first — `"<language>,<style1>,<style2>,…"`. The raw `styles`
    // string is what we persist as the song's `tags`.
    let language = req.language.trim().to_lowercase();
    let normalized_styles = req
        .styles
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(",");
    let tags = if language.is_empty() {
        normalized_styles
    } else {
        format!("{language},{normalized_styles}")
    };

    // Create the song row (`songs::create` inserts it with `status =
    // 'generating'`). Persist the raw styles as the song's tags plus the
    // chosen language; `lyrics` is handed to the engine unchanged.
    let song_id = melodie_db::songs::create(
        &state.db,
        NewSong {
            owner_id: user.id,
            title: None,
            tags: Some(&req.styles),
            lyrics: Some(&req.lyrics),
            language: &language,
            model: req.model.trim(),
        },
    )
    .await?;
    let lyrics = req.lyrics.clone();

    // One streaming clip up front so the UI has a row to follow; the worker
    // flips it to complete/error when the generation finishes.
    let clip_id = Uuid::new_v4().to_string();
    melodie_db::clips::upsert_many(
        &state.db,
        &[UpsertClip {
            id: clip_id.clone(),
            song_id,
            variant_index: 0,
            status: "streaming".into(),
            duration_s: None,
            image_url: None,
        }],
    )
    .await?;

    // Hand the work to the engine worker thread, which drives the clip/song
    // updates and the SSE broadcast itself.
    if let Err(e) = state.engine.send(crate::engine::EngineJob {
        song_id,
        clip_id,
        lyrics,
        tags,
        max_frames: 2250,
    }) {
        tracing::error!(error = %e, %song_id, "engine: failed to enqueue job");
    }

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
    let styles = req.styles.trim();
    if styles.is_empty() || styles.len() > TAGS_MAX {
        return Err(ApiError::BadRequest(format!(
            "styles must be 1-{TAGS_MAX} characters"
        )));
    }
    let lyrics = req.lyrics.trim();
    if lyrics.is_empty() || lyrics.len() > LYRICS_MAX {
        return Err(ApiError::BadRequest(format!(
            "lyrics must be 1-{LYRICS_MAX} characters"
        )));
    }
    if req.language.len() > TITLE_MAX {
        return Err(ApiError::BadRequest(format!(
            "language must be at most {TITLE_MAX} characters"
        )));
    }
    if req.model.trim().is_empty() {
        return Err(ApiError::BadRequest("model is required".into()));
    }
    Ok(())
}
