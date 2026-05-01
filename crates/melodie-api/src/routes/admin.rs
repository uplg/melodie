use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64URL;
use melodie_core::model::Role;
use melodie_db::invites::InviteListRow;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::extract::AdminUser;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/admin/suno-auth", post(set_suno_auth))
        .route("/admin/health", get(get_health))
        .route("/admin/invites", get(list_invites).post(create_invite))
        .route("/admin/songs", get(list_all_songs))
}

#[derive(Debug, Deserialize)]
struct SunoAuthRequest {
    clerk_cookie: String,
}

async fn set_suno_auth(
    State(state): State<AppState>,
    _admin: AdminUser,
    Json(req): Json<SunoAuthRequest>,
) -> ApiResult<StatusCode> {
    if req.clerk_cookie.trim().is_empty() {
        return Err(ApiError::BadRequest("clerk_cookie must not be empty".into()));
    }
    state.suno.replace_auth(req.clerk_cookie).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Serialize)]
struct HealthView {
    status: String,
    last_check: Option<String>,
    has_jwt: bool,
    has_clerk_cookie: bool,
}

async fn get_health(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> ApiResult<Json<HealthView>> {
    let row = melodie_db::suno_session::load(&state.db)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(HealthView {
        status: row.last_status,
        last_check: row.last_check,
        has_jwt: row.jwt.is_some(),
        has_clerk_cookie: row.clerk_cookie.is_some(),
    }))
}

#[derive(Debug, Serialize)]
struct InviteView {
    code: String,
    role: String,
    created_at: String,
    created_by: Option<String>,
    used_by: Option<String>,
}

impl From<InviteListRow> for InviteView {
    fn from(r: InviteListRow) -> Self {
        Self {
            code: r.code,
            role: r.role,
            created_at: r.created_at,
            created_by: r.created_by_name,
            used_by: r.used_by_name,
        }
    }
}

async fn list_invites(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> ApiResult<Json<Vec<InviteView>>> {
    let rows = melodie_db::invites::list(&state.db).await?;
    Ok(Json(rows.into_iter().map(InviteView::from).collect()))
}

#[derive(Debug, Deserialize)]
struct CreateInviteRequest {
    /// `"member"` (default) or `"admin"`.
    #[serde(default)]
    role: Option<String>,
}

async fn create_invite(
    State(state): State<AppState>,
    AdminUser(admin): AdminUser,
    Json(req): Json<CreateInviteRequest>,
) -> ApiResult<(StatusCode, Json<InviteView>)> {
    let role = match req.role.as_deref() {
        Some("admin") => Role::Admin,
        Some("member") | None => Role::Member,
        Some(other) => {
            return Err(ApiError::BadRequest(format!(
                "role must be 'member' or 'admin', got {other:?}"
            )));
        }
    };
    let mut bytes = [0u8; 24];
    rand::rng().fill_bytes(&mut bytes);
    let code = BASE64URL.encode(bytes);
    let invite = melodie_db::invites::create(&state.db, &code, admin.id, role).await?;
    Ok((
        StatusCode::CREATED,
        Json(InviteView {
            code: invite.code,
            role: invite.role,
            created_at: invite.created_at,
            created_by: Some(admin.display_name),
            used_by: None,
        }),
    ))
}

/// Admin feed: every user's songs, newest first. Reuses the regular SongView
/// shape with one extra field (`owner`) so the React island can lean on the
/// existing `<SongCard>` component.
#[derive(Debug, Serialize)]
struct AdminSongOwner {
    id: String,
    display_name: String,
}

#[derive(Debug, Serialize)]
struct AdminSongView {
    id: String,
    owner: AdminSongOwner,
    mode: String,
    title: Option<String>,
    tags: Option<String>,
    exclude_tags: Option<String>,
    lyrics: Option<String>,
    prompt: Option<String>,
    model: String,
    status: String,
    error: Option<String>,
    created_at: String,
    updated_at: String,
    clips: Vec<AdminClipView>,
}

#[derive(Debug, Serialize)]
struct AdminClipView {
    id: String,
    variant_index: i32,
    status: String,
    duration_s: Option<f64>,
    image_url: Option<String>,
}

const ADMIN_FEED_LIMIT: u32 = 100;

async fn list_all_songs(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> ApiResult<Json<Vec<AdminSongView>>> {
    let rows = melodie_db::songs::list_all_with_owner(&state.db, ADMIN_FEED_LIMIT).await?;
    let out: Vec<AdminSongView> = rows
        .into_iter()
        .map(|(song, owner_display_name)| AdminSongView {
            id: song.id.to_string(),
            owner: AdminSongOwner {
                id: song.owner_id.to_string(),
                display_name: owner_display_name,
            },
            mode: match song.mode {
                melodie_core::model::SongMode::Custom => "custom".into(),
                melodie_core::model::SongMode::Describe => "describe".into(),
            },
            title: song.title,
            tags: song.tags,
            exclude_tags: song.exclude_tags,
            lyrics: song.lyrics,
            prompt: song.prompt,
            model: song.model,
            status: match song.status {
                melodie_core::model::SongStatus::Pending => "pending".into(),
                melodie_core::model::SongStatus::Generating => "generating".into(),
                melodie_core::model::SongStatus::Complete => "complete".into(),
                melodie_core::model::SongStatus::Failed => "failed".into(),
            },
            error: song.error,
            created_at: song.created_at.to_rfc3339(),
            updated_at: song.updated_at.to_rfc3339(),
            clips: song
                .clips
                .into_iter()
                .map(|c| AdminClipView {
                    id: c.id,
                    variant_index: c.variant_index,
                    status: c.status,
                    duration_s: c.duration_s,
                    image_url: c.image_url,
                })
                .collect(),
        })
        .collect();
    Ok(Json(out))
}
