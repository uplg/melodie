use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64URL;
use melodie_core::ids::UserId;
use melodie_core::model::Role;
use melodie_db::invites::InviteListRow;
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::extract::AdminUser;
use crate::routes::songs::{DAILY_CAP, SongView};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/admin/invites", get(list_invites).post(create_invite))
        .route("/admin/songs", get(list_all_songs))
        .route("/admin/quotas", get(list_quotas).delete(reset_all_quotas))
        .route(
            "/admin/quotas/{user_id}",
            axum::routing::delete(reset_user_quota),
        )
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

/// Admin feed: every user's songs, newest first. Wraps the regular SongView
/// shape with one extra field (`owner`) so the React island can lean on the
/// existing `<SongCard>` component.
#[derive(Debug, Serialize)]
struct AdminSongOwner {
    id: String,
    display_name: String,
}

#[derive(Debug, Serialize)]
struct AdminSongView {
    owner: AdminSongOwner,
    #[serde(flatten)]
    song: SongView,
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
            owner: AdminSongOwner {
                id: song.owner_id.to_string(),
                display_name: owner_display_name,
            },
            song: SongView::from(&song),
        })
        .collect();
    Ok(Json(out))
}

// --- quotas ---

#[derive(Debug, Serialize)]
struct QuotaView {
    user_id: String,
    display_name: String,
    role: String,
    count_today: u32,
    /// `null` for admins (they bypass the daily cap entirely).
    cap: Option<u32>,
}

async fn list_quotas(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> ApiResult<Json<Vec<QuotaView>>> {
    let rows = melodie_db::quota::list_today_with_users(&state.db).await?;
    let out = rows
        .into_iter()
        .map(|r| QuotaView {
            cap: if r.role == "admin" {
                None
            } else {
                Some(DAILY_CAP)
            },
            user_id: r.user_id,
            display_name: r.display_name,
            role: r.role,
            count_today: r.count.max(0) as u32,
        })
        .collect();
    Ok(Json(out))
}

async fn reset_user_quota(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(user_id): Path<String>,
) -> ApiResult<StatusCode> {
    let id = Uuid::parse_str(&user_id)
        .map(UserId)
        .map_err(|_| ApiError::BadRequest("invalid user id".into()))?;
    melodie_db::quota::reset_user_today(&state.db, id).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn reset_all_quotas(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> ApiResult<StatusCode> {
    melodie_db::quota::reset_all_today(&state.db).await?;
    Ok(StatusCode::NO_CONTENT)
}
