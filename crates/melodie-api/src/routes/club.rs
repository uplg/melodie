//! Club proposals: friends flag clips worth archiving on the operator's
//! personal server. The operator reviews them in the admin UI; the actual
//! download/upload of audio is intentionally out of band — the proposals
//! table is just a TODO list.

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use melodie_core::authz::{self, Action, Resource};
use melodie_db::club::{ProposalRow, ProposeOutcome};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::extract::{AdminUser, AuthUser};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/clips/{clip_id}/club", post(propose))
        .route("/club/proposed", get(list_my_proposed))
        .route("/admin/club", get(list_admin))
        .route("/admin/club/{id}", axum::routing::delete(delete_admin))
}

#[derive(Debug, Deserialize)]
struct ProposeRequest {
    #[serde(default)]
    note: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProposeResponse {
    /// `true` when the proposal was just created; `false` when it already
    /// existed (the UI keeps "Proposed ✓" either way — we surface this for
    /// debugging and potential future analytics).
    created: bool,
}

const NOTE_MAX: usize = 500;

async fn propose(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(clip_id): Path<String>,
    body: Option<Json<ProposeRequest>>,
) -> ApiResult<(StatusCode, Json<ProposeResponse>)> {
    let note = body
        .and_then(|Json(b)| b.note)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(ref n) = note
        && n.len() > NOTE_MAX
    {
        return Err(ApiError::BadRequest(format!(
            "note must be at most {NOTE_MAX} characters"
        )));
    }

    let (clip, owner_id) = melodie_db::clips::find_with_song_owner(&state.db, &clip_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    if !authz::can(
        user.role,
        user.id,
        Action::Read,
        Resource::Song {
            owner_id,
            song_id: clip.song_id,
        },
    ) {
        return Err(ApiError::Forbidden);
    }

    let outcome =
        melodie_db::club::propose(&state.db, &clip_id, user.id, note.as_deref()).await?;
    let status = match outcome {
        ProposeOutcome::Created => StatusCode::CREATED,
        ProposeOutcome::AlreadyProposed => StatusCode::OK,
    };
    Ok((
        status,
        Json(ProposeResponse {
            created: matches!(outcome, ProposeOutcome::Created),
        }),
    ))
}

#[derive(Debug, Serialize)]
struct ProposedView {
    proposed_clip_ids: Vec<String>,
}

/// Clip ids the calling user has proposed. Used by the React islands to
/// render the "Proposed ✓" state on each clip card without an extra fetch
/// per card.
async fn list_my_proposed(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
) -> ApiResult<Json<ProposedView>> {
    let ids = melodie_db::club::list_proposed_clip_ids_for_user(&state.db, user.id).await?;
    Ok(Json(ProposedView {
        proposed_clip_ids: ids,
    }))
}

#[derive(Debug, Serialize)]
struct AdminProposalView {
    id: i64,
    clip_id: String,
    note: Option<String>,
    created_at: String,
    song_id: String,
    song_title: Option<String>,
    variant_index: i32,
    clip_duration_s: Option<f64>,
    clip_image_url: Option<String>,
    clip_status: String,
    proposer: AdminUserRef,
    owner: AdminUserRef,
}

#[derive(Debug, Serialize)]
struct AdminUserRef {
    id: String,
    display_name: String,
}

impl From<ProposalRow> for AdminProposalView {
    fn from(r: ProposalRow) -> Self {
        Self {
            id: r.id,
            clip_id: r.clip_id,
            note: r.note,
            created_at: r.created_at,
            song_id: r.song_id,
            song_title: r.song_title,
            variant_index: r.variant_index as i32,
            clip_duration_s: r.clip_duration_s,
            clip_image_url: r.clip_image_url,
            clip_status: r.clip_status,
            proposer: AdminUserRef {
                id: r.proposer_id,
                display_name: r.proposer_display_name,
            },
            owner: AdminUserRef {
                id: r.owner_id,
                display_name: r.owner_display_name,
            },
        }
    }
}

async fn list_admin(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> ApiResult<Json<Vec<AdminProposalView>>> {
    let rows = melodie_db::club::list(&state.db).await?;
    Ok(Json(rows.into_iter().map(AdminProposalView::from).collect()))
}

async fn delete_admin(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(id): Path<i64>,
) -> ApiResult<StatusCode> {
    let removed = melodie_db::club::delete(&state.db, id).await?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::NotFound)
    }
}
