use axum::Router;
use axum::extract::{Path, State};
use axum::response::Redirect;
use axum::routing::get;
use melodie_core::authz::{self, Action, Resource};
use suno_client::SunoError;

use crate::error::{ApiError, ApiResult};
use crate::extract::AuthUser;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/clips/{clip_id}/audio", get(audio))
}

/// 302 to a freshly-fetched Suno audio URL. We never cache the URL because
/// Suno's pre-signed S3 URLs expire — re-fetching `info` on each click costs
/// one extra round-trip but spares us a refresh layer.
async fn audio(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(clip_id): Path<String>,
) -> ApiResult<Redirect> {
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

    let client = state
        .suno
        .current()
        .await
        .ok_or_else(|| ApiError::Suno(SunoError::AuthMissing))?;
    let fresh = client.get_clips(std::slice::from_ref(&clip_id)).await?;
    let url = fresh
        .into_iter()
        .find(|c| c.id == clip_id)
        .and_then(|c| c.audio_url)
        .ok_or_else(|| {
            ApiError::Internal("clip has no audio_url yet — generation may still be pending".into())
        })?;
    Ok(Redirect::temporary(&url))
}
