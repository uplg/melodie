use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Redirect;
use axum::routing::{get, post};
use melodie_core::authz::{self, Action, Resource};
use serde::{Deserialize, Serialize};
use suno_client::SunoError;

use crate::error::{ApiError, ApiResult};
use crate::extract::AuthUser;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/clips/{clip_id}/audio", get(audio))
        .route("/clips/{clip_id}/push-to-live", post(push_to_live))
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

#[derive(Debug, Serialize)]
struct PushToLiveResponse {
    title: Option<String>,
    position: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct HomiePushReply {
    #[serde(default)]
    queued: bool,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    position: Option<usize>,
    #[serde(default)]
    error: Option<String>,
}

/// Push a clip's fresh Suno CDN URL to homie's loopback push server, which
/// drops it into the live music queue (same path as a viewer-typed `!yt
/// <direct-url>`). Owner-or-admin gated, same as the audio endpoint.
async fn push_to_live(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(clip_id): Path<String>,
) -> ApiResult<(StatusCode, Json<PushToLiveResponse>)> {
    let push = state
        .homie_push
        .as_ref()
        .ok_or_else(|| ApiError::ServiceUnavailable("push-to-live is not configured".into()))?;

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
    let (audio_url, title) = fresh
        .into_iter()
        .find(|c| c.id == clip_id)
        .map(|c| (c.audio_url, c.title))
        .ok_or_else(|| ApiError::Internal(format!("clip {clip_id} vanished from suno")))?;
    let audio_url = audio_url.ok_or_else(|| {
        ApiError::BadRequest(
            "clip has no audio_url yet — wait until generation is past the streaming step".into(),
        )
    })?;

    let body = serde_json::json!({
        "url": audio_url,
        "requested_by": user.display_name,
        "title": title,
    });
    let reply: HomiePushReply = reqwest::Client::new()
        .post(&push.url)
        .bearer_auth(&push.token)
        .json(&body)
        .send()
        .await
        .map_err(|err| ApiError::Internal(format!("homie push request failed: {err}")))?
        .error_for_status()
        .map_err(|err| {
            // 401/422 from homie surface as 502 here so the client knows the
            // upstream rejected it and can show the message.
            ApiError::Internal(format!("homie push rejected: {err}"))
        })?
        .json()
        .await
        .map_err(|err| ApiError::Internal(format!("homie push response decode: {err}")))?;

    if !reply.queued {
        return Err(ApiError::BadRequest(
            reply.error.unwrap_or_else(|| "homie rejected the push".into()),
        ));
    }

    Ok((
        StatusCode::OK,
        Json(PushToLiveResponse {
            title: reply.title,
            position: reply.position,
        }),
    ))
}
