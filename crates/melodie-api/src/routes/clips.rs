use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use axum::routing::{get, post};
use melodie_core::authz::{self, Action, Resource};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::extract::AuthUser;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/clips/{clip_id}/audio", get(audio))
        .route("/clips/{clip_id}/push-to-live", post(push_to_live))
}

/// Serve a clip's generated audio: stream `{audio_dir}/{clip_id}.mp3` from disk
/// with `Content-Type: audio/mpeg`, 404 if the file isn't there yet (still
/// generating, or generation failed).
async fn audio(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    Path(clip_id): Path<String>,
) -> ApiResult<Response> {
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

    let path = state.audio_dir.join(format!("{clip_id}.mp3"));
    let bytes = tokio::fs::read(&path).await.map_err(|_| ApiError::NotFound)?;
    let resp = Response::builder()
        .header(header::CONTENT_TYPE, "audio/mpeg")
        .body(Body::from(bytes))
        .map_err(|e| ApiError::Internal(format!("audio response build: {e}")))?;
    Ok(resp)
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

/// Push a clip's audio to homie's loopback push server, which drops it into the
/// live music queue (same path as a viewer-typed `!yt <direct-url>`). We hand
/// homie a URL pointing back at this server's own clip-audio endpoint, built
/// from the request `Host`. Owner-or-admin gated, same as the audio endpoint.
async fn push_to_live(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    headers: HeaderMap,
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

    // Make sure the audio actually exists on disk before we hand homie a URL.
    let path = state.audio_dir.join(format!("{clip_id}.mp3"));
    if tokio::fs::metadata(&path).await.is_err() {
        return Err(ApiError::BadRequest(
            "clip has no audio yet — wait until generation finishes".into(),
        ));
    }

    let song = melodie_db::songs::find_with_clips(&state.db, clip.song_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    let title = song.title.clone();

    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError::BadRequest("missing Host header".into()))?;
    let audio_url = format!("http://{host}/api/clips/{clip_id}/audio");

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
