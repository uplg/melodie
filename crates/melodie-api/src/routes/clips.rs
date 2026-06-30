use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use axum::routing::{get, post};
use melodie_core::authz::Action;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::AsyncReadExt;

use crate::error::{ApiError, ApiResult};
use crate::extract::{AuthUser, require_song_access};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/clips/{clip_id}/audio", get(audio))
        .route("/clips/{clip_id}/push-to-live", post(push_to_live))
}

/// Serve a clip's mp3. While the clip is still `streaming`, tail the growing file (chunked,
/// until generation completes) so the browser can play it mid-generation; once it's finished,
/// serve the whole file with HTTP Range support for seeking. Generating-but-no-bytes-yet ⇒ 202;
/// failed/missing ⇒ 404.
async fn audio(
    State(state): State<AppState>,
    AuthUser(user): AuthUser,
    headers: HeaderMap,
    Path(clip_id): Path<String>,
) -> ApiResult<Response> {
    let (clip, owner_id) = melodie_db::clips::find_with_song_owner(&state.db, &clip_id)
        .await?
        .ok_or(ApiError::NotFound)?;
    require_song_access(&user, owner_id, clip.song_id, Action::Read)?;

    let path = state.audio_dir.join(format!("{clip_id}.mp3"));
    let exists = tokio::fs::metadata(&path).await.is_ok();
    match (clip.status.as_str(), exists) {
        ("streaming", true) => serve_tail(state.db.clone(), clip_id, path).await,
        ("streaming", false) => Response::builder()
            .status(StatusCode::ACCEPTED)
            .body(Body::empty())
            .map_err(|e| ApiError::Internal(format!("audio 202: {e}"))),
        (_, true) => serve_range(&path, &headers).await,
        (_, false) => Err(ApiError::NotFound),
    }
}

/// Serve a finished file, honouring a `Range:` request with `206 Partial Content` for seeking.
async fn serve_range(path: &std::path::Path, headers: &HeaderMap) -> ApiResult<Response> {
    let data = tokio::fs::read(path).await.map_err(|_| ApiError::NotFound)?;
    let len = data.len() as u64;
    if let Some((start, end)) = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|r| parse_range(r, len))
    {
        let slice = data[start as usize..=end as usize].to_vec();
        return Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, "audio/mpeg")
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{len}"))
            .body(Body::from(slice))
            .map_err(|e| ApiError::Internal(format!("audio range: {e}")));
    }
    Response::builder()
        .header(header::CONTENT_TYPE, "audio/mpeg")
        .header(header::ACCEPT_RANGES, "bytes")
        .body(Body::from(data))
        .map_err(|e| ApiError::Internal(format!("audio response: {e}")))
}

/// Parse `bytes=START-END` (either end optional) into an inclusive `(start, end)` within `len`.
fn parse_range(raw: &str, len: u64) -> Option<(u64, u64)> {
    if len == 0 {
        return None;
    }
    let (s, e) = raw.strip_prefix("bytes=")?.split_once('-')?;
    let start: u64 = if s.is_empty() { 0 } else { s.parse().ok()? };
    let end: u64 = if e.is_empty() { len - 1 } else { e.parse::<u64>().ok()?.min(len - 1) };
    (start <= end).then_some((start, end))
}

/// Stream a still-growing mp3: send what's on disk, and at EOF poll the clip status — if it's
/// still `streaming`, wait and read more; once it leaves `streaming` we've sent everything.
async fn serve_tail(db: SqlitePool, clip_id: String, path: PathBuf) -> ApiResult<Response> {
    let stream = async_stream::stream! {
        let mut file = match tokio::fs::File::open(&path).await {
            Ok(f) => f,
            Err(e) => { yield Err(e); return; }
        };
        let mut idle = 0u32;
        loop {
            let mut buf = vec![0u8; 64 * 1024];
            match file.read(&mut buf).await {
                Ok(0) => {
                    // Nothing more right now — is generation still running?
                    let still = matches!(
                        melodie_db::clips::find_with_song_owner(&db, &clip_id).await,
                        Ok(Some((c, _))) if c.status == "streaming"
                    );
                    if !still || idle > 2000 {
                        break; // complete/failed, or a ~10 min safety cap (2000 × 300 ms)
                    }
                    idle += 1;
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
                Ok(n) => {
                    idle = 0;
                    buf.truncate(n);
                    yield Ok(Bytes::from(buf));
                }
                Err(e) => { yield Err(e); break; }
            }
        }
    };
    Response::builder()
        .header(header::CONTENT_TYPE, "audio/mpeg")
        .body(Body::from_stream(stream))
        .map_err(|e| ApiError::Internal(format!("audio stream: {e}")))
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
    require_song_access(&user, owner_id, clip.song_id, Action::Read)?;

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
