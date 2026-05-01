//! Background polling for in-flight Suno generations.
//!
//! When a `POST /api/songs` succeeds, we spawn one of these tasks per song.
//! It polls Suno's `/api/feed/?ids=...` until every clip is terminal
//! (`complete` or `error`) or a 10-minute deadline lapses, persisting clip
//! state on each tick and broadcasting [`SongEvent`] updates to anyone
//! subscribed via the SSE endpoint.
//!
//! Polling delay backs off from 3 → 15 seconds to mirror upstream's pacing
//! on `/api/feed/`. The DB writes are best-effort: a transient failure
//! doesn't bring down the loop.

use std::time::{Duration, Instant};

use melodie_core::ids::SongId;
use melodie_core::model::{Song, SongStatus};
use melodie_db::clips::UpsertClip;
use serde::Serialize;
use suno_client::SunoError;
use tokio::task::JoinHandle;

use crate::state::AppState;

/// Wire event broadcast through the SSE endpoint. Cheap to clone.
#[derive(Debug, Clone, Serialize)]
pub struct SongEvent {
    pub song_id: String,
    pub status: String,
    pub clips: Vec<ClipEventView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClipEventView {
    pub id: String,
    pub variant_index: i32,
    pub status: String,
    pub duration_s: Option<f64>,
    pub image_url: Option<String>,
}

impl SongEvent {
    pub fn from_song(song: &Song) -> Self {
        Self {
            song_id: song.id.to_string(),
            status: song_status_str(song.status).to_string(),
            clips: song
                .clips
                .iter()
                .map(|c| ClipEventView {
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

pub fn spawn(state: AppState, song_id: SongId, clip_ids: Vec<String>) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(e) = run(&state, song_id, &clip_ids).await {
            tracing::warn!(error = %e, %song_id, "poll task ended with error");
        }
    })
}

const POLL_DEADLINE: Duration = Duration::from_secs(10 * 60);
const POLL_DELAY_MIN: Duration = Duration::from_secs(3);
const POLL_DELAY_MAX: Duration = Duration::from_secs(15);

async fn run(state: &AppState, song_id: SongId, clip_ids: &[String]) -> Result<(), SunoError> {
    let deadline = Instant::now() + POLL_DEADLINE;
    let mut delay = POLL_DELAY_MIN;
    let mut last_status = SongStatus::Generating;

    while Instant::now() < deadline {
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(POLL_DELAY_MAX);

        let Some(client) = state.suno.current().await else {
            // No active client — operator hasn't re-upped. Keep waiting in
            // case they re-up mid-generation.
            tracing::debug!(%song_id, "poll: no Suno client, retrying");
            continue;
        };

        let suno_clips = match client.get_clips(clip_ids).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, %song_id, "poll: get_clips failed");
                continue;
            }
        };

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
        if let Err(e) = melodie_db::clips::upsert_many(&state.db, &upserts).await {
            tracing::warn!(error = %e, %song_id, "poll: clip upsert failed");
        }

        let new_status = aggregate(&suno_clips);
        if new_status != last_status {
            if let Err(e) =
                melodie_db::songs::set_status(&state.db, song_id, new_status, None).await
            {
                tracing::warn!(error = %e, %song_id, "poll: song status update failed");
            }
            last_status = new_status;
        }

        let event = SongEvent {
            song_id: song_id.to_string(),
            status: song_status_str(new_status).to_string(),
            clips: suno_clips
                .iter()
                .enumerate()
                .map(|(i, c)| ClipEventView {
                    id: c.id.clone(),
                    variant_index: i as i32,
                    status: c.status.clone(),
                    duration_s: c.metadata.duration,
                    image_url: c.image_url.clone(),
                })
                .collect(),
        };
        // No subscribers is normal — drop the error.
        let _ = state.events.send(event);

        if matches!(new_status, SongStatus::Complete | SongStatus::Failed) {
            return Ok(());
        }
    }

    // Deadline hit — leave a clear marker on the song row.
    let _ = melodie_db::songs::set_status(
        &state.db,
        song_id,
        SongStatus::Failed,
        Some("polling deadline exceeded"),
    )
    .await;
    let _ = state.events.send(SongEvent {
        song_id: song_id.to_string(),
        status: "failed".into(),
        clips: Vec::new(),
    });
    Ok(())
}

fn aggregate(clips: &[suno_client::types::Clip]) -> SongStatus {
    if clips.is_empty() {
        return SongStatus::Generating;
    }
    let all_terminal = clips
        .iter()
        .all(|c| matches!(c.status.as_str(), "complete" | "error"));
    if !all_terminal {
        return SongStatus::Generating;
    }
    let any_complete = clips.iter().any(|c| c.status == "complete");
    if any_complete {
        SongStatus::Complete
    } else {
        SongStatus::Failed
    }
}

fn song_status_str(s: SongStatus) -> &'static str {
    match s {
        SongStatus::Pending => "pending",
        SongStatus::Generating => "generating",
        SongStatus::Complete => "complete",
        SongStatus::Failed => "failed",
    }
}
