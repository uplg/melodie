//! Resume in-flight generations after a process restart.
//!
//! Poll tasks are tokio::spawn'd futures. They die with the process, leaving
//! their songs stuck in `status = 'generating'` forever. On boot we look for
//! such rows and either respawn a poll loop (if we have clip ids to query)
//! or mark them failed (clip-less rows are unrecoverable — we never inserted
//! clips because the backend crashed between the song INSERT and clip INSERT).

use std::sync::Arc;

use melodie_core::model::SongStatus;
use sqlx::SqlitePool;
use tokio::sync::broadcast;

use crate::poll::{self, SongEvent};
use crate::state::AppState;
use crate::suno::SunoBridge;

pub async fn resume_in_flight(
    pool: SqlitePool,
    suno: Arc<SunoBridge>,
    events: broadcast::Sender<SongEvent>,
) {
    let in_flight = match melodie_db::songs::list_in_flight(&pool).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "resume: failed to list in-flight songs");
            return;
        }
    };
    if in_flight.is_empty() {
        return;
    }

    let state = AppState::new(pool.clone(), suno, events);
    let mut respawned = 0usize;
    let mut buried = 0usize;

    for (song_id, clip_ids) in in_flight {
        if clip_ids.is_empty() {
            // No clips means we can't poll Suno for this row. Likely the
            // backend died between the song INSERT and the clip INSERT.
            // Bury it so it stops haunting `/app`.
            if let Err(e) = melodie_db::songs::set_status(
                &pool,
                song_id,
                SongStatus::Failed,
                Some("resume: no clips, killed mid-create"),
            )
            .await
            {
                tracing::warn!(error = %e, %song_id, "resume: bury failed");
            } else {
                buried += 1;
            }
            continue;
        }
        poll::spawn(state.clone(), song_id, clip_ids);
        respawned += 1;
    }

    tracing::info!(
        respawned,
        buried,
        "resumed in-flight songs after restart"
    );
}
