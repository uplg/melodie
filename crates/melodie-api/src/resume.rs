//! Reconcile in-flight generations after a process restart.
//!
//! Generation runs on an in-memory worker queue that is *not* persisted, so a
//! restart drops any queued or running jobs on the floor. Their song rows are
//! left stuck in `pending`/`generating` forever. We can't recover them, so on
//! boot we mark every such row `failed` with a clear reason.

use melodie_core::model::SongStatus;
use sqlx::SqlitePool;

pub async fn resume_in_flight(pool: SqlitePool) {
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

    let mut buried = 0usize;
    for (song_id, _clip_ids) in in_flight {
        if let Err(e) = melodie_db::songs::set_status(
            &pool,
            song_id,
            SongStatus::Failed,
            Some("interrupted by restart"),
        )
        .await
        {
            tracing::warn!(error = %e, %song_id, "resume: bury failed");
        } else {
            buried += 1;
        }
    }

    // The buried songs' clips are stuck `streaming` too — bury them as well, else the UI keeps
    // showing a forever-"generating" clip (progress bar / partial player) under a failed song.
    // Runs before the worker starts, so every `streaming` clip here is genuinely orphaned.
    if let Err(e) = sqlx::query("UPDATE clips SET status = 'error' WHERE status = 'streaming'")
        .execute(&pool)
        .await
    {
        tracing::warn!(error = %e, "resume: bury in-flight clips failed");
    }

    tracing::info!(buried, "marked interrupted in-flight songs as failed");
}
