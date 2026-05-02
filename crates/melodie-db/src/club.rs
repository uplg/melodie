//! Club proposals: per-user submissions of clips worth archiving. The
//! operator reviews them in the admin UI, then optionally pulls the audio
//! to a personal server out-of-band.

use melodie_core::ids::UserId;
use sqlx::SqlitePool;

use crate::DbError;

/// Outcome of a `propose` attempt. We treat duplicates as a non-error so the
/// UI can keep "Proposed ✓" idempotent across retries.
#[derive(Debug, PartialEq, Eq)]
pub enum ProposeOutcome {
    Created,
    AlreadyProposed,
}

#[derive(Debug, sqlx::FromRow)]
pub struct ProposalRow {
    pub id: i64,
    pub clip_id: String,
    pub note: Option<String>,
    pub created_at: String,
    pub song_id: String,
    pub song_title: Option<String>,
    pub variant_index: i64,
    pub clip_duration_s: Option<f64>,
    pub clip_image_url: Option<String>,
    pub clip_status: String,
    pub proposer_id: String,
    pub proposer_display_name: String,
    pub owner_id: String,
    pub owner_display_name: String,
}

pub async fn propose(
    pool: &SqlitePool,
    clip_id: &str,
    user_id: UserId,
    note: Option<&str>,
) -> Result<ProposeOutcome, DbError> {
    let result = sqlx::query(
        "INSERT INTO club_proposals (clip_id, user_id, note) VALUES (?, ?, ?) \
         ON CONFLICT(clip_id, user_id) DO NOTHING",
    )
    .bind(clip_id)
    .bind(user_id.to_string())
    .bind(note)
    .execute(pool)
    .await?;
    if result.rows_affected() == 0 {
        Ok(ProposeOutcome::AlreadyProposed)
    } else {
        Ok(ProposeOutcome::Created)
    }
}

pub async fn list(pool: &SqlitePool) -> Result<Vec<ProposalRow>, DbError> {
    let rows: Vec<ProposalRow> = sqlx::query_as(
        "SELECT \
            cp.id, cp.clip_id, cp.note, cp.created_at, \
            clips.song_id, songs.title AS song_title, \
            clips.variant_index, clips.duration_s AS clip_duration_s, \
            clips.image_url AS clip_image_url, clips.status AS clip_status, \
            cp.user_id AS proposer_id, proposer.display_name AS proposer_display_name, \
            songs.owner_id, owner.display_name AS owner_display_name \
         FROM club_proposals cp \
         JOIN clips ON clips.id = cp.clip_id \
         JOIN songs ON songs.id = clips.song_id \
         JOIN users proposer ON proposer.id = cp.user_id \
         JOIN users owner ON owner.id = songs.owner_id \
         ORDER BY cp.created_at DESC",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn delete(pool: &SqlitePool, id: i64) -> Result<bool, DbError> {
    let result = sqlx::query("DELETE FROM club_proposals WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

/// Per-user set of clip ids the user has already proposed. Used to render
/// "Proposed ✓" on the UI without an extra round-trip per card.
pub async fn list_proposed_clip_ids_for_user(
    pool: &SqlitePool,
    user_id: UserId,
) -> Result<Vec<String>, DbError> {
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT clip_id FROM club_proposals WHERE user_id = ?")
            .bind(user_id.to_string())
            .fetch_all(pool)
            .await?;
    Ok(rows.into_iter().map(|(id,)| id).collect())
}
