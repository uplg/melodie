use chrono::{DateTime, Utc};
use melodie_core::ids::{SongId, UserId};
use melodie_core::model::{Clip, Song, SongMode, SongStatus};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::DbError;

#[derive(Debug, sqlx::FromRow)]
struct SongRow {
    id: String,
    owner_id: String,
    mode: String,
    title: Option<String>,
    tags: Option<String>,
    exclude_tags: Option<String>,
    lyrics: Option<String>,
    prompt: Option<String>,
    model: String,
    status: String,
    error: Option<String>,
    created_at: String,
    updated_at: String,
}

impl SongRow {
    fn into_domain(self, clips: Vec<Clip>) -> Result<Song, DbError> {
        let id = SongId(parse_uuid(&self.id)?);
        let owner_id = UserId(parse_uuid(&self.owner_id)?);
        let mode = match self.mode.as_str() {
            "describe" => SongMode::Describe,
            _ => SongMode::Custom,
        };
        let status = parse_song_status(&self.status);
        Ok(Song {
            id,
            owner_id,
            mode,
            title: self.title,
            tags: self.tags,
            exclude_tags: self.exclude_tags,
            lyrics: self.lyrics,
            prompt: self.prompt,
            model: self.model,
            status,
            error: self.error,
            created_at: parse_ts(&self.created_at),
            updated_at: parse_ts(&self.updated_at),
            clips,
        })
    }
}

fn parse_uuid(s: &str) -> Result<Uuid, DbError> {
    Uuid::parse_str(s).map_err(|e| DbError::Sqlx(sqlx::Error::Decode(Box::new(e))))
}

fn parse_ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn parse_song_status(s: &str) -> SongStatus {
    match s {
        "pending" => SongStatus::Pending,
        "generating" => SongStatus::Generating,
        "complete" => SongStatus::Complete,
        "failed" => SongStatus::Failed,
        _ => SongStatus::Pending,
    }
}

pub fn song_status_str(s: SongStatus) -> &'static str {
    match s {
        SongStatus::Pending => "pending",
        SongStatus::Generating => "generating",
        SongStatus::Complete => "complete",
        SongStatus::Failed => "failed",
    }
}

pub struct NewSong<'a> {
    pub owner_id: UserId,
    pub mode: SongMode,
    pub title: Option<&'a str>,
    pub tags: Option<&'a str>,
    pub exclude_tags: Option<&'a str>,
    pub lyrics: Option<&'a str>,
    pub prompt: Option<&'a str>,
    pub vocal: Option<&'a str>,
    pub weirdness: Option<i32>,
    pub style_inf: Option<i32>,
    pub variation: Option<&'a str>,
    pub model: &'a str,
}

pub async fn create(pool: &SqlitePool, new: NewSong<'_>) -> Result<SongId, DbError> {
    let id = SongId::new();
    let mode_str = match new.mode {
        SongMode::Custom => "custom",
        SongMode::Describe => "describe",
    };
    sqlx::query(
        "INSERT INTO songs (id, owner_id, mode, title, tags, exclude_tags, lyrics, prompt, vocal, weirdness, style_inf, variation, model, status) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'generating')",
    )
    .bind(id.to_string())
    .bind(new.owner_id.to_string())
    .bind(mode_str)
    .bind(new.title)
    .bind(new.tags)
    .bind(new.exclude_tags)
    .bind(new.lyrics)
    .bind(new.prompt)
    .bind(new.vocal)
    .bind(new.weirdness)
    .bind(new.style_inf)
    .bind(new.variation)
    .bind(new.model)
    .execute(pool)
    .await?;
    // Mirror ownership in the relations table to keep the ReBAC model honest.
    sqlx::query(
        "INSERT INTO relations (subject_type, subject_id, relation, object_type, object_id) VALUES ('user', ?, 'owner', 'song', ?)",
    )
    .bind(new.owner_id.to_string())
    .bind(id.to_string())
    .execute(pool)
    .await?;
    Ok(id)
}

pub async fn set_status(
    pool: &SqlitePool,
    song_id: SongId,
    status: SongStatus,
    error: Option<&str>,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE songs SET status = ?, error = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
    )
    .bind(song_status_str(status))
    .bind(error)
    .bind(song_id.to_string())
    .execute(pool)
    .await?;
    Ok(())
}

/// Set the title only when the row currently has none. Used by the poll task
/// to lift Suno-generated titles into describe-mode rows that started with
/// `title = NULL`. Idempotent and cheap. Returns rows affected.
pub async fn set_title_if_missing(
    pool: &SqlitePool,
    song_id: SongId,
    title: &str,
) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE songs SET title = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
         WHERE id = ? AND (title IS NULL OR title = '')",
    )
    .bind(title)
    .bind(song_id.to_string())
    .execute(pool)
    .await?;
    Ok(r.rows_affected())
}

/// Unconditional title set, used by the manual rename endpoint.
pub async fn set_title(
    pool: &SqlitePool,
    song_id: SongId,
    title: &str,
) -> Result<u64, DbError> {
    let r = sqlx::query(
        "UPDATE songs SET title = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
    )
    .bind(title)
    .bind(song_id.to_string())
    .execute(pool)
    .await?;
    Ok(r.rows_affected())
}

pub async fn find_with_clips(
    pool: &SqlitePool,
    song_id: SongId,
) -> Result<Option<Song>, DbError> {
    let row: Option<SongRow> = sqlx::query_as(
        "SELECT id, owner_id, mode, title, tags, exclude_tags, lyrics, prompt, model, status, error, created_at, updated_at FROM songs WHERE id = ?",
    )
    .bind(song_id.to_string())
    .fetch_optional(pool)
    .await?;
    let Some(row) = row else { return Ok(None) };
    let clips = crate::clips::list_for_songs(pool, &[song_id]).await?;
    Ok(Some(row.into_domain(clips)?))
}

pub async fn list_by_owner(
    pool: &SqlitePool,
    owner_id: UserId,
    limit: u32,
) -> Result<Vec<Song>, DbError> {
    let rows: Vec<SongRow> = sqlx::query_as(
        "SELECT id, owner_id, mode, title, tags, exclude_tags, lyrics, prompt, model, status, error, created_at, updated_at FROM songs WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
    )
    .bind(owner_id.to_string())
    .bind(limit as i64)
    .fetch_all(pool)
    .await?;

    let song_ids: Vec<SongId> = rows
        .iter()
        .filter_map(|r| Uuid::parse_str(&r.id).ok().map(SongId))
        .collect();
    let mut clips = crate::clips::list_for_songs(pool, &song_ids).await?;
    // Group by song_id (stable iteration order — small N).
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let sid = SongId(parse_uuid(&row.id)?);
        let song_clips: Vec<Clip> = clips.extract_if(.., |c| c.song_id == sid).collect();
        out.push(row.into_domain(song_clips)?);
    }
    Ok(out)
}

pub async fn delete(pool: &SqlitePool, song_id: SongId) -> Result<u64, DbError> {
    let res = sqlx::query("DELETE FROM songs WHERE id = ?")
        .bind(song_id.to_string())
        .execute(pool)
        .await?;
    Ok(res.rows_affected())
}

/// Songs whose poll task should still be running but isn't (process restart).
/// Returns `(SongId, clip_ids)` so the caller can respawn `poll::spawn`.
/// Clip-less rows are returned with an empty `clip_ids` so the caller can
/// mark them failed — they can't be polled and would otherwise stay stuck.
pub async fn list_in_flight(
    pool: &SqlitePool,
) -> Result<Vec<(SongId, Vec<String>)>, DbError> {
    #[derive(sqlx::FromRow)]
    struct Row {
        song_id: String,
        clip_id: Option<String>,
    }
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT s.id AS song_id, c.id AS clip_id \
         FROM songs s \
         LEFT JOIN clips c ON c.song_id = s.id \
         WHERE s.status IN ('pending', 'generating') \
         ORDER BY s.created_at, c.variant_index",
    )
    .fetch_all(pool)
    .await?;

    let mut grouped: Vec<(SongId, Vec<String>)> = Vec::new();
    for r in rows {
        let sid = SongId(parse_uuid(&r.song_id)?);
        match grouped.last_mut() {
            Some((existing_id, clips)) if *existing_id == sid => {
                if let Some(c) = r.clip_id {
                    clips.push(c);
                }
            }
            _ => {
                let mut clips = Vec::new();
                if let Some(c) = r.clip_id {
                    clips.push(c);
                }
                grouped.push((sid, clips));
            }
        }
    }
    Ok(grouped)
}

#[derive(Debug, sqlx::FromRow)]
struct SongWithOwnerRow {
    id: String,
    owner_id: String,
    mode: String,
    title: Option<String>,
    tags: Option<String>,
    exclude_tags: Option<String>,
    lyrics: Option<String>,
    prompt: Option<String>,
    model: String,
    status: String,
    error: Option<String>,
    created_at: String,
    updated_at: String,
    owner_display_name: String,
}

/// Admin-only listing across all users. Returns `(Song, owner_display_name)`
/// pairs so the API layer can shape its own view without leaking the owner's
/// email. Newest first.
pub async fn list_all_with_owner(
    pool: &SqlitePool,
    limit: u32,
) -> Result<Vec<(Song, String)>, DbError> {
    let rows: Vec<SongWithOwnerRow> = sqlx::query_as(
        "SELECT s.id, s.owner_id, s.mode, s.title, s.tags, s.exclude_tags, \
                s.lyrics, s.prompt, s.model, s.status, s.error, \
                s.created_at, s.updated_at, \
                u.display_name AS owner_display_name \
         FROM songs s \
         JOIN users u ON u.id = s.owner_id \
         ORDER BY s.created_at DESC \
         LIMIT ?",
    )
    .bind(limit as i64)
    .fetch_all(pool)
    .await?;

    let song_ids: Vec<SongId> = rows
        .iter()
        .filter_map(|r| Uuid::parse_str(&r.id).ok().map(SongId))
        .collect();
    let mut clips = crate::clips::list_for_songs(pool, &song_ids).await?;

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let sid = SongId(parse_uuid(&row.id)?);
        let song_clips: Vec<Clip> = clips.extract_if(.., |c| c.song_id == sid).collect();
        let owner_name = row.owner_display_name.clone();
        let song_row = SongRow {
            id: row.id,
            owner_id: row.owner_id,
            mode: row.mode,
            title: row.title,
            tags: row.tags,
            exclude_tags: row.exclude_tags,
            lyrics: row.lyrics,
            prompt: row.prompt,
            model: row.model,
            status: row.status,
            error: row.error,
            created_at: row.created_at,
            updated_at: row.updated_at,
        };
        out.push((song_row.into_domain(song_clips)?, owner_name));
    }
    Ok(out)
}
