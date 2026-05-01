use melodie_core::ids::SongId;
use melodie_core::model::Clip;
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::DbError;

#[derive(Debug, sqlx::FromRow)]
struct ClipRow {
    id: String,
    song_id: String,
    variant_index: i64,
    status: String,
    duration_s: Option<f64>,
    image_url: Option<String>,
}

impl ClipRow {
    fn into_domain(self) -> Result<Clip, DbError> {
        let song_id = SongId(
            Uuid::parse_str(&self.song_id)
                .map_err(|e| DbError::Sqlx(sqlx::Error::Decode(Box::new(e))))?,
        );
        Ok(Clip {
            id: self.id,
            song_id,
            variant_index: self.variant_index as i32,
            status: self.status,
            duration_s: self.duration_s,
            image_url: self.image_url,
        })
    }
}

#[derive(Debug, Clone)]
pub struct UpsertClip {
    pub id: String,
    pub song_id: SongId,
    pub variant_index: i32,
    pub status: String,
    pub duration_s: Option<f64>,
    pub image_url: Option<String>,
}

/// Upsert clips by `id`. Suno returns the same clip IDs across the initial
/// `generate` response and subsequent `feed` polls, so we want to update
/// status/duration/image as the generation progresses.
pub async fn upsert_many(pool: &SqlitePool, clips: &[UpsertClip]) -> Result<(), DbError> {
    if clips.is_empty() {
        return Ok(());
    }
    let mut tx = pool.begin().await?;
    for c in clips {
        sqlx::query(
            "INSERT INTO clips (id, song_id, variant_index, status, duration_s, image_url) VALUES (?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET status = excluded.status, duration_s = COALESCE(excluded.duration_s, clips.duration_s), image_url = COALESCE(excluded.image_url, clips.image_url)",
        )
        .bind(&c.id)
        .bind(c.song_id.to_string())
        .bind(c.variant_index as i64)
        .bind(&c.status)
        .bind(c.duration_s)
        .bind(c.image_url.as_deref())
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

pub async fn list_for_songs(pool: &SqlitePool, song_ids: &[SongId]) -> Result<Vec<Clip>, DbError> {
    if song_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = vec!["?"; song_ids.len()].join(",");
    let sql = format!(
        "SELECT id, song_id, variant_index, status, duration_s, image_url FROM clips WHERE song_id IN ({placeholders}) ORDER BY song_id, variant_index"
    );
    let mut q = sqlx::query_as::<_, ClipRow>(&sql);
    for id in song_ids {
        q = q.bind(id.to_string());
    }
    let rows = q.fetch_all(pool).await?;
    rows.into_iter().map(ClipRow::into_domain).collect()
}

pub async fn find_with_song_owner(
    pool: &SqlitePool,
    clip_id: &str,
) -> Result<Option<(Clip, melodie_core::ids::UserId)>, DbError> {
    #[derive(sqlx::FromRow)]
    struct Joined {
        id: String,
        song_id: String,
        variant_index: i64,
        status: String,
        duration_s: Option<f64>,
        image_url: Option<String>,
        owner_id: String,
    }
    let row: Option<Joined> = sqlx::query_as(
        "SELECT clips.id, clips.song_id, clips.variant_index, clips.status, clips.duration_s, clips.image_url, songs.owner_id \
         FROM clips JOIN songs ON songs.id = clips.song_id WHERE clips.id = ?",
    )
    .bind(clip_id)
    .fetch_optional(pool)
    .await?;
    let Some(j) = row else { return Ok(None) };
    let song_id = SongId(
        Uuid::parse_str(&j.song_id)
            .map_err(|e| DbError::Sqlx(sqlx::Error::Decode(Box::new(e))))?,
    );
    let owner = melodie_core::ids::UserId(
        Uuid::parse_str(&j.owner_id)
            .map_err(|e| DbError::Sqlx(sqlx::Error::Decode(Box::new(e))))?,
    );
    let clip = Clip {
        id: j.id,
        song_id,
        variant_index: j.variant_index as i32,
        status: j.status,
        duration_s: j.duration_s,
        image_url: j.image_url,
    };
    Ok(Some((clip, owner)))
}
