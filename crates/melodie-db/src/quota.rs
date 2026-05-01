use chrono::Utc;
use melodie_core::ids::UserId;
use sqlx::SqlitePool;

use crate::DbError;

/// Atomically increment today's quota for `user_id`. Returns:
/// - `Ok(Some(new_count))` if the increment succeeded.
/// - `Ok(None)` if the cap would be exceeded — the row is left untouched.
pub async fn try_increment(
    pool: &SqlitePool,
    user_id: UserId,
    cap: u32,
) -> Result<Option<u32>, DbError> {
    let day_utc = Utc::now().format("%Y-%m-%d").to_string();
    // The DO UPDATE WHERE only fires on conflict; the WHERE filter blocks the
    // increment when count is already at the cap, in which case RETURNING
    // yields no row. The first-of-the-day INSERT path always returns `1`.
    let row: Option<(i64,)> = sqlx::query_as(
        "INSERT INTO generation_quota (user_id, day_utc, count) VALUES (?, ?, 1) \
         ON CONFLICT(user_id, day_utc) DO UPDATE SET count = count + 1 \
         WHERE generation_quota.count < ? \
         RETURNING count",
    )
    .bind(user_id.to_string())
    .bind(&day_utc)
    .bind(cap as i64)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|(c,)| c as u32))
}

pub async fn count_today(pool: &SqlitePool, user_id: UserId) -> Result<u32, DbError> {
    let day_utc = Utc::now().format("%Y-%m-%d").to_string();
    let n: Option<(i64,)> =
        sqlx::query_as("SELECT count FROM generation_quota WHERE user_id = ? AND day_utc = ?")
            .bind(user_id.to_string())
            .bind(&day_utc)
            .fetch_optional(pool)
            .await?;
    Ok(n.map(|(c,)| c as u32).unwrap_or(0))
}
