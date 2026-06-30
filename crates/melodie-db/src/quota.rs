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

/// Undo a `try_increment` that turned out not to consume a real generation
/// (e.g. the engine never picked up the job). No-ops if there's no row for
/// today, or if the count is already 0.
pub async fn decrement(pool: &SqlitePool, user_id: UserId) -> Result<(), DbError> {
    let day_utc = Utc::now().format("%Y-%m-%d").to_string();
    sqlx::query(
        "UPDATE generation_quota SET count = MAX(count - 1, 0) WHERE user_id = ? AND day_utc = ?",
    )
    .bind(user_id.to_string())
    .bind(&day_utc)
    .execute(pool)
    .await?;
    Ok(())
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

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct QuotaRow {
    pub user_id: String,
    pub display_name: String,
    pub role: String,
    pub count: i64,
}

/// Today's quota state for every user, including those with zero generations.
/// LEFT JOIN so a user who hasn't generated yet today still appears with `count=0`.
pub async fn list_today_with_users(pool: &SqlitePool) -> Result<Vec<QuotaRow>, DbError> {
    let day_utc = Utc::now().format("%Y-%m-%d").to_string();
    let rows = sqlx::query_as::<_, QuotaRow>(
        "SELECT u.id AS user_id, u.display_name, u.role, COALESCE(q.count, 0) AS count \
         FROM users u \
         LEFT JOIN generation_quota q ON q.user_id = u.id AND q.day_utc = ? \
         ORDER BY count DESC, u.display_name COLLATE NOCASE",
    )
    .bind(&day_utc)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Drop today's quota row for `user_id`. Returns rows affected (0 if the user
/// hadn't generated yet today, 1 otherwise).
pub async fn reset_user_today(pool: &SqlitePool, user_id: UserId) -> Result<u64, DbError> {
    let day_utc = Utc::now().format("%Y-%m-%d").to_string();
    let r = sqlx::query("DELETE FROM generation_quota WHERE user_id = ? AND day_utc = ?")
        .bind(user_id.to_string())
        .bind(&day_utc)
        .execute(pool)
        .await?;
    Ok(r.rows_affected())
}

/// Drop today's quota row for every user.
pub async fn reset_all_today(pool: &SqlitePool) -> Result<u64, DbError> {
    let day_utc = Utc::now().format("%Y-%m-%d").to_string();
    let r = sqlx::query("DELETE FROM generation_quota WHERE day_utc = ?")
        .bind(&day_utc)
        .execute(pool)
        .await?;
    Ok(r.rows_affected())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn insert_user(pool: &SqlitePool, id: UserId) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO users (id, email, display_name, password_hash) VALUES (?, ?, 'tester', 'x')",
        )
        .bind(id.to_string())
        .bind(format!("{id}@test.invalid"))
        .execute(pool)
        .await?;
        Ok(())
    }

    #[sqlx::test]
    async fn try_increment_under_cap_succeeds(pool: SqlitePool) -> Result<(), DbError> {
        let user = UserId::new();
        insert_user(&pool, user).await?;
        assert_eq!(try_increment(&pool, user, 4).await?, Some(1));
        assert_eq!(try_increment(&pool, user, 4).await?, Some(2));
        Ok(())
    }

    #[sqlx::test]
    async fn try_increment_blocks_at_cap(pool: SqlitePool) -> Result<(), DbError> {
        let user = UserId::new();
        insert_user(&pool, user).await?;
        assert_eq!(try_increment(&pool, user, 1).await?, Some(1));
        assert_eq!(try_increment(&pool, user, 1).await?, None);
        assert_eq!(count_today(&pool, user).await?, 1);
        Ok(())
    }

    /// The bug this guards against: two requests racing the same (user, day)
    /// row at cap=1 must not both be told they won — that would let a member
    /// generate twice on a daily cap of one.
    #[sqlx::test]
    async fn try_increment_race_has_exactly_one_winner(pool: SqlitePool) -> Result<(), DbError> {
        let user = UserId::new();
        insert_user(&pool, user).await?;
        let cap = 1;

        let (a, b) = tokio::join!(
            try_increment(&pool, user, cap),
            try_increment(&pool, user, cap),
        );
        let winners = [a, b]
            .into_iter()
            .filter(|r| matches!(r, Ok(Some(_))))
            .count();
        assert_eq!(
            winners, 1,
            "exactly one concurrent increment should succeed at cap=1"
        );
        assert_eq!(count_today(&pool, user).await?, 1);
        Ok(())
    }
}
