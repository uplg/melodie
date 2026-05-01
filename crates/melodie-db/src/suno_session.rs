use sqlx::SqlitePool;

use crate::DbError;

#[derive(Debug, Clone, Default, sqlx::FromRow)]
pub struct SunoSession {
    pub jwt: Option<String>,
    pub session_id: Option<String>,
    pub device_id: Option<String>,
    pub clerk_cookie: Option<String>,
    pub last_status: String,
    pub last_check: Option<String>,
}

pub async fn load(pool: &SqlitePool) -> Result<SunoSession, DbError> {
    let s = sqlx::query_as::<_, SunoSession>(
        "SELECT jwt, session_id, device_id, clerk_cookie, last_status, last_check FROM suno_session WHERE id = 1",
    )
    .fetch_one(pool)
    .await?;
    Ok(s)
}

pub async fn save(
    pool: &SqlitePool,
    jwt: Option<&str>,
    session_id: Option<&str>,
    device_id: Option<&str>,
    clerk_cookie: Option<&str>,
) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE suno_session SET jwt = ?, session_id = ?, device_id = ?, clerk_cookie = ?, \
         last_check = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = 1",
    )
    .bind(jwt)
    .bind(session_id)
    .bind(device_id)
    .bind(clerk_cookie)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn set_status(pool: &SqlitePool, status: &str) -> Result<(), DbError> {
    sqlx::query(
        "UPDATE suno_session SET last_status = ?, last_check = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = 1",
    )
    .bind(status)
    .execute(pool)
    .await?;
    Ok(())
}
