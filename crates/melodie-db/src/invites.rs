use melodie_core::ids::UserId;
use melodie_core::model::Role;
use sqlx::SqlitePool;

use crate::DbError;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Invite {
    pub code: String,
    pub created_by: Option<String>,
    pub used_by: Option<String>,
    pub role: String,
    pub created_at: String,
    pub expires_at: Option<String>,
}

impl Invite {
    pub fn role(&self) -> Role {
        match self.role.as_str() {
            "admin" => Role::Admin,
            _ => Role::Member,
        }
    }
}

pub async fn find(pool: &SqlitePool, code: &str) -> Result<Option<Invite>, DbError> {
    let row = sqlx::query_as::<_, Invite>(
        "SELECT code, created_by, used_by, role, created_at, expires_at FROM invites WHERE code = ?",
    )
    .bind(code)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Inserts an invite if its code is not already present. Returns true if a row
/// was inserted, false if the code already existed (idempotent — used for the
/// bootstrap path where the same code may be supplied across restarts).
pub async fn upsert_idempotent(
    pool: &SqlitePool,
    code: &str,
    created_by: Option<UserId>,
    role: Role,
) -> Result<bool, DbError> {
    let role_str = match role {
        Role::Admin => "admin",
        Role::Member => "member",
    };
    let res = sqlx::query(
        "INSERT INTO invites (code, created_by, role) VALUES (?, ?, ?) ON CONFLICT(code) DO NOTHING",
    )
    .bind(code)
    .bind(created_by.map(|u| u.to_string()))
    .bind(role_str)
    .execute(pool)
    .await?;
    Ok(res.rows_affected() > 0)
}

pub async fn create(
    pool: &SqlitePool,
    code: &str,
    created_by: UserId,
    role: Role,
) -> Result<Invite, DbError> {
    let role_str = match role {
        Role::Admin => "admin",
        Role::Member => "member",
    };
    sqlx::query("INSERT INTO invites (code, created_by, role) VALUES (?, ?, ?)")
        .bind(code)
        .bind(created_by.to_string())
        .bind(role_str)
        .execute(pool)
        .await?;
    find(pool, code)
        .await?
        .ok_or_else(|| DbError::Sqlx(sqlx::Error::RowNotFound))
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct InviteListRow {
    pub code: String,
    pub role: String,
    pub created_at: String,
    pub created_by_name: Option<String>,
    pub used_by_name: Option<String>,
}

/// List invites with their creator/consumer display names. Newest first.
pub async fn list(pool: &SqlitePool) -> Result<Vec<InviteListRow>, DbError> {
    let rows = sqlx::query_as::<_, InviteListRow>(
        "SELECT \
            i.code, \
            i.role, \
            i.created_at, \
            u_creator.display_name AS created_by_name, \
            u_user.display_name AS used_by_name \
         FROM invites i \
         LEFT JOIN users u_creator ON u_creator.id = i.created_by \
         LEFT JOIN users u_user ON u_user.id = i.used_by \
         ORDER BY i.created_at DESC",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Atomically mark the invite as consumed by `user_id`. Returns true if the
/// invite was unused and is now bound to this user, false if it was already
/// consumed (or doesn't exist / is expired).
pub async fn consume(pool: &SqlitePool, code: &str, user_id: UserId) -> Result<bool, DbError> {
    let res = sqlx::query(
        "UPDATE invites SET used_by = ? WHERE code = ? AND used_by IS NULL \
         AND (expires_at IS NULL OR expires_at > strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
    )
    .bind(user_id.to_string())
    .bind(code)
    .execute(pool)
    .await?;
    Ok(res.rows_affected() == 1)
}
