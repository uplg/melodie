use chrono::{DateTime, Utc};
use melodie_core::ids::UserId;
use melodie_core::model::{Role, User};
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::DbError;

#[derive(Debug, sqlx::FromRow)]
struct UserRow {
    id: String,
    email: String,
    display_name: String,
    password_hash: String,
    role: String,
    created_at: String,
}

impl UserRow {
    fn into_domain(self) -> Result<(User, String), DbError> {
        let id = UserId(
            Uuid::parse_str(&self.id)
                .map_err(|e| DbError::Sqlx(sqlx::Error::Decode(Box::new(e))))?,
        );
        let role = match self.role.as_str() {
            "admin" => Role::Admin,
            _ => Role::Member,
        };
        let created_at = DateTime::parse_from_rfc3339(&self.created_at)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        Ok((
            User {
                id,
                email: self.email,
                display_name: self.display_name,
                role,
                created_at,
            },
            self.password_hash,
        ))
    }
}

pub async fn count(pool: &SqlitePool) -> Result<i64, DbError> {
    let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM users")
        .fetch_one(pool)
        .await?;
    Ok(n)
}

pub async fn find_by_email(
    pool: &SqlitePool,
    email: &str,
) -> Result<Option<(User, String)>, DbError> {
    let row: Option<UserRow> =
        sqlx::query_as("SELECT id, email, display_name, password_hash, role, created_at FROM users WHERE email = ?")
            .bind(email)
            .fetch_optional(pool)
            .await?;
    row.map(UserRow::into_domain).transpose()
}

pub async fn find_by_id(pool: &SqlitePool, id: UserId) -> Result<Option<User>, DbError> {
    let row: Option<UserRow> =
        sqlx::query_as("SELECT id, email, display_name, password_hash, role, created_at FROM users WHERE id = ?")
            .bind(id.to_string())
            .fetch_optional(pool)
            .await?;
    Ok(row.map(UserRow::into_domain).transpose()?.map(|(u, _)| u))
}

pub struct NewUser<'a> {
    pub email: &'a str,
    pub display_name: &'a str,
    pub password_hash: &'a str,
    pub role: Role,
}

pub async fn create(pool: &SqlitePool, new: NewUser<'_>) -> Result<User, DbError> {
    let id = UserId::new();
    let role_str = match new.role {
        Role::Admin => "admin",
        Role::Member => "member",
    };
    sqlx::query(
        "INSERT INTO users (id, email, display_name, password_hash, role) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(id.to_string())
    .bind(new.email)
    .bind(new.display_name)
    .bind(new.password_hash)
    .bind(role_str)
    .execute(pool)
    .await?;

    find_by_id(pool, id)
        .await?
        .ok_or_else(|| DbError::Sqlx(sqlx::Error::RowNotFound))
}
