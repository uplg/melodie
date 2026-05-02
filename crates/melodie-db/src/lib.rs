//! SQLite pool, migrations, and repositories.
//!
//! Migrations live in `crates/melodie-db/migrations/` and are applied on startup
//! via [`connect_and_migrate`].

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use std::str::FromStr;

pub mod clips;
pub mod club;
pub mod invites;
pub mod quota;
pub mod songs;
pub mod suno_session;
pub mod users;

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    Migrate(#[from] sqlx::migrate::MigrateError),
}

pub async fn connect_and_migrate(database_url: &str) -> Result<SqlitePool, DbError> {
    let opts = SqliteConnectOptions::from_str(database_url)?
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
        .foreign_keys(true)
        .busy_timeout(std::time::Duration::from_secs(5));

    // SQLite's `create_if_missing` covers the file but NOT its parent dir, so
    // a default path like `./data/melodie.db` blows up when `./data` doesn't
    // exist yet. Make sure the directory is there before we connect.
    let filename = opts.get_filename();
    if let Some(parent) = filename.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            DbError::Sqlx(sqlx::Error::Configuration(
                format!("could not create database parent dir {parent:?}: {e}").into(),
            ))
        })?;
    }

    let pool = SqlitePoolOptions::new()
        .max_connections(8)
        .connect_with(opts)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;
    Ok(pool)
}
