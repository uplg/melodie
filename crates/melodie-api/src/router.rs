use std::sync::Arc;

use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;
use sqlx::SqlitePool;
use time::Duration;
use tower_http::trace::TraceLayer;
use tower_sessions::cookie::SameSite;
use tower_sessions::{Expiry, SessionManagerLayer};
use tower_sessions_sqlx_store::SqliteStore;

use crate::config::AppConfig;
use crate::poll::SongEvent;
use crate::routes;
use crate::state::AppState;
use crate::suno::SunoBridge;

pub async fn build(
    cfg: &AppConfig,
    pool: SqlitePool,
    suno: Arc<SunoBridge>,
    events: tokio::sync::broadcast::Sender<SongEvent>,
) -> anyhow::Result<Router> {
    let session_store = SqliteStore::new(pool.clone());
    session_store.migrate().await?;

    // Session ID is a random opaque token stored in SQLite — we don't need
    // signed cookies because the cookie carries no user data, only the ID.
    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(cfg.cookie_secure)
        .with_http_only(true)
        .with_same_site(SameSite::Lax)
        .with_name("melodie.sid")
        .with_expiry(Expiry::OnInactivity(Duration::days(30)));

    let state = AppState::new(pool, suno, events);

    Ok(Router::new()
        .route("/healthz", get(|| async { (StatusCode::OK, "ok") }))
        .nest("/api", routes::api_router())
        .layer(TraceLayer::new_for_http())
        .layer(session_layer)
        .with_state(state))
}
