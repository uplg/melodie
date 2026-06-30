use std::sync::Arc;

use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;
use sqlx::SqlitePool;
use time::Duration;
use tower_http::trace::TraceLayer;
use tower_sessions::cookie::SameSite;
use tower_sessions::{Expiry, MemoryStore, SessionManagerLayer};

use crate::config::AppConfig;
use crate::engine::EngineHandle;
use crate::events::SongEvent;
use crate::routes;
use crate::state::AppState;

/// Returns the router plus the engine handle, so `main.rs` can wait for any
/// in-flight generation to finish before the process actually exits.
pub async fn build(
    cfg: &AppConfig,
    pool: SqlitePool,
    events: tokio::sync::broadcast::Sender<SongEvent>,
) -> anyhow::Result<(Router, EngineHandle)> {
    // In-memory: sessions don't survive a process restart, which is already
    // the norm here — `just live` mints a fresh cloudflared URL (and process)
    // per session, so re-login on restart is expected, not a regression.
    let session_store = MemoryStore::default();

    // Session ID is a random opaque token kept in memory — we don't need
    // signed cookies because the cookie carries no user data, only the ID.
    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(cfg.cookie_secure)
        .with_http_only(true)
        .with_same_site(SameSite::Lax)
        .with_name("melodie.sid")
        .with_expiry(Expiry::OnInactivity(Duration::days(30)));

    let homie_push = cfg.homie_push.clone().map(Arc::new);

    // Local engine: spawn the dedicated generation thread. It loads the model
    // asynchronously (~30s) on its own thread, so the HTTP server starts serving
    // immediately; the first job blocks until the load finishes.
    let audio_dir = Arc::new(cfg.engine.audio_dir.clone());
    let engine = crate::engine::spawn_worker(
        tokio::runtime::Handle::current(),
        pool.clone(),
        events.clone(),
        cfg.engine.engine_cfg.clone(),
        cfg.engine.audio_dir.clone(),
    );

    let engine_handle = engine.clone();
    let state = AppState::new(
        pool,
        events,
        homie_push,
        engine,
        audio_dir,
        Arc::from(cfg.local_base_url.as_str()),
    );

    let router = Router::new()
        .route("/healthz", get(|| async { (StatusCode::OK, "ok") }))
        .nest("/api", routes::api_router())
        .layer(TraceLayer::new_for_http())
        .layer(session_layer)
        .with_state(state);

    Ok((router, engine_handle))
}
