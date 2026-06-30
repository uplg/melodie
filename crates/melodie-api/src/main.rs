use tracing_subscriber::EnvFilter;

mod bootstrap;
mod config;
mod engine;
mod error;
mod events;
mod extract;
mod resume;
mod router;
mod routes;
mod state;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cfg = config::AppConfig::from_env()?;
    let pool = melodie_db::connect_and_migrate(&cfg.database_url).await?;
    bootstrap::ensure_bootstrap_invite(&pool, cfg.bootstrap_invite.as_deref()).await?;

    // Capacity: at low volume (handful of friends, ~10 concurrent generations)
    // 64 is plenty. Lagged subscribers drop frames silently and re-sync on
    // the next tick anyway.
    let (events_tx, _) = tokio::sync::broadcast::channel::<events::SongEvent>(64);

    // Any song still "generating" was dropped when the previous process died —
    // the worker queue isn't persisted, so bury those rows up front.
    resume::resume_in_flight(pool.clone()).await;

    let (app, engine) = router::build(&cfg, pool, events_tx).await?;
    let listener = tokio::net::TcpListener::bind(cfg.bind).await?;
    tracing::info!(bind = %cfg.bind, "melodie-api listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(engine))
        .await?;
    Ok(())
}

/// The engine can't be interrupted mid-job, so on SIGTERM/SIGINT we stop
/// accepting new connections (axum's graceful shutdown handles that once this
/// future resolves) but don't actually let the process exit until any
/// in-flight generation finishes — otherwise a restart silently discards
/// whatever GPU compute was in progress with no record of it ever existing
/// (the "generating" row only gets buried on the *next* boot, see
/// `resume::resume_in_flight`). Capped so a stuck engine can't block forever.
async fn shutdown_signal(engine: engine::EngineHandle) {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("SIGTERM received, shutting down"),
        _ = sigint.recv()  => tracing::info!("SIGINT received, shutting down"),
    }

    tracing::info!("waiting for in-flight generation to finish (up to 5 min)");
    if tokio::time::timeout(std::time::Duration::from_secs(5 * 60), engine.wait_idle())
        .await
        .is_err()
    {
        tracing::warn!("generation still running after 5 min, exiting anyway");
    }
}
