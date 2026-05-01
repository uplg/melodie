use std::sync::Arc;

use tracing_subscriber::EnvFilter;

mod bootstrap;
mod config;
mod error;
mod extract;
mod health;
mod notifier;
mod poll;
mod preflight;
mod router;
mod routes;
mod state;
mod suno;

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
    preflight::check_chrome();

    let suno_bridge = Arc::new(suno::SunoBridge::from_db(pool.clone()).await);
    let notifier = notifier::from_env();

    // Single-publisher shutdown signal that fans out to background tasks.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    // Capacity: at low volume (handful of friends, ~10 concurrent generations)
    // 64 is plenty. Lagged subscribers drop frames silently and re-sync on
    // the next tick anyway.
    let (events_tx, _) = tokio::sync::broadcast::channel::<poll::SongEvent>(64);

    let _health_task = health::spawn(
        suno_bridge.clone(),
        notifier.clone(),
        pool.clone(),
        shutdown_tx.subscribe(),
    );

    let app = router::build(&cfg, pool, suno_bridge, events_tx).await?;
    let listener = tokio::net::TcpListener::bind(cfg.bind).await?;
    tracing::info!(bind = %cfg.bind, "melodie-api listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_tx))
        .await?;
    Ok(())
}

async fn shutdown_signal(shutdown_tx: tokio::sync::broadcast::Sender<()>) {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("SIGTERM received, shutting down"),
        _ = sigint.recv()  => tracing::info!("SIGINT received, shutting down"),
    }
    let _ = shutdown_tx.send(());
}
