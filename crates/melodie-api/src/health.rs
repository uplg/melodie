//! Background loop that polls the Suno upstream every `HEALTH_INTERVAL` to
//! check the JWT is still good. Persists `last_status` to the DB and pings
//! Telegram on transitions to/from `expired`.

use std::sync::Arc;
use std::time::Duration;

use melodie_core::notif::Notifier;
use sqlx::SqlitePool;
use tokio::time::{MissedTickBehavior, interval};

use crate::suno::SunoBridge;

const HEALTH_INTERVAL: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Ok,
    Expired,
    Missing,
    Degraded,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Expired => "expired",
            Self::Missing => "missing",
            Self::Degraded => "degraded",
        }
    }
}

pub fn spawn(
    bridge: Arc<SunoBridge>,
    notifier: Arc<dyn Notifier>,
    pool: SqlitePool,
    mut shutdown: tokio::sync::broadcast::Receiver<()>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = interval(HEALTH_INTERVAL);
        // Skip burst-firing if we miss a tick window (e.g. system suspended).
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

        let mut last_status: Option<Status> = None;
        // Run an immediate check so the operator sees the verdict in the
        // first 15-minute window rather than after one.
        run_once(&bridge, &notifier, &pool, &mut last_status).await;

        loop {
            tokio::select! {
                _ = tick.tick() => {
                    run_once(&bridge, &notifier, &pool, &mut last_status).await;
                }
                _ = shutdown.recv() => {
                    tracing::debug!("health loop shutting down");
                    break;
                }
            }
        }
    })
}

async fn run_once(
    bridge: &SunoBridge,
    notifier: &Arc<dyn Notifier>,
    pool: &SqlitePool,
    last_status: &mut Option<Status>,
) {
    let status = check(bridge).await;
    if let Err(e) = melodie_db::suno_session::set_status(pool, status.as_str()).await {
        tracing::warn!(error = %e, "failed to persist suno status");
    }
    if status == Status::Ok
        && let Err(e) = bridge.checkpoint().await
    {
        tracing::warn!(error = %e, "failed to checkpoint suno session");
    }

    let prev = *last_status;
    *last_status = Some(status);

    let entered_expired = matches!(prev, Some(p) if p != Status::Expired) && status == Status::Expired;
    let recovered = matches!(prev, Some(Status::Expired)) && status == Status::Ok;
    let first_seen_expired = prev.is_none() && status == Status::Expired;

    if entered_expired || first_seen_expired {
        let msg = "🚨 Melodie: Suno auth expired. POST /api/admin/suno-auth with a fresh Clerk cookie.";
        if let Err(e) = notifier.alert(msg).await {
            tracing::warn!(error = %e, "notifier failed for expiry alert");
        } else {
            tracing::warn!("Suno auth expired — Telegram alert sent");
        }
    } else if recovered {
        let msg = "✅ Melodie: Suno auth back online.";
        if let Err(e) = notifier.alert(msg).await {
            tracing::warn!(error = %e, "notifier failed for recovery alert");
        }
    }
}

async fn check(bridge: &SunoBridge) -> Status {
    let Some(client) = bridge.current().await else {
        return Status::Missing;
    };
    match client.billing_info().await {
        Ok(_) => Status::Ok,
        Err(e) if e.is_auth() => Status::Expired,
        Err(e) => {
            tracing::warn!(error = %e, "suno health check failed");
            Status::Degraded
        }
    }
}
