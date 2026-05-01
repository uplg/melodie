use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::broadcast;

use crate::poll::SongEvent;
use crate::suno::SunoBridge;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub suno: Arc<SunoBridge>,
    /// Fan-out for poll-task progress updates. Subscribers connect via
    /// `GET /api/songs/{id}/events`. Lagged subscribers drop messages — we
    /// don't backfill; the SSE endpoint sends a fresh DB snapshot on connect
    /// so brief drops are recovered on the next tick.
    pub events: broadcast::Sender<SongEvent>,
}

impl AppState {
    pub fn new(
        db: SqlitePool,
        suno: Arc<SunoBridge>,
        events: broadcast::Sender<SongEvent>,
    ) -> Self {
        Self { db, suno, events }
    }
}
