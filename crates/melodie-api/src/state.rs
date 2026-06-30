use std::path::PathBuf;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::broadcast;

use crate::config::HomiePushConfig;
use crate::engine::EngineHandle;
use crate::events::SongEvent;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    /// Fan-out for engine-worker progress updates. Subscribers connect via
    /// `GET /api/songs/{id}/events`. Lagged subscribers drop messages — we
    /// don't backfill; the SSE endpoint sends a fresh DB snapshot on connect
    /// so brief drops are recovered on the next tick.
    pub events: broadcast::Sender<SongEvent>,
    /// When set, the `push-to-live` endpoint is enabled and the React UI
    /// shows a "Push to live" button on each playable clip. The bridge talks
    /// to homie's loopback push server.
    pub homie_push: Option<Arc<HomiePushConfig>>,
    /// Local-engine job submitter. `POST /api/songs` enqueues generation work
    /// onto the dedicated worker thread through this handle.
    pub engine: EngineHandle,
    /// Directory holding generated `.mp3` files, served by
    /// `GET /api/clips/{id}/audio`.
    pub audio_dir: Arc<PathBuf>,
}

impl AppState {
    pub fn new(
        db: SqlitePool,
        events: broadcast::Sender<SongEvent>,
        homie_push: Option<Arc<HomiePushConfig>>,
        engine: EngineHandle,
        audio_dir: Arc<PathBuf>,
    ) -> Self {
        Self {
            db,
            events,
            homie_push,
            engine,
            audio_dir,
        }
    }
}
