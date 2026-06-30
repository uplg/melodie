//! SSE progress events for song generation.
//!
//! The local engine worker (see `engine.rs`) broadcasts these as a generation
//! moves from `streaming` to `complete`/`error`. The SSE endpoint in
//! `routes/songs.rs` forwards them to the React UI. Shapes are part of the
//! public HTTP API — keep them stable.

use melodie_core::model::{Song, SongStatus};
use serde::Serialize;

/// Wire event broadcast through the SSE endpoint. Cheap to clone.
#[derive(Debug, Clone, Serialize)]
pub struct SongEvent {
    pub song_id: String,
    pub status: String,
    pub clips: Vec<ClipEventView>,
    /// Coarse generation progress, 0–100. `None` for terminal/non-progress events
    /// (e.g. a `complete`/`failed` broadcast or a snapshot built from a [`Song`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClipEventView {
    pub id: String,
    pub variant_index: i32,
    pub status: String,
    pub duration_s: Option<f64>,
    pub image_url: Option<String>,
}

impl SongEvent {
    pub fn from_song(song: &Song) -> Self {
        Self {
            song_id: song.id.to_string(),
            status: song_status_str(song.status).to_string(),
            progress: None,
            clips: song
                .clips
                .iter()
                .map(|c| ClipEventView {
                    id: c.id.clone(),
                    variant_index: c.variant_index,
                    status: c.status.clone(),
                    duration_s: c.duration_s,
                    image_url: c.image_url.clone(),
                })
                .collect(),
        }
    }
}

pub fn song_status_str(s: SongStatus) -> &'static str {
    match s {
        SongStatus::Pending => "pending",
        SongStatus::Generating => "generating",
        SongStatus::Complete => "complete",
        SongStatus::Failed => "failed",
    }
}
