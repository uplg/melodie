//! SSE progress events for song generation.
//!
//! The local engine worker (see `engine.rs`) broadcasts these as a generation
//! moves from `streaming` to `complete`/`error`. The SSE endpoint in
//! `routes/songs.rs` forwards them to the React UI. Shapes are part of the
//! public HTTP API — keep them stable.

use melodie_core::model::Song;
use serde::Serialize;

use crate::routes::songs::ClipView;

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

/// Same wire shape as [`ClipView`] — distinct name to keep the SSE payload
/// readable independent of the REST views module.
pub type ClipEventView = ClipView;

impl SongEvent {
    pub fn from_song(song: &Song) -> Self {
        Self {
            song_id: song.id.to_string(),
            status: song.status.as_str().to_string(),
            progress: None,
            clips: song.clips.iter().map(ClipEventView::from).collect(),
        }
    }
}
