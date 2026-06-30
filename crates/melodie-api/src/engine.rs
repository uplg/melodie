//! Local generation worker for the HeartMuLa engine ([`melodie_engine`]).
//!
//! The engine is blocking and pinned to a single Metal device, so it can't run
//! on the tokio runtime: we host it on a dedicated OS thread fed by an unbounded
//! channel, and generations are serialised one-at-a-time. When a job finishes,
//! [`finish`] persists the clip/song state and broadcasts a [`SongEvent`] so the
//! React UI follows along over SSE.
//!
//! Model load (~30s, ~15 GB read) happens on the worker thread *after* the HTTP
//! server is already serving — the first job simply blocks until load is done.
//! If load fails, the thread logs and exits; the server keeps running (local
//! generation just stays unavailable).

use std::path::{Path, PathBuf};

use melodie_core::ids::SongId;
use melodie_core::model::SongStatus;
use melodie_db::clips::UpsertClip;
use melodie_engine::{Audio, Engine, EngineConfig, EngineError, GenOptions};
use sqlx::SqlitePool;
use tokio::sync::broadcast;
use tokio::sync::mpsc::UnboundedSender;

use crate::events::{ClipEventView, SongEvent};

/// One local generation request handed to the worker thread.
pub struct EngineJob {
    pub song_id: SongId,
    pub clip_id: String,
    pub lyrics: String,
    pub tags: String,
    pub max_frames: usize,
}

/// Submit handle for local generation jobs. Clone-cheap; lives in `AppState`.
pub type EngineHandle = UnboundedSender<EngineJob>;

/// Spawn the dedicated engine thread and return a submit handle.
pub fn spawn_worker(
    rt: tokio::runtime::Handle,
    db: SqlitePool,
    events: broadcast::Sender<SongEvent>,
    cfg: EngineConfig,
    audio_dir: PathBuf,
) -> EngineHandle {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<EngineJob>();
    std::thread::spawn(move || {
        let engine = match Engine::load(&cfg) {
            Ok(e) => e,
            Err(e) => {
                tracing::error!(error = %e, "engine load failed; local generation disabled");
                return;
            }
        };
        tracing::info!("local engine loaded");
        while let Some(job) = rx.blocking_recv() {
            let opts = GenOptions { max_frames: job.max_frames, ..Default::default() };
            let result = engine.generate(&job.lyrics, &job.tags, &opts);
            rt.block_on(finish(&db, &events, &audio_dir, job, result));
        }
    });
    tx
}

/// Persist + broadcast the outcome of one generation, mirroring `poll.rs`.
async fn finish(
    db: &SqlitePool,
    events: &broadcast::Sender<SongEvent>,
    audio_dir: &Path,
    job: EngineJob,
    result: Result<Audio, EngineError>,
) {
    let song_id = job.song_id;
    let clip_id = job.clip_id;

    // Encode + write the mp3 before touching the DB so we never mark a clip
    // "complete" without a file behind it.
    let outcome: Result<f64, String> = match result {
        Ok(audio) => {
            let dur = audio.duration_secs();
            match write_mp3(audio_dir, &clip_id, &audio).await {
                Ok(()) => Ok(dur),
                Err(e) => Err(format!("mp3 write failed: {e}")),
            }
        }
        Err(e) => Err(e.to_string()),
    };

    match outcome {
        Ok(dur) => {
            let clip = UpsertClip {
                id: clip_id.clone(),
                song_id,
                variant_index: 0,
                status: "complete".into(),
                duration_s: Some(dur),
                image_url: None,
            };
            if let Err(e) = melodie_db::clips::upsert_many(db, std::slice::from_ref(&clip)).await {
                tracing::warn!(error = %e, %song_id, "engine: clip upsert failed");
            }
            if let Err(e) =
                melodie_db::songs::set_status(db, song_id, SongStatus::Complete, None).await
            {
                tracing::warn!(error = %e, %song_id, "engine: song status update failed");
            }
            let _ = events.send(SongEvent {
                song_id: song_id.to_string(),
                status: "complete".into(),
                clips: vec![ClipEventView {
                    id: clip_id,
                    variant_index: 0,
                    status: "complete".into(),
                    duration_s: Some(dur),
                    image_url: None,
                }],
            });
        }
        Err(msg) => {
            tracing::warn!(error = %msg, %song_id, "engine: generation failed");
            if let Err(e) =
                melodie_db::songs::set_status(db, song_id, SongStatus::Failed, Some(&msg)).await
            {
                tracing::warn!(error = %e, %song_id, "engine: song status update failed");
            }
            let clip = UpsertClip {
                id: clip_id.clone(),
                song_id,
                variant_index: 0,
                status: "error".into(),
                duration_s: None,
                image_url: None,
            };
            if let Err(e) = melodie_db::clips::upsert_many(db, std::slice::from_ref(&clip)).await {
                tracing::warn!(error = %e, %song_id, "engine: clip upsert failed");
            }
            let _ = events.send(SongEvent {
                song_id: song_id.to_string(),
                status: "failed".into(),
                clips: vec![ClipEventView {
                    id: clip_id,
                    variant_index: 0,
                    status: "error".into(),
                    duration_s: None,
                    image_url: None,
                }],
            });
        }
    }
}

/// Ensure `audio_dir` exists and write `{clip_id}.mp3` into it.
async fn write_mp3(audio_dir: &Path, clip_id: &str, audio: &Audio) -> Result<(), String> {
    tokio::fs::create_dir_all(audio_dir)
        .await
        .map_err(|e| e.to_string())?;
    // LAME encoding is CPU-bound and synchronous; we're already on the
    // dedicated worker thread (not a tokio worker), so it's safe to run inline.
    let bytes = encode_mp3(audio)?;
    let path = audio_dir.join(format!("{clip_id}.mp3"));
    tokio::fs::write(&path, bytes)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Encode interleaved-stereo f32 PCM (`[-1, 1]`) to a CBR 192 kbps MP3 with
/// LAME. Pure-Rust API over the vendored `libmp3lame` — no ffmpeg subprocess.
fn encode_mp3(audio: &Audio) -> Result<Vec<u8>, String> {
    use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, InterleavedPcm, Quality};

    let channels = audio.channels.max(1);
    let mut builder = Builder::new().ok_or_else(|| "mp3lame: builder alloc failed".to_string())?;
    builder
        .set_num_channels(channels as u8)
        .map_err(|e| format!("mp3lame channels: {e}"))?;
    builder
        .set_sample_rate(audio.sample_rate)
        .map_err(|e| format!("mp3lame sample_rate: {e}"))?;
    builder
        .set_brate(Bitrate::Kbps192)
        .map_err(|e| format!("mp3lame brate: {e}"))?;
    builder
        .set_quality(Quality::Good)
        .map_err(|e| format!("mp3lame quality: {e}"))?;
    let mut encoder = builder.build().map_err(|e| format!("mp3lame build: {e}"))?;

    let frames = audio.samples.len() / channels as usize;
    let mut mp3 = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(frames));
    encoder
        .encode_to_vec(InterleavedPcm(&audio.samples), &mut mp3)
        .map_err(|e| format!("mp3lame encode: {e}"))?;
    encoder
        .flush_to_vec::<FlushNoGap>(&mut mp3)
        .map_err(|e| format!("mp3lame flush: {e}"))?;
    Ok(mp3)
}
