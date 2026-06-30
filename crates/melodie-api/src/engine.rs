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
use std::time::{Duration, Instant};

use melodie_core::ids::SongId;
use melodie_core::model::SongStatus;
use melodie_db::clips::UpsertClip;
use melodie_engine::{Engine, EngineConfig, GenOptions, GenProgress, GenStage};
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
            let song_id = job.song_id.to_string();
            // RSS at start/end of every request: if it climbs across successive jobs
            // we're accumulating; if it just sits near Mélodie's ~13 GB the swap spikes
            // are the box's other apps, not us.
            tracing::info!("engine: song {song_id} start rss={}", rss_gb());
            let _ = std::fs::create_dir_all(&audio_dir);
            let path = audio_dir.join(format!("{}.mp3", job.clip_id));
            // Stream the mp3 to disk as each codec segment finalises, so the audio route can
            // serve a growing file and the browser plays it mid-generation.
            let result: Result<f64, String> = match StreamMp3::new(&path, 2, engine.sample_rate()) {
                Ok(mut enc) => {
                    let events = &events;
                    let song_id = song_id.as_str();
                    // Throttle the log + SSE to ~one per 10% bucket or per 2 s (NOT per frame).
                    let mut last_log = Instant::now();
                    let mut last_bucket: i32 = -1;
                    let mut codec_started = false;
                    let mut on = |p: GenProgress| {
                        let pct: u8 = match p.stage {
                            GenStage::Lm => ((p.done as f64 / p.total.max(1) as f64) * 80.0) as u8,
                            GenStage::Codec => {
                                (80.0 + (p.done as f64 / p.total.max(1) as f64) * 20.0) as u8
                            }
                        };
                        let now = Instant::now();
                        let bucket = (pct / 10) as i32;
                        if bucket == last_bucket
                            && now.duration_since(last_log) < Duration::from_secs(2)
                        {
                            return;
                        }
                        last_bucket = bucket;
                        last_log = now;
                        match p.stage {
                            GenStage::Lm => tracing::info!(
                                "engine: song {song_id} LM {}/{} ({}%)",
                                p.done,
                                p.total,
                                pct
                            ),
                            GenStage::Codec => {
                                if !codec_started {
                                    codec_started = true;
                                    tracing::info!(
                                        "engine: song {song_id} codec start rss={}",
                                        rss_gb()
                                    );
                                }
                                tracing::info!(
                                    "engine: song {song_id} codec segment {}/{}",
                                    p.done,
                                    p.total
                                );
                            }
                        }
                        let _ = events.send(SongEvent {
                            song_id: song_id.to_string(),
                            status: "generating".into(),
                            clips: Vec::new(),
                            progress: Some(pct),
                        });
                    };
                    let dur = {
                        let mut on_audio = |pcm: &[f32]| enc.write(pcm);
                        engine
                            .generate_streaming(
                                &job.lyrics,
                                &job.tags,
                                &opts,
                                &mut on_audio,
                                &mut on,
                            )
                            .map_err(|e| e.to_string())
                    };
                    match dur {
                        Ok(d) => enc.finish().map(|_| d),
                        Err(e) => {
                            let _ = std::fs::remove_file(&path);
                            Err(e)
                        }
                    }
                }
                Err(e) => Err(e),
            };
            tracing::info!("engine: song {song_id} done rss={}", rss_gb());
            rt.block_on(finish(&db, &events, job, result));
        }
    });
    tx
}

/// Best-effort resident set size of this process, formatted like `"13.4GB"`.
/// Shells out to `ps -o rss=` (KB on macOS/Linux) so we add no dependency.
fn rss_gb() -> String {
    let pid = std::process::id();
    match std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
    {
        Ok(out) => {
            let kb: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0.0);
            format!("{:.1}GB", kb / 1024.0 / 1024.0)
        }
        Err(_) => "?".to_string(),
    }
}

/// Incremental MP3 writer: feeds interleaved-stereo f32 PCM chunks through ONE LAME encoder as
/// the engine streams finalised segments, appending each chunk's frames to `{clip_id}.mp3` so
/// the file is playable while it's still growing. The first write/encode error is latched and
/// surfaced by [`StreamMp3::finish`].
struct StreamMp3 {
    encoder: mp3lame_encoder::Encoder,
    file: std::fs::File,
    err: Option<String>,
}
impl StreamMp3 {
    fn new(path: &Path, channels: u16, sample_rate: u32) -> Result<Self, String> {
        use mp3lame_encoder::{Bitrate, Builder, Quality};
        let mut b = Builder::new().ok_or_else(|| "mp3lame: builder alloc failed".to_string())?;
        b.set_num_channels(channels as u8).map_err(|e| format!("mp3lame channels: {e}"))?;
        b.set_sample_rate(sample_rate).map_err(|e| format!("mp3lame sample_rate: {e}"))?;
        b.set_brate(Bitrate::Kbps192).map_err(|e| format!("mp3lame brate: {e}"))?;
        b.set_quality(Quality::Good).map_err(|e| format!("mp3lame quality: {e}"))?;
        let encoder = b.build().map_err(|e| format!("mp3lame build: {e}"))?;
        let file =
            std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        Ok(Self { encoder, file, err: None })
    }
    /// Encode + append one interleaved-stereo chunk (`l,r,l,r,…`). Errors are latched.
    fn write(&mut self, pcm: &[f32]) {
        use mp3lame_encoder::InterleavedPcm;
        use std::io::Write;
        if self.err.is_some() {
            return;
        }
        let frames = pcm.len() / 2;
        let mut mp3 = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(frames));
        if let Err(e) = self.encoder.encode_to_vec(InterleavedPcm(pcm), &mut mp3) {
            self.err = Some(format!("mp3lame encode: {e}"));
            return;
        }
        if let Err(e) = self.file.write_all(&mp3) {
            self.err = Some(format!("mp3 write: {e}"));
        }
    }
    /// Flush LAME's tail frames + the OS buffer; returns the latched error if any.
    fn finish(mut self) -> Result<(), String> {
        use mp3lame_encoder::FlushNoGap;
        use std::io::Write;
        if let Some(e) = self.err {
            return Err(e);
        }
        // LAME's flush writes into the Vec's SPARE CAPACITY (it doesn't grow it). An empty Vec
        // handed `lame_encode_flush_nogap` a zero-capacity buffer, which it wrote past →
        // segfault. Reserve LAME's flush buffer (a frame's worth of buffered PCM + the ~7200 B
        // tail), exactly like the non-streaming path whose Vec was already sized by encode.
        let mut mp3 = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(1152));
        self.encoder
            .flush_to_vec::<FlushNoGap>(&mut mp3)
            .map_err(|e| format!("mp3lame flush: {e}"))?;
        self.file.write_all(&mp3).map_err(|e| format!("mp3 write: {e}"))?;
        self.file.flush().map_err(|e| format!("mp3 flush: {e}"))?;
        Ok(())
    }
}

/// Persist + broadcast the outcome of one generation, mirroring `poll.rs`.
async fn finish(
    db: &SqlitePool,
    events: &broadcast::Sender<SongEvent>,
    job: EngineJob,
    result: Result<f64, String>,
) {
    let song_id = job.song_id;
    let clip_id = job.clip_id;

    // The mp3 was already streamed to disk during generation; `result` is the duration (s) or
    // the failure reason.
    match result {
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
                progress: None,
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
                progress: None,
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
