//! High-level engine: load the models once, generate stereo audio from lyrics + tags.
//!
//! This is the integration surface `melodie-api` consumes in place of `suno-client`.
//! Loading is expensive (≈15 GB read → 7.5 GB bf16 resident on Metal) so an [`Engine`]
//! is built once and reused; generation is single-threaded and bound to one Metal
//! device, so callers must serialise calls (one generation at a time).

use std::path::{Path, PathBuf};

use candle_core::Device;
use tokenizers::Tokenizer;

use crate::codec::{CodecWeights, HeartCodec};
use crate::config::{GenConfig, HeartCodecConfig};
use crate::lm::{GenParams, HeartMuLaLm, LmWeights};
use crate::pipeline::{load_tokenizer, preprocess};
use crate::{EngineError, Result};

/// Filesystem locations of the model checkpoints.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// HeartMuLa LM directory (4 safetensors shards).
    pub lm_dir: PathBuf,
    /// HeartCodec directory (2 safetensors shards).
    pub codec_dir: PathBuf,
    /// Llama-3 `tokenizer.json`.
    pub tokenizer_path: PathBuf,
}

/// Per-request generation knobs.
#[derive(Clone, Copy, Debug)]
pub struct GenOptions {
    /// Hard cap on generated frames at 12.5 Hz (2250 ≈ 3 min). Generation also stops on EOS.
    pub max_frames: usize,
    /// Classifier-free guidance scale (1.0 = off, 1.5 = reference default).
    pub cfg_scale: f64,
    /// Top-k sampling constraint.
    pub topk: usize,
    /// Sampling temperature.
    pub temperature: f64,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self { max_frames: 2250, cfg_scale: 1.5, topk: 50, temperature: 1.0 }
    }
}

/// Generated stereo audio, channel-interleaved (`L,R,L,R,…`) f32 in `[-1, 1]`.
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Audio {
    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.channels == 0 || self.sample_rate == 0 {
            return 0.0;
        }
        (self.samples.len() as f64 / self.channels as f64) / self.sample_rate as f64
    }

    /// Encode to an in-memory 16-bit PCM WAV file (interleaved samples clamped to [-1, 1]).
    pub fn to_wav_bytes(&self) -> Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut w = hound::WavWriter::new(&mut cursor, spec).map_err(wav_err)?;
            for &s in &self.samples {
                w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(wav_err)?;
            }
            w.finalize().map_err(wav_err)?;
        }
        Ok(cursor.into_inner())
    }
}

fn wav_err(e: hound::Error) -> EngineError {
    EngineError::Config(format!("wav encode: {e}"))
}

/// A loaded HeartMuLa + HeartCodec pipeline pinned to one device.
pub struct Engine {
    tok: Tokenizer,
    lm: HeartMuLaLm,
    codec: HeartCodec,
    dev: Device,
    gcfg: GenConfig,
    codec_cfg: HeartCodecConfig,
}

impl Engine {
    /// Load the tokenizer, LM and codec. Prefers the Metal GPU, falling back to CPU.
    /// This reads ~15 GB and takes tens of seconds — do it once at startup.
    pub fn load(cfg: &EngineConfig) -> Result<Self> {
        let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
        Self::load_on(cfg, dev)
    }

    /// Load on a caller-chosen device (used by tests / CPU parity).
    pub fn load_on(cfg: &EngineConfig, dev: Device) -> Result<Self> {
        let tok = load_tokenizer(path_str(&cfg.tokenizer_path)?)?;
        let lm = {
            let w = LmWeights::load(&cfg.lm_dir, &dev)?;
            let lm = HeartMuLaLm::load(&w, &dev)?;
            drop(w); // free the f32 source before the codec loads
            lm
        };
        let cw = CodecWeights::load(&cfg.codec_dir, &dev)?;
        let codec_cfg = HeartCodecConfig::default();
        let codec = HeartCodec::load(&cw, &codec_cfg, &dev)?;
        Ok(Self { tok, lm, codec, dev, gcfg: GenConfig::default(), codec_cfg })
    }

    /// Generate a song from `lyrics` and style `tags`. Blocking and single-threaded.
    pub fn generate(&self, lyrics: &str, tags: &str, opts: &GenOptions) -> Result<Audio> {
        let p = preprocess(&self.tok, &self.gcfg, tags, lyrics, &self.dev)?;
        let max_frames = opts.max_frames;
        let codes = self.lm.generate_codes(
            &p.tokens,
            &p.mask,
            Some(p.muq_idx),
            &GenParams {
                cfg_scale: opts.cfg_scale,
                max_frames,
                topk: opts.topk,
                temperature: opts.temperature,
            },
        )?;
        let t = codes.dim(1)?;
        if t == 0 {
            return Err(EngineError::Config("model emitted EOS immediately (no audio)".into()));
        }
        // Release the LM's generation residency (per-frame intermediates + dropped KV cache)
        // before the codec runs. Otherwise the whole song's LM pool stays live on the GPU
        // through the first segment's decode, and on a memory-tight Metal device the decode's
        // working set tips it over the budget → OOM. The codec then starts from just the two
        // resident model weights.
        self.dev.synchronize()?;
        // Multi-segment overlap-add → full-length songs. The per-channel split + per-stage
        // synchronize in the decoder keep each segment's GPU residency bounded. `None` ⇒ each
        // segment draws its flow-matching latent with randn internally.
        let wav = self.codec.detokenize(
            &codes.unsqueeze(0)?,
            None,
            self.codec_cfg.segment_duration,
            self.codec_cfg.flow_num_steps,
            self.codec_cfg.flow_guidance_scale,
        )?; // [2, N] f32 @ 48 kHz
        let (channels, n) = wav.dims2()?;
        let ch0: Vec<f32> = wav.narrow(0, 0, 1)?.flatten_all()?.to_vec1()?;
        let ch1: Vec<f32> = wav.narrow(0, 1.min(channels - 1), 1)?.flatten_all()?.to_vec1()?;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            samples.push(ch0[i]);
            samples.push(ch1[i]);
        }
        Ok(Audio { samples, sample_rate: self.codec_cfg.sample_rate as u32, channels: 2.min(channels.max(1)) as u16 })
    }
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| EngineError::Config(format!("non-UTF-8 path: {}", p.display())))
}
