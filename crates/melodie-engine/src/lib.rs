//! # melodie-engine
//!
//! Local, pure-Rust ([candle]) inference engine for **HeartMuLa**, replacing the
//! Suno bridge (`suno-client`) inside Mélodie. Ported from the MLX reference at
//! `../heartlib-mlx` (verified architecture, June 2026).
//!
//! ## What it runs
//! - **HeartMuLa LM** (~3B): Llama-3.1 backbone + 300M *depth* decoder (an RQ-Transformer, Moshi/CSM family) emitting 8 audio codebooks @ 12.5 Hz via top-k Gumbel sampling + classifier-free guidance.
//! - **HeartCodec**: RVQ → DiT flow-matching (Euler ODE, 10 steps) → scalar-quant causal-conv vocoder @ 48 kHz, with overlap-add segmentation.
//!
//! Notably there are **no FFT/STFT/custom kernels** in either model, which is what
//! makes a candle port tractable.
//!
//! ## Phased port plan (each phase gated by parity tests, see [`parity`])
//! - **P0** scaffold + config + parity harness  ← *current*
//! - **P1** [`codec`] decoder (codes → waveform), validated in isolation
//! - **P2** [`lm`] HeartMuLa (backbone + depth decoder + sampling + CFG)
//! - **P3** [`pipeline`] end-to-end, parity vs the Python MLX reference
//! - **P4** integration into `melodie-api` (replace `suno-client`)

pub mod codec;
pub mod config;
pub mod engine;
pub mod error;
pub mod flow;
pub mod lm;
pub mod parity;
pub mod pipeline;

pub use engine::{Audio, Engine, EngineConfig, GenOptions};
pub use error::{EngineError, Result};
