//! # melodie-engine
//!
//! Local, pure-Rust ([candle]) inference engine for **HeartMuLa** — the music
//! generator that powers Mélodie. Runs generation fully on-device, no upstream API.
//!
//! ## What it runs
//! - **HeartMuLa LM** (~3B): Llama-3.1 backbone + 300M *depth* decoder (an RQ-Transformer, Moshi/CSM family) emitting 8 audio codebooks @ 12.5 Hz via top-k Gumbel sampling + classifier-free guidance.
//! - **HeartCodec**: RVQ → DiT flow-matching (Euler ODE, 10 steps) → scalar-quant causal-conv vocoder @ 48 kHz, with overlap-add segmentation.
//!
//! Notably there are **no FFT/STFT/custom kernels** in either model, which is what
//! makes a candle port tractable.
//!
//! ## Modules
//! - [`config`] static model configuration
//! - [`codec`] HeartCodec decoder (codes → waveform) + [`flow`] RVQ/DiT flow-matching
//! - [`lm`] HeartMuLa (backbone + depth decoder + sampling + CFG)
//! - [`pipeline`] end-to-end glue, wrapped by the high-level [`engine::Engine`]

pub mod codec;
pub mod config;
pub mod engine;
pub mod error;
pub mod flow;
pub mod lm;
pub mod pipeline;

pub use engine::{Audio, Engine, EngineConfig, GenOptions, GenProgress, GenStage};
pub use error::{EngineError, Result};
