//! End-to-end via the high-level `Engine` (the production path): lyrics+tags ->
//! multi-segment overlap-add detokenize -> WAV. Tests segment seams for long songs.
//! `cargo run --release --example gen_engine [frames]`  (400 ≈ 32 s ≈ 2 segments)
use std::path::PathBuf;

use melodie_engine::{Engine, EngineConfig, GenOptions, Result};

const CKPT: &str = "/Users/leonard/Github/heartlib-mlx/ckpt";
const LYRICS_FILE: &str = "crates/melodie-engine/reference/lyrics_input.txt";
const TAGS: &str =
    "french chanson, male vocal, upbeat feel-good pop, acoustic guitar, brass, 1960s";

fn main() -> Result<()> {
    let frames: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let cfg = EngineConfig {
        lm_dir: PathBuf::from(format!("{CKPT}/HeartMuLa-oss-3B")),
        codec_dir: PathBuf::from(format!("{CKPT}/HeartCodec-oss")),
        tokenizer_path: PathBuf::from(format!("{CKPT}/tokenizer.json")),
    };
    println!("loading engine...");
    let engine = Engine::load(&cfg)?;
    let lyrics = std::fs::read_to_string(LYRICS_FILE)?; // read locally; never printed
    println!("generating up to {frames} frames...");
    let opts = GenOptions {
        max_frames: frames,
        ..Default::default()
    };
    let audio = engine.generate(&lyrics, TAGS, &opts)?;
    std::fs::write("gen_engine.wav", audio.to_wav_bytes()?)?;
    println!(
        "wrote gen_engine.wav ({:.2} s, {} ch)",
        audio.duration_secs(),
        audio.channels
    );
    Ok(())
}
