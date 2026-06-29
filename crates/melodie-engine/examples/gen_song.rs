//! Real-lyrics generation: Llama-3 tokenizer + preprocess -> HeartMuLa codes ->
//! HeartCodec -> WAV. Lyrics are read from a local file and tokenized; they are
//! never printed. Run:  cargo run --release --example gen_song [frames]

use std::path::Path;

use candle_core::{Device, Tensor};
use melodie_engine::codec::{CodecWeights, HeartCodec};
use melodie_engine::config::{GenConfig, HeartCodecConfig};
use melodie_engine::lm::{HeartMuLaLm, LmWeights};
use melodie_engine::pipeline::{load_tokenizer, preprocess};
use melodie_engine::{EngineError, Result};

const LM: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartMuLa-oss-3B";
const CODEC: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";
const TOKENIZER: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/tokenizer.json";
const LYRICS_FILE: &str = "crates/melodie-engine/reference/lyrics_input.txt";
// style tags (mine — generic descriptive prompt, not lyrics)
const TAGS: &str = "french chanson, male vocal, upbeat feel-good pop, acoustic guitar, brass, 1960s";

fn write_wav(path: &str, wav: &Tensor) -> Result<()> {
    let ch0: Vec<f32> = wav.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let ch1: Vec<f32> = wav.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let spec = hound::WavSpec { channels: 2, sample_rate: 48000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut wtr = hound::WavWriter::create(path, spec).map_err(|e| EngineError::Config(e.to_string()))?;
    for i in 0..ch0.len() {
        for &s in &[ch0[i], ch1[i]] {
            wtr.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|e| EngineError::Config(e.to_string()))?;
        }
    }
    wtr.finalize().map_err(|e| EngineError::Config(e.to_string()))?;
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let frames: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(80);

    let tok = load_tokenizer(TOKENIZER)?;
    let gcfg = GenConfig::default();
    let lyrics = std::fs::read_to_string(LYRICS_FILE)?; // read locally; never printed
    let p = preprocess(&tok, &gcfg, TAGS, &lyrics, &dev)?;
    println!("prompt: {} tokens (muq_idx {})", p.tokens.dim(1)?, p.muq_idx);

    println!("loading HeartMuLa LM (15 GB)...");
    let codes = {
        let w = LmWeights::load(Path::new(LM), &dev)?;
        let lm = HeartMuLaLm::load(&w, &dev)?;
        println!("generating up to {frames} frames...");
        lm.generate_codes(&p.tokens, &p.mask, Some(p.muq_idx), 1.5, frames, 50, 1.0)?
    };
    let t = codes.dim(1)?;
    println!("generated codes [8, {t}]");
    if t == 0 {
        println!("(EOS immediately — no audio)");
        return Ok(());
    }

    println!("loading HeartCodec + detokenize...");
    let w = CodecWeights::load(Path::new(CODEC), &dev)?;
    let codec = HeartCodec::load(&w, &HeartCodecConfig::default(), &dev)?;
    let noise = Tensor::randn(0f32, 1f32, (1, 2 * t, 256), &dev)?;
    let wav = codec.detokenize_segment(&codes.unsqueeze(0)?, &noise, 10, 1.25)?;
    write_wav("gen_song.wav", &wav)?;
    println!("wrote gen_song.wav ({:.2} s) ✅", wav.dim(1)? as f32 / 48000.0);
    Ok(())
}
