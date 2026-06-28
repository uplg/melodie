//! End-to-end generation demo: synthetic prompt -> HeartMuLa multi-frame codes ->
//! HeartCodec detokenize -> WAV. Proves the full candle pipeline produces audio.
//! (Synthetic numeric prompt, not lyrics — real lyrics come with the P3 tokenizer.)
//!     cargo run --release --example gen_demo

use std::path::Path;

use candle_core::{Device, Tensor};
use melodie_engine::codec::{CodecWeights, HeartCodec};
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::lm::{HeartMuLaLm, LmWeights};
use melodie_engine::{EngineError, Result};

const LM: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartMuLa-oss-3B";
const CODEC: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";

fn write_wav(path: &str, wav: &Tensor) -> Result<()> {
    let ch0: Vec<f32> = wav.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let ch1: Vec<f32> = wav.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wtr = hound::WavWriter::create(path, spec).map_err(|e| EngineError::Config(e.to_string()))?;
    for i in 0..ch0.len() {
        for &s in &[ch0[i], ch1[i]] {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            wtr.write_sample(v).map_err(|e| EngineError::Config(e.to_string()))?;
        }
    }
    wtr.finalize().map_err(|e| EngineError::Config(e.to_string()))?;
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let frames = 24usize;
    let (s, ncb) = (8usize, 8usize);

    println!("loading HeartMuLa LM (15 GB)...");
    let codes = {
        let w = LmWeights::load(Path::new(LM), &dev)?;
        let lm = HeartMuLaLm::load(&w, &dev)?;
        // synthetic text-only prompt
        let text_ids = [128000i64, 100, 200, 300, 400, 500, 600, 128001];
        let mut tg = vec![0i64; s * (ncb + 1)];
        let mut mg = vec![0i64; s * (ncb + 1)];
        for i in 0..s {
            tg[i * (ncb + 1) + ncb] = text_ids[i];
            mg[i * (ncb + 1) + ncb] = 1;
        }
        let tokens = Tensor::from_vec(tg, (1, s, ncb + 1), &dev)?;
        let mask = Tensor::from_vec(mg, (1, s, ncb + 1), &dev)?;
        println!("generating {frames} frames (multi-frame loop)...");
        lm.generate_codes(&tokens, &mask, None, frames, 50, 1.0)?
    }; // LM dropped here, freeing ~15 GB before loading the codec

    let t = codes.dim(1)?;
    println!("generated codes {:?}", codes.dims());

    println!("loading HeartCodec...");
    let w = CodecWeights::load(Path::new(CODEC), &dev)?;
    let codec = HeartCodec::load(&w, &HeartCodecConfig::default(), &dev)?;
    let noise = Tensor::randn(0f32, 1f32, (1, 2 * t, 256), &dev)?;
    println!("detokenize...");
    let wav = codec.detokenize_segment(&codes.unsqueeze(0)?, &noise, 10, 1.25)?;
    println!("waveform {:?}  ({:.2} s)", wav.dims(), wav.dim(1)? as f32 / 48000.0);

    write_wav("gen_demo.wav", &wav)?;
    println!("wrote gen_demo.wav ✅");
    Ok(())
}
