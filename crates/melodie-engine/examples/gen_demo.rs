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
    let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("device: {dev:?}");
    let frames = 24usize;
    let s: usize = std::env::var("MELODIE_PROMPT_LEN").ok().and_then(|x| x.parse().ok()).unwrap_or(8);
    let ncb = 8usize;

    println!("loading HeartMuLa LM (15 GB)...");
    let codes = {
        let w = LmWeights::load(Path::new(LM), &dev)?;
        let lm = HeartMuLaLm::load(&w, &dev)?;
        drop(w); // free the 15 GB f32 source; the model keeps only its bf16 copy (~7.5 GB)
        // synthetic text-only prompt of length s (BOS … EOS)
        let text_ids: Vec<i64> = (0..s)
            .map(|i| if i == 0 { 128000 } else if i == s - 1 { 128001 } else { (100 + i) as i64 })
            .collect();
        let mut tg = vec![0i64; s * (ncb + 1)];
        let mut mg = vec![0i64; s * (ncb + 1)];
        for i in 0..s {
            tg[i * (ncb + 1) + ncb] = text_ids[i];
            mg[i * (ncb + 1) + ncb] = 1;
        }
        let tokens = Tensor::from_vec(tg, (1, s, ncb + 1), &dev)?;
        let mask = Tensor::from_vec(mg, (1, s, ncb + 1), &dev)?;
        let cfg: f64 = std::env::var("MELODIE_CFG").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        println!("generating {frames} frames (multi-frame loop, cfg={cfg})...");
        let t0 = std::time::Instant::now();
        let c = lm.generate_codes(&tokens, &mask, None, cfg, frames, 50, 1.0)?;
        let el = t0.elapsed().as_secs_f32();
        println!("  generation: {el:.1} s ({:.0} ms/frame)", el * 1000.0 / frames as f32);
        c
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
