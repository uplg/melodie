//! Parity for the full HeartCodec.detokenize (pad-to-segment + trim).
//!     cargo run --release --example parity_detok

use std::path::Path;

use candle_core::Device;
use melodie_engine::codec::{CodecWeights, HeartCodec};
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::parity::max_abs_diff;
use melodie_engine::Result;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/detok.safetensors";
const CKPT: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let g = candle_core::safetensors::load(GOLDEN, &dev)?;
    let codes = g["dtk_codes"].unsqueeze(0)?; // (8,16) -> (1,8,16)
    let fm_noise = &g["dtk_fm_noise"]; // (1,186,256) — single 7.44 s segment
    let wav_g = &g["dtk_waveform"]; // (2, target)

    println!("loading codec...");
    let w = CodecWeights::load(Path::new(CKPT), &dev)?;
    let codec = HeartCodec::load(&w, &HeartCodecConfig::default(), &dev)?;

    println!("detokenize (pad to segment + trim)...");
    let wav = codec.detokenize(&codes, Some(fm_noise), 7.44, 10, 1.25)?;
    let d = max_abs_diff(&wav, wav_g)?;
    let rms = wav_g.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!("waveform {:?} (golden {:?})  max|Δ|={d:.3e}  rms={rms:.3e}", wav.dims(), wav_g.dims());
    println!("{}", if d < 1e-3 { "detokenize PARITY OK ✅" } else { "PARITY OFF ❌" });
    Ok(())
}
