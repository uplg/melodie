//! End-to-end codec parity (P1b): codes + injected noise -> waveform, chaining the
//! verified FlowMatching and ScalarModel halves. Run:
//!     cargo run --release --example parity_codec_full

use std::path::Path;

use candle_core::Device;
use melodie_engine::Result;
use melodie_engine::codec::{CodecWeights, HeartCodec};
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::parity::max_abs_diff;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/codec_seg0.safetensors";
const CKPT: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let g = candle_core::safetensors::load(GOLDEN, &dev)?;
    let codes = g["codes"].unsqueeze(0)?; // (8,16) -> (1,8,16)
    let noise = &g["fm_noise"]; // (1,32,256)
    let wav_g = &g["waveform"]; // (2,61440)

    println!("loading codec...");
    let w = CodecWeights::load(Path::new(CKPT), &dev)?;
    let codec = HeartCodec::load(&w, &HeartCodecConfig::default(), &dev)?;

    println!("detokenize (FM -> reshape -> ScalarDecoder)...");
    let wav = codec.detokenize_segment(&codes, noise, 10, 1.25)?;
    let d = max_abs_diff(&wav, wav_g)?;
    let rms = wav_g.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!(
        "waveform {:?}  max|Δ|={d:.3e}  (golden rms={rms:.3e})",
        wav.dims()
    );
    println!(
        "{}",
        if d < 1e-3 {
            "HeartCodec end-to-end PARITY OK ✅"
        } else {
            "PARITY OFF ❌"
        }
    );
    Ok(())
}
