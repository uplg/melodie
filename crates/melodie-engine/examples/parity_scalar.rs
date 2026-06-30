//! Parity check for the ScalarModel decoder (P1b) against MLX golden vectors.
//!     cargo run --release --example parity_scalar

use std::path::Path;

use candle_core::{Device, Tensor};
use melodie_engine::Result;
use melodie_engine::codec::{CodecWeights, ScalarDecoder};
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::parity::max_abs_diff;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/codec_seg0.safetensors";
const CKPT: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let golden = candle_core::safetensors::load(GOLDEN, &dev)?;
    let latent_in = &golden["latent_in"]; // (2, 32, 128) NLC

    println!("loading codec weights (fp32, cpu)...");
    let w = CodecWeights::load(Path::new(CKPT), &dev)?;
    let dec = ScalarDecoder::load(&w, &HeartCodecConfig::default(), &dev)?;

    // per-stage parity localisation
    let taps = dec.decode_tapped(latent_in)?;
    let mut worst = 0f32;
    for (i, t) in taps.iter().enumerate() {
        let g = golden[&format!("dec{i}")].transpose(1, 2)?.contiguous()?;
        let d = max_abs_diff(t, &g)?;
        worst = worst.max(d);
        println!("  dec{i}: max|Δ|={d:.3e}");
    }

    let wav = dec.decode(latent_in)?;
    let d = max_abs_diff(&wav, &golden["waveform"])?;
    let rms = golden["waveform"]
        .sqr()?
        .mean_all()?
        .sqrt()?
        .to_scalar::<f32>()?;
    println!("waveform {:?}  max|Δ|={d:.3e}  (rms={rms:.3e})", wav.dims());
    println!(
        "{}",
        if d < 1e-3 && worst < 1e-3 {
            "ScalarModel decoder PARITY OK ✅"
        } else {
            "PARITY OFF ❌"
        }
    );

    // --- self-parity: streaming decode vs dense, on a longer random latent. The golden (32
    //     frames) is too short to exercise the conv stack's receptive field; this finds the R
    //     (context per chunk) at which streaming becomes bit-identical to the dense decode. ---
    let l = 256usize;
    let big = Tensor::randn(0f32, 1f32, (2, l, 128), &dev)?;
    let dense = dec.decode_one(&big)?;
    let dlen = dense.dim(1)?;
    println!("self-parity dense vs streaming, latent L={l} (dense len={dlen}):");
    for r in [16usize, 24, 32, 48, 64] {
        let s = dec.decode_streaming(&big, 64, r)?;
        let m = dlen.min(s.dim(1)?);
        let sd = max_abs_diff(&dense.narrow(1, 0, m)?, &s.narrow(1, 0, m)?)?;
        println!(
            "  CH=64 R={r:3} → max|Δ|={sd:.3e}  (stream len={})",
            s.dim(1)?
        );
    }
    Ok(())
}
