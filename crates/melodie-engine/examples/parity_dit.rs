//! Parity check for the FlowMatching DiT estimator (P1b) against the MLX golden
//! single-forward gate. Run: cargo run --release --example parity_dit

use std::path::Path;

use candle_core::Device;
use melodie_engine::Result;
use melodie_engine::codec::CodecWeights;
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::flow::Dit;
use melodie_engine::parity::max_abs_diff;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/codec_seg0.safetensors";
const CKPT: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let golden = candle_core::safetensors::load(GOLDEN, &dev)?;
    let est_in = &golden["est_in"]; // (2,32,1024)
    let est_t = &golden["est_t"]; // (2,)
    let est_out_g = &golden["est_out"]; // (2,32,256)

    println!("loading codec weights...");
    let w = CodecWeights::load(Path::new(CKPT), &dev)?;
    let dit = Dit::load(&w, &HeartCodecConfig::default().dit)?;

    println!("running DiT forward...");
    let out = dit.forward(est_in, est_t)?;
    let d = max_abs_diff(&out, est_out_g)?;
    let rms = est_out_g.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!(
        "DiT estimator: out {:?}  max|Δ|={d:.3e}  (golden rms={rms:.3e})",
        out.dims()
    );
    println!(
        "{}",
        if d < 1e-3 {
            "DiT PARITY OK ✅"
        } else {
            "DiT PARITY OFF ❌"
        }
    );
    Ok(())
}
