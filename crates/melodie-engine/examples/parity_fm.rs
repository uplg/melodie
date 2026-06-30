//! Parity check for the full FlowMatching half (RVQ + DiT + Euler/CFG) against the
//! MLX golden (`codes` + injected `fm_noise` -> `fm_latents`).
//!     cargo run --release --example parity_fm

use std::path::Path;

use candle_core::Device;
use melodie_engine::codec::CodecWeights;
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::flow::FlowMatching;
use melodie_engine::parity::max_abs_diff;
use melodie_engine::Result;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/codec_seg0.safetensors";
const CKPT: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let golden = candle_core::safetensors::load(GOLDEN, &dev)?;
    let codes = golden["codes"].unsqueeze(0)?; // (8,16) -> (1,8,16)
    let fm_noise = &golden["fm_noise"]; // (1,32,256)
    let fm_latents_g = &golden["fm_latents"]; // (1,32,256)

    println!("loading codec weights...");
    let w = CodecWeights::load(Path::new(CKPT), &dev)?;
    let fm = FlowMatching::load(&w, &HeartCodecConfig::default())?;

    println!("running FlowMatching (10 Euler steps, cfg 1.25)...");
    let out = fm.inference(&codes, fm_noise, 10, 1.25)?;
    let d = max_abs_diff(&out, fm_latents_g)?;
    let rms = fm_latents_g.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!("fm_latents {:?}  max|Δ|={d:.3e}  (golden rms={rms:.3e})", out.dims());
    println!("{}", if d < 1e-3 { "FlowMatching PARITY OK ✅" } else { "FlowMatching PARITY OFF ❌" });
    Ok(())
}
