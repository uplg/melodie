//! Parity check for the HeartMuLa LM (P2): backbone c0-logits gate + depth-decoder
//! ci-logits gate, against the MLX golden frame. Run:
//!     cargo run --release --example parity_lm

use std::path::Path;

use candle_core::Device;
use melodie_engine::Result;
use melodie_engine::lm::{HeartMuLaLm, LmWeights};
use melodie_engine::parity::max_abs_diff;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/lm_frame0.safetensors";
const LM: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartMuLa-oss-3B";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let g = candle_core::safetensors::load(GOLDEN, &dev)?;
    let tokens = &g["lm_tokens"]; // [1,8,9] i64
    let mask = &g["lm_mask"]; // [1,8,9] i64
    let c0_g = &g["lm_c0_logits"]; // [1,V]
    let ci_g = &g["lm_ci_logits"]; // [7,V]
    let samples = &g["lm_curr_sample"]; // [1,8] i64

    println!("loading HeartMuLa LM weights (fp32, cpu)... (15 GB)");
    let w = LmWeights::load(Path::new(LM), &dev)?;
    let lm = HeartMuLaLm::load(&w, &dev)?;

    println!("backbone forward...");
    let (last_h, c0) = lm.backbone_c0(tokens, mask)?;
    let d0 = max_abs_diff(&c0, c0_g)?;
    let rms0 = c0_g.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    println!(
        "backbone c0_logits {:?}  max|Δ|={d0:.3e}  (rms={rms0:.3e})",
        c0.dims()
    );

    println!("depth decoder (replaying samples)...");
    let ci = lm.depth_ci(&last_h, samples)?;
    let mut worst = d0;
    for (i, l) in ci.iter().enumerate() {
        let gi = ci_g.narrow(0, i, 1)?;
        let d = max_abs_diff(l, &gi)?;
        worst = worst.max(d);
        println!("  codebook {} ci_logits max|Δ|={d:.3e}", i + 1);
    }
    let logits_ok = worst < 1e-2;

    // full-frame generation (sampling) with injected uniforms -> exact tokens
    println!("generate_frame (top-k Gumbel, injected uniforms)...");
    let uniforms = &g["lm_uniforms"]; // [ncb, V]
    let curr_g: Vec<i64> = g["lm_curr_sample"].flatten_all()?.to_vec1::<i64>()?;
    let gen_tokens = lm.generate_frame(tokens, mask, 50, 1.0, uniforms)?;
    let gen_i: Vec<i64> = gen_tokens.iter().map(|&x| x as i64).collect();
    let samples_ok = gen_i == curr_g;
    println!("  rust samples   = {gen_i:?}");
    println!("  golden samples = {curr_g:?}  match={samples_ok}");

    println!(
        "{}",
        if logits_ok && samples_ok {
            "LM PARITY OK ✅"
        } else {
            "LM PARITY OFF ❌"
        }
    );
    Ok(())
}
