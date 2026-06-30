//! Parity for the CFG (cfg_scale=1.5) generate_frame: guided logits + exact tokens.
//!     cargo run --release --example parity_lm_cfg

use std::path::Path;

use candle_core::Device;
use melodie_engine::Result;
use melodie_engine::lm::{HeartMuLaLm, LmWeights};
use melodie_engine::parity::max_abs_diff;

const GOLDEN: &str = "crates/melodie-engine/reference/golden/lm_cfg.safetensors";
const LM: &str = "/Users/leonard/Github/heartlib-mlx/ckpt/HeartMuLa-oss-3B";

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let g = candle_core::safetensors::load(GOLDEN, &dev)?;
    let tokens = &g["cfg_tokens"]; // [1,S,9]
    let mask = &g["cfg_mask"];
    let uniforms = &g["cfg_uniforms"]; // [8,V]
    let c0g = &g["cfg_c0_guided"]; // [1,V]
    let cig = &g["cfg_ci_guided"]; // [7,V]
    let curr_g: Vec<i64> = g["cfg_curr_sample"].flatten_all()?.to_vec1::<i64>()?;

    println!("loading HeartMuLa LM (15 GB)...");
    let w = LmWeights::load(Path::new(LM), &dev)?;
    let lm = HeartMuLaLm::load(&w, &dev)?;

    println!("generate_frame_cfg (cfg=1.5)...");
    let (samples, c0_rs, ci_rs) = lm.generate_frame_cfg(tokens, mask, 1.5, 50, 1.0, uniforms)?;
    let d0 = max_abs_diff(&c0_rs, c0g)?;
    println!("c0 guided logits max|Δ|={d0:.3e}");
    let mut worst = d0;
    for (i, l) in ci_rs.iter().enumerate() {
        let d = max_abs_diff(l, &cig.narrow(0, i, 1)?)?;
        worst = worst.max(d);
        println!("  codebook {} guided max|Δ|={d:.3e}", i + 1);
    }
    let gen_i: Vec<i64> = samples.iter().map(|&x| x as i64).collect();
    let samples_ok = gen_i == curr_g;
    println!("  samples rust={gen_i:?} golden={curr_g:?} match={samples_ok}");
    println!(
        "{}",
        if worst < 1e-2 && samples_ok {
            "CFG PARITY OK ✅"
        } else {
            "CFG PARITY OFF ❌"
        }
    );
    Ok(())
}
