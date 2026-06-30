//! Parity for the full HeartCodec.detokenize (pad-to-segment + trim).
//!     cargo run --release --example parity_detok

use std::path::Path;

use candle_core::{Device, Tensor};
use melodie_engine::Result;
use melodie_engine::codec::{CodecWeights, DetokCb, HeartCodec};
use melodie_engine::config::HeartCodecConfig;
use melodie_engine::parity::max_abs_diff;

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
    let wav = codec.detokenize(&codes, Some(fm_noise), 7.44, 10, 1.25, DetokCb::default())?;
    let d = max_abs_diff(&wav, wav_g)?;
    let rms = wav_g.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    // The full-pipeline max is dominated by a handful of `round9` (= round(x·9)/9) quantisation
    // -boundary flips: the FM's ~1e-5 numerical error tips a few latent values to the adjacent
    // 1/9 level, which the decoder's conv spreads. So judge on the RMS (the meaningful metric),
    // not the sparse-spike max — the parts (FM ~9e-6, scalar decoder ~5e-6) are individually tight.
    let diff = (&wav - wav_g)?;
    let err_rms = diff.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let av: Vec<f32> = diff.abs()?.flatten_all()?.to_vec1()?;
    let n_big = av.iter().filter(|&&x| x > 1e-3).count();
    println!(
        "waveform {:?} (golden {:?})  max|Δ|={d:.3e}  rms={rms:.3e}",
        wav.dims(),
        wav_g.dims()
    );
    println!(
        "  error RMS={err_rms:.3e}; |Δ|>1e-3 on {n_big}/{} samples ({:.3}%) — sparse round9 flips",
        av.len(),
        100.0 * n_big as f32 / av.len() as f32
    );
    println!(
        "{}",
        if err_rms < 1e-3 && d < 5e-3 {
            "detokenize PARITY OK ✅ (max = sparse round9 flips; RMS tight)"
        } else {
            "PARITY OFF ❌"
        }
    );

    // Streaming self-parity: the `on_audio` chunks, concatenated, must equal the returned wav
    // (interleaved). Random multi-segment codes force the overlap-finalisation path; one call ⇒
    // deterministic, so the internal randn noise is irrelevant.
    println!("streaming self-parity (multi-segment)...");
    let many = Tensor::rand(0f32, 1000f32, (1, 8, 250), &dev)?; // 250 codes ⇒ several 7.44 s segments
    let mut streamed: Vec<f32> = Vec::new();
    let full = {
        let mut on_audio = |pcm: &[f32]| streamed.extend_from_slice(pcm);
        codec.detokenize(
            &many,
            None,
            7.44,
            10,
            1.25,
            DetokCb {
                on_audio: Some(&mut on_audio),
                ..Default::default()
            },
        )?
    };
    let interleaved: Vec<f32> = full
        .transpose(0, 1)?
        .contiguous()?
        .flatten_all()?
        .to_vec1()?;
    let sd = streamed
        .iter()
        .zip(&interleaved)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "streamed {} samples vs returned {}  max|Δ|={sd:.3e}  {}",
        streamed.len(),
        interleaved.len(),
        if streamed.len() == interleaved.len() && sd == 0.0 {
            "STREAM==RETURN ✅"
        } else {
            "STREAM MISMATCH ❌"
        }
    );
    Ok(())
}
