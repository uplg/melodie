//! HeartCodec decoder: discrete codes → 48 kHz waveform. **PHASE 1.**
//!
//! Ported from `heartcodec/{modeling_heartcodec.py, flow_matching.py, sq_codec.py}`.
//! This file currently implements the **ScalarModel decoder** (latent → waveform),
//! the pure-conv vocoder half. The RVQ + DiT flow-matching half is P1b-next.
//!
//! Layout note: the MLX reference works in (N, L, C); candle (like PyTorch) works
//! in (N, C, L). We load the **original PyTorch** weights — conv `(out,in,k)` and
//! conv-transpose `(in,out,k)` match candle exactly, so **no weight transpose** is
//! needed; we only fuse weight-norm at load time.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};

use crate::config::HeartCodecConfig;
use crate::{EngineError, Result};

// ---------------------------------------------------------------------------
// weight store + weight-norm fusion
// ---------------------------------------------------------------------------

/// Raw original-PyTorch HeartCodec tensors, keyed by their checkpoint names.
pub struct CodecWeights {
    map: HashMap<String, Tensor>,
}

impl CodecWeights {
    /// Load the 2-shard original safetensors from `ckpt/HeartCodec-oss`.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let mut map = HashMap::new();
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            let m = candle_core::safetensors::load(dir.join(shard), device)?;
            map.extend(m);
        }
        Ok(Self { map })
    }

    fn raw(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| EngineError::Config(format!("missing codec weight `{key}`")))
    }

    /// Public accessor for a raw (plain) tensor by its checkpoint key.
    pub fn tensor(&self, key: &str) -> Result<Tensor> {
        self.raw(key)
    }

    /// Effective weight from a weight-normed conv: `g * v / ||v||` (norm over dims 1,2).
    /// `prefix` e.g. `scalar_model.decoder.0` (looks up `.parametrizations.weight.original{0,1}`).
    fn fused(&self, prefix: &str) -> Result<Tensor> {
        let g = self.raw(&format!("{prefix}.parametrizations.weight.original0"))?;
        let v = self.raw(&format!("{prefix}.parametrizations.weight.original1"))?;
        // norm over all dims except 0 (here: 1 and 2), keepdim → (out,1,1)
        let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
        Ok(g.broadcast_mul(&v)?.broadcast_div(&norm)?)
    }

    fn prelu(&self, key: &str) -> Result<Tensor> {
        // PReLU weight is a single parameter [1]; reshape to (1,1,1) for NCL broadcast.
        self.raw(key)?.reshape((1, 1, 1)).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// conv primitives (NCL), matching sq_codec.py semantics
// ---------------------------------------------------------------------------

fn conv1d(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    causal: bool,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let k = w.dim(2)?;
    // causal = left-only pad of dilation*(k-1); non-causal = symmetric padding.
    let (padding, xin) = if causal {
        (0, x.pad_with_zeros(2, dilation * (k - 1), 0)?)
    } else {
        ((k * dilation - dilation) / 2, x.clone())
    };
    let cfg = Conv1dConfig { padding, stride, dilation, groups: 1, ..Default::default() };
    Ok(Conv1d::new(w.clone(), Some(b.clone()), cfg).forward(&xin)?)
}

/// Insert `stride-1` zeros between successive time samples (NCL, dim 2):
/// `(N,C,L)` → `(N,C,(L-1)*stride+1)`.
fn zero_stuff_time(x: &Tensor, stride: usize) -> Result<Tensor> {
    if stride == 1 {
        return Ok(x.clone());
    }
    let (n, c, l) = x.dims3()?;
    let x = x.unsqueeze(3)?.pad_with_zeros(3, 0, stride - 1)?; // (n,c,l,stride)
    let x = x.contiguous()?.reshape((n, c, l * stride))?;
    Ok(x.narrow(2, 0, (l - 1) * stride + 1)?)
}

/// Transposed conv written out explicitly (zero-stuff by `stride`, full-pad, then
/// cross-correlate with the flipped kernel). Equivalent to candle's built-in
/// `conv_transpose1d` (which is itself PyTorch-correct); kept in this explicit form
/// because it is verified bit-for-bit against the MLX reference across the decoder.
/// `w_inok` is `(in,out,k)`; output is causally trimmed of its last `stride` samples.
fn conv_transpose1d_causal_manual(
    x: &Tensor,
    w_inok: &Tensor,
    b: &Tensor,
    stride: usize,
) -> Result<Tensor> {
    let k = w_inok.dim(2)?;
    let xu = zero_stuff_time(x, stride)?.pad_with_zeros(2, k - 1, k - 1)?;
    let kern = flip_last_dim(&w_inok.permute((1, 0, 2))?.contiguous()?)?; // (out,in,k), k-flipped
    let cfg = Conv1dConfig { padding: 0, stride: 1, ..Default::default() };
    let y = Conv1d::new(kern, Some(b.clone()), cfg).forward(&xu)?;
    let l = y.dim(2)?;
    Ok(y.narrow(2, 0, l - stride)?)
}

/// PReLU: `x>=0 ? x : w*x`, computed as `relu(x) - w*relu(-x)`.
fn prelu(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = x.neg()?.relu()?;
    Ok(pos.sub(&neg.broadcast_mul(w)?)?)
}

/// Scalar quantisation: `round(9*x)/9`.
fn round9(x: &Tensor) -> Result<Tensor> {
    Ok(((x * 9.0)?.round()? / 9.0)?)
}

/// Reverse a 3-D tensor along its last dim (kernel time axis).
fn flip_last_dim(w: &Tensor) -> Result<Tensor> {
    let k = w.dim(2)?;
    let idx: Vec<u32> = (0..k as u32).rev().collect();
    let idx = Tensor::from_vec(idx, k, w.device())?;
    Ok(w.index_select(&idx, 2)?)
}

/// Nearest-neighbour ×2 upsample along the time axis (dim 2): repeat_interleave.
fn repeat_interleave2_time(x: &Tensor) -> Result<Tensor> {
    let (n, c, l) = x.dims3()?;
    let x = x.unsqueeze(3)?.broadcast_as((n, c, l, 2))?;
    Ok(x.contiguous()?.reshape((n, c, l * 2))?)
}

// ---------------------------------------------------------------------------
// decoder modules
// ---------------------------------------------------------------------------

struct ResidualUnit {
    c1_w: Tensor,
    c1_b: Tensor,
    dilation: usize,
    c2_w: Tensor,
    c2_b: Tensor,
    a1: Tensor,
    a2: Tensor,
}

impl ResidualUnit {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = conv1d(x, &self.c1_w, &self.c1_b, true, 1, self.dilation)?;
        let out = prelu(&out, &self.a1)?;
        let out = conv1d(&out, &self.c2_w, &self.c2_b, true, 1, 1)?;
        let out = prelu(&out, &self.a2)?;
        Ok((out + x)?)
    }
}

struct ResDecoderBlock {
    up_w: Tensor,
    up_b: Tensor,
    stride: usize,
    units: Vec<ResidualUnit>,
}

impl ResDecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // explicit transposed-conv form (verified against the MLX golden)
        let mut x = conv_transpose1d_causal_manual(x, &self.up_w, &self.up_b, self.stride)?;
        for u in &self.units {
            x = u.forward(&x)?;
        }
        Ok(x)
    }
}

struct PostProcessor {
    conv_w: Tensor,
    conv_b: Tensor,
    act: Tensor,
}

impl PostProcessor {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = repeat_interleave2_time(x)?;
        let x = conv1d(&x, &self.conv_w, &self.conv_b, true, 1, 1)?;
        prelu(&x, &self.act)
    }
}

/// The ScalarModel decoder: scalar-quantised latent (N,L,128) → waveform (N, L*1920).
pub struct ScalarDecoder {
    conv0_w: Tensor, // look-ahead, non-causal
    conv0_b: Tensor,
    blocks: Vec<ResDecoderBlock>,
    post: PostProcessor,
    conv7_w: Tensor, // final, causal
    conv7_b: Tensor,
}

impl ScalarDecoder {
    pub fn load(w: &CodecWeights, _cfg: &HeartCodecConfig, _device: &Device) -> Result<Self> {
        let up_factors = [5usize, 4, 4, 4, 3];
        let res_dils = [1usize, 3, 5, 7, 9];

        let conv0_w = w.fused("scalar_model.decoder.0")?;
        let conv0_b = w.raw("scalar_model.decoder.0.bias")?;

        let mut blocks = Vec::new();
        for (i, &stride) in up_factors.iter().enumerate() {
            let n = i + 1; // decoder.{1..5}
            let up_w = w.fused(&format!("scalar_model.decoder.{n}.up_conv.layer"))?;
            let up_b = w.raw(&format!("scalar_model.decoder.{n}.up_conv.layer.bias"))?;
            let mut units = Vec::new();
            for (j, &dilation) in res_dils.iter().enumerate() {
                let p = format!("scalar_model.decoder.{n}.convs.{j}");
                units.push(ResidualUnit {
                    c1_w: w.fused(&format!("{p}.conv1"))?,
                    c1_b: w.raw(&format!("{p}.conv1.bias"))?,
                    dilation,
                    c2_w: w.fused(&format!("{p}.conv2"))?,
                    c2_b: w.raw(&format!("{p}.conv2.bias"))?,
                    a1: w.prelu(&format!("{p}.activation1.weight"))?,
                    a2: w.prelu(&format!("{p}.activation2.weight"))?,
                });
            }
            blocks.push(ResDecoderBlock { up_w, up_b, stride, units });
        }

        let post = PostProcessor {
            conv_w: w.raw("scalar_model.decoder.6.conv.weight")?, // plain (no weight-norm)
            conv_b: w.raw("scalar_model.decoder.6.conv.bias")?,
            act: w.prelu("scalar_model.decoder.6.activation.weight")?,
        };

        let conv7_w = w.fused("scalar_model.decoder.7")?;
        let conv7_b = w.raw("scalar_model.decoder.7.bias")?;

        Ok(Self { conv0_w, conv0_b, blocks, post, conv7_w, conv7_b })
    }

    /// Decode a latent `(N, L, 128)` (MLX-layout, as dumped) → waveform `(N, L*1920)`.
    pub fn decode(&self, latent_nlc: &Tensor) -> Result<Tensor> {
        // (N, L, C) → (N, C, L) and scalar-quantise
        let x = latent_nlc.transpose(1, 2)?.contiguous()?;
        let x = round9(&x)?;
        let mut x = conv1d(&x, &self.conv0_w, &self.conv0_b, false, 1, 1)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        x = self.post.forward(&x)?;
        x = conv1d(&x, &self.conv7_w, &self.conv7_b, true, 1, 1)?;
        // (N, 1, L) → (N, L)
        Ok(x.squeeze(1)?)
    }

    /// Like [`decode`], but returns the NCL output of each decoder stage
    /// (conv0, blocks 0..4, postprocessor, final conv) for parity localisation.
    pub fn decode_tapped(&self, latent_nlc: &Tensor) -> Result<Vec<Tensor>> {
        let x = latent_nlc.transpose(1, 2)?.contiguous()?;
        let x = round9(&x)?;
        let mut taps = Vec::new();
        let mut x = conv1d(&x, &self.conv0_w, &self.conv0_b, false, 1, 1)?;
        taps.push(x.clone()); // dec0
        for blk in &self.blocks {
            x = blk.forward(&x)?;
            taps.push(x.clone()); // dec1..dec5
        }
        x = self.post.forward(&x)?;
        taps.push(x.clone()); // dec6
        x = conv1d(&x, &self.conv7_w, &self.conv7_b, true, 1, 1)?;
        taps.push(x.clone()); // dec7
        Ok(taps)
    }
}

// ---------------------------------------------------------------------------
// top-level (FM half still TODO)
// ---------------------------------------------------------------------------

/// HeartCodec: FlowMatching (codes→latent) + ScalarModel (latent→waveform).
pub struct HeartCodec {
    fm: crate::flow::FlowMatching,
    scalar: ScalarDecoder,
}

impl HeartCodec {
    pub fn load(w: &CodecWeights, cfg: &HeartCodecConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            fm: crate::flow::FlowMatching::load(w, cfg)?,
            scalar: ScalarDecoder::load(w, cfg, device)?,
        })
    }

    /// Decode one segment: codes `(1,Q,T)` + initial `noise` `(1,2T,256)` → waveform `(2, 2T*1920)`.
    /// (Single-segment path; multi-segment overlap-add is layered on top later.)
    pub fn detokenize_segment(&self, codes: &Tensor, noise: &Tensor, num_steps: usize, gs: f64) -> Result<Tensor> {
        let fm_latents = self.fm.inference(codes, noise, num_steps, gs)?; // (1,2T,256)
        let (b, t2, f) = fm_latents.dims3()?;
        // reshape (B,2T,256) -> (B,2T,2,128) -> (B,2,2T,128) -> (2B,2T,128)  (modeling_heartcodec.py:184-186)
        let latent = fm_latents
            .reshape((b, t2, 2, f / 2))?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b * 2, t2, f / 2))?;
        self.scalar.decode(&latent) // (2, 2T*1920)
    }

    /// Full detokenize for a clip shorter than one segment: pad codes to the
    /// `duration`-second segment (≈372 frames), flow-match + ScalarModel decode,
    /// then trim to the original length. Mirrors `HeartCodec.detokenize`
    /// (modeling_heartcodec.py:96-223) for the single-segment case. `fm_noise` is
    /// the injected `(1, 2*min_samples, 256)` latent noise.
    pub fn detokenize(&self, codes: &Tensor, fm_noise: &Tensor, duration: f64, num_steps: usize, gs: f64) -> Result<Tensor> {
        let t_orig = codes.dim(2)?;
        let min_samples = (duration * 12.5) as usize;
        let target_len = (t_orig as f64 / 12.5 * 48000.0) as usize;
        // tile codes to min_samples (repeat-and-truncate)
        let mut c = codes.clone();
        while c.dim(2)? < min_samples {
            c = Tensor::cat(&[&c, &c], 2)?;
        }
        let c = c.narrow(2, 0, min_samples)?;
        let wav = self.detokenize_segment(&c, fm_noise, num_steps, gs)?; // (2, min_samples*2*1920)
        Ok(wav.narrow(1, 0, target_len)?) // (2, target_len)
    }
}
