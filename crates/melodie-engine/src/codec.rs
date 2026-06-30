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
use crate::flow::SegmentCtx;
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

/// Row vector `linspace(0, 1, n)` of shape `(1, n)` — the overlap-add crossfade
/// window (`mx.linspace(0,1,ovlp)[None, :]` in modeling_heartcodec.py:175).
fn linspace01_row(n: usize, dev: &Device) -> Result<Tensor> {
    let denom = (n.max(2) - 1) as f32; // n>=2 in practice; guard n==1 against /0
    let v: Vec<f32> = (0..n).map(|i| i as f32 / denom).collect();
    Ok(Tensor::from_vec(v, (1, n), dev)?)
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
    /// Decode `(N, L, C)` → `(N, L*1920)`. Each batch item (e.g. the two stereo channels)
    /// is decoded independently and synchronized, so the ×1920 conv working set is bounded
    /// to ONE item — roughly halving the decode's peak GPU residency vs the batched form,
    /// which keeps long songs inside this Mac's memory budget. The conv stack is
    /// batch-independent, so this is bit-identical to decoding the full batch at once.
    pub fn decode(&self, latent_nlc: &Tensor) -> Result<Tensor> {
        let n = latent_nlc.dim(0)?;
        if n <= 1 {
            return self.decode_one(latent_nlc);
        }
        let mut outs = Vec::with_capacity(n);
        for i in 0..n {
            outs.push(self.decode_one(&latent_nlc.narrow(0, i, 1)?)?);
            latent_nlc.device().synchronize()?;
        }
        Ok(Tensor::cat(&outs, 0)?)
    }

    /// Decode a single-item batch `(1, L, C)` through the full conv stack.
    fn decode_one(&self, latent_nlc: &Tensor) -> Result<Tensor> {
        // (1, L, C) → (1, C, L) and scalar-quantise
        let x = latent_nlc.transpose(1, 2)?.contiguous()?;
        let x = round9(&x)?;
        let mut x = conv1d(&x, &self.conv0_w, &self.conv0_b, false, 1, 1)?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
            // Bound peak memory across the ×1920 upsampling. candle (eager) pools every
            // intermediate for reuse, but each stage's conv `im2col` is a different size so
            // nothing is reused — the pool otherwise grows to the SUM of all stages (~32 GB,
            // OOM → corrupted output). synchronize() flushes the GPU and drops the now-dead
            // pooled buffers (the candle analogue of the reference's per-segment `mx.eval`),
            // capping the peak at a single stage's working set. No-op on CPU.
            x.device().synchronize()?;
        }
        x = self.post.forward(&x)?;
        x = conv1d(&x, &self.conv7_w, &self.conv7_b, true, 1, 1)?;
        x.device().synchronize()?;
        // (1, 1, L) → (1, L)
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

    /// Full multi-segment detokenize — port of `HeartCodec.detokenize`
    /// (modeling_heartcodec.py:76-223). Pads codes to a whole number of
    /// `duration`-second segments, flow-matches each segment (carrying the previous
    /// segment's tail as in-context latents), ScalarModel-decodes, then linearly
    /// crossfades the segments together (overlap-add) and trims to the original length.
    ///
    /// `codes` is `(1, Q, T)`. `fm_noise` (optional) injects the **first** segment's
    /// initial latent `(1, 2*min_samples, 256)` for deterministic parity; when `None`
    /// every segment's latent (and the in-context random pad) is drawn with `randn` —
    /// matching the reference, which draws all randomness internally.
    pub fn detokenize(
        &self,
        codes: &Tensor,
        fm_noise: Option<&Tensor>,
        duration: f64,
        num_steps: usize,
        gs: f64,
    ) -> Result<Tensor> {
        let dev = codes.device();

        // --- segmentation constants (modeling_heartcodec.py:101-123) ---
        let min_samples = (duration * 12.5) as usize; // codes/segment (372 @ 29.76)
        let hop_samples = min_samples / 93 * 80; //                    (320)
        let ovlp_samples = min_samples - hop_samples; // codes overlap (52)
        let ovlp_frames = ovlp_samples * 2; // LATENT-frame overlap     (104)
        let latent_length = (duration * 25.0) as usize; // latent frames/segment (744)
        let sample_rate = 48000usize;

        // target_len uses the ORIGINAL (pre-pad) code length.
        let codes_len0 = codes.dim(2)?;
        let target_len = (codes_len0 as f64 / 12.5 * sample_rate as f64) as usize;

        // --- pad codes (modeling_heartcodec.py:108-122) ---
        let mut codes = codes.clone();
        if codes.dim(2)? < min_samples {
            while codes.dim(2)? < min_samples {
                codes = Tensor::cat(&[&codes, &codes], 2)?; // tile (concat to self)
            }
            codes = codes.narrow(2, 0, min_samples)?;
        }
        let codes_len = codes.dim(2)?;
        // Reference `(codes_len - ovlp_frames) % hop_samples > 0` (codes not segment-aligned).
        if !(codes_len - ovlp_frames).is_multiple_of(hop_samples) {
            // NB: condition uses `ovlp_frames`, formula uses `ovlp_samples` (verbatim).
            let len_codes = ((codes_len - ovlp_samples) as f64 / hop_samples as f64).ceil()
                as usize
                * hop_samples
                + ovlp_samples;
            while codes.dim(2)? < len_codes {
                codes = Tensor::cat(&[&codes, &codes], 2)?;
            }
            codes = codes.narrow(2, 0, len_codes)?;
        }
        let total = codes.dim(2)?;

        // --- audio overlap-add constants (used inside the segment loop) ---
        let min_samples_audio = (duration * sample_rate as f64) as usize; // 1428480
        let hop_samples_audio = min_samples_audio / 93 * 80; //              1228800
        let ovlp_samples_audio = min_samples_audio - hop_samples_audio; //    199680
        let win = if ovlp_samples_audio > 0 {
            Some(linspace01_row(ovlp_samples_audio, dev)?) // (1, ovlp)
        } else {
            None
        };

        // --- interleaved segment loop (modeling_heartcodec.py:126-223) ---
        // CRITICAL ordering: decode each segment IMMEDIATELY after its flow-matching pass,
        // then synchronize() to free the GPU, before moving to the next segment. The
        // reference (and the first port) ran ALL FM passes first and ALL decodes second; on
        // Metal that leaves every segment's FM residency live when the first decode runs, so
        // the decode's ×1920 working set tips over the GPU budget → OOM (localised to
        // `decode segment 0`). Interleaving keeps each segment's peak identical to a verified
        // single-segment decode, so songs of any length stay bounded. Only the previous
        // segment's latent is needed (for the in-context prefix), so we carry just that.
        let mut prev_latent: Option<Tensor> = None;
        let mut output: Option<Tensor> = None;
        let mut sinx = 0usize;
        while sinx + hop_samples <= total {
            let codes_input = codes.narrow(2, sinx, min_samples)?; // (1,Q,min_samples)
            let latents = if sinx == 0 || ovlp_frames == 0 {
                // First segment: incontext_length = 0 (true_latents unused; the
                // reference's `first_latent` randn only matters for its RNG draw).
                let noise = match fm_noise {
                    Some(n) if sinx == 0 => n.clone(),
                    _ => Tensor::randn(0f32, 1f32, (1, 2 * min_samples, 256), dev)?,
                };
                let first_latent = Tensor::randn(0f32, 1f32, (1, latent_length, 256), dev)?;
                let ctx = SegmentCtx {
                    true_latents: &first_latent,
                    latent_length,
                    incontext_length: 0,
                };
                self.fm.inference_codes(&codes_input, &ctx, &noise, num_steps, gs)?
            } else {
                // Subsequent segment: last `ovlp_frames` latents of the previous segment
                // become the in-context prefix, padded with randn to length.
                let prev = prev_latent.as_ref().unwrap();
                let prev_t = prev.dim(1)?;
                let true_tail = prev.narrow(1, prev_t - ovlp_frames, ovlp_frames)?; // (1,104,256)
                let len_add = latent_length - ovlp_frames; // 640
                let pad = Tensor::randn(0f32, 1f32, (1, len_add, 256), dev)?;
                let true_latent = Tensor::cat(&[&true_tail, &pad], 1)?; // (1,744,256)
                let noise = Tensor::randn(0f32, 1f32, (1, 2 * min_samples, 256), dev)?;
                let ctx = SegmentCtx {
                    true_latents: &true_latent,
                    latent_length,
                    incontext_length: ovlp_frames,
                };
                self.fm.inference_codes(&codes_input, &ctx, &noise, num_steps, gs)?
            };

            // Decode this segment NOW (interleaved). (B,T,256) → (B,T,2,128) → (2B,T,128).
            let (b, t_lat, f_lat) = latents.dims3()?;
            let latent_r = latents
                .reshape((b, t_lat, 2, f_lat / 2))?
                .permute((0, 2, 1, 3))?
                .contiguous()?
                .reshape((b * 2, t_lat, f_lat / 2))?;
            let mut cur = self.scalar.decode(&latent_r)?; // (2, L)
            if cur.dim(1)? > min_samples_audio {
                cur = cur.narrow(1, 0, min_samples_audio)?;
            }

            // Overlap-add this segment's audio into the running output.
            output = Some(match output {
                None => cur, // first segment
                Some(prev) => {
                    if ovlp_samples_audio == 0 {
                        Tensor::cat(&[&prev, &cur], 1)?
                    } else {
                        let win = win.as_ref().unwrap();
                        let win_inv = win.affine(-1.0, 1.0)?; // 1 - win
                        let prev_len = prev.dim(1)?;
                        let prev_head = prev.narrow(1, 0, prev_len - ovlp_samples_audio)?;
                        let prev_tail =
                            prev.narrow(1, prev_len - ovlp_samples_audio, ovlp_samples_audio)?;
                        let cur_head = cur.narrow(1, 0, ovlp_samples_audio)?;
                        let cur_rest =
                            cur.narrow(1, ovlp_samples_audio, cur.dim(1)? - ovlp_samples_audio)?;
                        // blended = prev_tail*(1-win) + cur_head*win
                        let blended = (prev_tail.broadcast_mul(&win_inv)?
                            + cur_head.broadcast_mul(win)?)?;
                        Tensor::cat(&[&prev_head, &blended, &cur_rest], 1)?
                    }
                }
            });

            prev_latent = Some(latents); // carry for the next segment's in-context prefix
            sinx += hop_samples;
            // Free this segment's FM + decode GPU residency before the next segment.
            dev.synchronize()?;
        }

        let output = output.unwrap();
        let tl = target_len.min(output.dim(1)?);
        Ok(output.narrow(1, 0, tl)?) // (2, target_len)
    }
}
