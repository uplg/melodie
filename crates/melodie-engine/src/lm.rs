//! HeartMuLa LM: RQ-Transformer that emits audio codes. **PHASE 2.**
//!
//! Ported from `heartmula/modeling_heartmula.py`. A Llama-3.1 backbone (~3B)
//! predicts codebook 0 per 12.5 Hz frame; a 300M depth decoder autoregresses
//! codebooks 1–7. GQA, Llama-3.1 *scaled* RoPE, RMSNorm, SwiGLU. Transformer
//! outputs are upcast to f32 (else codebooks 1–7 corrupt in bf16).
//!
//! candle is NCL/(out,in) like PyTorch, so the original safetensors load with no
//! transpose; norms are stored as `.scale`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::config::{HeartMuLaConfig, LlamaFlavor};
use crate::{EngineError, Result};

// --- weights -------------------------------------------------------------

pub struct LmWeights {
    map: HashMap<String, Tensor>,
    dtype: DType,
}
impl LmWeights {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        // bf16 on Metal (the reference runtime; matmuls are bandwidth-bound on the weights so
        // bf16 ≈ 2× f32). f32 on CPU — candle's CPU backend has no bf16 matmul, and CPU is
        // only used for bit-exact parity tests.
        let dtype = if matches!(device, Device::Metal(_)) {
            DType::BF16
        } else {
            DType::F32
        };
        // For bf16-on-Metal: cast f32→bf16 on the CPU and move only the 7.5 GB bf16 to Metal,
        // freeing each f32 shard before the next. Loading all 15 GB f32 on Metal and casting
        // there peaks at 22.5 GB (f32 + bf16) which OOMs Metal — and a failed Metal allocation
        // returns a ZERO buffer rather than erroring, so every weight silently became 0 and the
        // whole prompt collapsed. CPU cast is reliable; peak stays ≈ 11 GB.
        let bf16_metal = matches!(device, Device::Metal(_)) && dtype == DType::BF16;
        let mut map = HashMap::new();
        for shard in [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ] {
            if bf16_metal {
                for (k, v) in candle_core::safetensors::load(dir.join(shard), &Device::Cpu)? {
                    map.insert(k, v.to_dtype(dtype)?.to_device(device)?);
                }
            } else {
                map.extend(candle_core::safetensors::load(dir.join(shard), device)?);
            }
        }
        Ok(Self { map, dtype })
    }
    fn t(&self, key: &str) -> Result<Tensor> {
        // map is already at `dtype` (cast at load), so this shares the buffer — no extra copy.
        let r = self.raw(key)?;
        if r.dtype() == self.dtype {
            Ok(r)
        } else {
            Ok(r.to_dtype(self.dtype)?)
        }
    }
    /// Head/output weight: always f32 (runs on the f32-upcast transformer output, else
    /// codebooks 1-7 corrupt). On Metal, round to bf16 to match the reference (which
    /// promotes its bf16 weight to f32); on CPU keep raw f32 for bit-exact parity.
    fn t_head(&self, key: &str) -> Result<Tensor> {
        let raw = self.raw(key)?;
        if self.dtype == DType::BF16 {
            Ok(raw.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)
        } else {
            Ok(raw)
        }
    }
    fn raw(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| EngineError::Config(format!("missing LM weight `{key}`")))
    }
}

// --- helpers -------------------------------------------------------------

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}
fn rmsnorm(x: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    // candle's fused RMSNorm kernel: 1 Metal launch vs ~6 hand-rolled ops, and it
    // upcasts bf16/f16 to f32 internally for the reduction (same precision).
    Ok(candle_nn::ops::rms_norm(
        &x.contiguous()?,
        scale,
        eps as f32,
    )?)
}

struct Lin(Linear);
impl Lin {
    fn load(w: &LmWeights, key: &str) -> Result<Self> {
        Ok(Self(Linear::new(w.t(&format!("{key}.weight"))?, None)))
    }
    /// f32 head projection (bf16-rounded weight), runs on the f32 transformer output.
    fn load_head(w: &LmWeights, key: &str) -> Result<Self> {
        Ok(Self(Linear::new(w.t_head(&format!("{key}.weight"))?, None)))
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        // Collapse [B,S,D] -> [B*S,D] so candle issues ONE 2D gemm (weight read once)
        // instead of a batched 3D gemm that re-reads the weight per batch element
        // (~2.7x slower for B=2 / CFG). Reshape of a contiguous tensor is a free view.
        if let [b, s, d] = *x.dims() {
            let y = self.0.forward(&x.reshape((b * s, d))?)?;
            let n = y.dim(1)?;
            Ok(y.reshape((b, s, n))?)
        } else {
            Ok(self.0.forward(x)?)
        }
    }
}

// --- Llama-3.1 scaled RoPE ----------------------------------------------

fn apply_scaling(freq: f64, scale_factor: f64, low: f64, high: f64, old_ctx: f64) -> f64 {
    let low_wl = old_ctx / low;
    let high_wl = old_ctx / high;
    let wavelen = 2.0 * std::f64::consts::PI / freq;
    if wavelen < high_wl {
        freq
    } else if wavelen > low_wl {
        freq / scale_factor
    } else {
        let smooth = (old_ctx / wavelen - low) / (high - low);
        (1.0 - smooth) * freq / scale_factor + smooth * freq
    }
}

/// RoPE cache `[max_seq, dim/2, 2]` (cos, sin), Llama-3.1 scaled.
fn build_rope_cache(
    dim: usize,
    max_seq: usize,
    base: f64,
    scale: f64,
    dev: &Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let theta: Vec<f64> = (0..half)
        .map(|i| {
            let freq = 1.0 / base.powf(2.0 * i as f64 / dim as f64);
            apply_scaling(freq, scale, 1.0, 4.0, 8192.0)
        })
        .collect();
    let mut data = vec![0f32; max_seq * half * 2];
    for s in 0..max_seq {
        for i in 0..half {
            let a = s as f64 * theta[i];
            data[(s * half + i) * 2] = a.cos() as f32;
            data[(s * half + i) * 2 + 1] = a.sin() as f32;
        }
    }
    Ok(Tensor::from_vec(data, (max_seq, half, 2), dev)?)
}

// --- attention (GQA) + KV cache -----------------------------------------

/// Pre-allocated KV cache: new keys/values are written in place (`slice_set`) into a buffer
/// sized once, instead of `cat`-growing every frame. The `cat` form reallocated the whole
/// cache each step and left a different-sized freed buffer that candle's Metal pool cannot
/// reuse → multi-GB of churn per song (exactly what the per-frame `synchronize()` had to
/// drain). Pre-allocating removes the churn at its root.
struct LayerCache {
    inner: candle_nn::kv_cache::KvCache,
}
impl LayerCache {
    fn new(cap: usize) -> Self {
        // sequence axis is dim 2 in `[B, H, S, D]`.
        Self {
            inner: candle_nn::kv_cache::KvCache::new(2, cap),
        }
    }
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok(self.inner.append(k, v)?)
    }
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
}
impl Attn {
    fn load(w: &LmWeights, prefix: &str, f: &LlamaFlavor) -> Result<Self> {
        Ok(Self {
            q: Lin::load(w, &format!("{prefix}.q_proj"))?,
            k: Lin::load(w, &format!("{prefix}.k_proj"))?,
            v: Lin::load(w, &format!("{prefix}.v_proj"))?,
            o: Lin::load(w, &format!("{prefix}.output_proj"))?,
            n_heads: f.num_heads,
            n_kv: f.num_kv_heads,
            head_dim: f.head_dim,
        })
    }
    fn expand_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
        // [B,nkv,S,D] -> repeat_interleave on head axis -> [B,nkv*rep,S,D]
        let (b, nkv, s, d) = x.dims4()?;
        Ok(x.unsqueeze(2)?
            .broadcast_as((b, nkv, rep, s, d))?
            .contiguous()?
            .reshape((b, nkv * rep, s, d))?)
    }
    fn fwd(
        &self,
        x: &Tensor,
        rope_cache: &Tensor,
        positions: &Tensor, // u32 [Sx]
        add_mask: &Tensor,  // [Sx, Skv] additive (0 / -1e9)
        cache: &mut LayerCache,
    ) -> Result<Tensor> {
        let (b, sx, _) = x.dims3()?;
        let (h, nkv, dh) = (self.n_heads, self.n_kv, self.head_dim);
        let q = self
            .q
            .fwd(x)?
            .reshape((b, sx, h, dh))?
            .transpose(1, 2)?
            .contiguous()?; // [B,H,Sx,D]
        let k = self
            .k
            .fwd(x)?
            .reshape((b, sx, nkv, dh))?
            .transpose(1, 2)?
            .contiguous()?; // [B,nkv,Sx,D]
        let v = self
            .v
            .fwd(x)?
            .reshape((b, sx, nkv, dh))?
            .transpose(1, 2)?
            .contiguous()?;
        // fused interleaved RoPE (1 kernel vs ~10 hand-rolled ops); cos/sin [Sx, dh/2]
        let rc = rope_cache.index_select(positions, 0)?; // [Sx, dh/2, 2]
        let cos = rc
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .to_dtype(q.dtype())?
            .contiguous()?;
        let sin = rc
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .to_dtype(q.dtype())?
            .contiguous()?;
        let q = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;
        let k = Self::expand_kv(&k, h / nkv)?; // [B,H,Sx,D]
        let v = Self::expand_kv(&v, h / nkv)?;
        let (k, v) = cache.append(&k, &v)?; // [B,H,Skv,D]
        // Fused scaled-dot-product attention: one kernel for scores+softmax+(@v) and,
        // crucially, it never materialises k^T — the manual path copied the WHOLE KV
        // cache every frame (O(cache) → O(T²) over a song; ~7x slower at cache 500).
        // `do_causal` covers the prompt (Sx>1); a single generated token attends all.
        let attn = if matches!(q.device(), Device::Metal(_)) && dh <= 256 {
            // Metal: fused SDPA (no k^T materialisation; scores+softmax+@v in one kernel).
            // Used for the backbone (head_dim 128, large cache); the depth decoder
            // (head_dim 384, unsupported by Metal SDPA) falls to the manual path below —
            // its cache is ≤9 so the k^T copy is negligible there.
            // do_causal covers the prompt (Sx>1); a single generated token attends all.
            candle_nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                sx > 1,
                (1.0 / (dh as f64).sqrt()) as f32,
                1.0,
            )?
        } else {
            // CPU (parity only — SDPA has no CPU impl): manual attention with the mask.
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (dh as f64).sqrt())?;
            let skv = k.dim(2)?;
            let scores = scores.broadcast_add(
                &add_mask
                    .to_dtype(scores.dtype())?
                    .reshape((1, 1, sx, skv))?,
            )?;
            let w = candle_nn::ops::softmax(&scores, D::Minus1)?;
            w.matmul(&v)?
        };
        let out = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, sx, h * dh))?;
        self.o.fwd(&out)
    }
}

struct Mlp {
    w1: Lin,
    w2: Lin,
    w3: Lin,
}
impl Mlp {
    fn load(w: &LmWeights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: Lin::load(w, &format!("{prefix}.w1"))?,
            w2: Lin::load(w, &format!("{prefix}.w2"))?,
            w3: Lin::load(w, &format!("{prefix}.w3"))?,
        })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        self.w2.fwd(&(silu(&self.w1.fwd(x)?)? * self.w3.fwd(x)?)?)
    }
}

struct Layer {
    sa_norm: Tensor,
    attn: Attn,
    mlp_norm: Tensor,
    mlp: Mlp,
    eps: f64,
}
impl Layer {
    fn load(w: &LmWeights, prefix: &str, f: &LlamaFlavor, eps: f64) -> Result<Self> {
        Ok(Self {
            sa_norm: w.t(&format!("{prefix}.sa_norm.scale"))?,
            attn: Attn::load(w, &format!("{prefix}.attn"), f)?,
            mlp_norm: w.t(&format!("{prefix}.mlp_norm.scale"))?,
            mlp: Mlp::load(w, &format!("{prefix}.mlp"))?,
            eps,
        })
    }
    fn fwd(
        &self,
        x: &Tensor,
        rope: &Tensor,
        pos: &Tensor,
        mask: &Tensor,
        cache: &mut LayerCache,
    ) -> Result<Tensor> {
        let h = (self.attn.fwd(
            &rmsnorm(x, &self.sa_norm, self.eps)?,
            rope,
            pos,
            mask,
            cache,
        )? + x)?;
        Ok((self.mlp.fwd(&rmsnorm(&h, &self.mlp_norm, self.eps)?)? + &h)?)
    }
}

struct Transformer {
    layers: Vec<Layer>,
    norm: Tensor,
    rope_cache: Tensor,
    eps: f64,
}
impl Transformer {
    fn load(
        w: &LmWeights,
        prefix: &str,
        f: &LlamaFlavor,
        cfg: &HeartMuLaConfig,
        dev: &Device,
    ) -> Result<Self> {
        let layers = (0..f.num_layers)
            .map(|i| Layer::load(w, &format!("{prefix}.layers.{i}"), f, cfg.norm_eps))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            norm: w.t(&format!("{prefix}.norm.scale"))?,
            rope_cache: build_rope_cache(
                f.head_dim,
                f.max_seq_len,
                cfg.rope_base,
                cfg.rope_scale_factor,
                dev,
            )?,
            eps: cfg.norm_eps,
        })
    }
    fn fresh_caches(&self, cap: usize) -> Vec<LayerCache> {
        (0..self.layers.len())
            .map(|_| LayerCache::new(cap))
            .collect()
    }
    /// Forward `h [B,S,D]` at `positions [S]` with additive `mask [S,Skv]`; returns
    /// the normalised output upcast to f32.
    fn forward(
        &self,
        h: &Tensor,
        positions: &Tensor,
        mask: &Tensor,
        caches: &mut [LayerCache],
    ) -> Result<Tensor> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FWD: AtomicUsize = AtomicUsize::new(0);
        let is_bb = self.layers.len() > 10; // backbone (28 layers), not the depth decoder (3)
        let do_dbg = std::env::var("MELODIE_DBG").is_ok()
            && is_bb
            && FWD.fetch_add(1, Ordering::Relaxed) == 15;
        let mut h = h.clone();
        for (i, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            h = layer.fwd(&h, &self.rope_cache, positions, mask, cache)?;
            if do_dbg {
                let v: Vec<f32> = h.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let denorm = v
                    .iter()
                    .filter(|&&x| x != 0.0 && x.abs() < 1.18e-38)
                    .count();
                let small = v.iter().filter(|&&x| x != 0.0 && x.abs() < 1e-20).count();
                let maxa = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                eprintln!(
                    "[dbg] layer {i:2}: max={maxa:.2e} denorm={denorm} small(<1e-20)={small}/{}",
                    v.len()
                );
            }
        }
        Ok(rmsnorm(&h, &self.norm, self.eps)?.to_dtype(DType::F32)?)
    }
}

// --- additive causal masks ----------------------------------------------

/// Causal additive mask `[S,S]`: 0 if j<=i else -1e9.
fn causal_mask(s: usize, dev: &Device) -> Result<Tensor> {
    let mut d = vec![0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i {
                d[i * s + j] = -1e9;
            }
        }
    }
    Ok(Tensor::from_vec(d, (s, s), dev)?)
}

/// Top-k Gumbel sampling matching `_sample_topk` (modeling_heartmula.py:384-407),
/// with the uniform draw injected for reproducibility. B=1; returns the token id.
fn sample_topk(logits: &Tensor, topk: usize, temperature: f64, uniform: &Tensor) -> Result<u32> {
    let v: Vec<f32> = logits
        .affine(1.0 / temperature, 0.0)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let u: Vec<f32> = uniform.flatten_all()?.to_vec1::<f32>()?;
    // threshold = k-th largest value (mx.topk is ascending; keep logits >= threshold).
    // `total_cmp` (not `partial_cmp().unwrap()`) so a NaN logit — bf16 underflow on a
    // bad frame, say — sorts to a deterministic spot instead of panicking and taking
    // the whole engine worker thread down with it; see [`EngineError`] callers.
    let mut sorted = v.clone();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let thr = sorted[topk - 1];
    let masked: Vec<f32> = v.iter().map(|&x| if x < thr { -1e9 } else { x }).collect();
    let maxv = masked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = maxv + masked.iter().map(|&x| (x - maxv).exp()).sum::<f32>().ln();
    // Gumbel-max: argmax(exp(x-lse) / -ln(u))
    let (mut best, mut best_val) = (0usize, f32::NEG_INFINITY);
    for (i, &x) in masked.iter().enumerate() {
        let val = (x - lse).exp() / -(u[i].ln());
        if val > best_val {
            best_val = val;
            best = i;
        }
    }
    Ok(best as u32)
}

/// GPU-resident top-k Gumbel sampling: same math as [`sample_topk`] but stays on
/// the device (returns a `[1]` u32 token tensor) so the autoregressive depth loop
/// doesn't sync GPU→CPU 8× per frame. `uniform` is the injected/random draw `[1,V]`.
fn sample_topk_gpu(
    logits: &Tensor,
    topk: usize,
    temperature: f64,
    uniform: &Tensor,
) -> Result<Tensor> {
    let scaled = logits.affine(1.0 / temperature, 0.0)?; // [1,V]
    let masked = if std::env::var("MELODIE_NOTOPK").is_ok() {
        scaled.clone() // DIAGNOSTIC: skip the top-k sort
    } else {
        let (sorted, _) = scaled.sort_last_dim(false)?; // descending
        let threshold = sorted.narrow(D::Minus1, topk - 1, 1)?; // [1,1] = k-th largest
        let neg = scaled.affine(0.0, -1e9)?; // [1,V] all -1e9
        let keep = scaled.broadcast_ge(&threshold)?; // u8 mask: scaled >= threshold
        keep.where_cond(&scaled, &neg)?
    };
    let log_probs = candle_nn::ops::log_softmax(&masked, D::Minus1)?;
    let probs = log_probs.exp()?;
    let q = uniform.log()?.neg()?; // -log(u)
    let score = probs.broadcast_div(&q)?;
    score.argmax(D::Minus1).map_err(Into::into) // [1] u32
}

// --- HeartMuLa wrapper ---------------------------------------------------

/// Sampling knobs for [`HeartMuLaLm::generate_codes`] (grouped so the call stays
/// within clippy's argument-count limit).
#[derive(Clone, Copy, Debug)]
pub struct GenParams {
    /// Classifier-free guidance scale (`>1.0` ⇒ cond+uncond batch).
    pub cfg_scale: f64,
    /// Hard cap on generated frames (12.5 Hz); generation also stops on EOS.
    pub max_frames: usize,
    /// Top-k sampling constraint.
    pub topk: usize,
    /// Sampling temperature.
    pub temperature: f64,
}

pub struct HeartMuLaLm {
    backbone: Transformer,
    decoder: Transformer,
    text_embeddings: Tensor,  // [text_vocab, D]
    audio_embeddings: Tensor, // [audio_vocab*ncb, D]
    projection: Lin,
    codebook0_head: Lin,
    audio_head: Tensor,  // [ncb-1, D, V]
    muq_bias: Tensor,    // muq_linear bias (= muq_linear(zeros), the MuQ-slot embedding)
    uncond_text: Tensor, // unconditional_text_embedding [1, D] (CFG null token)
    cfg: HeartMuLaConfig,
}

impl HeartMuLaLm {
    pub fn load(w: &LmWeights, dev: &Device) -> Result<Self> {
        let cfg = HeartMuLaConfig::default();
        Ok(Self {
            backbone: Transformer::load(w, "backbone", &cfg.backbone, &cfg, dev)?,
            decoder: Transformer::load(w, "decoder", &cfg.decoder, &cfg, dev)?,
            text_embeddings: w.t("text_embeddings.weight")?,
            audio_embeddings: w.t("audio_embeddings.weight")?,
            projection: Lin::load(w, "projection")?,
            codebook0_head: Lin::load_head(w, "codebook0_head")?,
            audio_head: w.t_head("audio_head")?,
            muq_bias: w.t("muq_linear.bias")?,
            uncond_text: w.t("unconditional_text_embedding.weight")?,
            cfg,
        })
    }

    fn embed_audio(&self, codebook: usize, token: &Tensor) -> Result<Tensor> {
        // token [B,1] (int) -> [B,1,D]
        let (b, s) = token.dims2()?;
        let d = self.cfg.backbone.embed_dim;
        let off = (codebook * self.cfg.audio_vocab_size) as f64;
        let idx = token
            .to_dtype(DType::F32)?
            .affine(1.0, off)?
            .to_dtype(DType::U32)?
            .flatten_all()?;
        Ok(self
            .audio_embeddings
            .index_select(&idx, 0)?
            .reshape((b, s, d))?)
    }

    /// Embed a single token id duplicated across a batch of 2 (CFG): -> [2,1,D].
    fn embed_audio_dup(&self, codebook: usize, token: u32) -> Result<Tensor> {
        let t = Tensor::from_vec(
            vec![token as i64; 2],
            (2, 1),
            &self.text_embeddings.device().clone(),
        )?;
        self.embed_audio(codebook, &t)
    }

    /// Embed a `[B,S,ncb+1]` token grid (text in last channel) and mask-sum to `[B,S,D]`.
    fn embed_and_sum(&self, tokens: &Tensor, mask: &Tensor, uncond: bool) -> Result<Tensor> {
        let (b, s, _) = tokens.dims3()?;
        let ncb = self.cfg.audio_num_codebooks;
        let d = self.cfg.backbone.embed_dim;
        // text channel
        let text_idx = tokens
            .narrow(2, ncb, 1)?
            .squeeze(2)?
            .to_dtype(DType::U32)?
            .flatten_all()?;
        let mut text = self
            .text_embeddings
            .index_select(&text_idx, 0)?
            .reshape((b, s, 1, d))?;
        if uncond {
            // CFG: the uncond half (second half of the batch) uses the null text embedding
            let actual = b / 2;
            let ut = self
                .uncond_text
                .reshape((1, 1, 1, d))?
                .broadcast_as((b - actual, s, 1, d))?
                .contiguous()?;
            text = Tensor::cat(&[&text.narrow(0, 0, actual)?, &ut], 0)?;
        }
        // audio channels with per-codebook offset
        let offsets: Vec<f32> = (0..ncb)
            .map(|c| (c * self.cfg.audio_vocab_size) as f32)
            .collect();
        let offsets = Tensor::from_vec(offsets, (1, 1, ncb), tokens.device())?;
        let audio_tok = tokens
            .narrow(2, 0, ncb)?
            .to_dtype(DType::F32)?
            .broadcast_add(&offsets)?;
        let audio_idx = audio_tok.to_dtype(DType::U32)?.flatten_all()?;
        let audio = self
            .audio_embeddings
            .index_select(&audio_idx, 0)?
            .reshape((b, s, ncb, d))?;
        let embeds = Tensor::cat(&[&audio, &text], 2)?; // [B,S,ncb+1,D]
        let masked = embeds.broadcast_mul(&mask.to_dtype(embeds.dtype())?.unsqueeze(3)?)?;
        Ok(masked.sum(2)?) // [B,S,D]
    }

    /// Backbone gate: prompt tokens -> (last_h [B,D], codebook-0 logits [B,V]).
    pub fn backbone_c0(&self, tokens: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let dev = tokens.device();
        let s = tokens.dim(1)?;
        let h = self.embed_and_sum(tokens, mask, false)?; // [B,S,D]
        let pos: Vec<u32> = (0..s as u32).collect();
        let pos = Tensor::from_vec(pos, s, dev)?;
        let m = causal_mask(s, dev)?;
        let mut caches = self.backbone.fresh_caches(s);
        let out = self.backbone.forward(&h, &pos, &m, &mut caches)?; // [B,S,D] f32
        let last_h = out.narrow(1, s - 1, 1)?.squeeze(1)?.contiguous()?; // [B,D] (contiguous: Metal matmul mishandles strided B=2)
        let c0 = self.codebook0_head.fwd(&last_h)?; // [B,V]
        Ok((last_h, c0))
    }

    /// Depth gate: given backbone `last_h [B,D]` and the replayed per-codebook samples
    /// `samples [B,ncb]`, return the codebook 1..7 logits `[ncb-1, B, V]`.
    pub fn depth_ci(&self, last_h: &Tensor, samples: &Tensor) -> Result<Vec<Tensor>> {
        let dev = last_h.device();
        let b = last_h.dim(0)?;
        let ncb = self.cfg.audio_num_codebooks;
        let mut caches = self.decoder.fresh_caches(self.cfg.audio_num_codebooks + 1);
        let mut logits = Vec::new();

        // seed: [last_h, embed_audio(0, samples[:,0])]
        let c0_tok = samples.narrow(1, 0, 1)?; // [B,1]
        let c0_embed = self.embed_audio(0, &c0_tok)?; // [B,1,D]
        let curr_h = Tensor::cat(
            &[
                &last_h
                    .to_dtype(self.text_embeddings.dtype())?
                    .unsqueeze(1)?,
                &c0_embed,
            ],
            1,
        )?; // [B,2,D]
        let pos = Tensor::from_vec(vec![0u32, 1], 2, dev)?;
        let mask = causal_mask(2, dev)?;
        let dh = self
            .decoder
            .forward(&self.projection.fwd(&curr_h)?, &pos, &mask, &mut caches)?;
        logits.push(self.ci_logits(&dh, 0)?); // codebook 1

        for i in 2..ncb {
            let tok = samples.narrow(1, i - 1, 1)?; // sample of codebook i-1
            let emb = self.embed_audio(i - 1, &tok)?; // [B,1,D]
            let pos = Tensor::from_vec(vec![i as u32], 1, dev)?;
            let skv = i + 1; // cache already holds i positions, +1 new
            let mask = Tensor::zeros((1, skv), DType::F32, dev)?; // attend to all (causal-satisfied)
            let dh = self
                .decoder
                .forward(&self.projection.fwd(&emb)?, &pos, &mask, &mut caches)?;
            logits.push(self.ci_logits(&dh, i - 1)?); // codebook i
        }
        let _ = b;
        Ok(logits)
    }

    /// Generate one full frame: backbone → sample codebook 0 → depth decoder
    /// autoregressively sampling codebooks 1..7. `uniforms` `[ncb, V]` are the
    /// injected Gumbel draws. Returns the `ncb` sampled token ids.
    pub fn generate_frame(
        &self,
        tokens: &Tensor,
        mask: &Tensor,
        topk: usize,
        temperature: f64,
        uniforms: &Tensor,
    ) -> Result<Vec<u32>> {
        let (last_h, _c0) = self.backbone_c0(tokens, mask)?;
        self.sample_frame(&last_h, topk, temperature, uniforms)
    }

    /// Sample a full frame from a backbone hidden `last_h [B,D]`: codebook-0 head +
    /// depth-decoder autoregression. `uniforms [ncb,V]` are the Gumbel draws.
    fn sample_frame(
        &self,
        last_h: &Tensor,
        topk: usize,
        temperature: f64,
        uniforms: &Tensor,
    ) -> Result<Vec<u32>> {
        let dev = last_h.device();
        let ncb = self.cfg.audio_num_codebooks;
        let c0_logits = self.codebook0_head.fwd(last_h)?;
        let tok = |t: u32| -> Result<Tensor> { Ok(Tensor::from_vec(vec![t as i64], (1, 1), dev)?) };

        let mut samples: Vec<u32> = Vec::with_capacity(ncb);
        let c0 = sample_topk(&c0_logits, topk, temperature, &uniforms.narrow(0, 0, 1)?)?;
        samples.push(c0);

        let mut caches = self.decoder.fresh_caches(self.cfg.audio_num_codebooks + 1);
        // seed = [last_h (→bf16 for the bf16 decoder), embed_audio(0, c0)]
        let curr_h = Tensor::cat(
            &[
                &last_h
                    .to_dtype(self.text_embeddings.dtype())?
                    .unsqueeze(1)?,
                &self.embed_audio(0, &tok(c0)?)?,
            ],
            1,
        )?;
        let pos = Tensor::from_vec(vec![0u32, 1], 2, dev)?;
        let dh = self.decoder.forward(
            &self.projection.fwd(&curr_h)?,
            &pos,
            &causal_mask(2, dev)?,
            &mut caches,
        )?;
        let mut logits = self.ci_logits(&dh, 0)?; // codebook 1

        for cb in 1..ncb {
            let s = sample_topk(&logits, topk, temperature, &uniforms.narrow(0, cb, 1)?)?;
            samples.push(s);
            if cb == ncb - 1 {
                break;
            }
            // feed sample of codebook cb at position cb+1, read codebook cb+1 logits (head cb)
            let emb = self.embed_audio(cb, &tok(s)?)?;
            let pos = Tensor::from_vec(vec![(cb + 1) as u32], 1, dev)?;
            let m = Tensor::zeros((1, cb + 2), DType::F32, dev)?;
            let dh = self
                .decoder
                .forward(&self.projection.fwd(&emb)?, &pos, &m, &mut caches)?;
            logits = self.ci_logits(&dh, cb)?;
        }
        Ok(samples)
    }

    /// CFG `generate_frame`: doubles the prompt (cond + uncond null-text), guides each
    /// codebook's logits `uncond + (cond-uncond)*cfg`, then samples. Returns
    /// (samples, c0_guided_logits `[1,V]`, ci_guided_logits `[7][1,V]`).
    pub fn generate_frame_cfg(
        &self,
        tokens: &Tensor,
        mask: &Tensor,
        cfg: f64,
        topk: usize,
        temperature: f64,
        uniforms: &Tensor,
    ) -> Result<(Vec<u32>, Tensor, Vec<Tensor>)> {
        let dev = tokens.device();
        let s = tokens.dim(1)?;
        let tokens2 = Tensor::cat(&[tokens, tokens], 0)?;
        let mask2 = Tensor::cat(&[mask, mask], 0)?;
        let h0 = self.embed_and_sum(&tokens2, &mask2, true)?; // [2,S,D]
        let pos = Tensor::from_vec((0..s as u32).collect::<Vec<_>>(), s, dev)?;
        let mut caches = self.backbone.fresh_caches(s);
        let out = self
            .backbone
            .forward(&h0, &pos, &causal_mask(s, dev)?, &mut caches)?;
        let last_h = out.narrow(1, s - 1, 1)?.squeeze(1)?.contiguous()?; // [2,D]
        self.sample_frame_cfg(&last_h, cfg, topk, temperature, uniforms)
    }

    fn sample_frame_cfg(
        &self,
        last_h: &Tensor,
        cfg: f64,
        topk: usize,
        temperature: f64,
        uniforms: &Tensor,
    ) -> Result<(Vec<u32>, Tensor, Vec<Tensor>)> {
        let dev = last_h.device();
        let ncb = self.cfg.audio_num_codebooks;
        let guide = |logits: &Tensor| -> Result<Tensor> {
            let cond = logits.narrow(0, 0, 1)?;
            let uncond = logits.narrow(0, 1, 1)?;
            Ok((&uncond + ((cond - &uncond)? * cfg)?)?)
        };

        let c0_guided = guide(&self.codebook0_head.fwd(last_h)?)?; // [1,V]
        let mut samples = Vec::with_capacity(ncb);
        let mut ci_guided = Vec::new();
        samples.push(sample_topk(
            &c0_guided,
            topk,
            temperature,
            &uniforms.narrow(0, 0, 1)?,
        )?);

        let mut caches = self.decoder.fresh_caches(self.cfg.audio_num_codebooks + 1);
        let c0_emb = self.embed_audio_dup(0, samples[0])?; // [2,1,D]
        let curr_h = Tensor::cat(
            &[
                &last_h
                    .to_dtype(self.text_embeddings.dtype())?
                    .unsqueeze(1)?,
                &c0_emb,
            ],
            1,
        )?; // [2,2,D]
        let pos = Tensor::from_vec(vec![0u32, 1], 2, dev)?;
        let dh = self.decoder.forward(
            &self.projection.fwd(&curr_h)?,
            &pos,
            &causal_mask(2, dev)?,
            &mut caches,
        )?;
        let mut logits = self.ci_logits(&dh, 0)?; // [2,V] codebook 1

        for cb in 1..ncb {
            let g = guide(&logits)?;
            ci_guided.push(g.clone());
            let sv = sample_topk(&g, topk, temperature, &uniforms.narrow(0, cb, 1)?)?;
            samples.push(sv);
            if cb == ncb - 1 {
                break;
            }
            let emb = self.embed_audio_dup(cb, sv)?;
            let pos = Tensor::from_vec(vec![(cb + 1) as u32], 1, dev)?;
            let m = Tensor::zeros((1, cb + 2), DType::F32, dev)?;
            let dh = self
                .decoder
                .forward(&self.projection.fwd(&emb)?, &pos, &m, &mut caches)?;
            logits = self.ci_logits(&dh, cb)?;
        }
        Ok((samples, c0_guided, ci_guided))
    }

    /// GPU-resident frame sampling for fast generation: the sampled token of each
    /// codebook stays on the device and is fed straight back into the depth decoder
    /// (no per-codebook GPU→CPU sync). Handles CFG when `last_h` is `[2,D]`.
    /// Returns the frame's `[ncb]` u32 token tensor (on device).
    fn sample_frame_gpu(
        &self,
        last_h: &Tensor,
        cfg_scale: f64,
        topk: usize,
        temperature: f64,
        uniforms: &Tensor,
    ) -> Result<Tensor> {
        let dev = last_h.device();
        let ncb = self.cfg.audio_num_codebooks;
        let cfg = last_h.dim(0)? == 2;
        let guide = |logits: &Tensor| -> Result<Tensor> {
            if cfg {
                let cond = logits.narrow(0, 0, 1)?;
                let uncond = logits.narrow(0, 1, 1)?;
                Ok((&uncond + ((cond - &uncond)? * cfg_scale)?)?)
            } else {
                Ok(logits.clone())
            }
        };
        let embed = |cb: usize, tok: &Tensor| -> Result<Tensor> {
            let e = self.embed_audio(cb, &tok.reshape((1, 1))?)?; // [1,1,D]
            if cfg {
                Tensor::cat(&[&e, &e], 0).map_err(Into::into)
            } else {
                Ok(e)
            }
        };

        let prof2 = std::env::var("MELODIE_PROF2").is_ok();
        let (mut t_dec, mut t_smp, mut t_head) = (
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        );
        let sync_u = |t: &Tensor| -> Result<()> {
            t.narrow(0, 0, 1)?.to_vec1::<u32>()?;
            Ok(())
        };
        let sync_f = |t: &Tensor| -> Result<()> {
            t.sum_all()?.to_scalar::<f32>()?;
            Ok(())
        };

        let ts = std::time::Instant::now();
        let c0 = sample_topk_gpu(
            &guide(&self.codebook0_head.fwd(last_h)?)?,
            topk,
            temperature,
            &uniforms.narrow(0, 0, 1)?,
        )?;
        if prof2 {
            sync_u(&c0)?;
            t_smp += ts.elapsed();
        }
        let mut tokens = vec![c0.clone()];

        let mut caches = self.decoder.fresh_caches(self.cfg.audio_num_codebooks + 1);
        let last_hb = last_h
            .to_dtype(self.text_embeddings.dtype())?
            .unsqueeze(1)?;
        let curr_h = Tensor::cat(&[&last_hb, &embed(0, &c0)?], 1)?; // [B,2,D]
        let pos = Tensor::from_vec(vec![0u32, 1], 2, dev)?;
        let td = std::time::Instant::now();
        let dh = self.decoder.forward(
            &self.projection.fwd(&curr_h)?,
            &pos,
            &causal_mask(2, dev)?,
            &mut caches,
        )?;
        if prof2 {
            sync_f(&dh)?;
            t_dec += td.elapsed();
        }
        let mut logits = self.ci_logits(&dh, 0)?; // [B,V] codebook 1

        for cb in 1..ncb {
            let ts = std::time::Instant::now();
            let ci = sample_topk_gpu(
                &guide(&logits)?,
                topk,
                temperature,
                &uniforms.narrow(0, cb, 1)?,
            )?;
            if prof2 {
                sync_u(&ci)?;
                t_smp += ts.elapsed();
            }
            tokens.push(ci.clone());
            if cb == ncb - 1 {
                break;
            }
            let emb = embed(cb, &ci)?;
            let pos = Tensor::from_vec(vec![(cb + 1) as u32], 1, dev)?;
            let m = Tensor::zeros((1, cb + 2), DType::F32, dev)?;
            let td = std::time::Instant::now();
            let dh = self
                .decoder
                .forward(&self.projection.fwd(&emb)?, &pos, &m, &mut caches)?;
            if prof2 {
                sync_f(&dh)?;
                t_dec += td.elapsed();
            }
            let th = std::time::Instant::now();
            logits = self.ci_logits(&dh, cb)?;
            if prof2 {
                sync_f(&logits)?;
                t_head += th.elapsed();
            }
        }
        if prof2 {
            println!(
                "    [prof2] dec_fwd={:.1} sample={:.1} heads={:.1} ms/frame",
                t_dec.as_secs_f64() * 1000.0,
                t_smp.as_secs_f64() * 1000.0,
                t_head.as_secs_f64() * 1000.0
            );
        }
        Tensor::cat(&tokens, 0).map_err(Into::into) // [ncb] u32
    }

    /// Autoregressive multi-frame generation: prompt → codes `[ncb, T]`. Backbone
    /// KV cache persists across frames; each frame feeds the previous frame's audio
    /// tokens; stops at EOS (codebook-0 ≥ 8193) or `max_frames`. RNG is self-generated.
    ///
    /// `on_frame`, if set, is called as `(frames_done, max_frames)` every 8 frames —
    /// purely for progress reporting; it has no effect on the generated codes.
    pub fn generate_codes(
        &self,
        tokens: &Tensor,
        mask: &Tensor,
        muq_idx: Option<usize>,
        params: &GenParams,
        mut on_frame: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<Tensor> {
        let GenParams {
            cfg_scale,
            max_frames,
            topk,
            temperature,
        } = *params;
        let dev = tokens.device();
        let ncb = self.cfg.audio_num_codebooks;
        let v = self.cfg.audio_vocab_size;
        let d = self.cfg.backbone.embed_dim;
        let eos = 8193u32;
        let cfg = cfg_scale > 1.0; // B=2 (cond+uncond) when guiding
        let s = tokens.dim(1)?;
        let mut bcaches = self.backbone.fresh_caches(s + max_frames);

        // frame 0: prompt forward (doubled for CFG; MuQ slot scattered per row)
        let (t0, m0) = if cfg {
            (
                Tensor::cat(&[tokens, tokens], 0)?,
                Tensor::cat(&[mask, mask], 0)?,
            )
        } else {
            (tokens.clone(), mask.clone())
        };
        let mut h0 = self.embed_and_sum(&t0, &m0, cfg)?;
        if let Some(idx) = muq_idx {
            let mq = if cfg {
                Tensor::cat(
                    &[
                        &self.muq_bias.reshape((1, 1, d))?,
                        &self.uncond_text.reshape((1, 1, d))?,
                    ],
                    0,
                )?
            } else {
                self.muq_bias.reshape((1, 1, d))?
            };
            let before = h0.narrow(1, 0, idx)?;
            let after = h0.narrow(1, idx + 1, s - idx - 1)?;
            h0 = Tensor::cat(&[&before, &mq, &after], 1)?;
        }
        let pos0 = Tensor::from_vec((0..s as u32).collect::<Vec<_>>(), s, dev)?;
        let out = self
            .backbone
            .forward(&h0, &pos0, &causal_mask(s, dev)?, &mut bcaches)?;
        let mut last_h = out.narrow(1, s - 1, 1)?.squeeze(1)?.contiguous()?; // [B,D] (contiguous: Metal matmul mishandles strided B=2)

        let prof = std::env::var("MELODIE_PROFILE").is_ok();
        let (mut t_smp, mut t_bb) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut per_frame: Vec<f64> = Vec::new();
        let (mut pf_smp, mut pf_bb): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut nf = 0usize;
        // constant audio-active mask for generated frames ([1,1,ncb+1]: audio=1, text=0)
        let mut mvec0 = vec![1i64; ncb + 1];
        mvec0[ncb] = 0;
        let mg1 = Tensor::from_vec(mvec0, (1, 1, ncb + 1), dev)?;
        let mg = if cfg {
            Tensor::cat(&[&mg1, &mg1], 0)?
        } else {
            mg1
        };

        // GPU-resident sampling (1 readback/frame instead of 1 per codebook) is the
        // default — `MELODIE_CPU_SAMPLE` opts back into the scalar CPU path for
        // debugging. Token choices can diverge between the two (GPU vs CPU floating
        // point isn't bit-associative, same as the existing bf16/f32 and Metal/CPU
        // codec tradeoffs elsewhere in this engine), which is expected, not a bug.
        let gpu_sample = std::env::var("MELODIE_CPU_SAMPLE").is_err();
        let mut frames: Vec<Vec<u32>> = Vec::new();
        // All-zero generated-frame attention mask, allocated once and narrowed each step
        // (Metal sdpa ignores it; the CPU parity path adds 0 → no-op). Avoids a fresh
        // growing `(1, pos+1)` tensor every frame.
        let gen_mask = Tensor::zeros((1, s + max_frames), DType::F32, dev)?;
        // `pos` is the absolute KV position (starts past the `s`-token prompt).
        for pos in s..s + max_frames {
            let tf = std::time::Instant::now();
            let uniforms = Tensor::rand(0f32, 1f32, (ncb, v), dev)?;
            let t0 = std::time::Instant::now();
            let frame = if gpu_sample {
                // 8 codebooks sampled on-device (no per-codebook sync), 1 readback/frame
                self.sample_frame_gpu(&last_h, cfg_scale, topk, temperature, &uniforms)?
                    .to_vec1::<u32>()?
            } else if cfg {
                self.sample_frame_cfg(&last_h, cfg_scale, topk, temperature, &uniforms)?
                    .0
            } else {
                self.sample_frame(&last_h, topk, temperature, &uniforms)?
            };
            if prof {
                let e = t0.elapsed();
                t_smp += e;
                pf_smp.push(e.as_secs_f64() * 1000.0);
            }
            if frame[0] >= eos {
                break;
            }
            // next backbone input grid [1,1,ncb+1]: prev frame's audio tokens, text empty
            let mut grid = vec![0i64; ncb + 1];
            for (c, &tk) in frame.iter().enumerate() {
                grid[c] = tk as i64;
            }
            frames.push(frame);
            let tg1 = Tensor::from_vec(grid, (1, 1, ncb + 1), dev)?;
            let tg = if cfg {
                Tensor::cat(&[&tg1, &tg1], 0)?
            } else {
                tg1
            };
            let t1 = std::time::Instant::now();
            let h_t = self.embed_and_sum(&tg, &mg, false)?;
            let m = gen_mask.narrow(1, 0, pos + 1)?;
            let out = self.backbone.forward(
                &h_t,
                &Tensor::from_vec(vec![pos as u32], 1, dev)?,
                &m,
                &mut bcaches,
            )?;
            last_h = out.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
            if prof {
                let e = t1.elapsed();
                t_bb += e;
                pf_bb.push(e.as_secs_f64() * 1000.0);
                per_frame.push(tf.elapsed().as_secs_f64() * 1000.0);
            }
            if std::env::var("MELODIE_DBG").is_ok() && nf == 12 {
                let v: Vec<f32> = last_h
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let bad = v.iter().filter(|x| !x.is_finite()).count();
                let denorm = v
                    .iter()
                    .filter(|&&x| x != 0.0 && x.abs() < 1.18e-38)
                    .count();
                let maxabs = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
                let minabs = v
                    .iter()
                    .filter(|&&x| x != 0.0)
                    .fold(f32::INFINITY, |a, &x| a.min(x.abs()));
                eprintln!(
                    "[dbg] last_h n={} maxabs={maxabs:.3e} minabs={minabs:.3e} nonfinite={bad} denorm={denorm}",
                    v.len()
                );
            }
            nf += 1;
            if nf.is_multiple_of(8)
                && let Some(cb) = on_frame.as_deref_mut()
            {
                cb(nf, max_frames);
            }
        }
        if prof && nf > 0 {
            println!(
                "  [profile] sample={:.0} | backbone+embed={:.0} ms/frame (avg over n={nf})",
                t_smp.as_secs_f64() * 1000.0 / nf as f64,
                t_bb.as_secs_f64() * 1000.0 / nf as f64
            );
            if per_frame.len() > 5 {
                let mut w: Vec<f64> = per_frame[5..].to_vec();
                w.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let avg = w.iter().sum::<f64>() / w.len() as f64;
                println!(
                    "  [profile] WARM total (frames 6+): min={:.0} med={:.0} avg={:.0} max={:.0} ms/frame  (n={})",
                    w[0],
                    w[w.len() / 2],
                    avg,
                    w[w.len() - 1],
                    w.len()
                );
            }
        }

        let t = frames.len();
        let mut data = vec![0i64; ncb * t];
        for (fi, f) in frames.iter().enumerate() {
            for (c, &tok) in f.iter().enumerate() {
                data[c * t + fi] = tok as i64;
            }
        }
        Ok(Tensor::from_vec(data, (ncb, t), dev)?)
    }

    fn ci_logits(&self, decoder_h: &Tensor, head_idx: usize) -> Result<Tensor> {
        // decoder_h [B,S,D] -> last position @ audio_head[head_idx] [D,V] -> [B,V]
        let last = decoder_h
            .narrow(1, decoder_h.dim(1)? - 1, 1)?
            .squeeze(1)?
            .contiguous()?; // [B,D] (contiguous for Metal B=2 matmul)
        let head = self.audio_head.narrow(0, head_idx, 1)?.squeeze(0)?; // [D,V]
        Ok(last.matmul(&head)?)
    }
}
