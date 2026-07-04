//! HeartCodec FlowMatching DiT (the transformer half of the codec). **P1b.**
//!
//! Ported from `heartcodec/models/{flow_matching.py, transformer.py}`. The DiT
//! ("estimator") is a two-stage PixArt-style transformer with AdaLayerNorm-single
//! timestep conditioning, interleaved RoPE, and SwiGLU MLP. All weights are plain
//! (no weight-norm); conv `(out,in,k)` and linear `(out,in)` match candle directly.
//!
//! Layout: the reference runs in (B,T,C); we keep (B,T,C) here and only drop to
//! (B,C,T) inside the ProjectLayer convs.

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module};

use crate::Result;
use crate::codec::CodecWeights;
use crate::config::{DitConfig, HeartCodecConfig};

// --- small helpers -------------------------------------------------------

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// RMSNorm with weight, eps (matches mlx nn.RMSNorm).
fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    // candle's fused RMSNorm kernel (1 Metal launch vs ~6 hand-rolled ops); the
    // reduction runs in f32 — the DiT is f32 throughout, so numerics are unchanged.
    Ok(candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)?)
}

/// Parameter-free LayerNorm (elementwise_affine=False): (x-mean)/sqrt(var+eps).
/// The reduction runs in f32 even for bf16 inputs (mean/var in bf16 lose ~3 digits);
/// on the f32 path the upcasts are no-ops, so parity is unchanged.
fn layernorm_free(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xc.broadcast_div(&(var + eps)?.sqrt()?)?.to_dtype(dt)?)
}

struct Lin(Linear);
impl Lin {
    fn load(w: &CodecWeights, prefix: &str, bias: bool) -> Result<Self> {
        Self::load_dt(w, prefix, bias, DType::F32)
    }
    fn load_dt(w: &CodecWeights, prefix: &str, bias: bool, dt: DType) -> Result<Self> {
        let weight = w.tensor(&format!("{prefix}.weight"))?.to_dtype(dt)?;
        let b = if bias {
            Some(w.tensor(&format!("{prefix}.bias"))?.to_dtype(dt)?)
        } else {
            None
        };
        Ok(Self(Linear::new(weight, b)))
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(x)?)
    }
}

/// ProjectLayer: Conv1d(k) → *k^-0.5 → Linear (transformer.py:248-270).
struct ProjectLayer {
    conv_w: Tensor,
    conv_b: Tensor,
    ffn2: Lin,
    kernel: usize,
}
impl ProjectLayer {
    fn load(w: &CodecWeights, prefix: &str, kernel: usize, dt: DType) -> Result<Self> {
        Ok(Self {
            conv_w: w.tensor(&format!("{prefix}.ffn_1.weight"))?.to_dtype(dt)?, // (out,in,k)
            conv_b: w.tensor(&format!("{prefix}.ffn_1.bias"))?.to_dtype(dt)?,
            ffn2: Lin::load_dt(w, &format!("{prefix}.ffn_2"), true, dt)?,
            kernel,
        })
    }
    fn fwd(&self, x_btc: &Tensor) -> Result<Tensor> {
        let xn = x_btc.transpose(1, 2)?.contiguous()?; // (B,C,T)
        let cfg = Conv1dConfig {
            padding: self.kernel / 2,
            ..Default::default()
        };
        let y = Conv1d::new(self.conv_w.clone(), Some(self.conv_b.clone()), cfg).forward(&xn)?;
        let y = y.transpose(1, 2)?.contiguous()?; // (B,T,out)
        let y = (y * (self.kernel as f64).powf(-0.5))?;
        self.ffn2.fwd(&y)
    }
}

// --- RoPE (interleaved) --------------------------------------------------

/// cos/sin tables of shape (T, dim/2), base 10000 (transformer.py:20-28).
fn rope_tables(t: usize, dim: usize, base: f64, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / base.powf(2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let inv = Tensor::from_vec(inv, half, dev)?;
    let tt: Vec<f32> = (0..t).map(|i| i as f32).collect();
    let tt = Tensor::from_vec(tt, t, dev)?;
    let freqs = tt.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)?; // (T,half)
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Apply interleaved RoPE to (B,H,T,D) with cos/sin (T,D/2) (transformer.py:31-64).
/// Same math as the reference's hand-rolled form (even/odd pairs, x1·cos−x2·sin /
/// x1·sin+x2·cos) but via candle's fused kernel: 1 launch vs ~10.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, cos, sin)?)
}

// --- attention / mlp / block --------------------------------------------

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    n_heads: usize,
    head_dim: usize,
}
impl Attn {
    fn load(
        w: &CodecWeights,
        prefix: &str,
        n_heads: usize,
        head_dim: usize,
        dt: DType,
    ) -> Result<Self> {
        Ok(Self {
            q: Lin::load_dt(w, &format!("{prefix}.q_proj"), false, dt)?,
            k: Lin::load_dt(w, &format!("{prefix}.k_proj"), false, dt)?,
            v: Lin::load_dt(w, &format!("{prefix}.v_proj"), false, dt)?,
            o: Lin::load_dt(w, &format!("{prefix}.o_proj"), false, dt)?,
            n_heads,
            head_dim,
        })
    }
    fn fwd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (h, dh) = (self.n_heads, self.head_dim);
        let split = |z: Tensor| -> Result<Tensor> {
            Ok(z.reshape((b, t, h, dh))?.transpose(1, 2)?.contiguous()?) // (B,H,T,Dh)
        };
        let q = apply_rope(&split(self.q.fwd(x)?)?, cos, sin)?;
        let k = apply_rope(&split(self.k.fwd(x)?)?, cos, sin)?;
        let v = split(self.v.fwd(x)?)?;
        let scale = (dh as f64).powf(-0.5);
        // Fused SDPA on Metal — no T×T score materialisation. The FM denoiser is bidirectional,
        // so full attention (do_causal = false). The manual path stays for CPU / head_dim > 256
        // (Metal SDPA's limit). This attention dominates the codec runtime (~76% of it).
        let out = if matches!(q.device(), candle_core::Device::Metal(_)) && dh <= 256 {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?
        } else {
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // (B,H,T,T)
            let w = candle_nn::ops::softmax(&scores, D::Minus1)?;
            w.matmul(&v)? // (B,H,T,Dh)
        };
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, h * dh))?;
        self.o.fwd(&out)
    }
}

struct Mlp {
    gate: Lin,
    up: Lin,
    down: Lin,
}
impl Mlp {
    fn load(w: &CodecWeights, prefix: &str, dt: DType) -> Result<Self> {
        Ok(Self {
            gate: Lin::load_dt(w, &format!("{prefix}.gate"), false, dt)?,
            up: Lin::load_dt(w, &format!("{prefix}.up"), false, dt)?,
            down: Lin::load_dt(w, &format!("{prefix}.down"), false, dt)?,
        })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        self.down
            .fwd(&(silu(&self.gate.fwd(x)?)? * self.up.fwd(x)?)?)
    }
}

struct Block {
    attn_norm: Tensor,
    attn: Attn,
    mlp_norm: Tensor,
    mlp: Mlp,
    scale_shift_table: Tensor, // (6, D)
    dim: usize,
}
impl Block {
    fn load(
        w: &CodecWeights,
        prefix: &str,
        n_heads: usize,
        head_dim: usize,
        dim: usize,
        dt: DType,
    ) -> Result<Self> {
        Ok(Self {
            attn_norm: w
                .tensor(&format!("{prefix}.attn_norm.weight"))?
                .to_dtype(dt)?,
            attn: Attn::load(w, &format!("{prefix}.attn"), n_heads, head_dim, dt)?,
            mlp_norm: w
                .tensor(&format!("{prefix}.mlp_norm.weight"))?
                .to_dtype(dt)?,
            mlp: Mlp::load(w, &format!("{prefix}.mlp"), dt)?,
            scale_shift_table: w
                .tensor(&format!("{prefix}.scale_shift_table"))?
                .to_dtype(dt)?,
            dim,
        })
    }
    /// `tmod` is (B, 6*dim). cos/sin are the RoPE tables for head_dim.
    fn fwd(&self, x: &Tensor, tmod: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        // chunks = scale_shift_table[None] + tmod.reshape(B,6,D)  -> (B,6,D)
        let chunks = self
            .scale_shift_table
            .unsqueeze(0)?
            .broadcast_add(&tmod.reshape((b, 6, self.dim))?)?;
        let c = |i: usize| -> Result<Tensor> { Ok(chunks.narrow(1, i, 1)?) }; // (B,1,D)
        let (shift_msa, scale_msa, gate_msa) = (c(0)?, c(1)?, c(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (c(3)?, c(4)?, c(5)?);

        let norm_h = rmsnorm(x, &self.attn_norm, 1e-6)?;
        let norm_h = (norm_h.broadcast_mul(&(scale_msa + 1.0)?)?).broadcast_add(&shift_msa)?;
        let h = self.attn.fwd(&norm_h, cos, sin)?;
        let x = (x + h.broadcast_mul(&gate_msa)?)?;

        let norm_h = rmsnorm(&x, &self.mlp_norm, 1e-6)?;
        let norm_h = (norm_h.broadcast_mul(&(scale_mlp + 1.0)?)?).broadcast_add(&shift_mlp)?;
        let h = self.mlp.fwd(&norm_h)?;
        Ok((x + h.broadcast_mul(&gate_mlp)?)?)
    }
}

// --- timestep / AdaLN-single --------------------------------------------

/// Sinusoidal timestep embedding (transformer.py:297-308). t: (B,), out (B, dim).
fn timestep_sinusoid(t: &Tensor, dim: usize, dev: &Device) -> Result<Tensor> {
    let half = dim / 2;
    let max_period = 10000.0f64;
    let scale = 1000.0f64;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-max_period.ln() * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, half, dev)?; // (half,)
    let args = t
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?
        .affine(scale, 0.0)?; // (B,half)
    Ok(Tensor::cat(&[args.cos()?, args.sin()?], 1)?) // (B,dim)
}

struct AdaLnSingle {
    ts_l1: Lin,
    ts_l2: Lin,
    linear: Lin,
    flow_t_size: usize,
    dt: DType,
}
impl AdaLnSingle {
    fn load(w: &CodecWeights, prefix: &str, dt: DType) -> Result<Self> {
        Ok(Self {
            ts_l1: Lin::load_dt(
                w,
                &format!("{prefix}.emb.timestep_embedder.linear_1"),
                true,
                dt,
            )?,
            ts_l2: Lin::load_dt(
                w,
                &format!("{prefix}.emb.timestep_embedder.linear_2"),
                true,
                dt,
            )?,
            linear: Lin::load_dt(w, &format!("{prefix}.linear"), true, dt)?,
            flow_t_size: 512,
            dt,
        })
    }
    /// returns (timestep_mod (B,6*D), embedded_timestep (B,D)).
    fn fwd(&self, t: &Tensor, dev: &Device) -> Result<(Tensor, Tensor)> {
        let proj = timestep_sinusoid(t, self.flow_t_size, dev)?.to_dtype(self.dt)?; // (B,512)
        let embedded = self.ts_l2.fwd(&silu(&self.ts_l1.fwd(&proj)?)?)?; // (B,D)
        let tmod = self.linear.fwd(&silu(&embedded)?)?; // (B,6D)
        Ok((tmod, embedded))
    }
}

// --- the estimator (two-stage DiT) --------------------------------------

/// HeartCodec flow-matching DiT estimator.
pub struct Dit {
    proj_in: ProjectLayer,
    blocks1: Vec<Block>,
    sst1: Tensor, // (2, inner)
    adaln1: AdaLnSingle,
    connection_proj: ProjectLayer,
    blocks2: Vec<Block>,
    sst2: Tensor, // (2, inner2)
    adaln2: AdaLnSingle,
    proj_out: ProjectLayer,
    head_dim1: usize,
    head_dim2: usize,
    /// RoPE cos/sin tables for both stages, precomputed to `ROPE_T_MAX` at load —
    /// `forward` narrows to its `t` instead of rebuilding them every Euler step.
    rope1: (Tensor, Tensor),
    rope2: (Tensor, Tensor),
    /// Compute dtype: bf16 on Metal by default — speed-neutral on M1 Max (the
    /// estimator's gemms are large-M compute-bound) but ~1.5 GB less resident
    /// weight memory. `MELODIE_CODEC_F32=1` opts back into the f32 numerics for
    /// A/B listening; CPU is always f32 (bit-exact parity). The Euler solver
    /// stays f32 either way — `forward` casts at its boundary.
    dt: DType,
}

/// Longest sequence the precomputed DiT RoPE tables cover (segments are 744
/// latent frames; anything longer falls back to building tables on the fly).
const ROPE_T_MAX: usize = 2048;

impl Dit {
    pub fn load(w: &CodecWeights, cfg: &DitConfig) -> Result<Self> {
        let inner = cfg.num_heads * cfg.head_dim; // 1536
        let inner2 = inner * 2; // 3072
        let hd1 = cfg.head_dim; // 64
        let hd2 = cfg.head_dim * 2; // 128
        let p = "flow_matching.estimator";
        let sst1 = w.tensor(&format!("{p}.scale_shift_table"))?;
        let dev = sst1.device().clone();
        // bf16 measured speed-NEUTRAL on M1 Max (the DiT's gemms are large-M compute-bound,
        // and Apple GPUs don't run bf16 math faster than f32) but halves the estimator's
        // resident weights (~1.5 GB) — the default for the 32 GB target. MELODIE_CODEC_F32=1
        // restores the f32 numerics for A/B listening.
        let dt = if matches!(dev, Device::Metal(_)) && std::env::var("MELODIE_CODEC_F32").is_err() {
            DType::BF16
        } else {
            DType::F32
        };
        let load_blocks = |list: &str, n: usize, dim: usize, hd: usize| -> Result<Vec<Block>> {
            (0..n)
                .map(|i| Block::load(w, &format!("{p}.{list}.{i}"), cfg.num_heads, hd, dim, dt))
                .collect()
        };
        let rope_dt = |hd: usize| -> Result<(Tensor, Tensor)> {
            let (c, s) = rope_tables(ROPE_T_MAX, hd, 10000.0, &dev)?;
            Ok((c.to_dtype(dt)?, s.to_dtype(dt)?))
        };
        Ok(Self {
            proj_in: ProjectLayer::load(w, &format!("{p}.proj_in"), 3, dt)?,
            blocks1: load_blocks("transformer_blocks", cfg.num_layers_stage1, inner, hd1)?,
            sst1: sst1.to_dtype(dt)?,
            adaln1: AdaLnSingle::load(w, &format!("{p}.adaln_single"), dt)?,
            connection_proj: ProjectLayer::load(w, &format!("{p}.connection_proj"), 3, dt)?,
            blocks2: load_blocks("transformer_blocks_2", cfg.num_layers_stage2, inner2, hd2)?,
            sst2: w
                .tensor(&format!("{p}.scale_shift_table_2"))?
                .to_dtype(dt)?,
            adaln2: AdaLnSingle::load(w, &format!("{p}.adaln_single_2"), dt)?,
            proj_out: ProjectLayer::load(w, &format!("{p}.proj_out"), 3, dt)?,
            head_dim1: hd1,
            head_dim2: hd2,
            rope1: rope_dt(hd1)?,
            rope2: rope_dt(hd2)?,
            dt,
        })
    }

    /// Precomputed RoPE tables narrowed to `t` (or rebuilt if `t > ROPE_T_MAX`).
    fn rope_for(&self, stage: usize, t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let ((cos, sin), hd) = if stage == 1 {
            (&self.rope1, self.head_dim1)
        } else {
            (&self.rope2, self.head_dim2)
        };
        if t <= ROPE_T_MAX {
            Ok((cos.narrow(0, 0, t)?, sin.narrow(0, 0, t)?))
        } else {
            rope_tables(t, hd, 10000.0, dev)
        }
    }

    /// hidden (B,T,in_channels), timestep (B,) → (B,T,out_channels).
    /// Input/output stay f32 (the Euler solver's dtype); the estimator body runs
    /// at `self.dt` (bf16 on Metal) between the two boundary casts.
    pub fn forward(&self, hidden: &Tensor, timestep: &Tensor) -> Result<Tensor> {
        let dev = hidden.device();
        let t = hidden.dim(1)?;
        let in_dt = hidden.dtype();
        let hidden = &hidden.to_dtype(self.dt)?;

        let mut s = self.proj_in.fwd(hidden)?; // (B,T,inner)
        let (tmod1, emb1) = self.adaln1.fwd(timestep, dev)?;
        let (cos1, sin1) = self.rope_for(1, t, dev)?;
        for blk in &self.blocks1 {
            s = blk.fwd(&s, &tmod1, &cos1, &sin1)?;
        }
        // post-norm stage 1: split(sst1[None] + emb1[:,None,:], 2, axis=1)
        let mod1 = self.sst1.unsqueeze(0)?.broadcast_add(&emb1.unsqueeze(1)?)?; // (B,2,inner)
        let shift = mod1.narrow(1, 0, 1)?;
        let scale = mod1.narrow(1, 1, 1)?;
        s = (layernorm_free(&s, 1e-6)?.broadcast_mul(&(scale + 1.0)?)?).broadcast_add(&shift)?;

        let mut x = Tensor::cat(&[hidden, &s], D::Minus1)?; // (B,T,in+inner)
        x = self.connection_proj.fwd(&x)?; // (B,T,inner2)
        let (tmod2, emb2) = self.adaln2.fwd(timestep, dev)?;
        let (cos2, sin2) = self.rope_for(2, t, dev)?;
        for blk in &self.blocks2 {
            x = blk.fwd(&x, &tmod2, &cos2, &sin2)?;
        }
        let mod2 = self.sst2.unsqueeze(0)?.broadcast_add(&emb2.unsqueeze(1)?)?;
        let shift2 = mod2.narrow(1, 0, 1)?;
        let scale2 = mod2.narrow(1, 1, 1)?;
        x = (layernorm_free(&x, 1e-6)?.broadcast_mul(&(scale2 + 1.0)?)?).broadcast_add(&shift2)?;

        Ok(self.proj_out.fwd(&x)?.to_dtype(in_dt)?)
    }
}

/// Nearest-neighbour ×2 upsample along time (dim 1) of a (B,T,C) tensor.
fn upsample2_time_btc(x: &Tensor) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    let x = x.unsqueeze(2)?.broadcast_as((b, t, 2, c))?;
    Ok(x.contiguous()?.reshape((b, t * 2, c))?)
}

/// In-context conditioning for one [`FlowMatching::inference_codes`] segment
/// (groups the `true_latents` / `latent_length` / `incontext_length` args of the
/// reference `flow_matching.py::inference_codes`).
pub struct SegmentCtx<'a> {
    /// Context latents `(1, latent_length, 256)`; only the first `incontext_length`
    /// frames are read (the rest are masked to zero, matching `true_latents * mask`).
    pub true_latents: &'a Tensor,
    /// Valid latent frames for this segment (`== 2*T_codes` in this pipeline).
    pub latent_length: usize,
    /// Leading frames carried over as in-context (`0` ⇒ the in-context branch is inert
    /// and `inference_codes` reduces exactly to the verified seg0 path).
    pub incontext_length: usize,
}

/// FlowMatching: RVQ-conditioned flow-matching that turns codes → continuous latent.
pub struct FlowMatching {
    dit: Dit,
    codebooks: Vec<Tensor>, // each (codebook_size, codebook_dim)
    project_out: Lin,       // codebook_dim -> dim
    cond_feature_emb: Lin,  // dim -> dim
}

impl FlowMatching {
    pub fn load(w: &CodecWeights, cfg: &HeartCodecConfig) -> Result<Self> {
        let dit = Dit::load(w, &cfg.dit)?;
        let codebooks = (0..cfg.rvq.num_quantizers)
            .map(|q| {
                let e = w.tensor(&format!(
                    "flow_matching.vq_embed.layers.{q}._codebook.embed"
                ))?;
                Ok(e.squeeze(0)?) // (1,S,D) -> (S,D)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dit,
            codebooks,
            project_out: Lin::load(w, "flow_matching.vq_embed.project_out", true)?,
            cond_feature_emb: Lin::load(w, "flow_matching.cond_feature_emb", true)?,
        })
    }

    /// In-context-capable flow-matching inference — port of
    /// `flow_matching.py::inference_codes` (161-252).
    ///
    /// - `codes` (1,Q,T_codes) integer codes for this segment.
    /// - `ctx` the in-context conditioning ([`SegmentCtx`]): `true_latents`
    ///   (1, latent_length, 256) — only the first `incontext_length` frames are read
    ///   (the reference zeros the rest via the `mask==1` multiply).
    /// - `noise` (1, 2*T_codes, 256) the initial latent. The reference draws this
    ///   with `randn` *inside* `inference_codes`; we take it as an argument so the
    ///   caller controls randomness (and parity tests can inject a fixed latent).
    ///
    /// Returns (1, 2*T_codes, 256). With `ctx.incontext_length == 0` the in-context
    /// branch is inert and this reduces *exactly* to the verified seg0 path.
    pub fn inference_codes(
        &self,
        codes: &Tensor,
        ctx: &SegmentCtx,
        noise: &Tensor,
        num_steps: usize,
        gs: f64,
    ) -> Result<Tensor> {
        let true_latents = ctx.true_latents;
        let latent_length = ctx.latent_length;
        let incontext_length = ctx.incontext_length;
        let t = codes.dim(2)?;

        // RVQ codebook lookup + sum over quantizers → project_out → cond_feature_emb.
        let codes_u = codes.to_dtype(candle_core::DType::U32)?;
        let mut summed: Option<Tensor> = None;
        for (q, emb) in self.codebooks.iter().enumerate() {
            let idx = codes_u.narrow(1, q, 1)?.squeeze(1)?.flatten_all()?; // (T,)
            let g = emb.index_select(&idx, 0)?.reshape((1, t, emb.dim(1)?))?; // (1,T,D)
            summed = Some(match summed {
                Some(s) => (s + g)?,
                None => g,
            });
        }
        let cond = self
            .cond_feature_emb
            .fwd(&self.project_out.fwd(&summed.unwrap())?)?; // (1,T,512)
        // Nearest-neighbour ×2 time upsample → conditioning `mu`.
        //
        // The reference builds `latent_masks` (=2 where frame<latent_length, else 0;
        // =1 where frame<incontext_length) and zeros `mu` wherever mask==0 (replacing
        // it with `zero_cond_embedding1`, which is all zeros). In this pipeline
        // `latent_length == num_frames == 2*T_codes` always, so the mask is 2 (or 1 in
        // the in-context prefix) everywhere — never 0 — and the conditioning mask is the
        // identity. `mu` is therefore used in full. (flow_matching.py:203-220)
        let mu = upsample2_time_btc(&cond)?; // (1,2T,512)
        let num_frames = mu.dim(1)?;
        debug_assert_eq!(
            latent_length, num_frames,
            "pipeline invariant: latent_length (=int(d*25)) == 2*T_codes (=2*int(d*12.5))"
        );

        // In-context latents = `true_latents * (latent_masks == 1)`: keep the first
        // `incontext_length` frames, zero the rest (flow_matching.py:222-227). With
        // `latent_length == num_frames`, `incontext_length_actual == incontext_length`.
        let incontext_x = if incontext_length > 0 {
            let kept = true_latents.narrow(1, 0, incontext_length)?; // (1, ic, 256)
            let zeros_rest = Tensor::zeros(
                (1, num_frames - incontext_length, 256),
                kept.dtype(),
                kept.device(),
            )?;
            Tensor::cat(&[&kept, &zeros_rest], 1)?
        } else {
            noise.zeros_like()? // all-zero context (1,2T,256)
        };

        // Euler ODE solve (with the per-step in-context blend), then restore the
        // in-context prefix (flow_matching.py:233-250).
        let solved = self.solve_euler(noise, &incontext_x, incontext_length, &mu, num_steps, gs)?;
        if incontext_length > 0 {
            let head = incontext_x.narrow(1, 0, incontext_length)?;
            let tail = solved.narrow(1, incontext_length, num_frames - incontext_length)?;
            Ok(Tensor::cat(&[&head, &tail], 1)?)
        } else {
            Ok(solved)
        }
    }

    /// Fixed-step Euler ODE solver with classifier-free guidance and the optional
    /// in-context blend — port of `flow_matching.py::_solve_euler` (254-312).
    ///
    /// `x_init` is the initial latent; it is also the `noise` reference the
    /// in-context blend reads each step. `incontext_length == 0` skips the blend,
    /// leaving the verified single-segment Euler/CFG body untouched.
    fn solve_euler(
        &self,
        x_init: &Tensor,
        incontext_x: &Tensor,
        incontext_length: usize,
        mu: &Tensor,
        num_steps: usize,
        gs: f64,
    ) -> Result<Tensor> {
        let dev = x_init.device();
        let num_frames = x_init.dim(1)?;
        let mu_zeros = mu.zeros_like()?;
        let mut x = x_init.clone(); // (1,2T,256)
        // CFG batch halves that don't change across Euler steps, concatenated once.
        let (ic2, mu2) = if gs > 1.0 {
            (
                Some(Tensor::cat(&[incontext_x, incontext_x], 0)?),
                Some(Tensor::cat(&[&mu_zeros, mu], 0)?), // [uncond=zeros ; cond=mu]
            )
        } else {
            (None, None)
        };
        // t_span = linspace(0,1,num_steps+1) ⇒ uniform dt = 1/num_steps, t = step*dt.
        let dt = 1.0f64 / num_steps as f64;
        for step in 0..num_steps {
            let tval = step as f64 * dt;

            // In-context blend: x[:, :ic] = blend*noise[:, :ic] + t*incontext[:, :ic],
            // blend = 1 - (1-1e-6)*t ; frames [ic:] are left as the current x.
            if incontext_length > 0 {
                let blend = 1.0 - (1.0 - 1e-6) * tval;
                let head = ((x_init.narrow(1, 0, incontext_length)? * blend)?
                    + (incontext_x.narrow(1, 0, incontext_length)? * tval)?)?;
                let tail = x.narrow(1, incontext_length, num_frames - incontext_length)?;
                x = Tensor::cat(&[&head, &tail], 1)?;
            }

            let dphi = if gs > 1.0 {
                // Classifier-free guidance: batch [uncond(mu=0) ; cond(mu)].
                let x2 = Tensor::cat(&[&x, &x], 0)?;
                let (ic2, mu2) = (ic2.as_ref().unwrap(), mu2.as_ref().unwrap());
                let combined = Tensor::cat(&[&x2, ic2, mu2], 2)?; // (2,2T,1024)
                let t_input = Tensor::from_vec(vec![tval as f32; 2], 2, dev)?;
                let dphi = self.dit.forward(&combined, &t_input)?; // (2,2T,256)
                let uncond = dphi.narrow(0, 0, 1)?;
                let cond_d = dphi.narrow(0, 1, 1)?;
                (&uncond + ((cond_d - &uncond)? * gs)?)? // (1,2T,256)
            } else {
                let combined = Tensor::cat(&[&x, incontext_x, mu], 2)?; // (1,2T,1024)
                let t_input = Tensor::from_vec(vec![tval as f32; 1], 1, dev)?;
                self.dit.forward(&combined, &t_input)?
            };

            x = (x + (dphi * dt)?)?;
        }
        Ok(x)
    }

    /// `codes` (1,Q,T) integer codes, `noise` (1,2T,256) initial latent.
    /// Returns fm_latents (1,2T,256). The seg0 path: all frames conditioned,
    /// `incontext_length=0`, CFG `gs` over `num_steps` Euler steps. Thin wrapper
    /// over [`Self::inference_codes`].
    pub fn inference(
        &self,
        codes: &Tensor,
        noise: &Tensor,
        num_steps: usize,
        gs: f64,
    ) -> Result<Tensor> {
        let num_frames = noise.dim(1)?; // 2T
        let ctx = SegmentCtx {
            true_latents: noise,
            latent_length: num_frames,
            incontext_length: 0,
        };
        self.inference_codes(codes, &ctx, noise, num_steps, gs)
    }
}
