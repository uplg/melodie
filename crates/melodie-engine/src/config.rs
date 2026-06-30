//! Static model configuration, ported from `../heartlib-mlx`
//! (`configuration_heartmula.py`, `configuration_heartcodec.py`, `gen_config.json`).
//!
//! Values are the `FLAVORS` / dataclass defaults for the `HeartMuLa-oss-3B`
//! checkpoint; the real `config.json` is filtered against these at load time.

use serde::Deserialize;

/// One Llama-style transformer "flavor" (the temporal backbone OR the depth decoder).
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaFlavor {
    pub num_layers: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
}

impl LlamaFlavor {
    /// `llama-3B` temporal backbone — configuration_heartmula.py:53-60.
    pub fn backbone_3b() -> Self {
        Self {
            num_layers: 28,
            embed_dim: 3072,
            num_heads: 24,
            num_kv_heads: 8, // GQA, q_per_kv = 3
            head_dim: 128,   // 3072 / 24
            intermediate_dim: 8192,
            max_seq_len: 8192,
        }
    }

    /// `llama-300M` depth/"sub" decoder — configuration_heartmula.py:61-68.
    /// Note the unusually large head_dim (3072/8 = 384); RoPE is built for it.
    pub fn decoder_300m() -> Self {
        Self {
            num_layers: 3,
            embed_dim: 3072,
            num_heads: 8,
            num_kv_heads: 4, // GQA, q_per_kv = 2
            head_dim: 384,   // 3072 / 8
            intermediate_dim: 8192,
            max_seq_len: 2048,
        }
    }
}

/// HeartMuLa LM config (the RQ-Transformer wrapper) — configuration_heartmula.py:17-49.
#[derive(Debug, Clone)]
pub struct HeartMuLaConfig {
    pub backbone: LlamaFlavor,
    pub decoder: LlamaFlavor,
    pub text_vocab_size: usize,     // 128256 (Llama-3 tokenizer)
    pub audio_vocab_size: usize,    // 8197
    pub audio_num_codebooks: usize, // 8
    pub muq_dim: usize,             // 512 (style/reference embedding)
    // Llama-3.1 scaled RoPE
    pub rope_base: f64,              // 500000
    pub rope_scale_factor: f64,      // 32
    pub rope_low_freq_factor: f64,   // 1
    pub rope_high_freq_factor: f64,  // 4
    pub rope_old_context_len: usize, // 8192
    pub norm_eps: f64,               // 1e-5 (RMSNorm)
}

impl Default for HeartMuLaConfig {
    fn default() -> Self {
        Self {
            backbone: LlamaFlavor::backbone_3b(),
            decoder: LlamaFlavor::decoder_300m(),
            text_vocab_size: 128256,
            audio_vocab_size: 8197,
            audio_num_codebooks: 8,
            muq_dim: 512,
            rope_base: 500_000.0,
            rope_scale_factor: 32.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_old_context_len: 8192,
            norm_eps: 1e-5,
        }
    }
}

/// Residual-VQ used by HeartCodec for the conditioning codes — configuration_heartcodec.py:18-25.
#[derive(Debug, Clone)]
pub struct RvqConfig {
    pub num_quantizers: usize, // 8
    pub codebook_size: usize,  // 8192
    pub codebook_dim: usize,   // 32
    pub dim: usize,            // 512 (project_out target)
}

/// DiT flow-matching estimator — configuration_heartcodec.py:26-34, transformer.py.
#[derive(Debug, Clone)]
pub struct DitConfig {
    pub num_layers_stage1: usize, // 24
    pub num_layers_stage2: usize, // 6
    pub num_heads: usize,         // 24
    pub head_dim: usize,          // 64 (stage1); stage2 doubles to 128
    pub rope_base: f64,           // 10000
    pub in_channels: usize,       // 1024 = [noisy x 256 | incontext 256 | cond mu 512]
    pub out_channels: usize,      // 256
    pub timestep_dim: usize,      // 512 (flow_t_size)
    pub norm_eps: f64,            // 1e-6
}

/// HeartCodec config — configuration_heartcodec.py:17-49, modeling_heartcodec.py.
#[derive(Debug, Clone)]
pub struct HeartCodecConfig {
    pub sample_rate: usize,       // 48000
    pub causal: bool,             // true
    pub latent_hidden_dim: usize, // 128 (per-stream SQ latent)
    /// SQ encoder/decoder up/down ratios; product * num_samples = 1920 (→ 25 Hz).
    pub ratios: [usize; 5], // encoder [3,4,4,4,5]; decoder is reversed
    pub num_samples: usize,       // 2 (Pre/PostProcessor avgpool/repeat)
    pub rvq: RvqConfig,
    pub dit: DitConfig,
    pub flow_num_steps: usize,    // 10 (Euler ODE)
    pub flow_guidance_scale: f64, // 1.25 (CFG)
    pub codes_frame_rate: f64,    // 12.5 Hz (RVQ codes from the LM)
    pub latent_frame_rate: f64,   // 25 Hz (FM latent)
    pub segment_duration: f64,    // 29.76 s per decode segment
}

impl Default for HeartCodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            causal: true,
            latent_hidden_dim: 128,
            ratios: [3, 4, 4, 4, 5],
            num_samples: 2,
            rvq: RvqConfig {
                num_quantizers: 8,
                codebook_size: 8192,
                codebook_dim: 32,
                dim: 512,
            },
            dit: DitConfig {
                num_layers_stage1: 24,
                num_layers_stage2: 6,
                num_heads: 24,
                head_dim: 64,
                rope_base: 10_000.0,
                in_channels: 1024,
                out_channels: 256,
                timestep_dim: 512,
                norm_eps: 1e-6,
            },
            flow_num_steps: 10,
            flow_guidance_scale: 1.25,
            codes_frame_rate: 12.5,
            latent_frame_rate: 25.0,
            segment_duration: 29.76,
        }
    }
}

/// Special token IDs — `gen_config.json` / HeartMuLaGenConfig (music_generation.py:31-34).
#[derive(Debug, Clone, Deserialize)]
pub struct GenConfig {
    pub text_bos_id: u32,  // 128000
    pub text_eos_id: u32,  // 128001
    pub audio_eos_id: u32, // 8193 (codebook-0 token >= this => EOS)
    pub empty_id: u32,     // 0
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            text_bos_id: 128000,
            text_eos_id: 128001,
            audio_eos_id: 8193,
            empty_id: 0,
        }
    }
}
