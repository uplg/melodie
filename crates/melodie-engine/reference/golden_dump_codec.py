#!/usr/bin/env python3
"""Dump golden vectors for HeartCodec decoder parity (port phase P1b).

Instead of the full `detokenize` (which forces a ~30 s segment and overruns the
macOS Metal GPU watchdog), we call the core single-segment path directly on a
SMALL input: FlowMatching.inference_codes -> reshape -> ScalarModel.decode.
Small command buffers run fine on the GPU, and the goldens are quick to check
in Rust.

The only stochastic input is `mx.random.normal` at flow_matching.py:201; we
monkeypatch it with a seeded numpy RNG and record the draw so the exact noise
can be replayed in candle.

Run:
    uv run --project /Users/leonard/Github/heartlib-mlx \
        python crates/melodie-engine/reference/golden_dump_codec.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

CODEC_DIR = Path("/Users/leonard/Github/heartlib-mlx/convert/HeartCodec-oss")
OUT = Path(__file__).parent / "golden"
OUT.mkdir(parents=True, exist_ok=True)

T_CODES = 16            # small single segment: 16 code frames -> 32 latent frames
NUM_STEPS = 10
GUIDANCE = 1.25
NOISE_SEED = 1234

from heartlib_mlx.heartcodec import HeartCodec  # noqa: E402
from heartlib_mlx.heartcodec.models.sq_codec import round_func9  # noqa: E402

# --- deterministic, recorded noise for mx.random.normal (flow_matching.py:201) ---
rng = np.random.default_rng(NOISE_SEED)
recorded: list[np.ndarray] = []


def patched_normal(shape, *args, **kwargs):  # noqa: ANN001
    arr = rng.standard_normal(tuple(int(s) for s in shape)).astype(np.float32)
    recorded.append(arr)
    return mx.array(arr)


print("loading HeartCodec (mlx, fp32)...")
codec = HeartCodec.from_pretrained(str(CODEC_DIR), dtype=mx.float32)

# Install the recording RNG AFTER load so model-init randoms (e.g. the AdaLN
# scale_shift_table) aren't captured — only the inference noise is recorded.
mx.random.normal = patched_normal

# Capture the first DiT estimator forward (combined input, timestep, output) as an
# isolated parity gate for the transformer half. __call__ is resolved on the class,
# so patch the class method.
from heartlib_mlx.heartcodec.models.transformer import LlamaTransformer  # noqa: E402

_orig_est = LlamaTransformer.__call__
est_cap: dict[str, np.ndarray] = {}


def _wrap_est(self, hidden_states, timestep=None):  # noqa: ANN001
    out = _orig_est(self, hidden_states, timestep=timestep)
    if "in" not in est_cap:
        est_cap["est_in"] = np.array(hidden_states).astype(np.float32)
        est_cap["est_t"] = np.array(timestep).astype(np.float32)
        est_cap["est_out"] = np.array(out).astype(np.float32)
        est_cap["in"] = est_cap["est_in"]  # sentinel
    return out


LlamaTransformer.__call__ = _wrap_est

# fixed input codes in valid RVQ index range [0, codebook_size=8192)
codes_np = rng.integers(0, 8192, size=(8, T_CODES)).astype(np.int64)
codes = mx.array(codes_np.astype(np.int32))[None]  # (1, 8, T)

latent_length = 2 * T_CODES                      # all frames conditioned (mask==2)
true_latents = mx.zeros((1, 2 * T_CODES, 256))   # masked out (incontext_length=0)

print(f"inference_codes: codes (1,8,{T_CODES}), steps={NUM_STEPS}, gs={GUIDANCE} ...")
fm_latents = codec.flow_matching.inference_codes(
    [codes],
    true_latents,
    latent_length,
    0,  # incontext_length
    guidance_scale=GUIDANCE,
    num_steps=NUM_STEPS,
    scenario="other_seg",
)
mx.eval(fm_latents)

# reshape FM latent -> ScalarModel decode input (modeling_heartcodec.py:184-186)
B, T_lat, F_lat = fm_latents.shape
latent_in = fm_latents.reshape(B, T_lat, 2, F_lat // 2)
latent_in = latent_in.transpose(0, 2, 1, 3)
latent_in = latent_in.reshape(B * 2, T_lat, F_lat // 2)  # (2, 2T, 128)

print("scalar_model.decode (tapped) ...")
x = round_func9(latent_in)
taps: dict[str, np.ndarray] = {}
for i, layer in enumerate(codec.scalar_model.decoder):
    x = layer(x)
    mx.eval(x)
    taps[f"dec{i}"] = np.array(x).astype(np.float32)  # (N, L_i, C_i) MLX layout
wav = x.squeeze(-1)                        # (2, L)
mx.eval(wav)

# isolate block-1 up_conv (transposed conv) to localise the dec1 divergence
_dec0 = mx.array(taps["dec0"])
_up1 = codec.scalar_model.decoder[1].up_conv(_dec0)
mx.eval(_up1)
taps["up1"] = np.array(_up1).astype(np.float32)  # (2, 160, 1024) NLC

fm_noise = recorded[0].astype(np.float32)   # the one normal draw at line 201
assert fm_noise.shape == tuple(fm_latents.shape), f"{fm_noise.shape} vs {tuple(fm_latents.shape)}"

fm_latents_np = np.array(fm_latents).astype(np.float32)
latent_in_np = np.array(latent_in).astype(np.float32)
wav_np = np.array(wav).astype(np.float32)

print(
    "shapes:",
    "codes", codes_np.shape,
    "| fm_noise", fm_noise.shape,
    "| fm_latents", fm_latents_np.shape,
    "| latent_in", latent_in_np.shape,
    "| waveform", wav_np.shape,
)

tensors = {
    "codes": codes_np,                  # (8, T) i64
    "fm_noise": fm_noise,               # (1, 2T, 256) f32  -- inject in candle
    "fm_latents": fm_latents_np,        # (1, 2T, 256) f32  -- FM output gate
    "latent_in": latent_in_np,          # (2, 2T, 128) f32  -- ScalarModel.decode input gate
    "waveform": wav_np,                 # (2, L) f32        -- final gate
}
tensors.update(taps)                    # dec0..dec7 intermediate taps (MLX N,L,C layout)
for kk in ("est_in", "est_t", "est_out"):  # DiT estimator parity gate (first forward)
    tensors[kk] = est_cap[kk]
print("estimator gate:", {kk: est_cap[kk].shape for kk in ("est_in", "est_t", "est_out")})
save_file(tensors, str(OUT / "codec_seg0.safetensors"))
(OUT / "codec_seg0.json").write_text(
    json.dumps(
        {
            "t_codes": T_CODES,
            "num_steps": NUM_STEPS,
            "guidance_scale": GUIDANCE,
            "noise_seed": NOISE_SEED,
            "noise_call_index": 0,
            "latent_dim": 256,
        },
        indent=2,
    )
)
print("wrote", OUT / "codec_seg0.safetensors")
