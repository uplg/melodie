#!/usr/bin/env python3
"""Dump golden gates for the HeartMuLa LM (port phase P2).

Runs one `generate_frame` on a fixed synthetic text-only prompt and records the
deterministic logits gates: codebook-0 logits from the backbone, and each
codebook 1..7 logits from the depth decoder. Sampling is stochastic, so we also
record the sampled tokens (`curr_sample`) and the candle port replays them into
the depth path — making both gates fully deterministic given the same inputs.

Run:
    uv run --project /Users/leonard/Github/heartlib-mlx \
        python crates/melodie-engine/reference/golden_dump_lm.py
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

LM_DIR = Path("/Users/leonard/Github/heartlib-mlx/convert/HeartMuLa-oss-3B")
OUT = Path(__file__).parent / "golden"
OUT.mkdir(parents=True, exist_ok=True)

import heartlib_mlx.heartmula.modeling_heartmula as M  # noqa: E402

NCB = 8  # audio_num_codebooks
S = 8

print("loading HeartMuLa LM (mlx, fp32)...")
model = M.HeartMuLa.from_pretrained(str(LM_DIR), dtype=mx.float32)
model.setup_caches(1)

# fixed synthetic text-only prompt: text in the last channel, audio channels empty
text_ids = np.array([128000, 100, 200, 300, 400, 500, 600, 128001], dtype=np.int32)
tokens = np.zeros((1, S, NCB + 1), dtype=np.int32)
tokens[0, :, NCB] = text_ids
mask = np.zeros((1, S, NCB + 1), dtype=bool)
mask[0, :, NCB] = True
input_pos = np.arange(S, dtype=np.int32)[None]  # [1,S]

# Inject + record the uniform draws used by Gumbel sampling so the candle port can
# replay them and reproduce the exact tokens (deterministic full-frame gate).
_urng = np.random.default_rng(777)
uniforms: list[np.ndarray] = []
_orig_uniform = mx.random.uniform


def _wrap_uniform(*a, shape=None, **kw):  # noqa: ANN001
    if shape is None and a:
        shape = a[0]
    arr = _urng.random(tuple(int(x) for x in shape)).astype(np.float32)
    uniforms.append(arr)
    return mx.array(arr)


mx.random.uniform = _wrap_uniform

# record (logits, sample) for every _sample_topk call (call 0 = c0, calls 1..7 = ci)
recs: list[tuple[np.ndarray, np.ndarray]] = []
_orig_sample = M._sample_topk


def _wrap_sample(logits, topk, temperature):  # noqa: ANN001
    s = _orig_sample(logits, topk, temperature)
    recs.append((np.array(logits).astype(np.float32), np.array(s).astype(np.int64)))
    return s


M._sample_topk = _wrap_sample

print("generate_frame...")
curr = model.generate_frame(
    mx.array(tokens),
    mx.array(mask),
    mx.array(input_pos),
    temperature=1.0,
    topk=50,
    cfg_scale=1.0,
)
mx.eval(curr)
curr = np.array(curr).astype(np.int64)  # [1, NCB]

c0_logits = recs[0][0]  # [1, V]
ci_logits = np.stack([recs[i][0][0] for i in range(1, NCB)], 0)  # [7, V]
lm_uniforms = np.stack([u[0] for u in uniforms[:NCB]], 0)  # [NCB, V]
print(f"calls={len(recs)} tokens={tokens.shape} c0={c0_logits.shape} ci={ci_logits.shape} curr={curr.shape}")

save_file(
    {
        "lm_tokens": tokens.astype(np.int64),
        "lm_mask": mask.astype(np.int64),
        "lm_input_pos": input_pos.astype(np.int64),
        "lm_c0_logits": c0_logits,        # [1, V]  backbone gate
        "lm_ci_logits": ci_logits,        # [7, V]  depth-decoder gate
        "lm_curr_sample": curr,           # [1, NCB] deterministic samples
        "lm_uniforms": lm_uniforms,       # [NCB, V] injected Gumbel uniforms
    },
    str(OUT / "lm_frame0.safetensors"),
)
print("wrote", OUT / "lm_frame0.safetensors")
