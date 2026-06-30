#!/usr/bin/env python3
"""Golden for a classifier-free-guidance (cfg_scale=1.5) generate_frame (P2/P3 quality).

B=2 (cond + uncond): the uncond half replaces text with the unconditional_text
embedding; codebook logits are guided = uncond + (cond-uncond)*cfg before sampling.
Uniforms are injected/recorded so the candle port reproduces the exact tokens.

Run:
    uv run --project /Users/leonard/Github/heartlib-mlx \
        python crates/melodie-engine/reference/golden_dump_lm_cfg.py
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

NCB = 8
S = 8
CFG = 1.5

print("loading HeartMuLa LM (mlx, fp32)...")
model = M.HeartMuLa.from_pretrained(str(LM_DIR), dtype=mx.float32)
model.setup_caches(2)  # cfg => batch 2

text_ids = np.array([128000, 100, 200, 300, 400, 500, 600, 128001], dtype=np.int32)
tokens = np.zeros((1, S, NCB + 1), dtype=np.int32)
tokens[0, :, NCB] = text_ids
mask = np.zeros((1, S, NCB + 1), dtype=bool)
mask[0, :, NCB] = True
# double for cfg (cond, uncond)
tokens2 = np.concatenate([tokens, tokens], 0)
mask2 = np.concatenate([mask, mask], 0)
input_pos = np.broadcast_to(np.arange(S, dtype=np.int32)[None], (2, S))

_urng = np.random.default_rng(777)
uniforms: list[np.ndarray] = []
_ou = mx.random.uniform


def _wu(*a, shape=None, **kw):  # noqa: ANN001
    if shape is None and a:
        shape = a[0]
    arr = _urng.random(tuple(int(x) for x in shape)).astype(np.float32)
    uniforms.append(arr)
    return mx.array(arr)


mx.random.uniform = _wu

recs: list[np.ndarray] = []
_os = M._sample_topk


def _ws(logits, topk, temperature):  # noqa: ANN001
    s = _os(logits, topk, temperature)
    recs.append(np.array(logits).astype(np.float32))  # guided logits [actual_B, V]
    return s


M._sample_topk = _ws

print("generate_frame cfg=1.5 ...")
curr = model.generate_frame(
    mx.array(tokens2),
    mx.array(mask2),
    mx.array(input_pos),
    temperature=1.0,
    topk=50,
    cfg_scale=CFG,
    continuous_segments=None,
    starts=None,
)
mx.eval(curr)
curr = np.array(curr).astype(np.int64)[0:1]  # [1, NCB]

c0_guided = recs[0]  # [1, V]
ci_guided = np.stack([recs[i][0] for i in range(1, NCB)], 0)  # [7, V]
uni = np.stack([u[0] for u in uniforms[:NCB]], 0)  # [NCB, V]
print(f"calls={len(recs)} c0g={c0_guided.shape} cig={ci_guided.shape} curr={curr.shape}")

save_file(
    {
        "cfg_tokens": tokens.astype(np.int64),     # [1,S,9] (candle doubles it)
        "cfg_mask": mask.astype(np.int64),
        "cfg_uniforms": uni,                        # [NCB, V]
        "cfg_c0_guided": c0_guided,                 # [1, V]
        "cfg_ci_guided": ci_guided,                 # [7, V]
        "cfg_curr_sample": curr,                    # [1, NCB]
    },
    str(OUT / "lm_cfg.safetensors"),
)
print("wrote", OUT / "lm_cfg.safetensors")
