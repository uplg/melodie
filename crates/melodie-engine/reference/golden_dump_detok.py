#!/usr/bin/env python3
"""Golden for the full HeartCodec.detokenize: pad codes to the 29.76 s segment,
flow-matching decode, ScalarModel decode, trim to target length. FM noise injected.

Run:
    uv run --project /Users/leonard/Github/heartlib-mlx \
        python crates/melodie-engine/reference/golden_dump_detok.py
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

# Force CPU: a full 29.76 s segment decode overruns the macOS Metal GPU watchdog.
mx.set_default_device(mx.cpu)

CODEC_DIR = Path("/Users/leonard/Github/heartlib-mlx/convert/HeartCodec-oss")
OUT = Path(__file__).parent / "golden"

T = 16  # original code frames -> target_len = 16/12.5*48000 = 61440
DURATION = 7.44  # smaller segment (min_samples=93) — same pad/trim logic, ~4x faster
NUM_STEPS = 10
GS = 1.25

from heartlib_mlx.heartcodec import HeartCodec  # noqa: E402

print("loading codec...")
codec = HeartCodec.from_pretrained(str(CODEC_DIR), dtype=mx.float32)

# inject + record FM noise AFTER load
rng = np.random.default_rng(1234)
recorded: list[np.ndarray] = []


def _patched(shape, *a, **k):  # noqa: ANN001
    arr = rng.standard_normal(tuple(int(s) for s in shape)).astype(np.float32)
    recorded.append(arr)
    return mx.array(arr)


mx.random.normal = _patched

codes_np = np.random.default_rng(99).integers(0, 8192, size=(8, T)).astype(np.int64)
codes = mx.array(codes_np.astype(np.int32))

print("detokenize...")
wav = codec.detokenize(codes, duration=DURATION, num_steps=NUM_STEPS, guidance_scale=GS)
mx.eval(wav)
wav_np = np.array(wav).astype(np.float32)  # [2, target_len]

# recorded[0] = first_latent (masked), recorded[1] = FM noise (the one that matters)
fm_noise = recorded[1].astype(np.float32)  # [1, 744, 256]
print(f"calls={len(recorded)} codes={codes_np.shape} fm_noise={fm_noise.shape} wav={wav_np.shape}")

save_file(
    {
        "dtk_codes": codes_np,        # [8, T]
        "dtk_fm_noise": fm_noise,     # [1, 2*min_samples, 256]
        "dtk_waveform": wav_np,       # [2, target_len]
    },
    str(OUT / "detok.safetensors"),
)
print("wrote", OUT / "detok.safetensors")
