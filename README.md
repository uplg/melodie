# Melodie

Self-hosted music generator for a small group of friends, built on a local
HeartMuLa inference engine. Designed to run on the operator's machine and
exposed publicly via `cloudflared` only during "live" sessions, avoiding
running anything 24/7.

## Stack

- Backend: Rust 2024, axum 0.8, SQLite (sqlx), tower-sessions, argon2id
- Engine: `melodie-engine`, a pure-Rust (candle, Metal backend) port of
  HeartMuLa — runs generation fully on-device, no upstream API
- Frontend: Astro 7 SSR (Node) + React 19 islands, TypeScript strict, Tailwind v4

## Layout

```
crates/
  melodie-core/   # domain types + traits
  melodie-db/     # sqlx pool + migrations
  melodie-engine/ # local HeartMuLa inference engine (candle)
  melodie-api/    # axum binary
web/              # Astro 7 app
```

## Prerequisites

- Rust stable (edition 2024 — see `rust-toolchain.toml`)
- macOS with Apple Silicon, ≥ 32 GB unified memory (the engine targets candle's
  Metal backend; ~14 GB resident once loaded — 7.5 GB LM bf16 + 6.3 GB codec —
  with a ~16 GB transient peak while the checkpoints load)
- HeartMuLa checkpoints + tokenizer on disk — see [Models](#models)
- `cloudflared` on `PATH` — only when sharing a session with friends
- Node ≥ 22, bun

## Models

The engine loads the **original** HeartMuLa safetensors with candle. 
Three artifacts are needed
(~21 GB total, ~15 GB LM + ~6 GB codec):

| Piece         | Hugging Face repo                          | Files                                   |
| ------------- | ------------------------------------------ | --------------------------------------- |
| LM (3B)       | `HeartMuLa/HeartMuLa-oss-3B-happy-new-year`| `model-0000{1..4}-of-00004.safetensors` |
| Codec         | `HeartMuLa/HeartCodec-oss-20260123`        | `model-0000{1..2}-of-00002.safetensors` |
| Tokenizer     | `HeartMuLa/HeartMuLaGen`                   | `tokenizer.json`                        |

Fetch everything with curl (no Python, no `hf` CLI; resumable and idempotent —
re-run it after an interrupted download and it picks up where it left off):

```bash
just fetch-models   # → data/models/ (override with MELODIE_MODELS_DIR)
```

`.env.example` already points `MELODIE_LM_DIR`, `MELODIE_CODEC_DIR` and
`MELODIE_TOKENIZER` at `data/models/`.

## Engine knobs

All optional. The defaults ARE the fastest and most memory-efficient measured
setup on Apple Silicon (bf16 LM + GQA KV cache, Q8_0 depth decoder, bf16 codec
DiT — LM ≈ 59 ms/frame on an M1 Max, faster than realtime): only set these to
A/B against the defaults or to debug.

| Variable              | Effect                                                                     |
| --------------------- | -------------------------------------------------------------------------- |
| `MELODIE_NO_Q8=1`     | depth decoder back to dense bf16 instead of Q8_0 (by-ear A/B; ~11% slower) |
| `MELODIE_CODEC_F32=1` | codec DiT back to f32 (by-ear A/B; same speed, ~1.5 GB more resident)      |
| `MELODIE_PROFILE=1`   | per-frame LM timing summary on stdout                                      |
| `MELODIE_MODELS_DIR`  | target dir of `just fetch-models` (default `data/models`)                  |

Deeper debug switches (numerics hunts, not for normal use): `MELODIE_PROF2`
(synced per-frame breakdown), `MELODIE_CPU_SAMPLE` (scalar CPU sampling path),
`MELODIE_DBG` (layer-activation stats), `MELODIE_NOTOPK` (skip top-k, diagnostic),
`MELODIE_DECODE_CH` / `MELODIE_DECODE_R` (codec streaming-decode chunk/context).

## Dev

```bash
cp .env.example .env
just check        # cargo check across the workspace
just dev          # api + web in parallel (HMR; engine deps built optimized)
just live         # prod build + one-shot cloudflared tunnel (see below)
```

## Going live

```bash
just live
```

`cloudflared` prints a `*.trycloudflare.com` URL — share it with your friends
for the duration of the session. Each launch produces a fresh URL, so users
re-login each time. That's intentional; we'll move to a named tunnel if the
"live" cadence becomes regular.

## Bootstrap admin

On first start, if no users exist, the API ensures one admin invite is in the
DB:

- If `MELODIE_BOOTSTRAP_INVITE` is set, that exact code is registered as an
  admin invite (idempotent across restarts).
- Otherwise a random URL-safe code is generated and logged at WARN level —
  copy it from the logs, sign up with it, you become admin.
