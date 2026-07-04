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
- macOS with Apple Silicon (the engine targets candle's Metal backend)
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

## Dev

```bash
cp .env.example .env
just check        # cargo check across the workspace
just run          # run just the API
just dev          # api + web in parallel
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
