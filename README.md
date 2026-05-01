# Melodie

Self-hosted Suno wrapper for a small group of friends. One Suno account
upstream, N app users in front, per-user scoping, daily quotas, MP3 streaming.

Designed to run locally on the operator's machine and exposed publicly via
`cloudflared` only during "live" sessions — keeps the Suno hCaptcha solver
happy (it needs a real Chrome with a warm browser fingerprint) and avoids
running anything 24/7.

## Stack

- Backend: Rust 2024, axum 0.8, SQLite (sqlx), tower-sessions, argon2id
- Frontend: Astro 6 SSR (Node) + React 19 islands, TypeScript strict, Tailwind v4
- Suno bridge: `crates/suno-client` (vendored from `paperfoot/suno-cli`)

## Layout

```
crates/
  suno-client/   # vendored Suno HTTP client + auth + hCaptcha (library)
  melodie-core/  # domain types + traits
  melodie-db/    # sqlx pool + migrations
  melodie-api/   # axum binary
web/             # Astro 6 app
```

## Prerequisites

- Rust stable (edition 2024 — see `rust-toolchain.toml`)
- `chromium` or `google-chrome` on `PATH` — required by the Suno hCaptcha solver
- `cloudflared` on `PATH` — only when sharing a session with friends
- (later) Node ≥ 22, bun

## Dev

```bash
cp .env.example .env
just check        # cargo check across the workspace
just run          # run just the API
just dev          # api + web in parallel (web lands in P3)
```

## Going live

```bash
just live                                  # in one terminal
cloudflared tunnel --url http://localhost:3000   # in another
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
