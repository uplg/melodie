set dotenv-load := true
set shell := ["bash", "-cu"]

default:
    @just --list

# --- Backend ---

check:
    cargo check --workspace --all-targets

build:
    cargo build --workspace

test:
    cargo test --workspace

fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets -- -D warnings

run:
    cargo run -p melodie-api

# --- Frontend ---

web-install:
    cd web && bun install

web-dev:
    cd web && bun run dev

web-build:
    cd web && bun run build

# --- Full dev: HMR-friendly api + Vite dev server. Use this while iterating. ---

dev:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    cargo run -p melodie-api &
    (cd web && bun run dev) &
    wait

# --- Live: prod-build the frontend (Astro SSR Node bundle), release-build the
#     API, run both, expose the front through a one-shot cloudflared tunnel.
#     Slower to start than `dev` (one-time builds) but matches the perf the
#     friends will see, and avoids Vite-only host-check / HMR quirks. ---

live:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "▶ building backend (release)…"
    cargo build --release -p melodie-api

    echo "▶ building frontend (astro build)…"
    (cd web && bun run build)

    echo "▶ starting api + web (prod) + cloudflared…"
    trap 'kill 0' EXIT
    ./target/release/melodie-api &
    (cd web && HOST=127.0.0.1 PORT=3000 node dist/server/entry.mjs) &
    cloudflared tunnel --url http://localhost:3000 &
    wait
