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

# --- Frontend (lands in P3) ---

web-install:
    cd web && bun install

web-dev:
    cd web && bun dev

web-build:
    cd web && bun build

# --- Full dev: api + web in parallel ---

dev:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    cargo run -p melodie-api &
    (cd web && bun dev) &
    wait

# --- Going live: same as `dev` but exposes web through a one-shot
#     cloudflared tunnel. Cloudflared prints the public URL on stdout. ---

live:
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    cargo run -p melodie-api &
    (cd web && bun dev) &
    cloudflared tunnel --url http://localhost:3000 &
    wait
