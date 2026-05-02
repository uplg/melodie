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

    LIVE_URL_FILE="${MELODIE_URL_FILE:-$HOME/.melodie/live_url}"
    mkdir -p "$(dirname "$LIVE_URL_FILE")"

    echo "▶ starting api + web (prod) + cloudflared…"
    echo "▶ live URL will be written to $LIVE_URL_FILE for homie's !melodie"
    trap 'rm -f "$LIVE_URL_FILE"; kill 0' EXIT
    ./target/release/melodie-api &
    (cd web && HOST=127.0.0.1 PORT=3000 node dist/server/entry.mjs) &
    cloudflared tunnel --url http://localhost:3000 2>&1 | awk -v file="$LIVE_URL_FILE" '
      { print; fflush() }
      !wrote && match($0, /https:\/\/[a-z0-9][a-z0-9-]*\.trycloudflare\.com/) {
        url = substr($0, RSTART, RLENGTH)
        tmp = file ".tmp"
        printf "%s\n", url > tmp
        close(tmp)
        system("mv -f \"" tmp "\" \"" file "\"")
        printf "▶ live URL written: %s\n", url
        fflush()
        wrote = 1
      }
    ' &
    wait
