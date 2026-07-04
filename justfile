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

# Unused deps, license/advisory/source policy (see deny.toml).
deny:
    cargo machete
    cargo deny check

# Everything an agent should run before calling a backend change done.
# Mirrors CI; read-only (fmt --check, not `just fmt`, so it never mutates).
gate:
    cargo fmt --all -- --check
    just check
    just lint
    just test
    just deny

# --- Models: fetch the HeartMuLa checkpoints straight from Hugging Face.
#     curl-only (no Python/hf CLI), resumable, idempotent. ~21 GB total.
#     The engine reads the original safetensors — no MLX conversion step. ---

fetch-models:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="${MELODIE_MODELS_DIR:-data/models}"
    fetch() { # <hf-repo> <remote-file> <dest>
        local url="https://huggingface.co/$1/resolve/main/$2" dest="$3" remote
        mkdir -p "$(dirname "$dest")"
        remote=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
        if [[ -f "$dest" && "$(stat -f%z "$dest")" -eq "$remote" ]]; then
            echo "✓ $dest (already complete)"
            return
        fi
        echo "▶ $dest ($remote bytes)"
        curl -fL --retry 3 -C - --progress-bar -o "$dest" "$url"
    }
    for i in 1 2 3 4; do
        fetch HeartMuLa/HeartMuLa-oss-3B-happy-new-year \
            "model-0000$i-of-00004.safetensors" \
            "$dir/HeartMuLa-oss-3B/model-0000$i-of-00004.safetensors"
    done
    for i in 1 2; do
        fetch HeartMuLa/HeartCodec-oss-20260123 \
            "model-0000$i-of-00002.safetensors" \
            "$dir/HeartCodec-oss/model-0000$i-of-00002.safetensors"
    done
    fetch HeartMuLa/HeartMuLaGen tokenizer.json "$dir/tokenizer.json"
    echo "✓ all models under $dir/ — MELODIE_LM_DIR/MELODIE_CODEC_DIR/MELODIE_TOKENIZER in .env"

# --- Frontend ---

web-install:
    cd web && bun install

web-dev:
    cd web && bun run dev

web-build:
    cd web && bun run build

# --- Full dev: HMR-friendly api + Vite dev server. Use this while iterating.
#     (Deps are compiled at opt-level 3 — see Cargo.toml — so the engine runs at
#     near-release speed even in dev builds.) ---

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
