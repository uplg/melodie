# Vendoring notice

This crate is vendored from [`paperfoot/suno-cli`](https://github.com/paperfoot/suno-cli),
licensed MIT (see `LICENSE`).

| Field | Value |
|---|---|
| Source SHA | `9032fd436081a227ebebf8eacfb4d1b2c40f6c67` |
| Source date | 2026-04-17 |
| Vendored on | 2026-05-01 |

## What was kept

- `src/api/` (entire module) — HTTP wrappers around Suno's web API.
- `src/auth.rs` — Clerk session/JWT exchange + JWT staleness check.
- `src/errors.rs` → `src/error.rs` — error type, renamed `CliError` → `SunoError`.
- `src/captcha.rs` — gated behind the `captcha` cargo feature, off by default.
- `src/download.rs` — gated behind the `id3` cargo feature, off by default.

## What was dropped or rewritten

- `src/main.rs`, `src/cli.rs`, `src/output/` — CLI-only.
- `src/config.rs` — figment config tree, replaced by Melodie's own config layer.
- `auth.rs::load`/`save`/`path`/`extract_clerk_cookie` — filesystem persistence
  and browser-cookie scraping (`directories`, `rookie`). Melodie persists
  `AuthState` in SQLite via its own store; the `SunoClient` only holds it in
  memory.
- `download.rs` progress bars (`indicatif`) — replaced by `tracing::debug!` calls.
- `api/mod.rs::refresh_jwt` no longer calls `auth.save()` — persistence is the
  caller's responsibility.

## Porting upstream fixes

When Suno changes its API mid-cycle:

1. Look at upstream commits since the SHA above touching `src/api/`, `src/auth.rs`,
   or `src/captcha.rs`.
2. Cherry-pick the relevant body changes into the corresponding files here.
3. Update the SHA above.
