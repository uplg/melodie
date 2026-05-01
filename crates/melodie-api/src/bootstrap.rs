//! First-run bootstrap.
//!
//! When the `users` table is empty we want exactly one admin invite to exist
//! so the operator can sign themselves up. Two paths:
//!
//! - `MELODIE_BOOTSTRAP_INVITE` is set: upsert that exact code as an admin
//!   invite. Idempotent across restarts — useful for `docker-compose` deployments
//!   where the operator wants a deterministic code.
//! - Not set: generate a random URL-safe code, log it at WARN level, and
//!   upsert it. The operator copies it from the logs.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64URL;
use melodie_core::model::Role;
use rand::RngCore;
use sqlx::SqlitePool;

pub async fn ensure_bootstrap_invite(
    pool: &SqlitePool,
    configured_code: Option<&str>,
) -> anyhow::Result<()> {
    let user_count = melodie_db::users::count(pool).await?;
    if user_count > 0 {
        return Ok(());
    }

    let code: String = match configured_code {
        Some(c) if !c.trim().is_empty() => c.trim().to_string(),
        _ => {
            let mut bytes = [0u8; 24];
            rand::rng().fill_bytes(&mut bytes);
            BASE64URL.encode(bytes)
        }
    };

    let inserted = melodie_db::invites::upsert_idempotent(pool, &code, None, Role::Admin).await?;
    if inserted {
        if configured_code.is_some() {
            tracing::warn!(
                "Bootstrap admin invite registered from MELODIE_BOOTSTRAP_INVITE. Sign up at /signup with this code to claim admin."
            );
        } else {
            tracing::warn!(
                code = %code,
                "Bootstrap admin invite generated. Sign up at /signup with this code to claim admin."
            );
        }
    } else {
        tracing::info!("Bootstrap admin invite already present, leaving as-is");
    }

    Ok(())
}
