//! Vendored and trimmed from `paperfoot/suno-cli` `src/auth.rs`.
//!
//! This module retains the *protocol* pieces:
//! - [`AuthState`] — the in-memory bag of credentials a [`SunoClient`](crate::SunoClient) needs.
//! - [`is_jwt_expired`](AuthState::is_jwt_expired) — Suno's aggressive 30-min staleness window.
//! - [`browser_token`] — dynamic header value Suno expects.
//! - [`clerk_token_exchange`] / [`clerk_refresh_jwt`] — Clerk REST calls.
//!
//! It does *not* persist anything to disk (the upstream CLI wrote to
//! `~/.config/suno-cli/auth.json` via `directories`) and it does *not* scrape
//! browser cookies (upstream used `rookie`). Persistence and cookie acquisition
//! are the caller's responsibility — Melodie stores the cookie + JWT in SQLite
//! and accepts the cookie via an admin endpoint.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64URL;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::SunoError;

const CLERK_BASE: &str = "https://auth.suno.com";
const CLERK_JS_VERSION: &str = "5.117.0";

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct AuthState {
    pub jwt: Option<String>,
    pub session_id: Option<String>,
    pub device_id: Option<String>,
    /// The `__client` cookie value from `auth.suno.com` — long-lived (~7 days).
    pub clerk_client_cookie: Option<String>,
}

impl AuthState {
    pub fn is_jwt_expired(&self) -> bool {
        let Some(jwt) = &self.jwt else { return true };
        let parts: Vec<&str> = jwt.split('.').collect();
        if parts.len() != 3 {
            return true;
        }
        let claims = parts[1];
        let Ok(decoded) = BASE64URL.decode(claims) else {
            return true;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(&decoded) else {
            return true;
        };
        let Some(exp) = value.get("exp").and_then(|v| v.as_u64()) else {
            return true;
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Suno issues 1-hour JWTs but their generation endpoint silently
        // rejects tokens older than ~30 min with "Token validation failed."
        // even when the JWT's own `exp` claim is still valid. Refresh well
        // before that boundary.
        now + 1800 >= exp
    }
}

/// Generate the dynamic `browser-token` header value Suno expects.
pub fn browser_token() -> String {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let payload = format!(r#"{{"timestamp":{ms}}}"#);
    let encoded = BASE64.encode(payload.as_bytes());
    format!(r#"{{"token":"{encoded}"}}"#)
}

/// Exchange the `__client` Clerk cookie for `(session_id, jwt)`.
pub async fn clerk_token_exchange(
    client: &reqwest::Client,
    clerk_cookie: &str,
) -> Result<(String, String), SunoError> {
    let resp = client
        .get(format!(
            "{CLERK_BASE}/v1/client?_clerk_js_version={CLERK_JS_VERSION}"
        ))
        .header("cookie", format!("__client={clerk_cookie}"))
        .send()
        .await
        .map_err(SunoError::Http)?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(SunoError::Api {
            code: "clerk_exchange_failed",
            message: format!("Clerk token exchange failed ({status}): {body}"),
        });
    }

    let body: serde_json::Value = resp.json().await.map_err(SunoError::Http)?;
    let session_id = body
        .get("response")
        .and_then(|r| r.get("last_active_session_id"))
        .and_then(|s| s.as_str())
        .ok_or_else(|| SunoError::Api {
            code: "no_session",
            message: "No active Clerk session — the cookie may be invalid or logged out".into(),
        })?
        .to_string();

    let jwt = clerk_refresh_jwt(client, clerk_cookie, &session_id).await?;
    Ok((session_id, jwt))
}

/// Refresh JWT using stored Clerk cookie + session ID.
pub async fn clerk_refresh_jwt(
    client: &reqwest::Client,
    clerk_cookie: &str,
    session_id: &str,
) -> Result<String, SunoError> {
    let resp = client
        .post(format!(
            "{CLERK_BASE}/v1/client/sessions/{session_id}/tokens?_clerk_js_version={CLERK_JS_VERSION}"
        ))
        .header("cookie", format!("__client={clerk_cookie}"))
        .header("content-type", "application/x-www-form-urlencoded")
        .send()
        .await
        .map_err(SunoError::Http)?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(SunoError::Api {
            code: "clerk_refresh_failed",
            message: format!("Clerk JWT refresh failed ({status}): {body}"),
        });
    }

    let body: serde_json::Value = resp.json().await.map_err(SunoError::Http)?;
    body.get("jwt")
        .and_then(|j| j.as_str())
        .map(String::from)
        .ok_or_else(|| SunoError::Api {
            code: "no_jwt",
            message: "Clerk returned no JWT — the session may have been revoked".into(),
        })
}
