//! Adapted from `paperfoot/suno-cli` `src/api/mod.rs`.
//!
//! Differences from upstream:
//! - `eprintln!` calls replaced with `tracing` macros.
//! - `refresh_jwt()` no longer calls `auth.save()` — persistence is the
//!   responsibility of the embedding application via [`SunoClient::auth_snapshot`].
//! - `new_with_refresh` returns immediately on a fresh JWT and only refreshes
//!   if `is_jwt_expired()` is true; on refresh failure it returns
//!   [`SunoError::AuthExpired`] without printing.

pub mod billing;
pub mod concat;
pub mod cover;
pub mod delete;
pub mod feed;
pub mod generate;
pub mod lyrics;
pub mod metadata;
pub mod persona;
pub mod remaster;
pub mod stems;
pub mod types;

use std::sync::Mutex;

use reqwest::Client;

use crate::auth::{self, AuthState};
use crate::error::SunoError;

pub struct SunoClient {
    client: Client,
    /// Auth state behind a sync mutex so `&self` methods can transparently
    /// refresh the JWT mid-request when Suno returns "Token validation failed."
    /// (their server-side staleness threshold kicks in well before the JWT's
    /// own `exp` claim). The lock is only held briefly to read/clone auth
    /// fields — never across awaits.
    auth: Mutex<AuthState>,
}

const BASE_URL: &str = "https://studio-api-prod.suno.com";

impl SunoClient {
    /// Build a client. If the JWT is expired but a Clerk cookie is available,
    /// transparently refresh once before returning.
    pub async fn new_with_refresh(mut auth: AuthState) -> Result<Self, SunoError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
            .build()
            .map_err(|e| SunoError::Config(format!("HTTP client: {e}")))?;

        if auth.is_jwt_expired() {
            if let (Some(cookie), Some(session_id)) =
                (auth.clerk_client_cookie.as_deref(), auth.session_id.as_deref())
            {
                tracing::debug!("JWT expired, refreshing via Clerk");
                match auth::clerk_refresh_jwt(&client, cookie, session_id).await {
                    Ok(jwt) => {
                        auth.jwt = Some(jwt);
                        tracing::debug!("JWT refreshed successfully");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Clerk JWT refresh failed");
                        return Err(SunoError::AuthExpired);
                    }
                }
            } else {
                return Err(SunoError::AuthExpired);
            }
        }

        Ok(Self {
            client,
            auth: Mutex::new(auth),
        })
    }

    /// Snapshot the current auth state. Useful for persisting refreshed JWTs.
    pub fn auth_snapshot(&self) -> AuthState {
        self.auth.lock().expect("auth mutex poisoned").clone()
    }

    pub(crate) fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.client
            .get(format!("{BASE_URL}{path}"))
            .headers(self.headers())
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{BASE_URL}{path}"))
            .headers(self.headers())
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        let (jwt, device) = {
            let auth = self.auth.lock().expect("auth mutex poisoned");
            (
                auth.jwt.clone(),
                auth.device_id
                    .clone()
                    .unwrap_or_else(|| "00000000-0000-0000-0000-000000000000".to_string()),
            )
        };
        if let Some(jwt) = jwt
            && let Ok(val) = format!("Bearer {jwt}").parse()
        {
            headers.insert("authorization", val);
        }
        if let Ok(val) = device.parse() {
            headers.insert("device-id", val);
        }
        if let Ok(val) = auth::browser_token().parse() {
            headers.insert("browser-token", val);
        }
        if let Ok(val) = "https://suno.com".parse() {
            headers.insert("origin", val);
        }
        if let Ok(val) = "https://suno.com/".parse() {
            headers.insert("referer", val);
        }
        headers
    }

    /// Refresh the JWT via the stored Clerk session cookie. Used by the
    /// in-process retry path in `with_auth_retry` when Suno's server-side
    /// staleness check fires mid-request despite a still-valid `exp` claim.
    pub(crate) async fn refresh_jwt(&self) -> Result<(), SunoError> {
        let (cookie, session_id) = {
            let auth = self.auth.lock().expect("auth mutex poisoned");
            (
                auth.clerk_client_cookie
                    .clone()
                    .ok_or(SunoError::AuthExpired)?,
                auth.session_id.clone().ok_or(SunoError::AuthExpired)?,
            )
        };
        let jwt = auth::clerk_refresh_jwt(&self.client, &cookie, &session_id).await?;
        {
            let mut auth = self.auth.lock().expect("auth mutex poisoned");
            auth.jwt = Some(jwt);
        }
        Ok(())
    }

    /// Run an async API call once. If it fails with `AuthExpired`, refresh
    /// the JWT and try a single retry. Wraps the write/poll paths so
    /// long-running waits (5–30+ minute generation queues) survive Suno's
    /// JWT staleness window.
    pub(crate) async fn with_auth_retry<F, Fut, T>(&self, mut f: F) -> Result<T, SunoError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, SunoError>>,
    {
        match f().await {
            Err(SunoError::AuthExpired) => {
                self.refresh_jwt().await?;
                f().await
            }
            other => other,
        }
    }

    pub async fn check_response(
        &self,
        resp: reqwest::Response,
    ) -> Result<reqwest::Response, SunoError> {
        let status = resp.status();
        if status == 401 {
            return Err(SunoError::AuthExpired);
        }
        if status == 403 {
            let body = resp.text().await.unwrap_or_default();
            return Err(SunoError::Api {
                code: "forbidden",
                message: format!("HTTP 403 Forbidden: {body}"),
            });
        }
        if status == 429 {
            return Err(SunoError::RateLimited);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            // `Token validation failed` = Suno's server-side staleness threshold
            // (~30 min) firing despite a still-valid JWT exp claim. Treat as
            // AuthExpired so `with_auth_retry` refreshes via the Clerk cookie.
            if body.contains("Token validation failed") {
                return Err(SunoError::AuthExpired);
            }
            if body.contains("'loc': ['body', 'params'")
                || body.contains("\"loc\": [\"body\", \"params\"")
            {
                return Err(SunoError::Api {
                    code: "schema_drift",
                    message: format!(
                        "HTTP {status}: Suno's request schema has changed — suno-client needs an update. Body: {body}"
                    ),
                });
            }
            return Err(SunoError::Api {
                code: "api_error",
                message: format!("HTTP {status}: {body}"),
            });
        }
        Ok(resp)
    }
}
