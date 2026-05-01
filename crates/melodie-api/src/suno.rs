//! Suno bridge — owns the single shared [`SunoClient`] and persists its auth
//! state in SQLite so a process restart picks up where we left off.
//!
//! The bridge has three states:
//!
//! 1. **Empty** — no auth in DB yet. `current()` returns `None` and any
//!    `/api/songs` write must 503 with `suno_auth`. Operator runs
//!    `POST /api/admin/suno-auth` with a fresh Clerk cookie.
//! 2. **Loaded** — `current()` returns an `Arc<SunoClient>` ready to use. The
//!    underlying client transparently refreshes the JWT mid-request via the
//!    Clerk cookie (see `SunoClient::with_auth_retry`).
//! 3. **Stale** — last health check failed. The `last_status` column reads
//!    `expired`. The bridge still hands out `current()` (for read paths like
//!    listing past generations), but generation handlers should refuse early.
//!    *Health-check loop lands in the next iteration, alongside Telegram.*

use std::sync::Arc;

use sqlx::SqlitePool;
use suno_client::{AuthState, SunoClient, SunoError, clerk_token_exchange};
use tokio::sync::RwLock;

pub struct SunoBridge {
    pool: SqlitePool,
    client: RwLock<Option<Arc<SunoClient>>>,
}

impl SunoBridge {
    /// Build a bridge and try to rehydrate from `suno_session`. A failure to
    /// rehydrate is logged at WARN level but does not fail startup — the
    /// operator can re-up via `/api/admin/suno-auth`.
    pub async fn from_db(pool: SqlitePool) -> Self {
        let bridge = Self {
            pool,
            client: RwLock::new(None),
        };
        match bridge.load_from_db().await {
            Ok(true) => tracing::info!("Suno session rehydrated from DB"),
            Ok(false) => tracing::warn!(
                "No usable Suno session in DB — POST /api/admin/suno-auth with a Clerk cookie before any /api/songs request"
            ),
            Err(e) => tracing::warn!(error = %e, "Failed to rehydrate Suno session from DB"),
        }
        bridge
    }

    /// Returns true if a client was loaded, false if the DB has nothing usable.
    async fn load_from_db(&self) -> Result<bool, SunoError> {
        let row = melodie_db::suno_session::load(&self.pool)
            .await
            .map_err(|e| SunoError::Config(format!("DB read suno_session: {e}")))?;

        // We need at least one of: a non-expired JWT, or (cookie + session_id)
        // for a refresh.
        let has_jwt = row.jwt.is_some();
        let can_refresh = row.clerk_cookie.is_some() && row.session_id.is_some();
        if !has_jwt && !can_refresh {
            return Ok(false);
        }

        let auth = AuthState {
            jwt: row.jwt,
            session_id: row.session_id,
            device_id: row.device_id,
            clerk_client_cookie: row.clerk_cookie,
        };
        let client = SunoClient::new_with_refresh(auth).await?;
        // The client may have refreshed the JWT during construction — persist
        // the snapshot so the next process start uses the fresh token.
        self.persist(&client.auth_snapshot()).await?;
        let mut slot = self.client.write().await;
        *slot = Some(Arc::new(client));
        Ok(true)
    }

    /// Snapshot of the currently active client, if any. Cheap (clones an Arc).
    pub async fn current(&self) -> Option<Arc<SunoClient>> {
        self.client.read().await.clone()
    }

    /// Persist the in-memory auth state to SQLite. Called by the health-check
    /// loop on each `Ok` tick so a refreshed JWT survives a restart.
    pub async fn checkpoint(&self) -> Result<(), SunoError> {
        let snapshot = match self.client.read().await.as_ref() {
            Some(client) => client.auth_snapshot(),
            None => return Ok(()),
        };
        self.persist(&snapshot).await
    }

    /// Replace the current Suno auth with a freshly-pasted Clerk cookie.
    /// Performs a Clerk token exchange, persists the resulting state, and
    /// swaps the in-memory client. Returns the upstream error verbatim if the
    /// cookie is bogus — callers map it to a 5xx response.
    pub async fn replace_auth(&self, cookie: String) -> Result<(), SunoError> {
        let cookie = cookie.trim().to_string();
        if cookie.is_empty() {
            return Err(SunoError::Config("clerk_cookie is empty".into()));
        }
        let http = reqwest::Client::new();
        let (session_id, jwt) = clerk_token_exchange(&http, &cookie).await?;
        let auth = AuthState {
            jwt: Some(jwt),
            session_id: Some(session_id),
            device_id: None,
            clerk_client_cookie: Some(cookie),
        };
        let client = SunoClient::new_with_refresh(auth).await?;
        self.persist(&client.auth_snapshot()).await?;
        let _ = melodie_db::suno_session::set_status(&self.pool, "ok").await;
        let mut slot = self.client.write().await;
        *slot = Some(Arc::new(client));
        tracing::info!("Suno session replaced via /api/admin/suno-auth");
        Ok(())
    }

    async fn persist(&self, auth: &AuthState) -> Result<(), SunoError> {
        melodie_db::suno_session::save(
            &self.pool,
            auth.jwt.as_deref(),
            auth.session_id.as_deref(),
            auth.device_id.as_deref(),
            auth.clerk_client_cookie.as_deref(),
        )
        .await
        .map_err(|e| SunoError::Config(format!("DB save suno_session: {e}")))?;
        Ok(())
    }
}
