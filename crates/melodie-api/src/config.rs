use std::net::SocketAddr;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub bind: SocketAddr,
    pub database_url: String,
    pub bootstrap_invite: Option<String>,
    pub cookie_secure: bool,
    /// Optional bridge to homie's loopback push server. When both URL and
    /// token are set, the `POST /api/clips/{id}/push-to-live` endpoint is
    /// enabled and the UI shows a "Push to live" button on each clip.
    pub homie_push: Option<HomiePushConfig>,
}

#[derive(Debug, Clone)]
pub struct HomiePushConfig {
    /// Full URL of homie's `POST /push` endpoint, typically
    /// `http://127.0.0.1:7878/push`.
    pub url: String,
    pub token: String,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let bind = std::env::var("MELODIE_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .parse()?;

        let database_url = std::env::var("MELODIE_DATABASE_URL")
            .unwrap_or_else(|_| "sqlite://./data/melodie.db?mode=rwc".into());

        let bootstrap_invite = std::env::var("MELODIE_BOOTSTRAP_INVITE").ok();

        // In dev (HTTP) we can't set Secure on the cookie or browsers drop it.
        // Default to off; flip on via MELODIE_COOKIE_SECURE=1 behind nginx+TLS.
        let cookie_secure = std::env::var("MELODIE_COOKIE_SECURE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let homie_push = HomiePushConfig::from_env();

        Ok(Self {
            bind,
            database_url,
            bootstrap_invite,
            cookie_secure,
            homie_push,
        })
    }
}

impl HomiePushConfig {
    /// Returns `Some(_)` only when `HOMIE_PUSH_TOKEN` is set. The URL defaults
    /// to `http://127.0.0.1:7878/push` (matching homie's default port) — set
    /// `HOMIE_PUSH_URL` to override (e.g. when running homie on a different
    /// host or port).
    fn from_env() -> Option<Self> {
        let token = std::env::var("HOMIE_PUSH_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())?;
        let url = std::env::var("HOMIE_PUSH_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:7878/push".to_string());
        Some(Self { url, token })
    }
}
