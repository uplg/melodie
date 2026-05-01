use std::net::SocketAddr;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub bind: SocketAddr,
    pub database_url: String,
    pub bootstrap_invite: Option<String>,
    pub cookie_secure: bool,
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

        Ok(Self {
            bind,
            database_url,
            bootstrap_invite,
            cookie_secure,
        })
    }
}
