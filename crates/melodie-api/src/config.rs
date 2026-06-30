use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub bind: SocketAddr,
    /// Base URL this server is reachable at from *other local services*
    /// (currently: homie fetching clip audio for push-to-live). Defaults to
    /// loopback on `bind`'s port — never derived from a request, since the
    /// client-supplied `Host` header is trivially spoofable.
    pub local_base_url: String,
    pub database_url: String,
    pub bootstrap_invite: Option<String>,
    pub cookie_secure: bool,
    /// Optional bridge to homie's loopback push server. When both URL and
    /// token are set, the `POST /api/clips/{id}/push-to-live` endpoint is
    /// enabled and the UI shows a "Push to live" button on each clip.
    pub homie_push: Option<HomiePushConfig>,
    /// Local HeartMuLa engine settings. `POST /api/songs` always generates
    /// on-device through this engine.
    pub engine: EngineSettings,
}

/// Configuration for the local `melodie-engine` generator.
#[derive(Debug, Clone)]
pub struct EngineSettings {
    /// Checkpoint locations handed to `Engine::load`.
    pub engine_cfg: melodie_engine::EngineConfig,
    /// Directory where generated `.mp3` files are written and served from.
    pub audio_dir: PathBuf,
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
        let bind: SocketAddr = std::env::var("MELODIE_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8080".into())
            .parse()?;

        let local_base_url = std::env::var("MELODIE_LOCAL_URL")
            .ok()
            .map(|v| v.trim().trim_end_matches('/').to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| format!("http://127.0.0.1:{}", bind.port()));

        let database_url = std::env::var("MELODIE_DATABASE_URL")
            .unwrap_or_else(|_| "sqlite://./data/melodie.db?mode=rwc".into());

        let bootstrap_invite = std::env::var("MELODIE_BOOTSTRAP_INVITE").ok();

        // In dev (HTTP) we can't set Secure on the cookie or browsers drop it.
        // Default to off; flip on via MELODIE_COOKIE_SECURE=1 behind nginx+TLS.
        let cookie_secure = std::env::var("MELODIE_COOKIE_SECURE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let homie_push = HomiePushConfig::from_env();

        let engine = EngineSettings::from_env();

        Ok(Self {
            bind,
            local_base_url,
            database_url,
            bootstrap_invite,
            cookie_secure,
            homie_push,
            engine,
        })
    }
}

impl EngineSettings {
    fn from_env() -> Self {
        let lm_dir = std::env::var("MELODIE_LM_DIR")
            .unwrap_or_else(|_| "/Users/leonard/Github/heartlib-mlx/ckpt/HeartMuLa-oss-3B".into());
        let codec_dir = std::env::var("MELODIE_CODEC_DIR")
            .unwrap_or_else(|_| "/Users/leonard/Github/heartlib-mlx/ckpt/HeartCodec-oss".into());
        let tokenizer = std::env::var("MELODIE_TOKENIZER")
            .unwrap_or_else(|_| "/Users/leonard/Github/heartlib-mlx/ckpt/tokenizer.json".into());
        let audio_dir = std::env::var("MELODIE_AUDIO_DIR").unwrap_or_else(|_| "data/audio".into());

        Self {
            engine_cfg: melodie_engine::EngineConfig {
                lm_dir: PathBuf::from(lm_dir),
                codec_dir: PathBuf::from(codec_dir),
                tokenizer_path: PathBuf::from(tokenizer),
            },
            audio_dir: PathBuf::from(audio_dir),
        }
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
