//! Vendored from `paperfoot/suno-cli` `src/errors.rs`. Renamed `CliError` →
//! `SunoError` since this crate is no longer a CLI. CLI-only variants
//! (`Update`) and the `exit_code()` mapping were dropped.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SunoError {
    #[error("API error: {message}")]
    Api { code: &'static str, message: String },

    #[error("Authentication missing — provide a Clerk cookie")]
    AuthMissing,

    #[error("JWT expired and could not be refreshed")]
    AuthExpired,

    #[error("Rate limited by Suno — wait and retry")]
    RateLimited,

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Download failed: {0}")]
    Download(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error(transparent)]
    Http(#[from] reqwest::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl SunoError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Api { code, .. } => code,
            Self::AuthMissing => "auth_missing",
            Self::AuthExpired => "auth_expired",
            Self::RateLimited => "rate_limited",
            Self::Config(_) => "config_error",
            Self::GenerationFailed(_) => "generation_failed",
            Self::Download(_) => "download_error",
            Self::NotFound(_) => "not_found",
            Self::Http(_) => "http_error",
            Self::Io(_) => "io_error",
            Self::Json(_) => "json_error",
        }
    }

    /// True if the error is recoverable by re-authing (refreshing the JWT
    /// from the Clerk cookie).
    pub fn is_auth(&self) -> bool {
        matches!(self, Self::AuthMissing | Self::AuthExpired)
    }
}
