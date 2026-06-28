//! Engine error type.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("config error: {0}")]
    Config(String),

    /// Placeholder for code paths that are scaffolded but not yet ported.
    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, EngineError>;
