use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

// Some variants are wired in later phases (admin endpoints, generation
// quota); keep them defined now to avoid churn on the IntoResponse impl.
#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not authenticated")]
    Unauthorized,

    #[error("forbidden")]
    Forbidden,

    #[error("not found")]
    NotFound,

    #[error("invalid input: {0}")]
    BadRequest(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("rate limited")]
    TooManyRequests,

    #[error(transparent)]
    Db(#[from] melodie_db::DbError),

    #[error(transparent)]
    Suno(#[from] suno_client::SunoError),

    #[error(transparent)]
    Session(#[from] tower_sessions::session::Error),

    #[error("internal: {0}")]
    Internal(String),
}

impl ApiError {
    fn status_and_code(&self) -> (StatusCode, &'static str) {
        match self {
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            Self::Forbidden => (StatusCode::FORBIDDEN, "forbidden"),
            Self::NotFound => (StatusCode::NOT_FOUND, "not_found"),
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            Self::Conflict(_) => (StatusCode::CONFLICT, "conflict"),
            Self::TooManyRequests => (StatusCode::TOO_MANY_REQUESTS, "rate_limited"),
            Self::Db(_) | Self::Session(_) | Self::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal")
            }
            Self::Suno(e) if e.is_auth() => (StatusCode::SERVICE_UNAVAILABLE, "suno_auth"),
            Self::Suno(_) => (StatusCode::BAD_GATEWAY, "suno_upstream"),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = self.status_and_code();
        if status.is_server_error() {
            tracing::error!(error = %self, "request failed");
        } else {
            tracing::debug!(error = %self, status = %status, "request failed");
        }
        let body = Json(json!({ "error": { "code": code, "message": self.to_string() } }));
        (status, body).into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
