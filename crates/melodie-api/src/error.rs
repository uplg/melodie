use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

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

    #[error("service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error(transparent)]
    Db(#[from] melodie_db::DbError),

    /// Raw sqlx errors from transactions opened directly in a handler (e.g.
    /// `routes/auth.rs::signup`'s invite-claim transaction), as opposed to
    /// `Db`, which wraps errors from melodie-db's query functions.
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),

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
            Self::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable"),
            Self::Db(_) | Self::Sqlx(_) | Self::Session(_) | Self::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal")
            }
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
        // 5xx variants wrap raw driver/library errors (sqlx, argon2, reqwest)
        // that can carry file paths, hostnames, or query fragments — the
        // detailed text goes to the log above, never to the client.
        let message = if status.is_server_error() {
            "internal error".to_string()
        } else {
            self.to_string()
        };
        let body = Json(json!({ "error": { "code": code, "message": message } }));
        (status, body).into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
