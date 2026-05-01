//! Suno HTTP client, vendored from `paperfoot/suno-cli` (MIT). See
//! [`UPSTREAM.md`](../UPSTREAM.md) for the source SHA and what was changed.
//!
//! ## Usage sketch
//!
//! ```no_run
//! # async fn run() -> Result<(), suno_client::SunoError> {
//! use suno_client::{AuthState, SunoClient, clerk_token_exchange};
//!
//! let http = reqwest::Client::new();
//! let cookie = std::env::var("SUNO_CLERK_COOKIE").unwrap();
//! let (session_id, jwt) = clerk_token_exchange(&http, &cookie).await?;
//!
//! let auth = AuthState {
//!     jwt: Some(jwt),
//!     session_id: Some(session_id),
//!     clerk_client_cookie: Some(cookie),
//!     device_id: None,
//! };
//! let suno = SunoClient::new_with_refresh(auth).await?;
//! let info = suno.billing_info().await?;
//! println!("{} credits left", info.total_credits_left);
//! # Ok(()) }
//! ```

pub mod api;
pub mod auth;
pub mod error;

#[cfg(feature = "captcha")]
pub mod captcha;

pub use api::SunoClient;
pub use api::types;
pub use auth::{AuthState, browser_token, clerk_refresh_jwt, clerk_token_exchange};
pub use error::SunoError;
