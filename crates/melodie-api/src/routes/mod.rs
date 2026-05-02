use axum::Router;

use crate::state::AppState;

pub mod admin;
pub mod auth;
pub mod clips;
pub mod club;
pub mod songs;

pub fn api_router() -> Router<AppState> {
    Router::new()
        .merge(auth::router())
        .merge(admin::router())
        .merge(songs::router())
        .merge(clips::router())
        .merge(club::router())
}
