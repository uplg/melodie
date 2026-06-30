use axum::extract::{FromRef, FromRequestParts};
use axum::http::request::Parts;
use melodie_core::authz::{self, Action, Resource};
use melodie_core::ids::{SongId, UserId};
use melodie_core::model::User;
use tower_sessions::Session;
use uuid::Uuid;

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;

pub const SESSION_USER_KEY: &str = "user_id";

/// Authenticated user, fetched from the session cookie.
#[derive(Debug, Clone)]
pub struct AuthUser(pub User);

impl<S> FromRequestParts<S> for AuthUser
where
    AppState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let session = Session::from_request_parts(parts, state)
            .await
            .map_err(|_| ApiError::Unauthorized)?;
        let raw: Option<String> = session.get(SESSION_USER_KEY).await?;
        let id_str = raw.ok_or(ApiError::Unauthorized)?;
        let id = Uuid::parse_str(&id_str)
            .map(UserId)
            .map_err(|_| ApiError::Unauthorized)?;

        let app = AppState::from_ref(state);
        let user = melodie_db::users::find_by_id(&app.db, id)
            .await?
            .ok_or(ApiError::Unauthorized)?;
        Ok(AuthUser(user))
    }
}

/// Gate `action` on `user`'s access to the song owned by `owner_id`, the
/// check every song/clip handler needs after loading its row. Owners and
/// admins pass; everyone else gets [`ApiError::Forbidden`].
pub fn require_song_access(
    user: &User,
    owner_id: UserId,
    song_id: SongId,
    action: Action,
) -> ApiResult<()> {
    if authz::can(
        user.role,
        user.id,
        action,
        Resource::Song { owner_id, song_id },
    ) {
        Ok(())
    } else {
        Err(ApiError::Forbidden)
    }
}

/// Same as [`AuthUser`] but rejects non-admin users.
#[derive(Debug, Clone)]
pub struct AdminUser(pub User);

impl<S> FromRequestParts<S> for AdminUser
where
    AppState: FromRef<S>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let AuthUser(user) = AuthUser::from_request_parts(parts, state).await?;
        if user.role != melodie_core::model::Role::Admin {
            return Err(ApiError::Forbidden);
        }
        Ok(AdminUser(user))
    }
}
