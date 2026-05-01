use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString, rand_core::OsRng};
use argon2::Argon2;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use melodie_core::model::{Role, User};
use melodie_db::{invites, users};
use serde::{Deserialize, Serialize};
use tower_sessions::Session;

use crate::error::{ApiError, ApiResult};
use crate::extract::{AuthUser, SESSION_USER_KEY};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/auth/signup", post(signup))
        .route("/auth/login", post(login))
        .route("/auth/logout", post(logout))
        .route("/me", get(me))
}

#[derive(Debug, Deserialize)]
pub struct SignupRequest {
    pub invite_code: String,
    pub email: String,
    pub password: String,
    pub display_name: String,
}

#[derive(Debug, Serialize)]
pub struct UserView {
    pub id: String,
    pub email: String,
    pub display_name: String,
    pub role: String,
}

impl From<&User> for UserView {
    fn from(u: &User) -> Self {
        Self {
            id: u.id.to_string(),
            email: u.email.clone(),
            display_name: u.display_name.clone(),
            role: match u.role {
                Role::Admin => "admin".into(),
                Role::Member => "member".into(),
            },
        }
    }
}

async fn signup(
    State(state): State<AppState>,
    session: Session,
    Json(req): Json<SignupRequest>,
) -> ApiResult<impl IntoResponse> {
    validate_email(&req.email)?;
    validate_password(&req.password)?;
    validate_display_name(&req.display_name)?;

    let invite = invites::find(&state.db, req.invite_code.trim())
        .await?
        .ok_or_else(|| ApiError::BadRequest("invalid invite code".into()))?;
    if invite.used_by.is_some() {
        return Err(ApiError::BadRequest("invite already used".into()));
    }

    if users::find_by_email(&state.db, &req.email).await?.is_some() {
        return Err(ApiError::Conflict("email already registered".into()));
    }

    let password_hash = hash_password(&req.password)?;
    let user = users::create(
        &state.db,
        users::NewUser {
            email: &req.email,
            display_name: &req.display_name,
            password_hash: &password_hash,
            role: invite.role(),
        },
    )
    .await?;

    let consumed = invites::consume(&state.db, &invite.code, user.id).await?;
    if !consumed {
        // Race: someone else consumed it between find() and now. Return 409 so
        // the client knows to retry with a fresh invite.
        return Err(ApiError::Conflict("invite already used".into()));
    }

    session.insert(SESSION_USER_KEY, user.id.to_string()).await?;
    Ok((StatusCode::CREATED, Json(UserView::from(&user))))
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

async fn login(
    State(state): State<AppState>,
    session: Session,
    Json(req): Json<LoginRequest>,
) -> ApiResult<Json<UserView>> {
    let (user, password_hash) = users::find_by_email(&state.db, &req.email)
        .await?
        .ok_or(ApiError::Unauthorized)?;
    verify_password(&req.password, &password_hash)?;
    // Cycle the session id on auth state change (defense in depth against
    // session fixation).
    session.cycle_id().await?;
    session.insert(SESSION_USER_KEY, user.id.to_string()).await?;
    Ok(Json(UserView::from(&user)))
}

async fn logout(session: Session) -> ApiResult<StatusCode> {
    session.flush().await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn me(AuthUser(user): AuthUser) -> Json<UserView> {
    Json(UserView::from(&user))
}

// --- helpers ---

fn validate_email(email: &str) -> ApiResult<()> {
    let trimmed = email.trim();
    if trimmed.len() < 3 || trimmed.len() > 254 || !trimmed.contains('@') {
        return Err(ApiError::BadRequest("invalid email".into()));
    }
    Ok(())
}

fn validate_password(pw: &str) -> ApiResult<()> {
    // 8 char minimum is the floor we care about for a friends-only app — if
    // people insist on weak passwords, that's their call. We do enforce a
    // ceiling because Argon2 has its own input length limit and we don't want
    // to DoS ourselves on huge inputs.
    if pw.len() < 8 {
        return Err(ApiError::BadRequest(
            "password must be at least 8 characters".into(),
        ));
    }
    if pw.len() > 1024 {
        return Err(ApiError::BadRequest("password too long".into()));
    }
    Ok(())
}

fn validate_display_name(name: &str) -> ApiResult<()> {
    let n = name.trim();
    if n.is_empty() || n.len() > 64 {
        return Err(ApiError::BadRequest(
            "display name must be 1-64 characters".into(),
        ));
    }
    Ok(())
}

fn hash_password(pw: &str) -> ApiResult<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(pw.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| ApiError::Internal(format!("argon2: {e}")))
}

fn verify_password(pw: &str, hash: &str) -> ApiResult<()> {
    let parsed = PasswordHash::new(hash).map_err(|_| ApiError::Unauthorized)?;
    Argon2::default()
        .verify_password(pw.as_bytes(), &parsed)
        .map_err(|_| ApiError::Unauthorized)
}
