use crate::ids::{SongId, UserId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Admin,
    Member,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: UserId,
    pub email: String,
    pub display_name: String,
    pub role: Role,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SongMode {
    Custom,
    Describe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SongStatus {
    Pending,
    Generating,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Song {
    pub id: SongId,
    pub owner_id: UserId,
    pub mode: SongMode,
    pub title: Option<String>,
    pub tags: Option<String>,
    pub exclude_tags: Option<String>,
    pub lyrics: Option<String>,
    pub prompt: Option<String>,
    pub model: String,
    pub status: SongStatus,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub clips: Vec<Clip>,
}

/// `status` is the raw Suno status string (`submitted` | `queued` | `streaming`
/// | `complete` | `error`). We keep it as a string rather than an enum so the
/// upstream is free to add new states without breaking deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub id: String,
    pub song_id: SongId,
    pub variant_index: i32,
    pub status: String,
    pub duration_s: Option<f64>,
    pub image_url: Option<String>,
}
