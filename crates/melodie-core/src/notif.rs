use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum NotifError {
    #[error("notifier transport error: {0}")]
    Transport(String),
}

#[async_trait]
pub trait Notifier: Send + Sync {
    async fn alert(&self, message: &str) -> Result<(), NotifError>;
}
