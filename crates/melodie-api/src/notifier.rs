//! [`Notifier`] implementations.
//!
//! Two flavours:
//! - [`TelegramNotifier`] — POSTs to the Bot API. Keyed by `TELEGRAM_BOT_TOKEN`
//!   + `TELEGRAM_ADMIN_CHAT_ID`.
//! - [`NoopNotifier`] — logs the alert and returns Ok. Used when the env vars
//!   aren't set, so the rest of the system can call `notifier.alert(...)`
//!   unconditionally.

use std::sync::Arc;

use async_trait::async_trait;
use melodie_core::notif::{NotifError, Notifier};
use serde_json::json;

pub struct TelegramNotifier {
    client: reqwest::Client,
    bot_token: String,
    chat_id: String,
}

impl TelegramNotifier {
    pub fn new(bot_token: String, chat_id: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("default reqwest client"),
            bot_token,
            chat_id,
        }
    }
}

#[async_trait]
impl Notifier for TelegramNotifier {
    async fn alert(&self, message: &str) -> Result<(), NotifError> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.bot_token);
        let resp = self
            .client
            .post(&url)
            .json(&json!({ "chat_id": self.chat_id, "text": message }))
            .send()
            .await
            .map_err(|e| NotifError::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(NotifError::Transport(format!("telegram {status}: {body}")));
        }
        Ok(())
    }
}

pub struct NoopNotifier;

#[async_trait]
impl Notifier for NoopNotifier {
    async fn alert(&self, message: &str) -> Result<(), NotifError> {
        tracing::info!(%message, "notifier (noop) alert");
        Ok(())
    }
}

pub fn from_env() -> Arc<dyn Notifier> {
    let token = std::env::var("TELEGRAM_BOT_TOKEN").ok();
    let chat = std::env::var("TELEGRAM_ADMIN_CHAT_ID").ok();
    match (token, chat) {
        (Some(t), Some(c)) if !t.trim().is_empty() && !c.trim().is_empty() => {
            tracing::info!("Telegram notifier configured");
            Arc::new(TelegramNotifier::new(t, c))
        }
        _ => {
            tracing::warn!(
                "Telegram credentials missing — alerts will be logged only. Set TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID to enable."
            );
            Arc::new(NoopNotifier)
        }
    }
}
