use super::SunoClient;
use super::types::{Clip, ConcatRequest};
use crate::error::SunoError;

impl SunoClient {
    pub async fn concat(&self, clip_id: &str) -> Result<Clip, SunoError> {
        let resp = self
            .post("/api/generate/concat/v2/")
            .json(&ConcatRequest {
                clip_id: clip_id.to_string(),
            })
            .send()
            .await?;
        let resp = self.check_response(resp).await?;
        Ok(resp.json().await?)
    }
}
