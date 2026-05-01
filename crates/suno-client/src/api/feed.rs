use super::SunoClient;
use super::types::{FeedFilters, FeedResponse, FeedV3Request};
use crate::error::SunoError;

impl SunoClient {
    /// List songs using feed/v3 with optional search and filters.
    pub async fn feed(&self, page: u32) -> Result<FeedResponse, SunoError> {
        let req = FeedV3Request {
            cursor: if page > 0 {
                Some(page.to_string())
            } else {
                None
            },
            limit: Some(20),
            filters: None,
        };
        let resp = self.post("/api/feed/v3").json(&req).send().await?;
        let resp = self.check_response(resp).await?;
        Ok(resp.json().await?)
    }

    /// Search songs using feed/v3 native searchText filter.
    pub async fn search(&self, query: &str) -> Result<FeedResponse, SunoError> {
        let req = FeedV3Request {
            cursor: None,
            limit: Some(50),
            filters: Some(FeedFilters {
                search_text: Some(query.to_string()),
                trashed: Some("False".to_string()),
                full_song: None,
                stem: None,
            }),
        };
        let resp = self.post("/api/feed/v3").json(&req).send().await?;
        let resp = self.check_response(resp).await?;
        Ok(resp.json().await?)
    }
}
