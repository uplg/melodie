use super::SunoClient;
use super::types::BillingInfo;
use crate::error::SunoError;

impl SunoClient {
    pub async fn billing_info(&self) -> Result<BillingInfo, SunoError> {
        let resp = self.get("/api/billing/info/").send().await?;
        let resp = self.check_response(resp).await?;
        Ok(resp.json().await?)
    }
}
