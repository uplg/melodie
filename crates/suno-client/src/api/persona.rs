use super::SunoClient;
use super::types::{PersonaInfo, PersonaResponse};
use crate::error::SunoError;

impl SunoClient {
    /// Fetch voice persona details.
    /// GET /api/persona/get-persona-paginated/{persona_id}/?page=0
    pub async fn get_persona(&self, persona_id: &str) -> Result<PersonaInfo, SunoError> {
        let resp = self
            .get(&format!(
                "/api/persona/get-persona-paginated/{persona_id}/?page=0"
            ))
            .send()
            .await?;
        let resp = self.check_response(resp).await?;
        let body: PersonaResponse = resp.json().await?;
        Ok(body.persona)
    }
}
