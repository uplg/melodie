//! End-to-end generation pipeline: lyrics + style → token grid. **PHASE 3.**
//!
//! Ported from `pipelines/music_generation.py` `preprocess`. Tokenizes the style
//! tags and lyrics with the Llama-3 tokenizer and builds the 9-channel token grid
//! the LM consumes (text in the last channel, audio channels empty, a single MuQ
//! slot between tags and lyrics).

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::config::GenConfig;
use crate::{EngineError, Result};

/// The prompt tensors the LM consumes.
pub struct Prompt {
    pub tokens: Tensor, // [1, prompt_len, 9] i64
    pub mask: Tensor,   // [1, prompt_len, 9] i64
    pub muq_idx: usize, // position of the MuQ slot (= len(tags_ids))
}

/// Load the Llama-3 tokenizer from a `tokenizer.json`.
pub fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| EngineError::Tokenizer(e.to_string()))
}

fn encode(tok: &Tokenizer, s: &str, gcfg: &GenConfig) -> Result<Vec<i64>> {
    let mut ids: Vec<i64> = tok
        .encode(s, false)
        .map_err(|e| EngineError::Tokenizer(e.to_string()))?
        .get_ids()
        .iter()
        .map(|&x| x as i64)
        .collect();
    if ids.first() != Some(&(gcfg.text_bos_id as i64)) {
        ids.insert(0, gcfg.text_bos_id as i64);
    }
    if ids.last() != Some(&(gcfg.text_eos_id as i64)) {
        ids.push(gcfg.text_eos_id as i64);
    }
    Ok(ids)
}

/// Build the prompt from `tags` (style, e.g. "french chanson, male vocal") and `lyrics`.
/// Mirrors `HeartMuLaGenPipeline.preprocess` (music_generation.py:66-147).
pub fn preprocess(tok: &Tokenizer, gcfg: &GenConfig, tags: &str, lyrics: &str, dev: &Device) -> Result<Prompt> {
    let mut tags_s = tags.to_lowercase();
    if !tags_s.starts_with("<tag>") {
        tags_s = format!("<tag>{tags_s}");
    }
    if !tags_s.ends_with("</tag>") {
        tags_s = format!("{tags_s}</tag>");
    }
    let tags_ids = encode(tok, &tags_s, gcfg)?;
    let muq_idx = tags_ids.len();
    let lyrics_ids = encode(tok, &lyrics.to_lowercase(), gcfg)?;

    // text column = [tags_ids, 0 (MuQ slot), lyrics_ids]
    let mut text_col = tags_ids;
    text_col.push(gcfg.empty_id as i64);
    text_col.extend_from_slice(&lyrics_ids);
    let plen = text_col.len();

    let ncb1 = 9; // audio_num_codebooks + 1
    let mut grid = vec![0i64; plen * ncb1];
    let mut mvec = vec![0i64; plen * ncb1];
    for (p, &tk) in text_col.iter().enumerate() {
        grid[p * ncb1 + (ncb1 - 1)] = tk; // text in last channel
        mvec[p * ncb1 + (ncb1 - 1)] = 1; // only text channel active
    }
    Ok(Prompt {
        tokens: Tensor::from_vec(grid, (1, plen, ncb1), dev)?,
        mask: Tensor::from_vec(mvec, (1, plen, ncb1), dev)?,
        muq_idx,
    })
}
