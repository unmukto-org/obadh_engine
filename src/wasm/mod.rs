use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use wasm_bindgen::prelude::*;
use web_sys::Performance;

use crate::autosuggest::{
    repetition_observation_for_raw_token, session_repetition_guard_pool_limit,
    AutosuggestRepetitionHistory,
};
use crate::{
    key_slip_repaired_outputs, roman_repaired_outputs, AutocorrectConfig, AutocorrectDecision,
    AutocorrectEngine, AutosuggestArtifactError, AutosuggestCandidateId, AutosuggestCandidatePrior,
    AutosuggestContext, AutosuggestContextPriorOptions, AutosuggestLm, AutosuggestMetadata,
    AutosuggestOptions, AutosuggestSource, CandidateFeatures, CorrectionCandidate,
    CorrectionSource, FstCandidate, FstLexicon, FstLoanwordMatch, FstRepairedBaseline,
    FstSuggestOptions, Lexicon, LexiconEntry, LexiconStats, LoanwordLexicon, LoanwordSearchOptions,
    ObadhEngine, PersonalAutosuggest, PersonalAutosuggestConfig, PersonalAutosuggestContext,
    PersonalAutosuggestSuggestion, PersonalAutosuggestTextSuggestion, RomanRepairOptions,
    RomanRepairedOutput, DEFAULT_AUTOSUGGEST_CANDIDATES, DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
};

const AUTOCORRECT_RERANK_POOL_SIZE: usize = 512;
const AUTOCORRECT_RESPONSE_CANDIDATES: usize = 24;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// Helper function to get the performance object
fn get_performance() -> Option<Performance> {
    web_sys::window()?.performance()
}

// Helper function to measure time
fn now() -> f64 {
    get_performance().map_or(0.0, |performance| performance.now())
}

fn reserve_vec_to<T>(values: &mut Vec<T>, capacity: usize) {
    if values.capacity() < capacity {
        // Vec reservations are additional to len, not to existing capacity.
        values.reserve_exact(capacity - values.len());
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::reserve_vec_to;

    #[test]
    fn growing_scratch_reserves_the_requested_total_capacity() {
        for len in [0, 4, 16] {
            let mut values = Vec::with_capacity(16);
            values.extend(0..len);
            reserve_vec_to(&mut values, 24);
            assert!(values.capacity() >= 24);
            assert_eq!(values, (0..len).collect::<Vec<_>>());
            let pointer = values.as_ptr();
            values.extend(len..24);
            assert_eq!(values.as_ptr(), pointer);
        }
    }

    #[test]
    fn sufficient_scratch_capacity_is_reused() {
        let mut values = Vec::with_capacity(24);
        values.extend(0..4);
        let pointer = values.as_ptr();
        reserve_vec_to(&mut values, 16);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values, vec![0, 1, 2, 3]);
    }
}

/// Output options for the transliteration
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct TransliterationOptions {
    /// Output performance metrics
    pub debug: bool,
    /// Include token analysis in output
    pub verbose: bool,
}

#[wasm_bindgen]
impl TransliterationOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            debug: false,
            verbose: false,
        }
    }
}

impl Default for TransliterationOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for the transliteration process
#[derive(Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_ms: f64,
    pub sanitize_ms: f64,
    pub tokenize_ms: f64,
    pub transliterate_ms: f64,
}

/// Detailed token analysis for verbose output
#[derive(Serialize, Deserialize)]
pub struct TokenAnalysis {
    pub content: String,
    pub position: usize,
    pub r#type: String, // "type" is a reserved keyword in JS
    pub transliterated: Option<String>,
    pub phonetic_units: Option<Vec<PhoneticUnitInfo>>,
}

/// Information about a phonetic unit
#[derive(Serialize, Deserialize)]
pub struct PhoneticUnitInfo {
    pub text: String,
    pub position: usize,
    pub r#type: String, // "type" is a reserved keyword in JS
}

/// Complete transliteration result
#[derive(Serialize, Deserialize)]
pub struct TransliterationResult {
    pub input: String,
    pub output: String,
    pub performance: Option<PerformanceMetrics>,
    pub token_analysis: Option<Vec<TokenAnalysis>>,
}

#[derive(Clone, Copy, Serialize)]
pub struct AutocorrectLexiconStats {
    pub artifact_kind: &'static str,
    pub entries: usize,
    pub loanword_entries: usize,
    pub trie_nodes: usize,
    pub trie_edges: usize,
    pub skeleton_keys: usize,
    pub unique_skeletons: usize,
    pub skeleton_delete_keys: usize,
}

#[derive(Serialize)]
pub struct AutocorrectCandidateInfo {
    pub text: String,
    pub source: &'static str,
    pub edit_cost: u16,
    pub frequency: u64,
    pub score: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roman_repair: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roman_repair_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roman_repair_cost: Option<u16>,
    pub features: [i16; crate::AUTOCORRECT_FEATURE_DIM],
}

#[derive(Serialize)]
pub struct AutocorrectLabResult {
    pub roman_input: String,
    pub obadh_output: String,
    pub input: String,
    pub elapsed_ms: f64,
    pub replacement: Option<AutocorrectCandidateInfo>,
    pub candidates: Vec<AutocorrectCandidateInfo>,
    pub lexicon: AutocorrectLexiconStats,
}

#[derive(Clone, Copy, Serialize)]
pub struct AutosuggestStats {
    pub artifact_kind: &'static str,
    pub backend: &'static str,
    pub artifact_fingerprint_high: u32,
    pub artifact_fingerprint_low: u32,
    pub vocab_size: usize,
    pub vocab_fingerprint: u32,
    pub unigram_count: usize,
    pub bigram_rows: usize,
    pub trigram_rows: usize,
    pub fourgram_rows: usize,
    pub candidate_record_len: usize,
    pub model_bytes: usize,
    pub load_elapsed_ms: f64,
    pub last_elapsed_ms: Option<f64>,
}

#[derive(Serialize)]
pub struct AutosuggestCandidateInfo {
    pub text: String,
    pub token_id: u32,
    pub source: &'static str,
    pub count: u32,
    pub score: i32,
}

#[derive(Serialize)]
pub struct AutosuggestLabResult {
    pub context: Vec<String>,
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
    pub elapsed_ms: f64,
    pub model: AutosuggestStats,
    pub candidates: Vec<AutosuggestCandidateInfo>,
}

#[derive(Serialize)]
pub struct AutosuggestContextPriorInfo {
    pub candidate_index: usize,
    pub text: String,
    pub token_id: u32,
    pub prior_rank: usize,
    pub source: &'static str,
    pub count: u32,
    pub score: i32,
}

#[derive(Serialize)]
pub struct AutosuggestContextPriorResult {
    pub context: Vec<String>,
    pub context_token_count: usize,
    pub matched_context_token_count: usize,
    pub prior_candidate_count: usize,
    pub matched_candidate_count: usize,
    pub elapsed_ms: f64,
    pub model: AutosuggestStats,
    pub priors: Vec<AutosuggestContextPriorInfo>,
}

/// ObdahWasm is the main WASM interface to the Obadh engine
#[wasm_bindgen]
pub struct ObadhaWasm {
    engine: ObadhEngine,
}

#[wasm_bindgen]
impl ObadhaWasm {
    /// Create a new instance of the Obadh engine
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            engine: ObadhEngine::new(),
        }
    }

    /// Transliterate text from Roman to Bengali
    #[wasm_bindgen]
    pub fn transliterate(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        self.engine.transliterate(text)
    }

    /// Exact lookup for intact ASCII emoticons; does not alter transliteration.
    #[wasm_bindgen(js_name = emoticonEmoji)]
    pub fn emoticon_emoji(&self, input: &str) -> Option<String> {
        crate::emoticon_emoji(input).map(str::to_owned)
    }

    /// Transliterate text after dropping unsupported characters
    #[wasm_bindgen]
    pub fn transliterate_lenient(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        self.engine.transliterate_lenient(text)
    }

    /// Transliterate with options for debug/verbose output
    #[wasm_bindgen]
    pub fn transliterate_with_options(
        &self,
        text: &str,
        options_js: JsValue,
    ) -> Result<JsValue, JsValue> {
        if text.is_empty() {
            let empty_result = TransliterationResult {
                input: String::new(),
                output: String::new(),
                performance: None,
                token_analysis: None,
            };
            return Ok(to_value(&empty_result)?);
        }

        // Convert JS options to Rust struct
        let options: TransliterationOptions = match from_value(options_js) {
            Ok(opts) => opts,
            Err(e) => {
                return Err(JsValue::from_str(&format!(
                    "Failed to parse options: {}",
                    e
                )));
            }
        };

        // Create result object
        let mut result = TransliterationResult {
            input: text.to_string(),
            output: String::new(),
            performance: None,
            token_analysis: None,
        };

        // For debug/verbose modes, measure performance
        if options.debug || options.verbose {
            // Measure sanitization performance
            let sanitize_start = now();
            let sanitized = self.engine.sanitize(text);
            let sanitize_duration = now() - sanitize_start;
            let Ok(sanitized) = sanitized else {
                result.output = text.to_string();
                result.performance = Some(PerformanceMetrics {
                    sanitize_ms: sanitize_duration,
                    tokenize_ms: 0.0,
                    transliterate_ms: 0.0,
                    total_ms: sanitize_duration,
                });
                if options.verbose {
                    result.token_analysis = Some(Vec::new());
                }
                return Ok(to_value(&result)?);
            };

            // Measure tokenization performance
            let tokenize_start = now();
            let tokens = self.engine.tokenize(&sanitized);
            let tokenize_duration = now() - tokenize_start;

            // Measure transliteration performance
            let transliterate_start = now();
            result.output = self.engine.transliterate_tokens(&tokens);
            let transliterate_duration = now() - transliterate_start;

            // Calculate total duration
            let total_duration = sanitize_duration + tokenize_duration + transliterate_duration;

            // Populate performance metrics with actual measurements
            result.performance = Some(PerformanceMetrics {
                sanitize_ms: sanitize_duration,
                tokenize_ms: tokenize_duration,
                transliterate_ms: transliterate_duration,
                total_ms: total_duration,
            });

            // Add token analysis if verbose is enabled
            if options.verbose {
                let mut token_analysis = Vec::new();

                for (index, token) in tokens.iter().enumerate() {
                    let mut analysis = TokenAnalysis {
                        content: token.content.clone(),
                        position: token.position,
                        r#type: format!("{:?}", token.token_type),
                        transliterated: self.engine.transliterate_token_at(&tokens, index),
                        phonetic_units: None,
                    };

                    // Add phonetic units for Word tokens
                    if let crate::TokenType::Word = token.token_type {
                        let phonetic_units = self.engine.tokenize_phonetic(&token.content);

                        if !phonetic_units.is_empty() {
                            let mut units_info = Vec::new();

                            for unit in phonetic_units {
                                units_info.push(PhoneticUnitInfo {
                                    text: unit.text.clone(),
                                    position: unit.position,
                                    r#type: format!("{:?}", unit.unit_type),
                                });
                            }

                            analysis.phonetic_units = Some(units_info);
                        }
                    }

                    token_analysis.push(analysis);
                }

                result.token_analysis = Some(token_analysis);
            }
        } else {
            // Simple transliteration without metrics
            result.output = self.engine.transliterate(text);
        }

        // Convert to JsValue and return
        match to_value(&result) {
            Ok(val) => Ok(val),
            Err(e) => Err(JsValue::from_str(&format!(
                "Failed to serialize result: {}",
                e
            ))),
        }
    }

    /// Get version information
    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

impl Default for ObadhaWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
pub struct ObadhAutosuggestWasm {
    lm: AutosuggestLm<Vec<u8>>,
    personal: PersonalAutosuggest,
    context: AutosuggestContext,
    personal_context: PersonalAutosuggestContext,
    repetition_history: AutosuggestRepetitionHistory,
    personal_scratch: Vec<PersonalAutosuggestSuggestion>,
    personal_text_scratch: Vec<PersonalAutosuggestTextSuggestion>,
    model_id_scratch: Vec<AutosuggestCandidateId>,
    candidate_id_scratch: Vec<AutosuggestCandidateId>,
    context_prior_scratch: Vec<AutosuggestCandidatePrior>,
    stats: AutosuggestStats,
}

#[wasm_bindgen]
impl ObadhAutosuggestWasm {
    #[wasm_bindgen(js_name = fromNgramArtifact)]
    pub fn from_ngram_artifact(bytes: &[u8]) -> Result<ObadhAutosuggestWasm, JsValue> {
        let start = now();
        let bytes = bytes.to_vec();
        let lm = AutosuggestLm::from_bytes(bytes)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let model = lm.model_info();
        let artifact_fingerprint = lm.artifact_fingerprint();
        let stats = AutosuggestStats {
            artifact_kind: model.artifact_kind,
            backend: "wasm/rust",
            artifact_fingerprint_high: (artifact_fingerprint >> 32) as u32,
            artifact_fingerprint_low: artifact_fingerprint as u32,
            vocab_size: model.vocab_size,
            vocab_fingerprint: model.vocab_fingerprint,
            unigram_count: model.unigram_count,
            bigram_rows: model.bigram_rows,
            trigram_rows: model.trigram_rows,
            fourgram_rows: model.fourgram_rows,
            candidate_record_len: model.candidate_record_len,
            model_bytes: model.artifact_bytes,
            load_elapsed_ms: now() - start,
            last_elapsed_ms: None,
        };
        Ok(Self {
            lm,
            personal: PersonalAutosuggest::new(PersonalAutosuggestConfig::default()),
            context: AutosuggestContext::new(),
            personal_context: PersonalAutosuggestContext::new(),
            repetition_history: AutosuggestRepetitionHistory::new(),
            personal_scratch: Vec::with_capacity(DEFAULT_AUTOSUGGEST_CANDIDATES),
            personal_text_scratch: Vec::with_capacity(DEFAULT_AUTOSUGGEST_CANDIDATES),
            model_id_scratch: Vec::with_capacity(DEFAULT_AUTOSUGGEST_CANDIDATES),
            candidate_id_scratch: Vec::with_capacity(DEFAULT_AUTOSUGGEST_CANDIDATES),
            context_prior_scratch: Vec::with_capacity(DEFAULT_AUTOSUGGEST_CANDIDATES),
            stats,
        })
    }

    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        to_value(&self.stats).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen]
    pub fn suggest(
        &mut self,
        context_tokens_js: JsValue,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let start = now();
        let context_tokens: Vec<String> = from_value(context_tokens_js)
            .map_err(|error| JsValue::from_str(&format!("invalid context tokens: {error}")))?;
        let mut context = AutosuggestContext::new();
        let mut personal_context = PersonalAutosuggestContext::new();
        let mut repetition_history = AutosuggestRepetitionHistory::new();
        for token in &context_tokens {
            match self
                .lm
                .token_id(token)
                .map_err(|error| JsValue::from_str(&error.to_string()))?
            {
                Some(token_id) => {
                    context.push_token_id(Some(token_id));
                    personal_context.push_word_token_id(token_id);
                    repetition_history.observe_resolved_token(Some(token_id), true, false);
                }
                None => {
                    context.push_unknown();
                    personal_context.push_text(token);
                    repetition_history.observe_resolved_token(None, true, false);
                }
            }
        }
        let options = AutosuggestOptions {
            max_candidates: limit.max(1),
        };
        let pool_limit = session_repetition_guard_pool_limit(options.max_candidates);
        reserve_vec_to(&mut self.personal_scratch, pool_limit);
        reserve_vec_to(&mut self.model_id_scratch, pool_limit);
        reserve_vec_to(&mut self.candidate_id_scratch, options.max_candidates);
        let metadata = self
            .personal
            .suggest_ids_with_lm_for_personal_context_into(
                &self.lm,
                context,
                personal_context,
                repetition_history.recent_token_ids(),
                options,
                &mut self.personal_scratch,
                &mut self.model_id_scratch,
                &mut self.candidate_id_scratch,
            )
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        self.personal.suggest_text_for_personal_context_into(
            personal_context,
            options.max_candidates,
            &mut self.personal_text_scratch,
        );
        let elapsed_ms = now() - start;
        let mut stats = self.stats;
        stats.last_elapsed_ms = Some(elapsed_ms);
        let lab_result = AutosuggestLabResult {
            context: context_tokens,
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            elapsed_ms,
            model: stats,
            candidates: autosuggest_candidate_infos_with_text(
                &self.lm,
                &self.personal,
                &self.personal_text_scratch,
                &self.candidate_id_scratch,
                options.max_candidates,
            )
            .map_err(|error| JsValue::from_str(&error.to_string()))?,
        };
        to_value(&lab_result).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen(js_name = contextPriors)]
    pub fn context_priors(
        &mut self,
        context_tokens_js: JsValue,
        candidate_texts_js: JsValue,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let start = now();
        let context_tokens: Vec<String> = from_value(context_tokens_js)
            .map_err(|error| JsValue::from_str(&format!("invalid context tokens: {error}")))?;
        let candidate_texts: Vec<String> = from_value(candidate_texts_js)
            .map_err(|error| JsValue::from_str(&format!("invalid candidate texts: {error}")))?;
        let mut context = AutosuggestContext::new();
        for token in &context_tokens {
            self.lm
                .push_context_token(&mut context, token)
                .map_err(|error| JsValue::from_str(&error.to_string()))?;
        }

        let candidate_text_refs = candidate_texts
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let metadata = self
            .lm
            .context_priors_for_candidate_texts_into(
                context,
                &candidate_text_refs,
                AutosuggestContextPriorOptions {
                    max_prior_candidates: limit.max(1),
                },
                &mut self.model_id_scratch,
                &mut self.context_prior_scratch,
            )
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let elapsed_ms = now() - start;
        let mut stats = self.stats;
        stats.last_elapsed_ms = Some(elapsed_ms);
        let priors = self
            .context_prior_scratch
            .iter()
            .map(|prior| autosuggest_context_prior_info(&self.lm, &candidate_texts, *prior))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let lab_result = AutosuggestContextPriorResult {
            context: context_tokens,
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            prior_candidate_count: metadata.prior_candidate_count,
            matched_candidate_count: metadata.matched_candidate_count,
            elapsed_ms,
            model: stats,
            priors,
        };
        to_value(&lab_result).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen(js_name = clearSession)]
    pub fn clear_session(&mut self) {
        self.context.clear();
        self.personal_context.clear();
        self.repetition_history.clear();
    }

    #[wasm_bindgen(js_name = clearPersonal)]
    pub fn clear_personal(&mut self) {
        self.personal.clear();
    }

    #[wasm_bindgen(js_name = personalSnapshotLen)]
    pub fn personal_snapshot_len(&self) -> usize {
        self.personal.compact_snapshot_len()
    }

    #[wasm_bindgen(js_name = exportPersonalSnapshot)]
    pub fn export_personal_snapshot(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.personal.compact_snapshot_len());
        self.personal
            .write_compact_bytes_with_model_fingerprint_into(
                &mut bytes,
                self.lm.vocab_fingerprint(),
            );
        bytes
    }

    #[wasm_bindgen(js_name = importPersonalSnapshot)]
    pub fn import_personal_snapshot(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.personal = PersonalAutosuggest::from_compact_bytes_for_model(
            PersonalAutosuggestConfig::default(),
            bytes,
            self.lm.vocab_size(),
            self.lm.vocab_fingerprint(),
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(())
    }

    #[wasm_bindgen(js_name = resolveTokenId)]
    pub fn resolve_token_id(&self, token: &str) -> Result<i32, JsValue> {
        let token_id = self
            .lm
            .token_id(token)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(token_id.map_or(-1, |id| id as i32))
    }

    #[wasm_bindgen(js_name = commitToken)]
    pub fn commit_token(&mut self, raw_token: &str) -> Result<bool, JsValue> {
        let (token_id, had_text, boundary_after) =
            repetition_observation_for_raw_token(&self.lm, raw_token)
                .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let learned = self
            .personal
            .observe_committed_token_with_personal_context(
                &self.lm,
                &mut self.context,
                &mut self.personal_context,
                raw_token,
            )
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        self.repetition_history
            .observe_resolved_token(token_id, had_text, boundary_after);
        Ok(learned)
    }

    #[wasm_bindgen(js_name = commitTokenId)]
    pub fn commit_token_id(
        &mut self,
        token_id: i32,
        boundary_after: bool,
    ) -> Result<bool, JsValue> {
        let resolved = if token_id < 0 {
            None
        } else {
            let token_id = token_id as u32;
            if !self.lm.is_word_token_id(token_id) {
                return Err(JsValue::from_str(&format!(
                    "autosuggest artifact references invalid token id {token_id}"
                )));
            }
            Some(token_id)
        };
        let learned = self
            .personal
            .observe_resolved_token_id_with_personal_context(
                &mut self.context,
                &mut self.personal_context,
                resolved,
                boundary_after,
            );
        self.repetition_history.observe_resolved_token(
            resolved,
            resolved.is_some(),
            boundary_after,
        );
        Ok(learned)
    }

    #[wasm_bindgen(js_name = commitUnknown)]
    pub fn commit_unknown(&mut self, boundary_after: bool) {
        self.personal
            .observe_resolved_token_id_with_personal_context(
                &mut self.context,
                &mut self.personal_context,
                None,
                boundary_after,
            );
        self.repetition_history
            .observe_resolved_token(None, true, boundary_after);
    }

    #[wasm_bindgen(js_name = suggestSession)]
    pub fn suggest_session(&mut self, limit: usize) -> Result<JsValue, JsValue> {
        let start = now();
        let metadata = autosuggest_session_candidate_ids_into(
            &self.lm,
            &self.personal,
            self.context,
            self.personal_context,
            self.repetition_history.recent_token_ids(),
            &mut self.personal_scratch,
            limit,
            &mut self.model_id_scratch,
            &mut self.candidate_id_scratch,
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        self.personal.suggest_text_for_personal_context_into(
            self.personal_context,
            limit.max(1),
            &mut self.personal_text_scratch,
        );
        let elapsed_ms = now() - start;
        let mut stats = self.stats;
        stats.last_elapsed_ms = Some(elapsed_ms);
        let lab_result = AutosuggestLabResult {
            context: autosuggest_context_labels(&self.lm, self.context)
                .map_err(|error| JsValue::from_str(&error.to_string()))?,
            context_token_count: metadata.context_token_count,
            matched_context_token_count: metadata.matched_context_token_count,
            elapsed_ms,
            model: stats,
            candidates: autosuggest_candidate_infos_with_text(
                &self.lm,
                &self.personal,
                &self.personal_text_scratch,
                &self.candidate_id_scratch,
                limit.max(1),
            )
            .map_err(|error| JsValue::from_str(&error.to_string()))?,
        };
        to_value(&lab_result).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen(js_name = suggestSessionCandidates)]
    pub fn suggest_session_candidates(&mut self, limit: usize) -> Result<JsValue, JsValue> {
        autosuggest_session_candidate_ids_into(
            &self.lm,
            &self.personal,
            self.context,
            self.personal_context,
            self.repetition_history.recent_token_ids(),
            &mut self.personal_scratch,
            limit,
            &mut self.model_id_scratch,
            &mut self.candidate_id_scratch,
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        self.personal.suggest_text_for_personal_context_into(
            self.personal_context,
            limit.max(1),
            &mut self.personal_text_scratch,
        );
        let candidate_infos = autosuggest_candidate_infos_with_text(
            &self.lm,
            &self.personal,
            &self.personal_text_scratch,
            &self.candidate_id_scratch,
            limit.max(1),
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
        to_value(&candidate_infos).map_err(|error| JsValue::from_str(&error.to_string()))
    }
}

#[wasm_bindgen]
pub struct ObadhAutocorrectWasm {
    obadh: ObadhEngine,
    backend: AutocorrectBackend,
    stats: AutocorrectLexiconStats,
}

enum AutocorrectBackend {
    Compact(AutocorrectEngine),
    Fst {
        lexicon: FstLexicon<Vec<u8>>,
        loanwords: Option<LoanwordLexicon<Vec<u8>>>,
    },
}

#[wasm_bindgen]
impl ObadhAutocorrectWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::from_lexicon(Lexicon::default())
    }

    #[wasm_bindgen(js_name = fromCompactLexicon)]
    pub fn from_compact_lexicon(bytes: &[u8]) -> Result<ObadhAutocorrectWasm, JsValue> {
        let lexicon = Lexicon::from_compact_bytes(bytes)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self::from_lexicon(lexicon))
    }

    #[wasm_bindgen(js_name = fromFstLexicon)]
    pub fn from_fst_lexicon(bytes: &[u8]) -> Result<ObadhAutocorrectWasm, JsValue> {
        let lexicon = FstLexicon::from_bytes(bytes.to_vec())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self::from_fst(lexicon, None))
    }

    #[wasm_bindgen(js_name = fromFstLexiconWithLoanwords)]
    pub fn from_fst_lexicon_with_loanwords(
        bytes: &[u8],
        loanword_bytes: &[u8],
    ) -> Result<ObadhAutocorrectWasm, JsValue> {
        let lexicon = FstLexicon::from_bytes(bytes.to_vec())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        let loanwords = LoanwordLexicon::from_bytes(loanword_bytes.to_vec())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Self::from_fst(lexicon, Some(loanwords)))
    }

    #[wasm_bindgen(js_name = fromTsv)]
    pub fn from_tsv(lexicon_tsv: &str) -> Result<ObadhAutocorrectWasm, JsValue> {
        let entries = parse_lexicon_tsv(lexicon_tsv).map_err(|error| JsValue::from_str(&error))?;
        Ok(Self::from_lexicon(Lexicon::new(entries)))
    }

    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        to_value(&self.stats).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen]
    pub fn suggest(&self, roman_input: &str) -> Result<JsValue, JsValue> {
        let start = now();
        if let Some(result) = crate::emoticon_suggestions(roman_input, 1) {
            let literal = AutocorrectCandidateInfo {
                text: result.baseline.clone(),
                source: "literal",
                edit_cost: 0,
                frequency: 0,
                score: 0,
                roman_repair: None,
                roman_repair_kind: None,
                roman_repair_cost: None,
                features: [0; crate::AUTOCORRECT_FEATURE_DIM],
            };
            let mut candidates = vec![literal];
            candidates.extend(
                result
                    .candidates
                    .into_iter()
                    .map(fst_autocorrect_candidate_info),
            );
            let result = AutocorrectLabResult {
                roman_input: roman_input.to_owned(),
                obadh_output: result.baseline.clone(),
                input: result.baseline,
                elapsed_ms: now() - start,
                replacement: None,
                candidates,
                lexicon: self.stats,
            };
            return to_value(&result).map_err(|error| JsValue::from_str(&error.to_string()));
        }
        if roman_input.trim().is_empty() {
            let empty_result = AutocorrectLabResult {
                roman_input: String::new(),
                obadh_output: String::new(),
                input: String::new(),
                elapsed_ms: now() - start,
                replacement: None,
                candidates: Vec::new(),
                lexicon: self.stats,
            };
            return to_value(&empty_result).map_err(|error| JsValue::from_str(&error.to_string()));
        }

        let result = match &self.backend {
            AutocorrectBackend::Compact(engine) => {
                let request = self.obadh.autocorrect_request(roman_input);
                autocorrect_result(
                    roman_input,
                    engine.decide(request),
                    now() - start,
                    self.stats,
                )
            }
            AutocorrectBackend::Fst { lexicon, loanwords } => fst_autocorrect_result(
                roman_input,
                &self.obadh,
                lexicon,
                loanwords.as_ref(),
                now() - start,
            )?,
        };
        to_value(&result).map_err(|error| JsValue::from_str(&error.to_string()))
    }
}

impl Default for ObadhAutocorrectWasm {
    fn default() -> Self {
        Self::new()
    }
}

impl ObadhAutocorrectWasm {
    fn from_lexicon(lexicon: Lexicon) -> Self {
        let stats = lexicon.stats();
        Self {
            obadh: ObadhEngine::new(),
            backend: AutocorrectBackend::Compact(AutocorrectEngine::with_config(
                lexicon,
                autocorrect_lab_config(),
            )),
            stats: AutocorrectLexiconStats::from(stats),
        }
    }

    fn from_fst(lexicon: FstLexicon<Vec<u8>>, loanwords: Option<LoanwordLexicon<Vec<u8>>>) -> Self {
        let stats = AutocorrectLexiconStats::from_fst(
            lexicon.len(),
            loanwords.as_ref().map_or(0, |l| l.len()),
        );
        Self {
            obadh: ObadhEngine::new(),
            backend: AutocorrectBackend::Fst { lexicon, loanwords },
            stats,
        }
    }
}

impl AutocorrectLexiconStats {
    fn from_fst(entries: usize, loanword_entries: usize) -> Self {
        Self {
            artifact_kind: if loanword_entries == 0 {
                "fst"
            } else {
                "fst+loan"
            },
            entries,
            loanword_entries,
            trie_nodes: 0,
            trie_edges: 0,
            skeleton_keys: 0,
            unique_skeletons: 0,
            skeleton_delete_keys: 0,
        }
    }
}

impl From<LexiconStats> for AutocorrectLexiconStats {
    fn from(stats: LexiconStats) -> Self {
        Self {
            artifact_kind: "compact",
            entries: stats.entries,
            loanword_entries: 0,
            trie_nodes: stats.trie_nodes,
            trie_edges: stats.trie_edges,
            skeleton_keys: stats.skeleton_keys,
            unique_skeletons: stats.unique_skeletons,
            skeleton_delete_keys: stats.skeleton_delete_keys,
        }
    }
}

fn autocorrect_lab_config() -> AutocorrectConfig {
    AutocorrectConfig {
        max_candidates: AUTOCORRECT_RERANK_POOL_SIZE,
        search_known_input: true,
        max_prefix_candidates: AUTOCORRECT_RESPONSE_CANDIDATES,
        max_skeleton_candidates: AUTOCORRECT_RERANK_POOL_SIZE,
        ..AutocorrectConfig::default()
    }
}

fn fst_autocorrect_lab_options() -> FstSuggestOptions {
    FstSuggestOptions {
        max_distance: crate::FST_MAX_LEVENSHTEIN_DISTANCE,
        max_candidates: AUTOCORRECT_RERANK_POOL_SIZE,
        response_candidates: AUTOCORRECT_RESPONSE_CANDIDATES,
        max_prefix_candidates: AUTOCORRECT_RESPONSE_CANDIDATES,
        ..FstSuggestOptions::default()
    }
}

fn autocorrect_result(
    roman_input: &str,
    decision: AutocorrectDecision,
    elapsed_ms: f64,
    stats: AutocorrectLexiconStats,
) -> AutocorrectLabResult {
    AutocorrectLabResult {
        roman_input: roman_input.to_string(),
        obadh_output: decision.input.clone(),
        input: decision.input,
        elapsed_ms,
        replacement: decision
            .replacement
            .as_ref()
            .map(autocorrect_candidate_info),
        candidates: autocorrect_candidate_infos(&decision.candidates),
        lexicon: stats,
    }
}

fn fst_autocorrect_result(
    roman_input: &str,
    obadh: &ObadhEngine,
    lexicon: &FstLexicon<Vec<u8>>,
    loanwords: Option<&LoanwordLexicon<Vec<u8>>>,
    elapsed_ms: f64,
) -> Result<AutocorrectLabResult, JsValue> {
    let obadh_output = obadh.transliterate(roman_input);
    let mut repair_records = fst_roman_repairs(roman_input, obadh, &obadh_output);
    // QWERTY fat-finger (key-slip) repairs — gated to non-word baselines and lexicon-validated
    // inside the helper. Mirrors the CLI path so the browser corrects identically.
    repair_records.extend(key_slip_repaired_outputs(
        roman_input,
        &obadh_output,
        lexicon.exact_frequency(&obadh_output),
        |roman| obadh.transliterate(roman),
        |word| lexicon.exact_frequency(word).is_some(),
    ));
    let repaired_baselines = repair_records
        .iter()
        .map(|repair| FstRepairedBaseline {
            roman_input: repair.roman_input.as_str(),
            bangla_output: repair.bangla_output.as_str(),
            repair_kind: repair.repair_kind,
            repair_cost: repair.repair_cost,
        })
        .collect::<Vec<_>>();
    let loanword_suggestions = loanwords
        .map(|loanwords| {
            loanwords.suggestions(roman_input, LoanwordSearchOptions::for_input(roman_input))
        })
        .transpose()
        .map_err(|error| JsValue::from_str(&error.to_string()))?
        .unwrap_or_default();
    let fst_loanword_matches = loanword_suggestions
        .iter()
        .map(|entry| FstLoanwordMatch {
            roman_input,
            roman_repair: entry.english.as_str(),
            bangla_output: entry.bangla.as_str(),
            frequency: entry.frequency,
            repair_kind: entry.kind.as_str(),
            repair_cost: entry.edit_cost,
        })
        .collect::<Vec<_>>();
    let decision = lexicon
        .suggest_with_repaired_baselines_and_loanwords(
            &obadh_output,
            &repaired_baselines,
            &fst_loanword_matches,
            fst_autocorrect_lab_options(),
        )
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    Ok(AutocorrectLabResult {
        roman_input: roman_input.to_string(),
        obadh_output: obadh_output.clone(),
        input: obadh_output,
        elapsed_ms,
        replacement: None,
        candidates: decision
            .candidates
            .into_iter()
            .take(AUTOCORRECT_RESPONSE_CANDIDATES)
            .map(fst_autocorrect_candidate_info)
            .collect(),
        lexicon: AutocorrectLexiconStats::from_fst(
            lexicon.len(),
            loanwords.map_or(0, |loanwords| loanwords.len()),
        ),
    })
}

fn autocorrect_candidate_infos(
    candidates: &[CorrectionCandidate],
) -> Vec<AutocorrectCandidateInfo> {
    candidates
        .iter()
        .take(AUTOCORRECT_RESPONSE_CANDIDATES)
        .map(autocorrect_candidate_info)
        .collect()
}

fn autocorrect_candidate_info(candidate: &CorrectionCandidate) -> AutocorrectCandidateInfo {
    AutocorrectCandidateInfo {
        text: candidate.text.clone(),
        source: correction_source_name(candidate.source),
        edit_cost: candidate.edit_cost.0,
        frequency: candidate.frequency as u64,
        score: candidate.score as i64,
        roman_repair: None,
        roman_repair_kind: None,
        roman_repair_cost: None,
        features: candidate_features_array(candidate.features),
    }
}

fn fst_autocorrect_candidate_info(candidate: FstCandidate) -> AutocorrectCandidateInfo {
    AutocorrectCandidateInfo {
        text: candidate.text,
        source: candidate.source.as_str(),
        edit_cost: candidate.edit_cost,
        frequency: candidate.frequency,
        score: candidate.score,
        roman_repair: candidate.roman_repair,
        roman_repair_kind: candidate.roman_repair_kind,
        roman_repair_cost: candidate.roman_repair_cost,
        features: [0; crate::AUTOCORRECT_FEATURE_DIM],
    }
}

fn autosuggest_session_candidate_ids_into(
    lm: &AutosuggestLm<Vec<u8>>,
    personal: &PersonalAutosuggest,
    context: AutosuggestContext,
    personal_context: PersonalAutosuggestContext,
    repetition_context_ids: &[u32],
    personal_scratch: &mut Vec<PersonalAutosuggestSuggestion>,
    limit: usize,
    model_scratch: &mut Vec<AutosuggestCandidateId>,
    output: &mut Vec<AutosuggestCandidateId>,
) -> Result<AutosuggestMetadata, AutosuggestArtifactError> {
    let limit = limit.max(1);
    let pool_limit = session_repetition_guard_pool_limit(limit);
    reserve_vec_to(personal_scratch, pool_limit);
    reserve_vec_to(model_scratch, pool_limit);
    reserve_vec_to(output, limit);
    personal.suggest_ids_with_lm_for_personal_context_into(
        lm,
        context,
        personal_context,
        repetition_context_ids,
        AutosuggestOptions {
            max_candidates: limit,
        },
        personal_scratch,
        model_scratch,
        output,
    )
}

fn fst_roman_repairs(
    input: &str,
    obadh: &ObadhEngine,
    obadh_output: &str,
) -> Vec<RomanRepairedOutput> {
    roman_repaired_outputs(
        input,
        obadh_output,
        RomanRepairOptions {
            max_repairs: DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
        },
        |repair| obadh.transliterate(repair),
    )
}

fn correction_source_name(source: CorrectionSource) -> &'static str {
    match source {
        CorrectionSource::NoChange => "no_change",
        CorrectionSource::LexiconEdit => "lexicon_edit",
        CorrectionSource::PrefixCompletion => "prefix_completion",
        CorrectionSource::PhoneticSkeleton => "phonetic_skeleton",
    }
}

fn candidate_features_array(features: CandidateFeatures) -> [i16; crate::AUTOCORRECT_FEATURE_DIM] {
    features.as_i16_array()
}

fn autosuggest_candidate_id_info(
    lm: &AutosuggestLm<Vec<u8>>,
    candidate: AutosuggestCandidateId,
) -> Result<AutosuggestCandidateInfo, AutosuggestArtifactError> {
    Ok(AutosuggestCandidateInfo {
        text: lm.token_text(candidate.token_id)?.to_string(),
        token_id: candidate.token_id,
        source: autosuggest_source_name(candidate.source),
        count: candidate.count,
        score: candidate.score,
    })
}

fn autosuggest_candidate_infos_with_text(
    lm: &AutosuggestLm<Vec<u8>>,
    personal: &PersonalAutosuggest,
    text_suggestions: &[PersonalAutosuggestTextSuggestion],
    id_candidates: &[AutosuggestCandidateId],
    limit: usize,
) -> Result<Vec<AutosuggestCandidateInfo>, AutosuggestArtifactError> {
    let limit = limit.max(1);
    let mut candidates: Vec<AutosuggestCandidateInfo> = Vec::with_capacity(limit);

    push_autosuggest_text_infos(personal, text_suggestions, true, limit, &mut candidates);

    for candidate in id_candidates {
        if candidates.len() >= limit {
            break;
        }
        let info = autosuggest_candidate_id_info(lm, *candidate)?;
        if candidates
            .iter()
            .any(|candidate| candidate.text == info.text)
        {
            continue;
        }
        candidates.push(info);
    }

    push_autosuggest_text_infos(personal, text_suggestions, false, limit, &mut candidates);

    Ok(candidates)
}

fn push_autosuggest_text_infos(
    personal: &PersonalAutosuggest,
    text_suggestions: &[PersonalAutosuggestTextSuggestion],
    contextual: bool,
    limit: usize,
    candidates: &mut Vec<AutosuggestCandidateInfo>,
) {
    for suggestion in text_suggestions {
        if candidates.len() >= limit {
            break;
        }
        if (suggestion.context_len > 0) != contextual {
            continue;
        }
        let Some(text) = personal.text_suggestion_text(*suggestion) else {
            continue;
        };
        if candidates.iter().any(|candidate| candidate.text == text) {
            continue;
        }
        candidates.push(AutosuggestCandidateInfo {
            text: text.to_string(),
            token_id: 0,
            source: "autosuggest_personal_oov",
            count: u32::from(suggestion.count),
            score: suggestion.score,
        });
    }
}

fn autosuggest_context_prior_info(
    lm: &AutosuggestLm<Vec<u8>>,
    candidate_texts: &[String],
    prior: AutosuggestCandidatePrior,
) -> Result<AutosuggestContextPriorInfo, AutosuggestArtifactError> {
    let text = match candidate_texts.get(prior.candidate_index) {
        Some(text) => text.clone(),
        None => lm.token_text(prior.token_id)?.to_string(),
    };
    Ok(AutosuggestContextPriorInfo {
        candidate_index: prior.candidate_index,
        text,
        token_id: prior.token_id,
        prior_rank: prior.prior_rank,
        source: autosuggest_source_name(prior.source),
        count: prior.count,
        score: prior.score,
    })
}

fn autosuggest_context_labels(
    lm: &AutosuggestLm<Vec<u8>>,
    context: AutosuggestContext,
) -> Result<Vec<String>, crate::AutosuggestArtifactError> {
    context
        .recent_token_ids()
        .iter()
        .map(|token_id| lm.token_text(*token_id).map(str::to_string))
        .collect()
}

fn autosuggest_source_name(source: AutosuggestSource) -> &'static str {
    match source {
        AutosuggestSource::Personal => "autosuggest_personal",
        AutosuggestSource::Fourgram => "autosuggest_ngram_fourgram",
        AutosuggestSource::Trigram => "autosuggest_ngram_trigram",
        AutosuggestSource::Bigram => "autosuggest_ngram_bigram",
        AutosuggestSource::Unigram => "autosuggest_ngram_unigram",
    }
}

fn parse_lexicon_tsv(lexicon_tsv: &str) -> Result<Vec<LexiconEntry>, String> {
    let mut entries = Vec::new();

    for (line_index, line) in lexicon_tsv.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split('\t');
        let word = parts.next().unwrap_or_default().trim();
        if word.is_empty() {
            return Err(format!("empty lexicon word at line {line_number}"));
        }

        let frequency = match parts.next().map(str::trim).filter(|part| !part.is_empty()) {
            Some(raw) => raw
                .parse::<u32>()
                .map_err(|error| format!("invalid frequency at line {line_number}: {error}"))?,
            None => 1,
        };

        if parts.next().is_some() {
            return Err(format!("expected word<TAB>frequency at line {line_number}"));
        }

        entries.push(LexiconEntry::new(word, frequency));
    }

    if entries.is_empty() {
        return Err("lexicon TSV did not contain any entries".to_string());
    }

    Ok(entries)
}
