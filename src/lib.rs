//! Obadh Engine — a deterministic Roman→Bangla transliteration engine and runtime
//! SDK for Bangla typing systems.
//!
//! The core transliterator is rule-based: Roman input becomes Bengali because an
//! Obadh rule says so, not because a dictionary or model guessed a word. Higher
//! runtime layers sit above that core as separate, optional stages:
//!
//! - [`engine`] — the deterministic tokenizer and transliterator ([`ObadhEngine`]).
//! - [`autocorrect`] — active-word candidate generation over memory-mapped FST
//!   artifacts, each candidate carrying its provenance ([`FstCandidate`]).
//!   Auto-insert *policy* is the caller's; the engine supplies the signals
//!   (source, edit costs, and lexicon frequency).
//! - [`autosuggest`] — next-word runtime over committed Bengali text, with a
//!   bounded personal overlay and an optional neural scorer/generator handoff.
//!
//! The default feature set is empty, so native downstreams pay for none of the
//! CLI or browser tooling. A stable, versioned C ABI is available behind the
//! `cabi` feature (see the `cabi` module and `include/obadh.h`); the
//! [obadh-ios](https://github.com/nsssayom/obadh-ios) keyboard is the reference
//! integrator.
//!
//! See the [README](https://github.com/nsssayom/obadh_engine#readme) for the
//! Roman rule contract, the reference auto-insert policy, and performance numbers.

pub mod autocorrect;
pub mod autosuggest;
#[cfg(all(feature = "cabi", not(target_arch = "wasm32")))]
pub mod cabi;
pub mod definitions;
pub mod engine;
pub mod fingerprint;
pub mod gesture;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export commonly used types for convenience
pub use autocorrect::{
    build_loanword_bytes, emoticon_emoji, emoticon_suggestions, is_loanword_key,
    key_slip_repaired_outputs, roman_repair_beam, roman_repaired_outputs, weighted_edit_distance,
    AutocorrectConfig, AutocorrectDecision, AutocorrectEngine, CandidateFeatures,
    CorrectionCandidate, CorrectionRequest, CorrectionSource, EditCost, FstCandidate,
    FstCandidateSource, FstLexicon, FstLoanwordMatch, FstRepairedBaseline, FstSuggestError,
    FstSuggestOptions, FstSuggestResult, Lexicon, LexiconArtifactError, LexiconEntry, LexiconStats,
    LoanwordArtifactError, LoanwordEntry, LoanwordLexicon, LoanwordMatch, LoanwordSearchOptions,
    LoanwordSuggestion, LoanwordSuggestionKind, RomanRepair, RomanRepairKind, RomanRepairOptions,
    RomanRepairedOutput, AUTOCORRECT_FEATURE_DIM, DEFAULT_FST_MAX_DISTANCE,
    DEFAULT_FST_PREFIX_CANDIDATES, DEFAULT_LOANWORD_FUZZY_CANDIDATES,
    DEFAULT_ROMAN_REPAIR_BEAM_SIZE, FST_MAX_LEVENSHTEIN_DISTANCE, LOANWORD_FUZZY_MAX_DISTANCE,
};
#[cfg(not(target_arch = "wasm32"))]
pub use autosuggest::AutosuggestLoadError;
pub use autosuggest::{
    accept_open_vocab_texts_into, materialize_generated_candidates_into,
    materialize_merged_candidates_into, materialize_scored_union_candidates_into,
    merge_static_and_generated_candidates_into,
    merge_static_generated_and_open_vocab_candidates_into,
    rerank_candidate_ids_with_fixed_scores_into, rerank_candidate_ids_with_scores_into,
    scored_union_static_and_generated_candidates_into, scorer_candidate_i32s_for_candidates_into,
    scorer_candidate_ids_for_candidates_into, validate_open_vocab_text, AutosuggestArtifactError,
    AutosuggestCandidate, AutosuggestCandidateId, AutosuggestCandidatePrior, AutosuggestContext,
    AutosuggestContextPriorMetadata, AutosuggestContextPriorOptions,
    AutosuggestGeneratedCandidateId, AutosuggestGeneratorCompatibility, AutosuggestGeneratorFile,
    AutosuggestGeneratorHandoff, AutosuggestGeneratorHandoffError, AutosuggestGeneratorManifest,
    AutosuggestGeneratorManifestError, AutosuggestGeneratorMergeOptions,
    AutosuggestGeneratorMergedQuality, AutosuggestGeneratorModel, AutosuggestGeneratorNgram,
    AutosuggestGeneratorQuality, AutosuggestGeneratorQualityMetrics,
    AutosuggestGeneratorRuntimeContract, AutosuggestGeneratorScoredUnionPolicy,
    AutosuggestGeneratorScoredUnionQuality, AutosuggestGeneratorSession, AutosuggestLm,
    AutosuggestMaterializedGeneratedCandidate, AutosuggestMaterializedMergedCandidate,
    AutosuggestMaterializedScoredCandidate, AutosuggestMaterializedScoredUnionCandidate,
    AutosuggestMergedCandidateId, AutosuggestMergedCandidateSource, AutosuggestMetadata,
    AutosuggestModelInfo, AutosuggestOpenVocabError, AutosuggestOpenVocabPolicy,
    AutosuggestOpenVocabRejectionKind, AutosuggestOpenVocabValidationReport, AutosuggestOptions,
    AutosuggestRerankError, AutosuggestRerankInputMetadata, AutosuggestRerankOptions,
    AutosuggestResult, AutosuggestScoredCandidateId, AutosuggestScoredUnionCandidateId,
    AutosuggestScorerCompatibility, AutosuggestScorerFile, AutosuggestScorerHandoff,
    AutosuggestScorerHandoffError, AutosuggestScorerManifest, AutosuggestScorerManifestError,
    AutosuggestScorerModel, AutosuggestScorerNgram, AutosuggestScorerQuality,
    AutosuggestScorerQualityMetrics, AutosuggestScorerRuntimeContract, AutosuggestScorerSession,
    AutosuggestSession, AutosuggestSource, AutosuggestUnifiedCandidate,
    AutosuggestUnifiedCandidateKind, AutosuggestValidatedTextCandidate, PersonalAutosuggest,
    PersonalAutosuggestConfig, PersonalAutosuggestContext, PersonalAutosuggestError,
    PersonalAutosuggestSnapshotError, PersonalAutosuggestSuggestion,
    PersonalAutosuggestTextSuggestion, AUTOSUGGEST_ARTIFACT_KIND, AUTOSUGGEST_BOS_I32,
    AUTOSUGGEST_BOS_ID, AUTOSUGGEST_GENERATOR_COREML_INPUT_DTYPE,
    AUTOSUGGEST_GENERATOR_MANIFEST_VERSION, AUTOSUGGEST_GENERATOR_ONNX_INPUT_DTYPE,
    AUTOSUGGEST_GENERATOR_PACKAGE_KIND, AUTOSUGGEST_GENERATOR_RUNTIME_ROLE,
    AUTOSUGGEST_GENERATOR_SCORE_DTYPE, AUTOSUGGEST_GENERATOR_TOKEN_ID_DTYPE, AUTOSUGGEST_PAD_I32,
    AUTOSUGGEST_PAD_ID, AUTOSUGGEST_SCORER_COREML_INPUT_DTYPE, AUTOSUGGEST_SCORER_MANIFEST_VERSION,
    AUTOSUGGEST_SCORER_ONNX_INPUT_DTYPE, AUTOSUGGEST_SCORER_PACKAGE_KIND,
    AUTOSUGGEST_SCORER_RUNTIME_ROLE, AUTOSUGGEST_SCORER_SCORE_DTYPE,
    AUTOSUGGEST_SCORER_TOKEN_ID_DTYPE, AUTOSUGGEST_UNK_I32, AUTOSUGGEST_UNK_ID,
    DEFAULT_AUTOSUGGEST_CANDIDATES, DEFAULT_AUTOSUGGEST_GENERATOR_LOCKED_STATIC_PREFIX,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_RANK_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_RANK_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_CANDIDATES, DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_SCALAR_RUN,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_WORD_CHARS, DEFAULT_AUTOSUGGEST_OPEN_VOCAB_OVERLAP_BONUS,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_BONUS,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_LOG_COUNT_SCALE,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_RANK_PENALTY,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_SOURCE_BONUS,
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_TEXT_HEAP_BYTES, DEFAULT_AUTOSUGGEST_RERANK_LOCKED_PREFIX,
    DEFAULT_AUTOSUGGEST_RERANK_RANK_PENALTY, DEFAULT_PERSONAL_AUTOSUGGEST_ENTRIES,
    DEFAULT_PERSONAL_AUTOSUGGEST_MIN_COUNT, MAX_AUTOSUGGEST_CONTEXT_TOKENS,
    MAX_AUTOSUGGEST_RERANK_CONTEXT_TOKENS,
};
#[cfg(not(target_arch = "wasm32"))]
pub use autosuggest::{
    AutosuggestGeneratorAsset, AutosuggestGeneratorAssetReport, AutosuggestScorerAsset,
    AutosuggestScorerAssetReport,
};
pub use engine::{PhoneticUnit, PhoneticUnitType, Token, TokenType, Tokenizer};
pub use engine::{SanitizeResult, Sanitizer};
pub use fingerprint::{artifact_fingerprint, verify_artifact_fingerprint, FingerprintMismatch};
#[cfg(feature = "wasm")]
pub use wasm::ObadhaWasm;

/// Main entry point for the Obadh transliteration engine
pub struct ObadhEngine {
    transliterator: engine::Transliterator,
}

impl ObadhEngine {
    /// Create a new engine with default settings
    pub fn new() -> Self {
        Self {
            transliterator: engine::Transliterator::new(),
        }
    }

    /// Transliterate Roman text to Bengali
    pub fn transliterate(&self, text: &str) -> String {
        self.transliterator.transliterate(text)
    }

    /// Transliterate Roman text after dropping unsupported characters.
    pub fn transliterate_lenient(&self, text: &str) -> String {
        self.transliterator.transliterate_lenient(text)
    }

    /// Sanitize input text to ensure it contains only valid characters
    pub fn sanitize(&self, text: &str) -> SanitizeResult {
        self.transliterator.sanitize(text)
    }

    /// Tokenize input text into words and other tokens
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        self.transliterator.tokenize(text)
    }

    /// Transliterate already-tokenized input to Bengali
    pub fn transliterate_tokens(&self, tokens: &[Token]) -> String {
        self.transliterator.transliterate_tokens(tokens)
    }

    /// Transliterate one token using its surrounding tokenized context.
    pub fn transliterate_token_at(&self, tokens: &[Token], index: usize) -> Option<String> {
        self.transliterator.transliterate_token_at(tokens, index)
    }

    /// Tokenize a word into phonetic units for Bengali transliteration
    pub fn tokenize_phonetic(&self, word: &str) -> Vec<PhoneticUnit> {
        self.transliterator.tokenize_phonetic(word)
    }

    /// Tokenize a word into a caller-owned phonetic-unit buffer.
    ///
    /// The buffer is cleared before use and then reused. Prefer this method for
    /// high-frequency editor or typing integrations that repeatedly analyze the
    /// active word.
    pub fn tokenize_phonetic_into(&self, word: &str, units: &mut Vec<PhoneticUnit>) {
        self.transliterator.tokenize_phonetic_into(word, units);
    }

    /// Get a new tokenizer instance for custom tokenization
    pub fn get_tokenizer(&self) -> Tokenizer {
        Tokenizer::new()
    }

    /// Build an autocorrect request from Roman text by preserving both the raw
    /// typing buffer and Obadh's deterministic Bangla output.
    pub fn autocorrect_request(&self, roman_input: &str) -> CorrectionRequest {
        let output = self.transliterate(roman_input);
        CorrectionRequest::new(output.clone())
            .with_roman_input(roman_input)
            .with_obadh_output(output)
    }
}

impl Default for ObadhEngine {
    fn default() -> Self {
        Self::new()
    }
}
