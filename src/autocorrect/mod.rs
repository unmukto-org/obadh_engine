//! Runtime autocorrect primitives for Bangla text produced above Obadh.
//!
//! This module implements the deterministic runtime shell for autocorrect:
//! candidate generation, Bangla-aware edit scoring, and conservative ranking.
//! Learned model weights can be added behind the same candidate/ranker boundary
//! later without changing the transliteration hot path.

mod artifact;
mod bangla;
mod edit;
mod emoticon;
mod fst_lexicon;
mod lexicon;
mod loanword;
mod morphology;
mod phoneme;
mod qwerty;
mod ranker;
mod roman_repair;
mod skeleton;

pub use artifact::LexiconArtifactError;
pub use edit::{weighted_edit_distance, EditCost};
pub use emoticon::{emoticon_emoji, emoticon_suggestions};
pub use fst_lexicon::{
    FstCandidate, FstCandidateSource, FstLexicon, FstLoanwordMatch, FstRepairedBaseline,
    FstSuggestError, FstSuggestOptions, FstSuggestResult, DEFAULT_FST_MAX_DISTANCE,
    DEFAULT_FST_PREFIX_CANDIDATES, FST_MAX_LEVENSHTEIN_DISTANCE,
};
pub use lexicon::{Lexicon, LexiconEntry, LexiconStats};
pub use loanword::{
    build_loanword_bytes, default_loanword_fuzzy_distance, is_loanword_key, LoanwordArtifactError,
    LoanwordEntry, LoanwordLexicon, LoanwordMatch, LoanwordSearchOptions, LoanwordSuggestion,
    LoanwordSuggestionKind, DEFAULT_LOANWORD_FUZZY_CANDIDATES, LOANWORD_FUZZY_MAX_DISTANCE,
};
pub use ranker::{
    AutocorrectConfig, AutocorrectDecision, AutocorrectEngine, CandidateFeatures,
    CorrectionCandidate, CorrectionRequest, CorrectionSource, AUTOCORRECT_FEATURE_DIM,
};
pub use roman_repair::{
    roman_repair_beam, roman_repaired_outputs, RomanRepair, RomanRepairKind, RomanRepairOptions,
    RomanRepairedOutput, DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
};
// `skeleton` (the dropped-vowel channel) and `phoneme` (graded consonant confusion) are
// internal to the FST suggest path — they expose no public surface, only new
// `FstCandidateSource` variants on the results.
pub use qwerty::key_slip_repaired_outputs;
