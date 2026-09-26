use serde::Serialize;
use std::error::Error;
use std::fmt;

use super::artifact::AutosuggestArtifactError;
use super::generator::AutosuggestGeneratedCandidateId;
use super::lm::{AutosuggestCandidateId, AutosuggestLm, AutosuggestSource};

pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_CANDIDATES: usize = 5;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_WORD_CHARS: usize = 32;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_TEXT_HEAP_BYTES: usize =
    DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_WORD_CHARS * 4;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_SCALAR_RUN: usize = 3;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_PENALTY: f32 = 0.75;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_RANK_PENALTY: f32 = 0.15;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_PENALTY: f32 = 0.25;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_RANK_PENALTY: f32 = 0.10;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_BONUS: f32 = 1.0;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_RANK_PENALTY: f32 = 0.25;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_LOG_COUNT_SCALE: f32 = 0.05;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_SOURCE_BONUS: f32 = 0.10;
pub const DEFAULT_AUTOSUGGEST_OPEN_VOCAB_OVERLAP_BONUS: f32 = 0.50;
const OPEN_VOCAB_LOCKED_STATIC_BONUS: f32 = 1.0e6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutosuggestOpenVocabPolicy {
    pub max_candidates: usize,
    pub max_word_chars: usize,
    pub max_repeated_scalar_run: usize,
    pub min_model_score: f32,
    pub locked_static_prefix: usize,
    pub static_bonus: f32,
    pub static_rank_penalty: f32,
    pub static_log_count_scale: f32,
    pub static_source_bonus: f32,
    pub generated_token_penalty: f32,
    pub generated_token_rank_penalty: f32,
    pub generated_text_penalty: f32,
    pub generated_text_rank_penalty: f32,
    pub overlap_bonus: f32,
}

impl Default for AutosuggestOpenVocabPolicy {
    fn default() -> Self {
        Self {
            max_candidates: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_CANDIDATES,
            max_word_chars: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_WORD_CHARS,
            max_repeated_scalar_run: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_MAX_SCALAR_RUN,
            min_model_score: f32::NEG_INFINITY,
            locked_static_prefix: 1,
            static_bonus: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_BONUS,
            static_rank_penalty: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_RANK_PENALTY,
            static_log_count_scale: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_LOG_COUNT_SCALE,
            static_source_bonus: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_STATIC_SOURCE_BONUS,
            generated_token_penalty: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_PENALTY,
            generated_token_rank_penalty:
                DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TOKEN_RANK_PENALTY,
            generated_text_penalty: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_PENALTY,
            generated_text_rank_penalty: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_GENERATED_TEXT_RANK_PENALTY,
            overlap_bonus: DEFAULT_AUTOSUGGEST_OPEN_VOCAB_OVERLAP_BONUS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutosuggestOpenVocabRejectionKind {
    Empty,
    TooLong,
    ScoreBelowThreshold,
    NonFiniteScore,
    LeadingOrTrailingWhitespace,
    ContainsNonBanglaScalar,
    ContainsDigit,
    ContainsWhitespace,
    ContainsPunctuation,
    StartsWithMark,
    EndsWithHasant,
    ConsecutiveHasant,
    MarkAfterHasant,
    InvalidZwnj,
    MissingLetter,
    RepeatedScalarRun,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AutosuggestOpenVocabValidationReport {
    pub accepted: bool,
    pub char_len: usize,
    pub rejection: Option<AutosuggestOpenVocabRejectionKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutosuggestValidatedTextCandidate {
    pub text: String,
    pub model_rank: usize,
    pub model_score: f32,
    pub char_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutosuggestUnifiedCandidateKind {
    Static,
    GeneratedToken,
    GeneratedText,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutosuggestUnifiedCandidate {
    pub text: String,
    pub token_id: Option<u32>,
    pub kind: AutosuggestUnifiedCandidateKind,
    pub static_rank: Option<usize>,
    pub static_candidate: Option<AutosuggestCandidateId>,
    pub token_model_rank: Option<usize>,
    pub token_model_score: Option<f32>,
    pub text_model_rank: Option<usize>,
    pub text_model_score: Option<f32>,
    pub final_score: f32,
}

impl AutosuggestUnifiedCandidate {
    pub fn has_static_signal(&self) -> bool {
        self.static_candidate.is_some()
    }

    pub fn has_generated_token_signal(&self) -> bool {
        self.token_model_rank.is_some()
    }

    pub fn has_generated_text_signal(&self) -> bool {
        self.text_model_rank.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutosuggestOpenVocabError {
    Artifact(AutosuggestArtifactError),
    InvalidBuffer {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for AutosuggestOpenVocabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => error.fmt(f),
            Self::InvalidBuffer {
                field,
                expected,
                actual,
            } => write!(
                f,
                "autosuggest open-vocab handoff expected {field} length {expected}, got {actual}"
            ),
        }
    }
}

impl Error for AutosuggestOpenVocabError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::InvalidBuffer { .. } => None,
        }
    }
}

impl From<AutosuggestArtifactError> for AutosuggestOpenVocabError {
    fn from(error: AutosuggestArtifactError) -> Self {
        Self::Artifact(error)
    }
}

pub fn validate_open_vocab_text(
    text: &str,
    model_score: f32,
    policy: AutosuggestOpenVocabPolicy,
) -> AutosuggestOpenVocabValidationReport {
    if !model_score.is_finite() {
        return rejected(0, AutosuggestOpenVocabRejectionKind::NonFiniteScore);
    }
    if model_score < policy.min_model_score {
        return rejected(0, AutosuggestOpenVocabRejectionKind::ScoreBelowThreshold);
    }
    if text.is_empty() {
        return rejected(0, AutosuggestOpenVocabRejectionKind::Empty);
    }
    if text.trim() != text {
        return rejected(
            text.chars().count(),
            AutosuggestOpenVocabRejectionKind::LeadingOrTrailingWhitespace,
        );
    }

    let char_len = text.chars().count();
    if char_len > policy.max_word_chars.max(1) {
        return rejected(char_len, AutosuggestOpenVocabRejectionKind::TooLong);
    }

    let mut saw_letter = false;
    let mut previous = '\0';
    let mut repeated = 0usize;
    let mut previous_was_hasant = false;
    let mut previous_was_zwnj = false;

    for (index, scalar) in text.chars().enumerate() {
        if scalar == previous {
            repeated += 1;
            if repeated > policy.max_repeated_scalar_run.max(1) {
                return rejected(
                    char_len,
                    AutosuggestOpenVocabRejectionKind::RepeatedScalarRun,
                );
            }
        } else {
            previous = scalar;
            repeated = 1;
        }

        if scalar == '\u{200c}' || scalar == '\u{200d}' {
            // Both joiners (ZWNJ and the reph-ya ZWJ, U+200D) follow the same
            // well-formedness rule: never leading, trailing, or doubled.
            if index == 0 || index + 1 == char_len || previous_was_zwnj {
                return rejected(char_len, AutosuggestOpenVocabRejectionKind::InvalidZwnj);
            }
            // Preserve the pending-hasant state across a joiner: a hasant followed
            // by a joiner and then a kar (ক + ্ + ZWJ + া) is still malformed. A
            // joiner only legitimizes a following letter (handled below), not a mark.
            previous_was_zwnj = true;
            continue;
        }
        previous_was_zwnj = false;

        if scalar.is_whitespace() {
            return rejected(
                char_len,
                AutosuggestOpenVocabRejectionKind::ContainsWhitespace,
            );
        }
        if scalar.is_ascii_punctuation() || is_bengali_punctuation(scalar) {
            return rejected(
                char_len,
                AutosuggestOpenVocabRejectionKind::ContainsPunctuation,
            );
        }
        if scalar.is_ascii_digit() || is_bengali_digit(scalar) {
            return rejected(char_len, AutosuggestOpenVocabRejectionKind::ContainsDigit);
        }
        if !is_bengali_word_scalar(scalar) {
            return rejected(
                char_len,
                AutosuggestOpenVocabRejectionKind::ContainsNonBanglaScalar,
            );
        }

        if is_bengali_letter(scalar) {
            saw_letter = true;
            previous_was_hasant = false;
            continue;
        }

        if index == 0 {
            return rejected(char_len, AutosuggestOpenVocabRejectionKind::StartsWithMark);
        }
        if previous_was_hasant {
            if scalar == '\u{09cd}' {
                return rejected(
                    char_len,
                    AutosuggestOpenVocabRejectionKind::ConsecutiveHasant,
                );
            }
            return rejected(char_len, AutosuggestOpenVocabRejectionKind::MarkAfterHasant);
        }
        previous_was_hasant = scalar == '\u{09cd}';
    }

    if previous_was_hasant {
        return rejected(char_len, AutosuggestOpenVocabRejectionKind::EndsWithHasant);
    }
    if !saw_letter {
        return rejected(char_len, AutosuggestOpenVocabRejectionKind::MissingLetter);
    }

    AutosuggestOpenVocabValidationReport {
        accepted: true,
        char_len,
        rejection: None,
    }
}

pub fn accept_open_vocab_texts_into<S: AsRef<str>>(
    texts: &[S],
    model_scores: &[f32],
    policy: AutosuggestOpenVocabPolicy,
    output: &mut Vec<AutosuggestValidatedTextCandidate>,
) -> Result<(), AutosuggestOpenVocabError> {
    if texts.len() != model_scores.len() {
        output.clear();
        return Err(AutosuggestOpenVocabError::InvalidBuffer {
            field: "open_vocab_model_scores",
            expected: texts.len(),
            actual: model_scores.len(),
        });
    }

    output.clear();
    output.reserve(policy.max_candidates.max(1).min(texts.len()));
    for (rank, (text, &score)) in texts.iter().zip(model_scores.iter()).enumerate() {
        let score = finite_score(score);
        let report = validate_open_vocab_text(text.as_ref(), score, policy);
        if !report.accepted {
            continue;
        }
        push_or_merge_text_candidate(
            text.as_ref(),
            rank,
            score,
            report.char_len,
            policy.max_candidates.max(1),
            output,
        );
    }
    Ok(())
}

pub fn merge_static_generated_and_open_vocab_candidates_into<D: AsRef<[u8]>>(
    lm: &AutosuggestLm<D>,
    static_candidates: &[AutosuggestCandidateId],
    generated_tokens: &[AutosuggestGeneratedCandidateId],
    generated_text: &[AutosuggestValidatedTextCandidate],
    policy: AutosuggestOpenVocabPolicy,
    output: &mut Vec<AutosuggestUnifiedCandidate>,
) -> Result<(), AutosuggestOpenVocabError> {
    output.clear();
    let limit = policy.max_candidates.max(1);
    output.reserve(
        static_candidates
            .len()
            .saturating_add(generated_tokens.len())
            .saturating_add(generated_text.len())
            .min(limit.saturating_mul(3)),
    );

    for (rank, candidate) in static_candidates.iter().copied().enumerate() {
        let text = lm.token_text(candidate.token_id)?;
        push_or_merge_static_unified(text, rank, candidate, policy, output);
    }
    for candidate in generated_tokens {
        let text = lm.token_text(candidate.token_id)?;
        push_or_merge_generated_token_unified(text, *candidate, policy, output);
    }
    for candidate in generated_text {
        push_or_merge_generated_text_unified(candidate, policy, output);
    }

    output.sort_by(unified_candidate_order);
    output.truncate(limit);
    Ok(())
}

fn push_or_merge_text_candidate(
    text: &str,
    model_rank: usize,
    model_score: f32,
    char_len: usize,
    limit: usize,
    output: &mut Vec<AutosuggestValidatedTextCandidate>,
) {
    if let Some(existing) = output.iter_mut().find(|candidate| candidate.text == text) {
        if model_score > existing.model_score || model_rank < existing.model_rank {
            existing.model_rank = existing.model_rank.min(model_rank);
            existing.model_score = existing.model_score.max(model_score);
        }
        return;
    }
    if output.len() >= limit {
        return;
    }
    output.push(AutosuggestValidatedTextCandidate {
        text: text.to_string(),
        model_rank,
        model_score,
        char_len,
    });
}

fn push_or_merge_static_unified(
    text: &str,
    rank: usize,
    candidate: AutosuggestCandidateId,
    policy: AutosuggestOpenVocabPolicy,
    output: &mut Vec<AutosuggestUnifiedCandidate>,
) {
    if let Some(existing) = output.iter_mut().find(|existing| existing.text == text) {
        existing.token_id = Some(candidate.token_id);
        existing.static_rank = Some(rank);
        existing.static_candidate = Some(candidate);
        existing.kind = AutosuggestUnifiedCandidateKind::Mixed;
        existing.final_score = unified_candidate_score(existing, policy);
        return;
    }
    let mut unified = AutosuggestUnifiedCandidate {
        text: text.to_string(),
        token_id: Some(candidate.token_id),
        kind: AutosuggestUnifiedCandidateKind::Static,
        static_rank: Some(rank),
        static_candidate: Some(candidate),
        token_model_rank: None,
        token_model_score: None,
        text_model_rank: None,
        text_model_score: None,
        final_score: 0.0,
    };
    unified.final_score = unified_candidate_score(&unified, policy);
    output.push(unified);
}

fn push_or_merge_generated_token_unified(
    text: &str,
    candidate: AutosuggestGeneratedCandidateId,
    policy: AutosuggestOpenVocabPolicy,
    output: &mut Vec<AutosuggestUnifiedCandidate>,
) {
    if let Some(existing) = output.iter_mut().find(|existing| existing.text == text) {
        existing.token_id = Some(candidate.token_id);
        existing.token_model_rank = Some(candidate.model_rank);
        existing.token_model_score = Some(finite_score(candidate.model_score));
        existing.kind = AutosuggestUnifiedCandidateKind::Mixed;
        existing.final_score = unified_candidate_score(existing, policy);
        return;
    }
    let mut unified = AutosuggestUnifiedCandidate {
        text: text.to_string(),
        token_id: Some(candidate.token_id),
        kind: AutosuggestUnifiedCandidateKind::GeneratedToken,
        static_rank: None,
        static_candidate: None,
        token_model_rank: Some(candidate.model_rank),
        token_model_score: Some(finite_score(candidate.model_score)),
        text_model_rank: None,
        text_model_score: None,
        final_score: 0.0,
    };
    unified.final_score = unified_candidate_score(&unified, policy);
    output.push(unified);
}

fn push_or_merge_generated_text_unified(
    candidate: &AutosuggestValidatedTextCandidate,
    policy: AutosuggestOpenVocabPolicy,
    output: &mut Vec<AutosuggestUnifiedCandidate>,
) {
    if let Some(existing) = output
        .iter_mut()
        .find(|existing| existing.text == candidate.text)
    {
        existing.text_model_rank = Some(candidate.model_rank);
        existing.text_model_score = Some(finite_score(candidate.model_score));
        existing.kind = AutosuggestUnifiedCandidateKind::Mixed;
        existing.final_score = unified_candidate_score(existing, policy);
        return;
    }
    let mut unified = AutosuggestUnifiedCandidate {
        text: candidate.text.clone(),
        token_id: None,
        kind: AutosuggestUnifiedCandidateKind::GeneratedText,
        static_rank: None,
        static_candidate: None,
        token_model_rank: None,
        token_model_score: None,
        text_model_rank: Some(candidate.model_rank),
        text_model_score: Some(finite_score(candidate.model_score)),
        final_score: 0.0,
    };
    unified.final_score = unified_candidate_score(&unified, policy);
    output.push(unified);
}

fn unified_candidate_score(
    candidate: &AutosuggestUnifiedCandidate,
    policy: AutosuggestOpenVocabPolicy,
) -> f32 {
    let mut score = f32::NEG_INFINITY;
    if let (Some(rank), Some(static_candidate)) =
        (candidate.static_rank, candidate.static_candidate)
    {
        score = score.max(static_candidate_score(rank, static_candidate, policy));
    }
    if let (Some(rank), Some(model_score)) =
        (candidate.token_model_rank, candidate.token_model_score)
    {
        score = score.max(
            finite_score(model_score)
                - policy.generated_token_penalty
                - policy.generated_token_rank_penalty * rank as f32,
        );
    }
    if let (Some(rank), Some(model_score)) = (candidate.text_model_rank, candidate.text_model_score)
    {
        score = score.max(
            finite_score(model_score)
                - policy.generated_text_penalty
                - policy.generated_text_rank_penalty * rank as f32,
        );
    }

    let signal_count = candidate.static_candidate.iter().count()
        + candidate.token_model_rank.iter().count()
        + candidate.text_model_rank.iter().count();
    if signal_count > 1 {
        score += policy.overlap_bonus * (signal_count as f32 - 1.0);
    }
    score
}

fn static_candidate_score(
    rank: usize,
    candidate: AutosuggestCandidateId,
    policy: AutosuggestOpenVocabPolicy,
) -> f32 {
    policy.static_bonus - policy.static_rank_penalty * rank as f32
        + policy.static_log_count_scale * (candidate.count as f32).ln_1p()
        + policy.static_source_bonus * static_source_order(candidate.source)
        + if rank < policy.locked_static_prefix {
            OPEN_VOCAB_LOCKED_STATIC_BONUS
        } else {
            0.0
        }
}

fn static_source_order(source: AutosuggestSource) -> f32 {
    match source {
        AutosuggestSource::Personal => 4.0,
        AutosuggestSource::Fourgram => 3.0,
        AutosuggestSource::Trigram => 2.0,
        AutosuggestSource::Bigram => 1.0,
        AutosuggestSource::Unigram => 0.0,
    }
}

fn unified_candidate_order(
    left: &AutosuggestUnifiedCandidate,
    right: &AutosuggestUnifiedCandidate,
) -> std::cmp::Ordering {
    right
        .final_score
        .total_cmp(&left.final_score)
        .then_with(|| {
            left.static_rank
                .unwrap_or(usize::MAX)
                .cmp(&right.static_rank.unwrap_or(usize::MAX))
        })
        .then_with(|| {
            left.token_model_rank
                .unwrap_or(usize::MAX)
                .cmp(&right.token_model_rank.unwrap_or(usize::MAX))
        })
        .then_with(|| {
            left.text_model_rank
                .unwrap_or(usize::MAX)
                .cmp(&right.text_model_rank.unwrap_or(usize::MAX))
        })
        .then_with(|| left.text.cmp(&right.text))
}

fn rejected(
    char_len: usize,
    rejection: AutosuggestOpenVocabRejectionKind,
) -> AutosuggestOpenVocabValidationReport {
    AutosuggestOpenVocabValidationReport {
        accepted: false,
        char_len,
        rejection: Some(rejection),
    }
}

fn finite_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        f32::NEG_INFINITY
    }
}

fn is_bengali_word_scalar(scalar: char) -> bool {
    is_bengali_letter(scalar) || is_bengali_mark(scalar)
}

fn is_bengali_letter(scalar: char) -> bool {
    matches!(
        scalar as u32,
        0x0985..=0x098C
            | 0x098F..=0x0990
            | 0x0993..=0x09A8
            | 0x09AA..=0x09B0
            | 0x09B2
            | 0x09B6..=0x09B9
            | 0x09CE
            | 0x09DC..=0x09DD
            | 0x09DF..=0x09E1
            | 0x09F0..=0x09F1
    )
}

fn is_bengali_mark(scalar: char) -> bool {
    matches!(
        scalar as u32,
        0x0981..=0x0983 | 0x09BC | 0x09BE..=0x09C4 | 0x09C7..=0x09C8 | 0x09CB..=0x09CD | 0x09D7
    )
}

fn is_bengali_digit(scalar: char) -> bool {
    matches!(scalar as u32, 0x09E6..=0x09EF)
}

fn is_bengali_punctuation(scalar: char) -> bool {
    matches!(scalar as u32, 0x0964..=0x0965)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(token_id: u32, source: AutosuggestSource, count: u32) -> AutosuggestCandidateId {
        AutosuggestCandidateId {
            token_id,
            source,
            count,
            score: 0,
        }
    }

    #[test]
    fn validator_accepts_valid_bengali_words_without_dictionary_lookup() {
        let policy = AutosuggestOpenVocabPolicy::default();
        for word in ["গিয়েছিলাম", "রিয়াদ", "র\u{200d}্যাব", "করছিলাম"]
        {
            let report = validate_open_vocab_text(word, 1.0, policy);
            assert_eq!(report.rejection, None, "{word}");
            assert!(report.accepted, "{word}");
        }
    }

    #[test]
    fn validator_rejects_non_word_and_malformed_output() {
        let policy = AutosuggestOpenVocabPolicy::default();
        let cases = [
            ("", AutosuggestOpenVocabRejectionKind::Empty),
            (
                " hello",
                AutosuggestOpenVocabRejectionKind::LeadingOrTrailingWhitespace,
            ),
            (
                "hello",
                AutosuggestOpenVocabRejectionKind::ContainsNonBanglaScalar,
            ),
            ("বাংলা১", AutosuggestOpenVocabRejectionKind::ContainsDigit),
            (
                "বাংলা।",
                AutosuggestOpenVocabRejectionKind::ContainsPunctuation,
            ),
            ("াকা", AutosuggestOpenVocabRejectionKind::StartsWithMark),
            ("ক্", AutosuggestOpenVocabRejectionKind::EndsWithHasant),
            ("ক্া", AutosuggestOpenVocabRejectionKind::MarkAfterHasant),
            // A joiner between the hasant and the kar must not launder it.
            ("ক্\u{200d}া", AutosuggestOpenVocabRejectionKind::MarkAfterHasant),
            ("ক্\u{200c}া", AutosuggestOpenVocabRejectionKind::MarkAfterHasant),
            ("কককক", AutosuggestOpenVocabRejectionKind::RepeatedScalarRun),
        ];
        for (word, rejection) in cases {
            let report = validate_open_vocab_text(word, 1.0, policy);
            assert!(!report.accepted, "{word}");
            assert_eq!(report.rejection, Some(rejection), "{word}");
        }
    }

    #[test]
    fn accept_open_vocab_texts_dedupes_and_keeps_better_signal() {
        let mut output = Vec::new();
        accept_open_vocab_texts_into(
            &["গেলাম", "গেলাম", "hello", "করছিলাম"],
            &[0.1, 0.8, 9.0, 0.7],
            AutosuggestOpenVocabPolicy {
                max_candidates: 4,
                ..AutosuggestOpenVocabPolicy::default()
            },
            &mut output,
        )
        .unwrap();

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].text, "গেলাম");
        assert_eq!(output[0].model_rank, 0);
        assert_eq!(output[0].model_score, 0.8);
        assert_eq!(output[1].text, "করছিলাম");
    }

    #[test]
    fn unified_score_combines_static_token_and_open_vocab_signals() {
        let mut candidate = AutosuggestUnifiedCandidate {
            text: "গেলাম".to_string(),
            token_id: Some(5),
            kind: AutosuggestUnifiedCandidateKind::Mixed,
            static_rank: Some(1),
            static_candidate: Some(candidate(5, AutosuggestSource::Bigram, 10)),
            token_model_rank: Some(0),
            token_model_score: Some(1.0),
            text_model_rank: Some(0),
            text_model_score: Some(1.1),
            final_score: 0.0,
        };
        let without_overlap = AutosuggestOpenVocabPolicy {
            overlap_bonus: 0.0,
            locked_static_prefix: 0,
            ..AutosuggestOpenVocabPolicy::default()
        };
        let with_overlap = AutosuggestOpenVocabPolicy {
            overlap_bonus: 2.0,
            locked_static_prefix: 0,
            ..AutosuggestOpenVocabPolicy::default()
        };

        let base = unified_candidate_score(&candidate, without_overlap);
        candidate.final_score = unified_candidate_score(&candidate, with_overlap);

        assert!(candidate.final_score > base);
    }
}
