use std::collections::{btree_map::Entry, BTreeMap};
use std::error::Error;
use std::fmt;

use fst::automaton::{Automaton, Str};
use fst::{IntoStreamer, Streamer};
use serde::Serialize;

use super::bangla::{
    differs_only_by_nasal_or_breath_mark, differs_only_by_vowel_length,
    for_each_chandrabindu_variant, has_bengali_letter,
};
use super::edit::weighted_edit_distance;
use super::edit_row::BoundedEditRow;
use super::morphology::{stem_suffix_completions, StemSuffixCompletion};

pub const DEFAULT_FST_MAX_DISTANCE: u32 = 1;
pub const DEFAULT_FST_PREFIX_CANDIDATES: usize = 24;
pub const FST_MAX_LEVENSHTEIN_DISTANCE: u32 = 2;
const SCORE_FREQUENCY_SCALE: f64 = 1024.0;
const SURFACE_EDIT_COST_SCALE: i64 = 1536;
// Channel priors express evidence strength from the input path; they are not
// claims that the deterministic baseline is always the intended word.
const EXACT_BASELINE_CHANNEL_PRIOR: i64 = 16_384;
const EDIT_DISTANCE_CHANNEL_PENALTY: i64 = 2048;
const DIACRITIC_EDIT_CHANNEL_PRIOR: i64 = 12_288;
const DIACRITIC_EDIT_COST_SCALE: i64 = 256;
const ORTHOGRAPHIC_VOWEL_LENGTH_CHANNEL_PRIOR: i64 = 12_288;
const ORTHOGRAPHIC_VOWEL_LENGTH_COST_SCALE: i64 = 64;
const PREFIX_COMPLETION_CHANNEL_PENALTY: i64 = 192;
const STEM_SUFFIX_COMPLETION_PRIOR: i64 = 768;
const STEM_SUFFIX_COST_SCALE: i64 = 256;
const STEM_SUFFIX_SURFACE_COST_SCALE: i64 = 64;
// Skeleton (dropped-vowel) channel. Its prior sits far below the exact baseline prior
// (16_384), so a real word the user typed correctly always outranks any skeleton sibling
// — the channel only wins when the baseline is not itself a confident word.
const SKELETON_VOWEL_DROP_PRIOR: i64 = 4_096;
const SKELETON_VOWEL_DROP_COST_SCALE: i64 = 256;
// Only run the skeleton walk when the baseline is a weak signal (not a lexicon word,
// or a rare one) — an efficiency/noise gate; precision is guaranteed by the prior above.
const SKELETON_BASELINE_FREQUENCY_FLOOR: u64 = 2_048;
const SKELETON_MATCH_LIMIT: usize = 16;
// Bound on words collected from one skeleton walk before frequency ranking, so a very short
// (highly ambiguous) skeleton cannot make a single lookup unbounded.
const SKELETON_MATCH_SCAN_CAP: usize = 512;
// Consonant-confusion channel: substitute one baseline consonant with a near phoneme
// (graded by articulatory distance) and keep the real-word results. Prior is high (a
// same-sound spelling fix is high-precision when the sibling is a frequent word) but still
// below the exact prior, and each candidate pays its phoneme distance — so a correctly
// spelled word (whose confusion siblings are rare) is not demoted.
const CONSONANT_CONFUSION_PRIOR: i64 = 12_288;
const CONSONANT_CONFUSION_DISTANCE_SCALE: i64 = 512;
const CONSONANT_CONFUSION_MAX_DISTANCE: u16 = 4;
const CONFUSION_BASELINE_FREQUENCY_FLOOR: u64 = 2_048;
const ROMAN_SEPARATOR_REPAIR_EXACT_BONUS: i64 = 4_096;
const ROMAN_ASPIRATED_SPLIT_REPAIR_EXACT_BONUS: i64 = 2_048;
const ROMAN_FLAP_REPAIR_EXACT_BONUS: i64 = 2_048;
const ROMAN_PALATAL_NASAL_REPAIR_EXACT_BONUS: i64 = 768;
const ROMAN_VELAR_NASAL_REPAIR_EXACT_BONUS: i64 = 2_048;
// QWERTY fat-finger (key-slip) repair. Deliberately conservative: it roughly offsets a
// single nearest-key slip's cost so a plausible fat-finger to a real word ranks near that
// word's own frequency, but it never gets the higher bonus a deterministic linguistic repair
// earns. The channel is also gated to non-word baselines, so a validly-typed word is untouched.
const ROMAN_KEY_SLIP_REPAIR_EXACT_BONUS: i64 = 1_024;
const ROMAN_STRONG_VELAR_NASAL_REPAIR_EXACT_BONUS: i64 = 12_288;
const ROMAN_STRONG_REPAIR_FREQUENCY_FLOOR: u64 = 128;
const ROMAN_REPAIR_COST_SCALE: i64 = 1024;
const ROMAN_REPAIR_SURFACE_COST_SCALE: i64 = 128;
const ENGLISH_LOANWORD_EXACT_PRIOR: i64 = 10_240;
const ENGLISH_LOANWORD_FUZZY_PRIOR: i64 = 8_704;
const ENGLISH_LOANWORD_REPAIR_COST_SCALE: i64 = 1536;
const ENGLISH_LOANWORD_SURFACE_COST_SCALE: i64 = 64;
const ENGLISH_LOANWORD_PREFIX_TRUNCATION_PENALTY: i64 = 12_288;

#[derive(Debug)]
pub struct FstLexicon<D: AsRef<[u8]> = Vec<u8>> {
    map: fst::Map<D>,
}

impl<D: AsRef<[u8]>> FstLexicon<D> {
    pub fn from_map(map: fst::Map<D>) -> Self {
        Self { map }
    }

    /// Content fingerprint of the backing FST bytes. A downstream pinning
    /// `bn.fst` compares this against the value published for its crate version
    /// to catch a stale or mismatched artifact. Matches
    /// [`AutosuggestLm::artifact_fingerprint`](crate::AutosuggestLm::artifact_fingerprint).
    /// See [`crate::fingerprint`].
    pub fn artifact_fingerprint(&self) -> u64 {
        crate::fingerprint::artifact_fingerprint(self.map.as_fst().as_bytes())
    }

    /// Lexicon words sharing `baseline`'s consonant skeleton, highest-frequency first, up to
    /// `limit`. This reads skeleton-mates directly out of the lexicon fst via
    /// [`SkeletonAutomaton`] — no separate index — so it always sees the full vocabulary with
    /// real frequencies. Empty when the skeleton is not in the indexable length band.
    ///
    /// Cost: one automaton walk of the fst that dies the instant a consonant diverges from
    /// the skeleton, so it visits only the shared subtree of actual skeleton-mates.
    fn skeleton_matches(
        &self,
        baseline: &str,
        limit: usize,
    ) -> Vec<super::skeleton::SkeletonMatch> {
        use super::skeleton::{SkeletonAutomaton, SkeletonMatch};

        if limit == 0 {
            return Vec::new();
        }
        let Some(automaton) = SkeletonAutomaton::for_baseline(baseline) else {
            return Vec::new();
        };

        let mut stream = self.map.search(automaton).into_stream();
        let mut collected: Vec<SkeletonMatch> = Vec::new();
        while let Some((key, frequency)) = stream.next() {
            if collected.len() >= SKELETON_MATCH_SCAN_CAP {
                break;
            }
            if let Ok(word) = std::str::from_utf8(key) {
                collected.push(SkeletonMatch {
                    word: word.to_string(),
                    frequency,
                });
            }
        }
        collected.sort_by(|a, b| {
            b.frequency
                .cmp(&a.frequency)
                .then_with(|| a.word.cmp(&b.word))
        });
        collected.truncate(limit);
        collected
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn exact_frequency(&self, word: &str) -> Option<u64> {
        self.map.get(word.as_bytes())
    }

    pub fn suggest(
        &self,
        baseline: &str,
        options: FstSuggestOptions,
    ) -> Result<FstSuggestResult, FstSuggestError> {
        self.suggest_with_repaired_baselines(baseline, &[], options)
    }

    pub fn suggest_with_repaired_baselines(
        &self,
        baseline: &str,
        repaired_baselines: &[FstRepairedBaseline<'_>],
        options: FstSuggestOptions,
    ) -> Result<FstSuggestResult, FstSuggestError> {
        self.suggest_with_repaired_baselines_and_loanwords(
            baseline,
            repaired_baselines,
            &[],
            options,
        )
    }

    pub fn suggest_with_repaired_baselines_and_loanwords(
        &self,
        baseline: &str,
        repaired_baselines: &[FstRepairedBaseline<'_>],
        loanword_matches: &[FstLoanwordMatch<'_>],
        options: FstSuggestOptions,
    ) -> Result<FstSuggestResult, FstSuggestError> {
        if options.max_distance > FST_MAX_LEVENSHTEIN_DISTANCE {
            return Err(FstSuggestError::MaxDistanceTooLarge {
                requested: options.max_distance,
                max: FST_MAX_LEVENSHTEIN_DISTANCE,
            });
        }

        let exact_frequency = self.exact_frequency(baseline);
        let retrieval_limit = options
            .max_candidates
            .max(options.response_candidates)
            .max(1);
        let mut seeds = BTreeMap::<String, FstCandidate>::new();
        let mut truncated = false;

        if let Some(frequency) = exact_frequency {
            insert_fst_candidate(
                &mut seeds,
                baseline,
                frequency,
                FstCandidateSource::Exact,
                baseline,
                options.max_edit_cost,
            );

            for completion in stem_suffix_completions(baseline) {
                if let Some(frequency) = self.exact_frequency(&completion.text) {
                    insert_stem_suffix_candidate(&mut seeds, &completion, frequency, baseline);
                }
            }

            for_each_chandrabindu_variant(baseline, |variant| {
                if let Some(frequency) = self.exact_frequency(variant) {
                    insert_diacritic_candidate(&mut seeds, variant, frequency, baseline);
                }
            });
        }

        // Edit-distance retrieval only makes sense for a word-like baseline. A token
        // with no Bangla letter — punctuation, digits, symbols (`,` `।` `১` `ঁ`) — is not
        // a misspelling of any lexicon word, so its edit neighbourhood is just the
        // shortest, most frequent entries surfacing as noise (issue #34: `,` pulling in
        // ও/এ/অং). The grounded channels above (exact, roman-repair, loanword) already
        // captured anything real; prefix needs an entry that starts with the token and
        // the skeleton/consonant channels need consonants, so none of them fire here.
        if has_bengali_letter(baseline) {
            let levenshtein = UnicodeLevenshtein::new(baseline, options.max_distance);
            let mut edit_stream = self.map.search(levenshtein).into_stream();
            while let Some((key, frequency)) = edit_stream.next() {
                let text = std::str::from_utf8(key).map_err(FstSuggestError::InvalidUtf8)?;
                insert_fst_candidate(
                    &mut seeds,
                    text,
                    frequency,
                    FstCandidateSource::EditDistance,
                    baseline,
                    options.max_edit_cost,
                );
                if seeds.len() >= retrieval_limit {
                    truncated = true;
                    break;
                }
            }
        }

        if options.max_prefix_candidates > 0 && seeds.len() < retrieval_limit {
            let prefix = Str::new(baseline).starts_with();
            let mut prefix_stream = self.map.search(prefix).into_stream();
            let mut prefix_count = 0_usize;
            while let Some((key, frequency)) = prefix_stream.next() {
                let text = std::str::from_utf8(key).map_err(FstSuggestError::InvalidUtf8)?;
                insert_fst_candidate(
                    &mut seeds,
                    text,
                    frequency,
                    FstCandidateSource::PrefixCompletion,
                    baseline,
                    options.max_edit_cost,
                );
                prefix_count += 1;
                if prefix_count >= options.max_prefix_candidates || seeds.len() >= retrieval_limit {
                    truncated = seeds.len() >= retrieval_limit;
                    break;
                }
            }
        }

        for repair in repaired_baselines {
            if repair.repair_cost == 0 || repair.bangla_output == baseline {
                continue;
            }

            if let Some(frequency) = self.exact_frequency(repair.bangla_output) {
                insert_repaired_baseline_candidate(&mut seeds, repair, frequency, baseline);
            }

            for_each_chandrabindu_variant(repair.bangla_output, |variant| {
                if let Some(frequency) = self.exact_frequency(variant) {
                    insert_repaired_baseline_text_candidate(
                        &mut seeds,
                        repair.roman_input,
                        repair.repair_kind,
                        repair.repair_cost,
                        variant,
                        frequency,
                        baseline,
                    );
                }
            });
        }

        for loanword in loanword_matches {
            if loanword.bangla_output.is_empty() {
                continue;
            }
            if exact_frequency.is_some() && loanword.repair_cost > 0 {
                continue;
            }
            let frequency = self
                .exact_frequency(loanword.bangla_output)
                .unwrap_or(loanword.frequency);
            insert_english_loanword_candidate(&mut seeds, loanword, frequency, baseline);
        }

        // The recovery channels below (skeleton, consonant-confusion) are heuristic guesses.
        // They must never outrank a CONFIDENT reading of the input — an exact lexicon word, a
        // cheap roman-repair to one, or an exact loanword. Cap them just below the best such
        // candidate already collected; when none exists they are the only recovery and rank
        // freely (e.g. korlm → করলাম, where nothing else explains the input).
        let confident_ceiling = seeds
            .values()
            .filter(|candidate| candidate.source.is_confident())
            .map(|candidate| candidate.score)
            .max();

        // Skeleton (dropped-vowel) channel: when the baseline is a weak signal (not a
        // lexicon word, or a rare one), pull real words sharing its consonant skeleton
        // straight out of the lexicon fst (via the skeleton automaton — no separate index)
        // and rank them by corpus frequency.
        let baseline_is_weak =
            exact_frequency.map_or(true, |freq| freq < SKELETON_BASELINE_FREQUENCY_FLOOR);
        if baseline_is_weak {
            for candidate in self.skeleton_matches(baseline, SKELETON_MATCH_LIMIT) {
                if candidate.word == baseline {
                    continue;
                }
                insert_skeleton_candidate(
                    &mut seeds,
                    &candidate.word,
                    candidate.frequency,
                    baseline,
                    confident_ceiling,
                    options.max_edit_cost,
                );
            }
        }

        // Consonant-confusion channel: substitute one baseline consonant with a near
        // phoneme and keep the real-word results. Gated to weak baselines (not a confident
        // word) so it never floods the ribbon of a correctly spelled input; among fired
        // candidates the prior + phoneme distance still let only a far-more-frequent
        // same-sound sibling win.
        let baseline_is_weak =
            exact_frequency.map_or(true, |freq| freq < CONFUSION_BASELINE_FREQUENCY_FLOOR);
        if baseline_is_weak {
            let chars: Vec<(usize, char)> = baseline.char_indices().collect();
            for (position, &(byte_index, ch)) in chars.iter().enumerate() {
                // Skip phota-form base consonants (ড়/ঢ়/য়); they are whole units.
                if chars
                    .get(position + 1)
                    .is_some_and(|&(_, next)| next == '\u{09BC}')
                {
                    continue;
                }
                for (near, distance) in
                    super::phoneme::near_consonants(ch, CONSONANT_CONFUSION_MAX_DISTANCE)
                {
                    let mut variant = String::with_capacity(baseline.len() + 3);
                    variant.push_str(&baseline[..byte_index]);
                    variant.push(near);
                    variant.push_str(&baseline[byte_index + ch.len_utf8()..]);
                    if let Some(frequency) = self.exact_frequency(&variant) {
                        insert_confusion_candidate(
                            &mut seeds,
                            variant,
                            frequency,
                            distance,
                            baseline,
                            confident_ceiling,
                            options.max_edit_cost,
                        );
                    }
                }
            }
        }

        let mut ranked = seeds.into_values().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.edit_cost.cmp(&right.edit_cost))
                .then_with(|| right.frequency.cmp(&left.frequency))
                .then_with(|| left.text.cmp(&right.text))
        });
        let candidate_count = ranked.len();
        ranked.truncate(options.response_candidates);

        Ok(FstSuggestResult {
            baseline: baseline.to_string(),
            exact_frequency,
            max_distance: options.max_distance,
            max_edit_cost: options.max_edit_cost,
            candidate_count,
            returned_candidates: ranked.len(),
            truncated,
            candidates: ranked,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FstRepairedBaseline<'a> {
    pub roman_input: &'a str,
    pub bangla_output: &'a str,
    pub repair_kind: &'static str,
    pub repair_cost: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct FstLoanwordMatch<'a> {
    pub roman_input: &'a str,
    pub roman_repair: &'a str,
    pub bangla_output: &'a str,
    pub frequency: u64,
    pub repair_kind: &'static str,
    pub repair_cost: u16,
}

impl FstLexicon<Vec<u8>> {
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, fst::Error> {
        Ok(Self::from_map(fst::Map::new(bytes)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FstSuggestOptions {
    pub max_distance: u32,
    pub max_edit_cost: Option<u16>,
    pub max_candidates: usize,
    pub max_prefix_candidates: usize,
    pub response_candidates: usize,
}

impl Default for FstSuggestOptions {
    fn default() -> Self {
        Self {
            max_distance: DEFAULT_FST_MAX_DISTANCE,
            max_edit_cost: None,
            max_candidates: 512,
            max_prefix_candidates: DEFAULT_FST_PREFIX_CANDIDATES,
            response_candidates: DEFAULT_FST_PREFIX_CANDIDATES,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FstSuggestResult {
    pub baseline: String,
    pub exact_frequency: Option<u64>,
    pub max_distance: u32,
    pub max_edit_cost: Option<u16>,
    pub candidate_count: usize,
    pub returned_candidates: usize,
    pub truncated: bool,
    pub candidates: Vec<FstCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FstCandidate {
    pub text: String,
    pub source: FstCandidateSource,
    pub edit_cost: u16,
    pub frequency: u64,
    pub score: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roman_repair: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roman_repair_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roman_repair_cost: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive] // new suggestion channels may add variants without a breaking change
pub enum FstCandidateSource {
    Exact,
    EditDistance,
    DiacriticEdit,
    OrthographicVowelLengthEdit,
    PrefixCompletion,
    StemSuffixCompletion,
    SkeletonVowelDrop,
    ConsonantConfusion,
    RomanRepairExact,
    EnglishLoanwordExact,
    EnglishLoanwordFuzzy,
    /// Exact emoticon alternative; never scored as a spelling correction.
    EmoticonExact,
}

impl FstCandidateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "fst_exact",
            Self::EditDistance => "fst_edit_distance",
            Self::DiacriticEdit => "fst_diacritic_edit",
            Self::OrthographicVowelLengthEdit => "fst_orthographic_vowel_length_edit",
            Self::PrefixCompletion => "fst_prefix_completion",
            Self::StemSuffixCompletion => "fst_stem_suffix_completion",
            Self::SkeletonVowelDrop => "fst_skeleton_vowel_drop",
            Self::ConsonantConfusion => "fst_consonant_confusion",
            Self::RomanRepairExact => "fst_roman_repair_exact",
            Self::EnglishLoanwordExact => "fst_english_loanword_exact",
            Self::EnglishLoanwordFuzzy => "fst_english_loanword_fuzzy",
            Self::EmoticonExact => "emoticon_exact",
        }
    }

    /// Stable numeric code for the C ABI (`obadh_autocorrect_suggest_detailed`).
    ///
    /// **Frozen and append-only:** a downstream branches on these values, so a
    /// code is never renumbered and a new channel only ever takes the next
    /// unused number. Callers must treat an unknown code conservatively (not
    /// eligible for auto-replace), which lets a future channel be added without
    /// a coordinated release.
    pub fn stable_code(self) -> u8 {
        match self {
            Self::Exact => 0,
            Self::EditDistance => 1,
            Self::DiacriticEdit => 2,
            Self::OrthographicVowelLengthEdit => 3,
            Self::PrefixCompletion => 4,
            Self::StemSuffixCompletion => 5,
            Self::SkeletonVowelDrop => 6,
            Self::ConsonantConfusion => 7,
            Self::RomanRepairExact => 8,
            Self::EnglishLoanwordExact => 9,
            Self::EnglishLoanwordFuzzy => 10,
            Self::EmoticonExact => 11,
        }
    }

    /// Provenance authority for dedup. The skeleton and consonant-confusion channels are
    /// heuristic recovery generators (0); every direct channel — exact, edit, roman-repair,
    /// loanword — is authoritative (1). When two channels yield the same word, the word keeps
    /// the more authoritative provenance (see `insert_best_candidate`), so a recovery guess
    /// can never relabel a confidently-found word.
    fn authority(self) -> u8 {
        match self {
            Self::SkeletonVowelDrop | Self::ConsonantConfusion => 0,
            _ => 1,
        }
    }

    /// A *confident* reading of the input — one that resolves to a specific exact lexicon
    /// word (typed directly, reached by a cheap roman-repair, or an exact loanword). The
    /// recovery channels (skeleton, consonant-confusion) are capped below the best confident
    /// candidate so a heuristic guess can never outrank a word the input actually spells.
    fn is_confident(self) -> bool {
        matches!(
            self,
            Self::Exact
                | Self::RomanRepairExact
                | Self::EnglishLoanwordExact
                | Self::DiacriticEdit
                | Self::OrthographicVowelLengthEdit
                | Self::StemSuffixCompletion
        )
    }

    fn prior(self) -> i64 {
        match self {
            Self::Exact => EXACT_BASELINE_CHANNEL_PRIOR,
            Self::EditDistance => -EDIT_DISTANCE_CHANNEL_PENALTY,
            Self::DiacriticEdit => DIACRITIC_EDIT_CHANNEL_PRIOR,
            Self::OrthographicVowelLengthEdit => ORTHOGRAPHIC_VOWEL_LENGTH_CHANNEL_PRIOR,
            Self::PrefixCompletion => -PREFIX_COMPLETION_CHANNEL_PENALTY,
            Self::StemSuffixCompletion => STEM_SUFFIX_COMPLETION_PRIOR,
            Self::SkeletonVowelDrop => SKELETON_VOWEL_DROP_PRIOR,
            Self::ConsonantConfusion => CONSONANT_CONFUSION_PRIOR,
            Self::RomanRepairExact => 0,
            Self::EnglishLoanwordExact => ENGLISH_LOANWORD_EXACT_PRIOR,
            Self::EnglishLoanwordFuzzy => ENGLISH_LOANWORD_FUZZY_PRIOR,
            Self::EmoticonExact => 0,
        }
    }
}

#[derive(Debug)]
pub enum FstSuggestError {
    MaxDistanceTooLarge { requested: u32, max: u32 },
    InvalidUtf8(std::str::Utf8Error),
}

impl fmt::Display for FstSuggestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxDistanceTooLarge { requested, max } => write!(
                formatter,
                "max-distance {requested} is too large for keyboard-time FST lookup; use {max} or less"
            ),
            Self::InvalidUtf8(error) => write!(formatter, "invalid UTF-8 key in FST lexicon: {error}"),
        }
    }
}

impl Error for FstSuggestError {}

fn insert_fst_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    text: &str,
    frequency: u64,
    source: FstCandidateSource,
    baseline: &str,
    max_edit_cost: Option<u16>,
) {
    let edit_cost = weighted_edit_distance(baseline, text).0;
    if max_edit_cost.is_some_and(|limit| edit_cost > limit) {
        return;
    }
    let source = if source == FstCandidateSource::EditDistance
        && differs_only_by_nasal_or_breath_mark(baseline, text)
    {
        FstCandidateSource::DiacriticEdit
    } else if source == FstCandidateSource::EditDistance
        && differs_only_by_vowel_length(baseline, text)
    {
        FstCandidateSource::OrthographicVowelLengthEdit
    } else {
        source
    };
    let score = fst_candidate_score(source, edit_cost, frequency);
    let candidate = FstCandidate {
        text: text.to_string(),
        source,
        edit_cost,
        frequency,
        score,
        roman_repair: None,
        roman_repair_kind: None,
        roman_repair_cost: None,
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_repaired_baseline_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    repair: &FstRepairedBaseline<'_>,
    frequency: u64,
    baseline: &str,
) {
    insert_repaired_baseline_text_candidate(
        seeds,
        repair.roman_input,
        repair.repair_kind,
        repair.repair_cost,
        repair.bangla_output,
        frequency,
        baseline,
    );
}

fn insert_repaired_baseline_text_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    roman_input: &str,
    repair_kind: &'static str,
    repair_cost: u16,
    bangla_output: &str,
    frequency: u64,
    baseline: &str,
) {
    let edit_cost = weighted_edit_distance(baseline, bangla_output).0;
    let score = roman_repair_candidate_score(edit_cost, repair_cost, repair_kind, frequency);
    let candidate = FstCandidate {
        text: bangla_output.to_string(),
        source: FstCandidateSource::RomanRepairExact,
        edit_cost,
        frequency,
        score,
        roman_repair: Some(roman_input.to_string()),
        roman_repair_kind: Some(repair_kind),
        roman_repair_cost: Some(repair_cost),
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_stem_suffix_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    completion: &StemSuffixCompletion,
    frequency: u64,
    baseline: &str,
) {
    let edit_cost = weighted_edit_distance(baseline, &completion.text).0;
    let score = stem_suffix_candidate_score(edit_cost, completion.cost, frequency);
    let candidate = FstCandidate {
        text: completion.text.clone(),
        source: FstCandidateSource::StemSuffixCompletion,
        edit_cost,
        frequency,
        score,
        roman_repair: None,
        roman_repair_kind: None,
        roman_repair_cost: None,
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_english_loanword_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    loanword: &FstLoanwordMatch<'_>,
    frequency: u64,
    baseline: &str,
) {
    let edit_cost = weighted_edit_distance(baseline, loanword.bangla_output).0;
    let source = if loanword.repair_cost == 0 {
        FstCandidateSource::EnglishLoanwordExact
    } else {
        FstCandidateSource::EnglishLoanwordFuzzy
    };
    let mut score = english_loanword_candidate_score(
        source,
        edit_cost,
        loanword.repair_cost,
        frequency,
        loanword.roman_input,
        loanword.roman_repair,
    );
    if source == FstCandidateSource::EnglishLoanwordFuzzy
        && english_loanword_prefix_truncation_penalty(loanword.roman_input, loanword.roman_repair)
            > 0
    {
        // Dropping a typed suffix is weaker evidence than a closer Bangla edit.
        // Preserve that ordering independently of frequency and cost-scale
        // tuning. Repeated final-key typos are exempt from the truncation rule.
        if let Some(ceiling) = seeds
            .values()
            .filter(|candidate| {
                candidate.source == FstCandidateSource::EditDistance
                    && candidate.edit_cost < edit_cost
            })
            .map(|candidate| candidate.score)
            .max()
        {
            score = score.min(ceiling.saturating_sub(1));
        }
    }
    let candidate = FstCandidate {
        text: loanword.bangla_output.to_string(),
        source,
        edit_cost,
        frequency,
        score,
        roman_repair: Some(loanword.roman_repair.to_string()),
        roman_repair_kind: Some(loanword.repair_kind),
        roman_repair_cost: Some(loanword.repair_cost),
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_diacritic_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    text: &str,
    frequency: u64,
    baseline: &str,
) {
    let edit_cost = weighted_edit_distance(baseline, text).0;
    let score = fst_candidate_score(FstCandidateSource::DiacriticEdit, edit_cost, frequency);
    let candidate = FstCandidate {
        text: text.to_string(),
        source: FstCandidateSource::DiacriticEdit,
        edit_cost,
        frequency,
        score,
        roman_repair: None,
        roman_repair_kind: None,
        roman_repair_cost: None,
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_skeleton_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    text: &str,
    frequency: u64,
    baseline: &str,
    confident_ceiling: Option<i64>,
    max_edit_cost: Option<u16>,
) {
    let edit_cost = weighted_edit_distance(baseline, text).0;
    if max_edit_cost.is_some_and(|limit| edit_cost > limit) {
        return;
    }
    let mut score = frequency_score(frequency) + SKELETON_VOWEL_DROP_PRIOR
        - (edit_cost as i64 * SKELETON_VOWEL_DROP_COST_SCALE);
    // A skeleton match is a heuristic vowel-restoration guess; it must not outrank a confident
    // reading of the input (exact word / cheap roman-repair / exact loanword).
    if let Some(ceiling) = confident_ceiling {
        score = score.min(ceiling - 1);
    }
    let candidate = FstCandidate {
        text: text.to_string(),
        source: FstCandidateSource::SkeletonVowelDrop,
        edit_cost,
        frequency,
        score,
        roman_repair: None,
        roman_repair_kind: None,
        roman_repair_cost: None,
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_confusion_candidate(
    seeds: &mut BTreeMap<String, FstCandidate>,
    text: String,
    frequency: u64,
    phoneme_distance: u16,
    baseline: &str,
    confident_ceiling: Option<i64>,
    max_edit_cost: Option<u16>,
) {
    let edit_cost = weighted_edit_distance(baseline, &text).0;
    if max_edit_cost.is_some_and(|limit| edit_cost > limit) {
        return;
    }
    let mut score = frequency_score(frequency) + CONSONANT_CONFUSION_PRIOR
        - (phoneme_distance as i64 * CONSONANT_CONFUSION_DISTANCE_SCALE);
    // A confusion is a heuristic sound-based guess (be it a different-sound ট↔ত or a
    // same-sound শ↔ষ). It must not outrank a confident reading of the input — a typed word, a
    // deliberate roman-repair, an exact loanword — so cap it just below the best such
    // candidate. When none exists (e.g. মানুশ → মানুষ, a genuinely ambiguous same-sound
    // spelling with no confident competitor) the ceiling is absent and it ranks freely.
    if let Some(ceiling) = confident_ceiling {
        score = score.min(ceiling - 1);
    }
    let candidate = FstCandidate {
        text,
        source: FstCandidateSource::ConsonantConfusion,
        edit_cost,
        frequency,
        score,
        roman_repair: None,
        roman_repair_kind: None,
        roman_repair_cost: None,
    };
    insert_best_candidate(seeds, candidate);
}

fn insert_best_candidate(seeds: &mut BTreeMap<String, FstCandidate>, candidate: FstCandidate) {
    match seeds.entry(candidate.text.clone()) {
        Entry::Occupied(mut entry) => {
            // A word ranks by its best evidence (max score across channels) but is labelled by
            // the most authoritative channel that found it. Same authority → higher score wins.
            let existing_authority = entry.get().source.authority();
            let existing_score = entry.get().score;
            let incoming_authority = candidate.source.authority();
            let best_score = existing_score.max(candidate.score);
            let take_incoming = if incoming_authority != existing_authority {
                incoming_authority > existing_authority
            } else {
                candidate.score > existing_score
            };
            if take_incoming {
                let mut chosen = candidate;
                chosen.score = best_score;
                entry.insert(chosen);
            } else {
                entry.get_mut().score = best_score;
            }
        }
        Entry::Vacant(entry) => {
            entry.insert(candidate);
        }
    }
}

fn fst_candidate_score(source: FstCandidateSource, edit_cost: u16, frequency: u64) -> i64 {
    if source == FstCandidateSource::DiacriticEdit {
        return frequency_score(frequency) + source.prior()
            - (edit_cost as i64 * DIACRITIC_EDIT_COST_SCALE);
    }
    if source == FstCandidateSource::OrthographicVowelLengthEdit {
        return frequency_score(frequency) + source.prior()
            - (edit_cost as i64 * ORTHOGRAPHIC_VOWEL_LENGTH_COST_SCALE);
    }

    frequency_score(frequency) - (edit_cost as i64 * SURFACE_EDIT_COST_SCALE) + source.prior()
}

fn roman_repair_candidate_score(
    edit_cost: u16,
    repair_cost: u16,
    repair_kind: &'static str,
    frequency: u64,
) -> i64 {
    frequency_score(frequency) + roman_repair_exact_bonus(repair_kind, frequency)
        - (repair_cost as i64 * ROMAN_REPAIR_COST_SCALE)
        - (edit_cost as i64 * ROMAN_REPAIR_SURFACE_COST_SCALE)
}

fn roman_repair_exact_bonus(repair_kind: &str, frequency: u64) -> i64 {
    match repair_kind {
        "palatal_nasal_ja_from_ng" | "palatal_nasal_ja_from_nz" => {
            ROMAN_PALATAL_NASAL_REPAIR_EXACT_BONUS
        }
        "velar_nasal_from_ng" if frequency >= ROMAN_STRONG_REPAIR_FREQUENCY_FLOOR => {
            ROMAN_STRONG_VELAR_NASAL_REPAIR_EXACT_BONUS
        }
        "velar_nasal_from_ng" => ROMAN_VELAR_NASAL_REPAIR_EXACT_BONUS,
        "split_aspirated_consonant" => ROMAN_ASPIRATED_SPLIT_REPAIR_EXACT_BONUS,
        "lowercase_r_to_flap" => ROMAN_FLAP_REPAIR_EXACT_BONUS,
        "qwerty_key_slip" => ROMAN_KEY_SLIP_REPAIR_EXACT_BONUS,
        _ => ROMAN_SEPARATOR_REPAIR_EXACT_BONUS,
    }
}

fn stem_suffix_candidate_score(edit_cost: u16, suffix_cost: u16, frequency: u64) -> i64 {
    frequency_score(frequency) + FstCandidateSource::StemSuffixCompletion.prior()
        - (suffix_cost as i64 * STEM_SUFFIX_COST_SCALE)
        - (edit_cost as i64 * STEM_SUFFIX_SURFACE_COST_SCALE)
}

fn english_loanword_candidate_score(
    source: FstCandidateSource,
    edit_cost: u16,
    repair_cost: u16,
    frequency: u64,
    roman_input: &str,
    roman_repair: &str,
) -> i64 {
    let prefix_truncation_penalty = if source == FstCandidateSource::EnglishLoanwordFuzzy {
        english_loanword_prefix_truncation_penalty(roman_input, roman_repair)
    } else {
        0
    };

    frequency_score(frequency) + source.prior()
        - (repair_cost as i64 * ENGLISH_LOANWORD_REPAIR_COST_SCALE)
        - (edit_cost as i64 * ENGLISH_LOANWORD_SURFACE_COST_SCALE)
        - prefix_truncation_penalty
}

fn english_loanword_prefix_truncation_penalty(roman_input: &str, roman_repair: &str) -> i64 {
    let input = roman_input.as_bytes();
    let repair = roman_repair.as_bytes();
    if input.len() <= repair.len() || !input[..repair.len()].eq_ignore_ascii_case(repair) {
        return 0;
    }

    let omitted = &input[repair.len()..];
    let Some(last_repair_byte) = repair.last().map(u8::to_ascii_lowercase) else {
        return 0;
    };
    if omitted
        .iter()
        .all(|byte| byte.to_ascii_lowercase() == last_repair_byte)
    {
        return 0;
    }

    ENGLISH_LOANWORD_PREFIX_TRUNCATION_PENALTY
}

fn frequency_score(frequency: u64) -> i64 {
    ((frequency.saturating_add(1) as f64).ln() * SCORE_FREQUENCY_SCALE).round() as i64
}

#[derive(Debug, Clone)]
struct UnicodeLevenshtein {
    query: Vec<char>,
    distance: u8,
}

#[derive(Debug, Clone)]
struct UnicodeLevenshteinState {
    row: BoundedEditRow,
    pending: [u8; 4],
    pending_len: u8,
    expected_len: u8,
    dead: bool,
}

impl UnicodeLevenshtein {
    fn new(query: &str, distance: u32) -> Self {
        debug_assert!(distance <= FST_MAX_LEVENSHTEIN_DISTANCE);
        Self {
            query: query.chars().collect(),
            distance: distance as u8,
        }
    }

    fn start_row(&self) -> BoundedEditRow {
        BoundedEditRow::initial(self.query.len(), self.distance + 1)
    }

    fn accept_char(&self, row: &[u8], ch: char) -> BoundedEditRow {
        let mut next = BoundedEditRow::with_capacity(self.query.len() + 1);
        next.push((row[0] + 1).min(self.distance + 1));
        for (index, query_ch) in self.query.iter().enumerate() {
            let substitution_cost = u8::from(*query_ch != ch);
            let substitution = row[index] + substitution_cost;
            let deletion = row[index + 1] + 1;
            let insertion = next[index] + 1;
            next.push(
                substitution
                    .min(deletion)
                    .min(insertion)
                    .min(self.distance + 1),
            );
        }
        next
    }

    fn accept_decoded_byte(
        &self,
        state: &UnicodeLevenshteinState,
        byte: u8,
    ) -> UnicodeLevenshteinState {
        if state.dead {
            return state.clone();
        }

        if state.pending_len == 0 {
            if byte <= 0x7f {
                return UnicodeLevenshteinState {
                    row: self.accept_char(&state.row, byte as char),
                    pending: [0; 4],
                    pending_len: 0,
                    expected_len: 0,
                    dead: false,
                };
            }

            let Some(expected_len) = utf8_expected_len(byte) else {
                return state.dead();
            };
            let mut next = state.clone();
            next.pending = [0; 4];
            next.pending[0] = byte;
            next.pending_len = 1;
            next.expected_len = expected_len;
            return next;
        }

        if !is_utf8_continuation(byte) {
            return state.dead();
        }

        let mut next = state.clone();
        next.pending[next.pending_len as usize] = byte;
        next.pending_len += 1;
        if next.pending_len < next.expected_len {
            return next;
        }

        let bytes = &next.pending[..next.pending_len as usize];
        let Ok(decoded) = std::str::from_utf8(bytes) else {
            return state.dead();
        };
        let Some(ch) = decoded.chars().next() else {
            return state.dead();
        };
        UnicodeLevenshteinState {
            row: self.accept_char(&state.row, ch),
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: false,
        }
    }
}

impl UnicodeLevenshteinState {
    fn dead(&self) -> Self {
        Self {
            row: self.row.clone(),
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: true,
        }
    }

    fn is_between_codepoints(&self) -> bool {
        self.pending_len == 0
    }
}

impl Automaton for UnicodeLevenshtein {
    type State = UnicodeLevenshteinState;

    fn start(&self) -> Self::State {
        UnicodeLevenshteinState {
            row: self.start_row(),
            pending: [0; 4],
            pending_len: 0,
            expected_len: 0,
            dead: false,
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        !state.dead
            && state.is_between_codepoints()
            && state
                .row
                .last()
                .is_some_and(|distance| *distance <= self.distance)
    }

    fn can_match(&self, state: &Self::State) -> bool {
        !state.dead
            && state
                .row
                .iter()
                .copied()
                .min()
                .is_some_and(|distance| distance <= self.distance)
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.accept_decoded_byte(state, byte)
    }
}

fn utf8_expected_len(byte: u8) -> Option<u8> {
    match byte {
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

fn is_utf8_continuation(byte: u8) -> bool {
    matches!(byte, 0x80..=0xbf)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use fst::automaton::Automaton;
    use fst::{IntoStreamer, Streamer};

    use super::{
        FstCandidateSource, FstLexicon, FstLoanwordMatch, FstRepairedBaseline, FstSuggestOptions,
        UnicodeLevenshtein,
    };

    fn reference_unicode_distance(left: &str, right: &str) -> u16 {
        let left = left.chars().collect::<Vec<_>>();
        let right = right.chars().collect::<Vec<_>>();
        let mut matrix = vec![vec![0_u16; right.len() + 1]; left.len() + 1];
        for (index, row) in matrix.iter_mut().enumerate() {
            row[0] = index as u16;
        }
        for (index, cell) in matrix[0].iter_mut().enumerate() {
            *cell = index as u16;
        }
        for i in 1..=left.len() {
            for j in 1..=right.len() {
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + u16::from(left[i - 1] != right[j - 1]));
            }
        }
        matrix[left.len()][right.len()]
    }

    fn assert_unicode_search_matches_reference(words: &[String]) {
        let map = fst::Map::from_iter(words.iter().map(|word| (word.as_bytes(), 1))).unwrap();
        for query in words {
            for limit in 0..=2 {
                let mut stream = map
                    .search_with_state(UnicodeLevenshtein::new(query, limit))
                    .into_stream();
                let mut actual = BTreeMap::new();
                while let Some((key, _, state)) = stream.next() {
                    actual.insert(key.to_vec(), u16::from(state.row[query.chars().count()]));
                }
                let expected = words
                    .iter()
                    .filter_map(|word| {
                        let distance = reference_unicode_distance(query, word);
                        (u32::from(distance) <= limit).then(|| (word.as_bytes().to_vec(), distance))
                    })
                    .collect::<BTreeMap<_, _>>();
                assert_eq!(actual, expected, "query={query:?}, limit={limit}");
            }
        }
    }

    #[test]
    fn compact_unicode_rows_match_exhaustive_scalar_distance() {
        let mut words = vec![String::new()];
        let mut level = vec![String::new()];
        for _ in 0..3 {
            level = level
                .iter()
                .flat_map(|prefix| ['x', 'é', 'ক', '😀'].map(|ch| format!("{prefix}{ch}")))
                .collect();
            words.extend(level.iter().cloned());
        }
        words.sort();
        assert_unicode_search_matches_reference(&words);
    }

    #[test]
    fn unicode_heap_fallback_matches_reference_across_inline_boundary() {
        let mut words = Vec::new();
        for len in [31, 32, 33, 64] {
            let prefix = "ক".repeat(len - 1);
            for suffix in ["", "ক", "খ", "কি", "😀"] {
                words.push(format!("{prefix}{suffix}"));
            }
        }
        words.sort();
        words.dedup();
        assert_unicode_search_matches_reference(&words);
    }

    #[test]
    fn unicode_automaton_rejects_invalid_or_incomplete_utf8() {
        let automaton = UnicodeLevenshtein::new("ক", 2);
        for bytes in [
            &[0x80][..],
            &[0xc0, 0xaf],
            &[0xe0, 0x80, 0x80],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0xe0, 0xa6],
        ] {
            let state = bytes.iter().fold(automaton.start(), |state, byte| {
                automaton.accept(&state, *byte)
            });
            assert!(!automaton.is_match(&state), "invalid UTF-8 {bytes:?}");
        }
    }

    #[test]
    fn fst_suggest_handles_bangla_unicode_edit_distance() {
        let lexicon = test_lexicon([("কমন", 20), ("কেমন", 100), ("যেমন", 30), ("বিজ্ঞান", 70)]);

        let result = lexicon
            .suggest(
                "ক্মন",
                FstSuggestOptions {
                    max_distance: 1,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        assert!(result
            .candidates
            .iter()
            .any(|candidate| candidate.text == "কেমন"));
    }

    #[test]
    fn non_word_baselines_get_no_edit_distance_noise() {
        // The short, high-frequency entries that leaked into the edit neighbourhood of
        // a comma or a bare digit before the word-like gate (issue #34): every 1-2 char
        // token is within edit distance of ও/এ/অং, so they surfaced as "corrections".
        let lexicon = test_lexicon([
            ("ও", 2_000_000),
            ("এ", 900_000),
            ("অং", 5_000),
            ("কে", 78_000),
            ("আমার", 50_000),
        ]);
        let options = || FstSuggestOptions {
            max_distance: 2,
            max_candidates: 32,
            max_prefix_candidates: 16,
            response_candidates: 16,
            ..FstSuggestOptions::default()
        };

        // Punctuation, digits, and a lone chandrabindu carry no Bangla letter, so no
        // lexicon word is a correction of them — the edit-distance channel stays silent.
        for non_word in [",", "।", "?", "()", "১", "২০১১", "ঁ"] {
            let result = lexicon.suggest(non_word, options()).expect("suggest");
            assert!(result.candidates.is_empty(), "{non_word:?} got corrections");
        }

        // A real word still gets its neighbourhood (the exact entry at least).
        let amar = lexicon.suggest("আমার", options()).expect("suggest");
        assert!(amar.candidates.iter().any(|c| c.text == "আমার"));
    }

    #[test]
    fn fst_suggest_returns_prefix_completions() {
        let lexicon = test_lexicon([("কমন", 20), ("কেমন", 100), ("যেমন", 30)]);

        let result = lexicon
            .suggest(
                "কেম",
                FstSuggestOptions {
                    max_distance: 0,
                    max_candidates: 16,
                    max_prefix_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST prefix lookup should succeed");

        assert!(result
            .candidates
            .iter()
            .any(|candidate| candidate.text == "কেমন"));
    }

    #[test]
    fn valid_baseline_channel_prior_prevents_frequency_only_overcorrection() {
        let lexicon = test_lexicon([("মাদার", 1017), ("তাদের", 133846), ("দাদার", 487)]);

        let result = lexicon
            .suggest(
                "মাদার",
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("valid baseline candidate should be returned");
        assert_eq!(first.text, "মাদার");
        assert_eq!(first.source, FstCandidateSource::Exact);

        let tader = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "তাদের")
            .expect("high-frequency edit candidate should still be visible");
        assert!(
            first.score > tader.score,
            "valid baseline score {} should beat frequency-only edit score {}",
            first.score,
            tader.score
        );
    }

    #[test]
    fn low_frequency_exact_baseline_stays_above_high_frequency_edits() {
        let lexicon = test_lexicon([("কটা", 1), ("কতা", 1_000_000), ("কাঠা", 500_000)]);

        let result = lexicon
            .suggest(
                "কটা",
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("valid baseline candidate should be returned");
        assert_eq!(first.text, "কটা");
        assert_eq!(first.source, FstCandidateSource::Exact);

        assert!(result
            .candidates
            .iter()
            .filter(|candidate| candidate.source == FstCandidateSource::EditDistance)
            .all(|candidate| first.score > candidate.score));
    }

    #[test]
    fn orthographic_vowel_length_candidate_can_beat_rare_exact_baseline() {
        let lexicon = test_lexicon([("সুশিল", 3), ("সুশীল", 550), ("সুনীল", 1_220)]);

        let result = lexicon
            .suggest(
                "সুশিল",
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("vowel-length candidate should be returned");
        assert_eq!(first.text, "সুশীল");
        assert_eq!(
            first.source,
            FstCandidateSource::OrthographicVowelLengthEdit
        );

        let unrelated = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "সুনীল")
            .expect("unrelated edit candidate should still be visible");
        assert!(
            first.score > unrelated.score,
            "vowel-length score {} should beat unrelated edit score {}",
            first.score,
            unrelated.score
        );
    }

    #[test]
    fn chandrabindu_variants_are_seeded_without_generic_edit_search() {
        let lexicon = test_lexicon([("চাদ", 301), ("চাঁদ", 2294), ("চাদর", 443)]);

        let result = lexicon
            .suggest(
                "চাদ",
                FstSuggestOptions {
                    max_distance: 0,
                    max_candidates: 8,
                    max_prefix_candidates: 8,
                    response_candidates: 8,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let marked = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "চাঁদ")
            .expect("nasal mark variant should be seeded before edit search");
        assert_eq!(marked.source, FstCandidateSource::DiacriticEdit);
    }

    #[test]
    fn high_frequency_diacritic_variant_can_beat_rare_exact_baseline() {
        let lexicon = test_lexicon([("দাড়িয়ে", 203), ("দাঁড়িয়ে", 18_053)]);

        let result = lexicon
            .suggest(
                "দাড়িয়ে",
                FstSuggestOptions {
                    max_distance: 1,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("diacritic candidate should be returned");
        assert_eq!(first.text, "দাঁড়িয়ে");
        assert_eq!(first.source, FstCandidateSource::DiacriticEdit);
    }

    #[test]
    fn exact_stem_can_surface_valid_suffix_completions() {
        let lexicon = test_lexicon([("নদী", 100), ("নদীটি", 30), ("নদীকে", 20), ("নদীকথা", 200)]);

        let result = lexicon
            .suggest(
                "নদী",
                FstSuggestOptions {
                    max_distance: 0,
                    max_candidates: 32,
                    max_prefix_candidates: 32,
                    response_candidates: 32,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let stem = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "নদী")
            .expect("exact stem should be returned");
        assert_eq!(stem.source, FstCandidateSource::Exact);

        let suffixed = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "নদীটি")
            .expect("valid stem suffix form should be returned");
        assert_eq!(suffixed.source, FstCandidateSource::StemSuffixCompletion);

        let compound = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "নদীকথা")
            .expect("ordinary prefix completion should still be returned");
        assert_eq!(compound.source, FstCandidateSource::PrefixCompletion);
    }

    #[test]
    fn suffix_channel_requires_exact_stem_in_lexicon() {
        let lexicon = test_lexicon([("নদীটি", 30)]);

        let result = lexicon
            .suggest(
                "নদী",
                FstSuggestOptions {
                    max_distance: 0,
                    max_candidates: 32,
                    max_prefix_candidates: 0,
                    response_candidates: 32,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        assert!(result.candidates.is_empty());
    }

    #[test]
    fn repaired_roman_baseline_can_beat_expensive_bangla_surface_edit() {
        let lexicon = test_lexicon([("অকালপক্ক", 18), ("অকাল্পক্কো", 1)]);
        let repairs = [FstRepairedBaseline {
            roman_input: "okalopokk",
            bangla_output: "অকালপক্ক",
            repair_kind: "inserted_separator_o",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines(
                "অকল্পক্ক",
                &repairs,
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("repair candidate should be returned");
        assert_eq!(first.text, "অকালপক্ক");
        assert_eq!(first.source, FstCandidateSource::RomanRepairExact);
        assert_eq!(first.roman_repair.as_deref(), Some("okalopokk"));
        assert_eq!(first.roman_repair_kind, Some("inserted_separator_o"));
        assert_eq!(first.roman_repair_cost, Some(1));
    }

    #[test]
    fn repaired_roman_baseline_seeds_diacritic_variants() {
        let lexicon = test_lexicon([("দাড়িয়ে", 203), ("দাঁড়িয়ে", 18_053), ("হারিয়ে", 27_441)]);
        let repairs = [FstRepairedBaseline {
            roman_input: "daRiye",
            bangla_output: "দাড়িয়ে",
            repair_kind: "lowercase_r_to_flap",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines(
                "দারিয়ে",
                &repairs,
                FstSuggestOptions {
                    max_distance: 1,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("repaired diacritic candidate should be returned");
        assert_eq!(first.text, "দাঁড়িয়ে");
        assert_eq!(first.source, FstCandidateSource::RomanRepairExact);
        assert_eq!(first.roman_repair.as_deref(), Some("daRiye"));
        assert_eq!(first.roman_repair_kind, Some("lowercase_r_to_flap"));
    }

    #[test]
    fn english_loanword_exact_channel_can_bridge_distant_obadh_output() {
        let lexicon = test_lexicon([("উনিভেরসিতা", 2), ("ইউনিভার্সিটি", 16), ("ইউনিয়ন", 4096)]);
        let loanwords = [FstLoanwordMatch {
            roman_input: "university",
            roman_repair: "university",
            bangla_output: "ইউনিভার্সিটি",
            frequency: 16,
            repair_kind: "english_loanword_exact",
            repair_cost: 0,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines_and_loanwords(
                "উনিভেরসিত্য",
                &[],
                &loanwords,
                FstSuggestOptions {
                    max_distance: 2,
                    response_candidates: 4,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("loanword candidate should be returned");
        assert_eq!(first.text, "ইউনিভার্সিটি");
        assert_eq!(first.source, FstCandidateSource::EnglishLoanwordExact);
        assert_eq!(first.roman_repair.as_deref(), Some("university"));
        assert_eq!(first.roman_repair_kind, Some("english_loanword_exact"));
    }

    #[test]
    fn english_loanword_fuzzy_channel_repairs_misspelled_english_keys() {
        let lexicon = test_lexicon([("উনিভেরসিতা", 2), ("ইউনিভার্সিটি", 9407)]);
        let loanwords = [FstLoanwordMatch {
            roman_input: "universty",
            roman_repair: "university",
            bangla_output: "ইউনিভার্সিটি",
            frequency: 16,
            repair_kind: "english_loanword_fuzzy",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines_and_loanwords(
                "উনিভেরস্ত্য",
                &[],
                &loanwords,
                FstSuggestOptions {
                    max_distance: 2,
                    response_candidates: 4,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("fuzzy loanword candidate should be returned");
        assert_eq!(first.text, "ইউনিভার্সিটি");
        assert_eq!(first.source, FstCandidateSource::EnglishLoanwordFuzzy);
        assert_eq!(first.roman_repair.as_deref(), Some("university"));
        assert_eq!(first.roman_repair_kind, Some("english_loanword_fuzzy"));
        assert_eq!(first.roman_repair_cost, Some(1));
    }

    #[test]
    fn fuzzy_english_loanword_prefix_truncation_does_not_beat_close_bangla_edit() {
        let lexicon = test_lexicon([("আমেরিকা", 10_018), ("আমেরিকান", 20_752)]);
        let loanwords = [FstLoanwordMatch {
            roman_input: "american",
            roman_repair: "america",
            bangla_output: "আমেরিকা",
            frequency: 16,
            repair_kind: "english_loanword_fuzzy",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines_and_loanwords(
                "আমেরিচান",
                &[],
                &loanwords,
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("Bangla edit candidate should be returned");
        assert_eq!(first.text, "আমেরিকান");
        assert_eq!(first.source, FstCandidateSource::EditDistance);

        let loanword = result
            .candidates
            .iter()
            .find(|candidate| candidate.text == "আমেরিকা")
            .expect("prefix-truncated loanword candidate should still be visible");
        assert_eq!(loanword.source, FstCandidateSource::EnglishLoanwordFuzzy);
        assert!(
            first.score > loanword.score,
            "close Bangla edit score {} should beat prefix-truncated loanword score {}",
            first.score,
            loanword.score
        );
    }

    #[test]
    fn suffix_truncated_loanword_channel_stays_below_closer_edit_across_frequencies() {
        for (america_frequency, american_frequency) in [(10_018, 20_752), (1_000_000, 10)] {
            let mut seeds = BTreeMap::new();
            super::insert_fst_candidate(
                &mut seeds,
                "আমেরিকান",
                american_frequency,
                FstCandidateSource::EditDistance,
                "আমেরিচান",
                None,
            );
            let loanword = FstLoanwordMatch {
                roman_input: "American",
                roman_repair: "america",
                bangla_output: "আমেরিকা",
                frequency: 16,
                repair_kind: "english_loanword_fuzzy",
                repair_cost: 1,
            };
            super::insert_english_loanword_candidate(
                &mut seeds,
                &loanword,
                america_frequency,
                "আমেরিচান",
            );
            let closer = &seeds["আমেরিকান"];
            let truncated = &seeds["আমেরিকা"];
            assert_eq!(truncated.source, FstCandidateSource::EnglishLoanwordFuzzy);
            assert!(closer.edit_cost < truncated.edit_cost);
            assert!(closer.score > truncated.score);
        }
    }

    #[test]
    fn fuzzy_english_loanword_repeated_final_key_typo_stays_strong() {
        let lexicon = test_lexicon([("উনিভেরসিতা", 2), ("ইউনিভার্সিটি", 9407)]);
        let loanwords = [FstLoanwordMatch {
            roman_input: "universityy",
            roman_repair: "university",
            bangla_output: "ইউনিভার্সিটি",
            frequency: 16,
            repair_kind: "english_loanword_fuzzy",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines_and_loanwords(
                "উনিভেরসিত্য্য",
                &[],
                &loanwords,
                FstSuggestOptions {
                    max_distance: 2,
                    response_candidates: 4,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("fuzzy loanword candidate should be returned");
        assert_eq!(first.text, "ইউনিভার্সিটি");
        assert_eq!(first.source, FstCandidateSource::EnglishLoanwordFuzzy);
    }

    #[test]
    fn fuzzy_english_loanword_does_not_beat_valid_bangla_exact_hit() {
        let lexicon = test_lexicon([("মাদার", 1017), ("রাডার", 9000)]);
        let loanwords = [FstLoanwordMatch {
            roman_input: "madar",
            roman_repair: "radar",
            bangla_output: "রাডার",
            frequency: 16,
            repair_kind: "english_loanword_fuzzy",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines_and_loanwords(
                "মাদার",
                &[],
                &loanwords,
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("valid exact candidate should be returned");
        assert_eq!(first.text, "মাদার");
        assert_eq!(first.source, FstCandidateSource::Exact);
        assert!(result
            .candidates
            .iter()
            .all(|candidate| candidate.source != FstCandidateSource::EnglishLoanwordFuzzy));
    }

    #[test]
    fn strong_velar_roman_repair_can_beat_low_frequency_exact_artifact() {
        let lexicon = test_lexicon([("জংই", 1), ("জঙ্গি", 938)]);
        let repairs = [FstRepairedBaseline {
            roman_input: "jonggi",
            bangla_output: "জঙ্গি",
            repair_kind: "velar_nasal_from_ng",
            repair_cost: 1,
        }];

        let result = lexicon
            .suggest_with_repaired_baselines(
                "জংই",
                &repairs,
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("repair candidate should be returned");
        assert_eq!(first.text, "জঙ্গি");
        assert_eq!(first.source, FstCandidateSource::RomanRepairExact);
        assert_eq!(first.roman_repair_kind, Some("velar_nasal_from_ng"));
    }

    #[test]
    fn rare_velar_roman_repair_stays_below_palatal_target() {
        let lexicon = test_lexicon([("জিঞ্জিরা", 61), ("জিঙ্গিরা", 1)]);
        let repairs = [
            FstRepairedBaseline {
                roman_input: "jinjira",
                bangla_output: "জিঞ্জিরা",
                repair_kind: "palatal_nasal_ja_from_ng",
                repair_cost: 2,
            },
            FstRepairedBaseline {
                roman_input: "jinggira",
                bangla_output: "জিঙ্গিরা",
                repair_kind: "velar_nasal_from_ng",
                repair_cost: 1,
            },
        ];

        let result = lexicon
            .suggest_with_repaired_baselines(
                "জিংইরা",
                &repairs,
                FstSuggestOptions {
                    max_distance: 2,
                    max_candidates: 16,
                    response_candidates: 16,
                    ..FstSuggestOptions::default()
                },
            )
            .expect("Bangla FST lookup should succeed");

        let first = result
            .candidates
            .first()
            .expect("repair candidate should be returned");
        assert_eq!(first.text, "জিঞ্জিরা");
        assert_eq!(first.source, FstCandidateSource::RomanRepairExact);
    }

    fn test_lexicon<const N: usize>(entries: [(&str, u64); N]) -> FstLexicon<Vec<u8>> {
        let mut entries = entries;
        entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        let mut builder = fst::MapBuilder::memory();
        for (word, frequency) in entries {
            builder
                .insert(word.as_bytes(), frequency)
                .expect("fixture key should insert");
        }
        FstLexicon::from_map(builder.into_map())
    }

    #[test]
    fn stable_source_codes_are_frozen() {
        // These codes are a C ABI contract; renumbering breaks every downstream.
        use FstCandidateSource::*;
        for (source, code) in [
            (Exact, 0),
            (EditDistance, 1),
            (DiacriticEdit, 2),
            (OrthographicVowelLengthEdit, 3),
            (PrefixCompletion, 4),
            (StemSuffixCompletion, 5),
            (SkeletonVowelDrop, 6),
            (ConsonantConfusion, 7),
            (RomanRepairExact, 8),
            (EnglishLoanwordExact, 9),
            (EnglishLoanwordFuzzy, 10),
            (EmoticonExact, 11),
        ] {
            assert_eq!(source.stable_code(), code, "{source:?}");
        }
    }

    /// Isolated timing of the skeleton automaton walk on the real 845k-word bn.fst. Ignored
    /// by default (needs resolved submodule data); run with:
    ///   cargo test --release --lib skeleton_walk_timing -- --ignored --nocapture
    #[test]
    #[ignore = "needs the resolved data/autocorrect/models/bn.fst; timing probe"]
    fn skeleton_walk_timing() {
        use std::time::Instant;
        let path = "data/autocorrect/models/bn.fst";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skip: {path} not resolved");
            return;
        };
        let lexicon = FstLexicon::from_bytes(bytes).expect("load fst");
        // Vowel-dropped baselines (the channel's real inputs) plus a 2-consonant worst case.
        for baseline in ["ক্রল্ম", "দখলম", "প্রয়জন", "ক্র"]
        {
            for _ in 0..200 {
                let _ = lexicon.skeleton_matches(baseline, super::SKELETON_MATCH_LIMIT);
            }
            let n = 5000;
            let start = Instant::now();
            let mut sink = 0usize;
            for _ in 0..n {
                sink += lexicon
                    .skeleton_matches(baseline, super::SKELETON_MATCH_LIMIT)
                    .len();
            }
            let per_us = start.elapsed().as_secs_f64() * 1e6 / n as f64;
            eprintln!(
                "skeleton_matches({baseline:12}) {per_us:7.1} µs/call  (mates≈{})",
                sink / n
            );
        }
    }

    /// The cabi `word_frequency` accessor returns `exact_frequency(word).unwrap_or(0)`
    /// and treats `0` as "not a lexicon word". That is only unambiguous if no real
    /// entry is stored with frequency 0. Prove it against the shipped 845k-word
    /// artifact by streaming every entry, and pin the rare-baseline counts the
    /// client ratio gate is designed around. Skips when the submodule is not
    /// resolved (CI), runs locally.
    #[test]
    fn no_lexicon_entry_has_frequency_zero_on_real_bn_fst() {
        use fst::Streamer;
        let path = "data/autocorrect/models/bn.fst";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skip: {path} not resolved");
            return;
        };
        let lexicon = FstLexicon::from_bytes(bytes).expect("load fst");

        // Every stored frequency is >= 1, so `word_frequency`'s 0 return means
        // exactly "absent" — the presence bit the removed is_lexicon_word gave.
        let mut stream = lexicon.map.stream();
        let mut entries = 0u64;
        let mut min_frequency = u64::MAX;
        while let Some((_key, frequency)) = stream.next() {
            entries += 1;
            min_frequency = min_frequency.min(frequency);
        }
        assert!(entries > 800_000, "sanity: streamed {entries} entries");
        assert!(
            min_frequency >= 1,
            "an entry has frequency 0; word_frequency's zero sentinel would be ambiguous"
        );

        // Rare real-word baselines whose common correction the client overrides by
        // ratio (verified via `suggest-fst`): মানুস 49 → মানুষ, বন্দু 25 → বন্ধু.
        assert_eq!(lexicon.exact_frequency("মানুস"), Some(49));
        assert_eq!(lexicon.exact_frequency("বন্দু"), Some(25));
        // A non-word baseline is None → the accessor's 0.
        assert_eq!(lexicon.exact_frequency("অআইঈউ"), None);
    }
}
