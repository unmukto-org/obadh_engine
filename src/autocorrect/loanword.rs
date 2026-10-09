use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use fst::automaton::Automaton;
use fst::{IntoStreamer, Streamer};

use super::edit_row::BoundedEditRow;

const LOANWORD_MAGIC: &[u8; 8] = b"OBLNW001";
const HEADER_LEN: usize = 24;
const RANGE_LEN: usize = 8;
const VALUE_LEN: usize = 12;
pub const DEFAULT_LOANWORD_FUZZY_CANDIDATES: usize = 16;
pub const LOANWORD_FUZZY_MAX_DISTANCE: u32 = 2;
const LOANWORD_FUZZY_MIN_BYTES: usize = 5;
const LOANWORD_FUZZY_DISTANCE_TWO_MIN_BYTES: usize = 8;

#[derive(Debug)]
pub struct LoanwordLexicon<D: AsRef<[u8]> = Vec<u8>> {
    map: fst::Map<D>,
    ranges: Vec<LoanwordRange>,
    values: Vec<LoanwordValue>,
    bangla: Vec<u8>,
    fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoanwordEntry {
    pub english: String,
    pub bangla: String,
    pub frequency: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoanwordMatch<'a> {
    pub english: &'a str,
    pub bangla: &'a str,
    pub frequency: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoanwordSearchOptions {
    pub max_distance: u32,
    pub max_candidates: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoanwordSuggestion {
    pub english: String,
    pub bangla: String,
    pub frequency: u64,
    pub edit_cost: u16,
    pub kind: LoanwordSuggestionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoanwordSuggestionKind {
    Exact,
    Transposition,
    Fuzzy,
}

#[derive(Debug, Clone, Copy)]
struct LoanwordRange {
    start: u32,
    len: u32,
}

#[derive(Debug, Clone, Copy)]
struct LoanwordValue {
    offset: u32,
    len: u32,
    frequency: u32,
}

impl LoanwordEntry {
    pub fn new(english: impl Into<String>, bangla: impl Into<String>, frequency: u32) -> Self {
        Self {
            english: english.into(),
            bangla: bangla.into(),
            frequency,
        }
    }
}

impl LoanwordSearchOptions {
    pub fn for_input(input: &str) -> Self {
        Self {
            max_distance: default_loanword_fuzzy_distance(input),
            max_candidates: DEFAULT_LOANWORD_FUZZY_CANDIDATES,
        }
    }
}

impl LoanwordSuggestionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "english_loanword_exact",
            Self::Transposition => "english_loanword_transposition",
            Self::Fuzzy => "english_loanword_fuzzy",
        }
    }
}

impl LoanwordLexicon<Vec<u8>> {
    pub fn from_entries(
        entries: impl IntoIterator<Item = LoanwordEntry>,
    ) -> Result<Self, LoanwordArtifactError> {
        Self::from_bytes(build_loanword_bytes(entries)?)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoanwordArtifactError> {
        let fingerprint = crate::fingerprint::artifact_fingerprint(&bytes);
        if bytes.len() < HEADER_LEN {
            return Err(LoanwordArtifactError::TruncatedHeader);
        }
        if &bytes[..LOANWORD_MAGIC.len()] != LOANWORD_MAGIC {
            return Err(LoanwordArtifactError::InvalidMagic);
        }

        let map_len = read_u32(&bytes, 8)? as usize;
        let range_len = read_u32(&bytes, 12)? as usize;
        let value_len = read_u32(&bytes, 16)? as usize;
        let bangla_len = read_u32(&bytes, 20)? as usize;
        let range_bytes = range_len
            .checked_mul(RANGE_LEN)
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        let value_bytes = value_len
            .checked_mul(VALUE_LEN)
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        let expected_len = HEADER_LEN
            .checked_add(map_len)
            .and_then(|len| len.checked_add(range_bytes))
            .and_then(|len| len.checked_add(value_bytes))
            .and_then(|len| len.checked_add(bangla_len))
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        if bytes.len() != expected_len {
            return Err(LoanwordArtifactError::LengthMismatch {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let mut cursor = HEADER_LEN;
        let map_bytes = bytes[cursor..cursor + map_len].to_vec();
        cursor += map_len;

        let mut ranges = Vec::with_capacity(range_len);
        for _ in 0..range_len {
            ranges.push(LoanwordRange {
                start: read_u32(&bytes, cursor)?,
                len: read_u32(&bytes, cursor + 4)?,
            });
            cursor += RANGE_LEN;
        }

        let mut values = Vec::with_capacity(value_len);
        for _ in 0..value_len {
            values.push(LoanwordValue {
                offset: read_u32(&bytes, cursor)?,
                len: read_u32(&bytes, cursor + 4)?,
                frequency: read_u32(&bytes, cursor + 8)?,
            });
            cursor += VALUE_LEN;
        }

        let bangla = bytes[cursor..cursor + bangla_len].to_vec();
        let map = fst::Map::new(map_bytes)?;
        validate_ranges(&ranges, values.len())?;
        validate_values(&values, bangla.len())?;

        Ok(Self {
            map,
            ranges,
            values,
            bangla,
            fingerprint,
        })
    }
}

impl<D: AsRef<[u8]>> LoanwordLexicon<D> {
    /// Content fingerprint of the loanword artifact bytes this lexicon was loaded
    /// from, for the same stale-artifact check as
    /// [`FstLexicon::artifact_fingerprint`](crate::FstLexicon::artifact_fingerprint).
    /// Captured at load over the whole artifact image. See [`crate::fingerprint`].
    pub fn artifact_fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn english_key_count(&self) -> usize {
        self.map.len()
    }

    pub fn exact_matches<'a>(&'a self, english: &'a str) -> Vec<LoanwordMatch<'a>> {
        if !is_loanword_key(english) {
            return Vec::new();
        }

        let Some(range_index) = self.map.get(english.as_bytes()) else {
            return Vec::new();
        };
        let Some(range) = self.ranges.get(range_index as usize).copied() else {
            return Vec::new();
        };
        let start = range.start as usize;
        let end = start.saturating_add(range.len as usize);
        if end > self.values.len() {
            return Vec::new();
        }

        self.values[start..end]
            .iter()
            .filter_map(|value| self.match_for_value(english, *value))
            .collect()
    }

    pub fn suggestions(
        &self,
        english: &str,
        options: LoanwordSearchOptions,
    ) -> Result<Vec<LoanwordSuggestion>, LoanwordArtifactError> {
        let Some(query) = loanword_query_key(english) else {
            return Ok(Vec::new());
        };
        if options.max_candidates == 0 {
            return Ok(Vec::new());
        }

        let english = query.as_ref();
        let max_distance = options.max_distance.min(LOANWORD_FUZZY_MAX_DISTANCE);
        let mut suggestions = Vec::<LoanwordSuggestion>::new();
        let mut seen = BTreeSet::<(String, String)>::new();
        self.push_suggestions_for_key(
            english,
            0,
            LoanwordSuggestionKind::Exact,
            &mut suggestions,
            &mut seen,
        )?;
        if !suggestions.is_empty() || max_distance == 0 {
            suggestions.truncate(options.max_candidates);
            return Ok(suggestions);
        }

        // Retrieve and score with the same OSA distance. A Levenshtein filter
        // loses candidates containing two swaps or a swap plus another edit.
        let automaton = AsciiOsa::new(english, max_distance);
        let mut stream = self.map.search_with_state(automaton).into_stream();
        while let Some((key, _range_index, state)) = stream.next() {
            let candidate_key =
                std::str::from_utf8(key).map_err(|_| LoanwordArtifactError::InvalidUtf8)?;
            if candidate_key == english {
                continue;
            }
            let edit_cost = u16::from(state.row[english.len()]);
            if edit_cost == 0 || edit_cost > max_distance as u16 {
                continue;
            }
            let kind = if edit_cost == 1 && is_adjacent_transposition(english, candidate_key) {
                LoanwordSuggestionKind::Transposition
            } else {
                LoanwordSuggestionKind::Fuzzy
            };
            self.push_suggestions_for_key(
                candidate_key,
                edit_cost,
                kind,
                &mut suggestions,
                &mut seen,
            )?;
        }

        suggestions.sort_by(|left, right| {
            left.edit_cost
                .cmp(&right.edit_cost)
                .then_with(|| left.kind.cmp(&right.kind))
                .then_with(|| right.frequency.cmp(&left.frequency))
                .then_with(|| left.english.cmp(&right.english))
                .then_with(|| left.bangla.cmp(&right.bangla))
        });
        suggestions.truncate(options.max_candidates);
        Ok(suggestions)
    }

    pub fn iter_entries(&self) -> Result<Vec<LoanwordEntry>, LoanwordArtifactError> {
        let mut entries = Vec::with_capacity(self.values.len());
        let mut stream = self.map.stream();
        while let Some((key, range_index)) = stream.next() {
            let english = std::str::from_utf8(key)
                .map_err(|_| LoanwordArtifactError::InvalidUtf8)?
                .to_string();
            let range = self
                .ranges
                .get(range_index as usize)
                .ok_or(LoanwordArtifactError::InvalidRangeIndex(range_index))?;
            let start = range.start as usize;
            let end = start.saturating_add(range.len as usize);
            if end > self.values.len() {
                return Err(LoanwordArtifactError::InvalidRangeIndex(range_index));
            }
            for value in &self.values[start..end] {
                let bangla = self.bangla_for_value(*value)?.to_string();
                entries.push(LoanwordEntry::new(english.clone(), bangla, value.frequency));
            }
        }
        Ok(entries)
    }

    fn match_for_value<'a>(
        &'a self,
        english: &'a str,
        value: LoanwordValue,
    ) -> Option<LoanwordMatch<'a>> {
        let bangla = self.bangla_for_value(value).ok()?;
        Some(LoanwordMatch {
            english,
            bangla,
            frequency: value.frequency as u64,
        })
    }

    fn bangla_for_value(&self, value: LoanwordValue) -> Result<&str, LoanwordArtifactError> {
        let start = value.offset as usize;
        let end = start
            .checked_add(value.len as usize)
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        let bytes = self
            .bangla
            .get(start..end)
            .ok_or(LoanwordArtifactError::ValueOutOfBounds)?;
        std::str::from_utf8(bytes).map_err(|_| LoanwordArtifactError::InvalidUtf8)
    }

    fn push_suggestions_for_key(
        &self,
        english: &str,
        edit_cost: u16,
        kind: LoanwordSuggestionKind,
        suggestions: &mut Vec<LoanwordSuggestion>,
        seen: &mut BTreeSet<(String, String)>,
    ) -> Result<(), LoanwordArtifactError> {
        let Some(range_index) = self.map.get(english.as_bytes()) else {
            return Ok(());
        };
        let range = self
            .ranges
            .get(range_index as usize)
            .ok_or(LoanwordArtifactError::InvalidRangeIndex(range_index))?;
        let start = range.start as usize;
        let end = start
            .checked_add(range.len as usize)
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        if end > self.values.len() {
            return Err(LoanwordArtifactError::InvalidRangeIndex(range_index));
        }

        for value in &self.values[start..end] {
            let bangla = self.bangla_for_value(*value)?;
            let dedupe_key = (english.to_string(), bangla.to_string());
            if !seen.insert(dedupe_key) {
                continue;
            }
            suggestions.push(LoanwordSuggestion {
                english: english.to_string(),
                bangla: bangla.to_string(),
                frequency: value.frequency as u64,
                edit_cost,
                kind,
            });
        }
        Ok(())
    }
}

pub fn build_loanword_bytes(
    entries: impl IntoIterator<Item = LoanwordEntry>,
) -> Result<Vec<u8>, LoanwordArtifactError> {
    let mut by_english = BTreeMap::<String, BTreeMap<String, u32>>::new();
    for entry in entries {
        if !is_loanword_key(&entry.english) {
            return Err(LoanwordArtifactError::InvalidEnglishKey(entry.english));
        }
        if entry.bangla.is_empty() {
            return Err(LoanwordArtifactError::EmptyBanglaValue);
        }
        by_english
            .entry(entry.english)
            .or_default()
            .entry(entry.bangla)
            .and_modify(|frequency| *frequency = (*frequency).max(entry.frequency))
            .or_insert(entry.frequency);
    }

    let mut builder = fst::MapBuilder::memory();
    let mut ranges = Vec::<LoanwordRange>::with_capacity(by_english.len());
    let mut values = Vec::<LoanwordValue>::new();
    let mut bangla = Vec::<u8>::new();

    for (english, bangla_values) in by_english {
        let range_index = ranges.len() as u64;
        builder.insert(english, range_index)?;
        let start = checked_u32(values.len())?;
        for (word, frequency) in bangla_values {
            let offset = checked_u32(bangla.len())?;
            let word_bytes = word.as_bytes();
            bangla.extend_from_slice(word_bytes);
            values.push(LoanwordValue {
                offset,
                len: checked_u32(word_bytes.len())?,
                frequency,
            });
        }
        ranges.push(LoanwordRange {
            start,
            len: checked_u32(values.len().saturating_sub(start as usize))?,
        });
    }

    let map_bytes = builder.into_inner()?;
    let mut bytes = Vec::with_capacity(
        HEADER_LEN
            + map_bytes.len()
            + ranges.len() * RANGE_LEN
            + values.len() * VALUE_LEN
            + bangla.len(),
    );
    bytes.extend_from_slice(LOANWORD_MAGIC);
    push_u32(&mut bytes, checked_u32(map_bytes.len())?);
    push_u32(&mut bytes, checked_u32(ranges.len())?);
    push_u32(&mut bytes, checked_u32(values.len())?);
    push_u32(&mut bytes, checked_u32(bangla.len())?);
    bytes.extend_from_slice(&map_bytes);
    for range in ranges {
        push_u32(&mut bytes, range.start);
        push_u32(&mut bytes, range.len);
    }
    for value in values {
        push_u32(&mut bytes, value.offset);
        push_u32(&mut bytes, value.len);
        push_u32(&mut bytes, value.frequency);
    }
    bytes.extend_from_slice(&bangla);
    Ok(bytes)
}

pub fn is_loanword_key(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
}

pub fn loanword_query_key(text: &str) -> Option<Cow<'_, str>> {
    if text.is_empty() {
        return None;
    }

    let mut has_uppercase = false;
    for byte in text.bytes() {
        if !byte.is_ascii_alphanumeric() {
            return None;
        }
        has_uppercase |= byte.is_ascii_uppercase();
    }

    if has_uppercase {
        Some(Cow::Owned(text.to_ascii_lowercase()))
    } else {
        Some(Cow::Borrowed(text))
    }
}

pub fn default_loanword_fuzzy_distance(input: &str) -> u32 {
    let Some(query) = loanword_query_key(input) else {
        return 0;
    };
    if query.len() < LOANWORD_FUZZY_MIN_BYTES {
        return 0;
    }
    if query.len() >= LOANWORD_FUZZY_DISTANCE_TWO_MIN_BYTES {
        2
    } else {
        1
    }
}

fn is_adjacent_transposition(left: &str, right: &str) -> bool {
    if left.len() != right.len() || left == right {
        return false;
    }
    let left = left.as_bytes();
    let right = right.as_bytes();
    let diffs = left
        .iter()
        .zip(right)
        .enumerate()
        .filter_map(|(index, (left, right))| (left != right).then_some(index))
        .collect::<Vec<_>>();
    matches!(
        diffs.as_slice(),
        [first, second]
            if *second == *first + 1
                && left[*first] == right[*second]
                && left[*second] == right[*first]
    )
}

// Optimal string alignment distance: each adjacent transposition is one edit,
// and a substring cannot participate in two overlapping transpositions.
#[derive(Debug, Clone)]
struct AsciiOsa<'q> {
    query: &'q [u8],
    distance: u8,
}

#[derive(Debug, Clone)]
struct AsciiOsaState {
    row: BoundedEditRow,
    previous_row: BoundedEditRow,
    previous_byte: Option<u8>,
}

impl<'q> AsciiOsa<'q> {
    fn new(query: &'q str, distance: u32) -> Self {
        debug_assert!(distance <= LOANWORD_FUZZY_MAX_DISTANCE);
        Self {
            query: query.as_bytes(),
            distance: distance as u8,
        }
    }
}

impl Automaton for AsciiOsa<'_> {
    type State = AsciiOsaState;

    fn start(&self) -> Self::State {
        let row = BoundedEditRow::initial(self.query.len(), self.distance + 1);
        AsciiOsaState {
            previous_row: row.clone(),
            row,
            previous_byte: None,
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        state.row[self.query.len()] <= self.distance
    }

    fn can_match(&self, state: &Self::State) -> bool {
        state.row.iter().any(|cost| *cost <= self.distance)
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        let ceiling = self.distance + 1;
        let mut row = BoundedEditRow::with_capacity(self.query.len() + 1);
        row.push((state.row[0] + 1).min(ceiling));
        for (index, query_byte) in self.query.iter().enumerate() {
            let substitution = state.row[index] + u8::from(*query_byte != byte);
            let deletion = state.row[index + 1] + 1;
            let insertion = row[index] + 1;
            let mut cost = substitution.min(deletion).min(insertion);
            if index > 0
                && state.previous_byte == Some(*query_byte)
                && self.query[index - 1] == byte
            {
                cost = cost.min(state.previous_row[index - 1] + 1);
            }
            row.push(cost.min(ceiling));
        }
        AsciiOsaState {
            row,
            previous_row: state.row.clone(),
            previous_byte: Some(byte),
        }
    }
}

fn validate_ranges(
    ranges: &[LoanwordRange],
    value_len: usize,
) -> Result<(), LoanwordArtifactError> {
    for range in ranges {
        let start = range.start as usize;
        let end = start
            .checked_add(range.len as usize)
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        if end > value_len {
            return Err(LoanwordArtifactError::ValueOutOfBounds);
        }
    }
    Ok(())
}

fn validate_values(
    values: &[LoanwordValue],
    bangla_len: usize,
) -> Result<(), LoanwordArtifactError> {
    for value in values {
        let start = value.offset as usize;
        let end = start
            .checked_add(value.len as usize)
            .ok_or(LoanwordArtifactError::LengthOverflow)?;
        if end > bangla_len {
            return Err(LoanwordArtifactError::ValueOutOfBounds);
        }
    }
    Ok(())
}

fn checked_u32(value: usize) -> Result<u32, LoanwordArtifactError> {
    value
        .try_into()
        .map_err(|_| LoanwordArtifactError::LengthOverflow)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, LoanwordArtifactError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(LoanwordArtifactError::TruncatedHeader)?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[derive(Debug)]
pub enum LoanwordArtifactError {
    EmptyBanglaValue,
    InvalidEnglishKey(String),
    InvalidMagic,
    InvalidRangeIndex(u64),
    InvalidUtf8,
    LengthMismatch { expected: usize, actual: usize },
    LengthOverflow,
    TruncatedHeader,
    ValueOutOfBounds,
    Fst(fst::Error),
}

impl fmt::Display for LoanwordArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBanglaValue => write!(formatter, "empty Bangla loanword value"),
            Self::InvalidEnglishKey(key) => {
                write!(formatter, "invalid English loanword key: {key}")
            }
            Self::InvalidMagic => write!(formatter, "invalid loanword lexicon magic bytes"),
            Self::InvalidRangeIndex(index) => {
                write!(formatter, "invalid loanword range index: {index}")
            }
            Self::InvalidUtf8 => write!(formatter, "invalid UTF-8 in loanword lexicon"),
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "loanword lexicon length mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::LengthOverflow => write!(formatter, "loanword lexicon length overflow"),
            Self::TruncatedHeader => write!(formatter, "truncated loanword lexicon header"),
            Self::ValueOutOfBounds => write!(formatter, "loanword lexicon value out of bounds"),
            Self::Fst(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for LoanwordArtifactError {}

impl From<fst::Error> for LoanwordArtifactError {
    fn from(error: fst::Error) -> Self {
        Self::Fst(error)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use fst::{IntoStreamer, Streamer};

    use super::{
        build_loanword_bytes, AsciiOsa, LoanwordEntry, LoanwordLexicon, LoanwordSearchOptions,
        LoanwordSuggestionKind,
    };

    #[test]
    fn fuzzy_lookup_retrieves_combined_transposition_edits() {
        let lexicon =
            LoanwordLexicon::from_entries([LoanwordEntry::new("keyboard", "কীবোর্ড", 16)]).unwrap();
        for (query, distance, kind) in [
            ("ekyboard", 1, LoanwordSuggestionKind::Transposition),
            ("keyborad", 1, LoanwordSuggestionKind::Transposition),
            ("ekyborad", 2, LoanwordSuggestionKind::Fuzzy),
            ("ekyboarx", 2, LoanwordSuggestionKind::Fuzzy),
            ("ekyboardx", 2, LoanwordSuggestionKind::Fuzzy),
        ] {
            let suggestions = lexicon
                .suggestions(query, LoanwordSearchOptions::for_input(query))
                .unwrap();
            let first = suggestions
                .first()
                .expect("valid OSA correction should be retrieved");
            assert_eq!(first.english, "keyboard", "{query}");
            assert_eq!(first.edit_cost, distance, "{query}");
            assert_eq!(first.kind, kind, "{query}");
        }
        assert!(lexicon
            .suggestions(
                "ekyborad",
                LoanwordSearchOptions {
                    max_distance: 1,
                    ..LoanwordSearchOptions::for_input("ekyborad")
                }
            )
            .unwrap()
            .is_empty());
    }

    // Independent, uncapped matrix oracle. It deliberately shares neither row
    // storage nor pruning with the production automaton.
    fn reference_osa(left: &[u8], right: &[u8]) -> u16 {
        let mut matrix = vec![vec![0_u16; right.len() + 1]; left.len() + 1];
        for (index, row) in matrix.iter_mut().enumerate() {
            row[0] = index as u16;
        }
        for (index, cell) in matrix[0].iter_mut().enumerate() {
            *cell = index as u16;
        }
        for i in 1..=left.len() {
            for j in 1..=right.len() {
                let mut cost = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + u16::from(left[i - 1] != right[j - 1]));
                if i > 1 && j > 1 && left[i - 1] == right[j - 2] && left[i - 2] == right[j - 1] {
                    cost = cost.min(matrix[i - 2][j - 2] + 1);
                }
                matrix[i][j] = cost;
            }
        }
        matrix[left.len()][right.len()]
    }

    fn assert_osa_search_matches_reference(words: &[String]) {
        let map = fst::Map::from_iter(words.iter().map(|word| (word.as_bytes(), 1))).unwrap();
        for query in words {
            for limit in 0..=2 {
                let mut stream = map
                    .search_with_state(AsciiOsa::new(query, limit))
                    .into_stream();
                let mut actual = BTreeMap::new();
                while let Some((key, _, state)) = stream.next() {
                    actual.insert(key.to_vec(), u16::from(state.row[query.len()]));
                }
                let expected = words
                    .iter()
                    .filter_map(|word| {
                        let distance = reference_osa(query.as_bytes(), word.as_bytes());
                        (u32::from(distance) <= limit).then(|| (word.as_bytes().to_vec(), distance))
                    })
                    .collect::<BTreeMap<_, _>>();
                assert_eq!(actual, expected, "query={query:?}, limit={limit}");
            }
        }
    }

    #[test]
    fn osa_fst_retrieval_and_costs_match_exhaustive_reference() {
        let mut words = vec![String::new()];
        let mut level = vec![String::new()];
        for _ in 0..4 {
            level = level
                .iter()
                .flat_map(|prefix| ['a', 'b', 'c'].map(|ch| format!("{prefix}{ch}")))
                .collect();
            words.extend(level.iter().cloned());
        }
        words.sort();
        assert_eq!(
            reference_osa(b"ca", b"abc"),
            3,
            "OSA is not unrestricted Damerau distance"
        );
        assert_osa_search_matches_reference(&words);
    }

    #[test]
    fn osa_heap_fallback_matches_reference_across_inline_boundary() {
        let mut words = Vec::new();
        for len in [31, 32, 33, 64] {
            let prefix = "a".repeat(len - 4);
            for suffix in ["abcd", "badc", "bacx", "bac", "bacdx"] {
                words.push(format!("{prefix}{suffix}"));
            }
        }
        words.sort();
        assert_osa_search_matches_reference(&words);
    }

    #[test]
    fn exact_lookup_preserves_multiple_bangla_variants() {
        let bytes = build_loanword_bytes([
            LoanwordEntry::new("train", "ট্রেন", 16),
            LoanwordEntry::new("train", "ট্রেইন", 16),
            LoanwordEntry::new("api", "এপিআই", 16),
        ])
        .expect("loanword artifact should build");
        let lexicon = LoanwordLexicon::from_bytes(bytes).expect("artifact should load");

        let matches = lexicon.exact_matches("train");
        let words = matches.iter().map(|entry| entry.bangla).collect::<Vec<_>>();

        assert_eq!(lexicon.english_key_count(), 2);
        assert_eq!(matches.len(), 2);
        assert!(words.contains(&"ট্রেন"));
        assert!(words.contains(&"ট্রেইন"));
        assert!(lexicon.exact_matches("Train").is_empty());
    }

    #[test]
    fn fuzzy_lookup_repairs_long_english_loanword_typos() {
        let bytes = build_loanword_bytes([
            LoanwordEntry::new("university", "ইউনিভার্সিটি", 16),
            LoanwordEntry::new("engineering", "ইঞ্জিনিয়ারিং", 16),
            LoanwordEntry::new("internet", "ইন্টারনেট", 16),
        ])
        .expect("loanword artifact should build");
        let lexicon = LoanwordLexicon::from_bytes(bytes).expect("artifact should load");

        let suggestions = lexicon
            .suggestions("universty", LoanwordSearchOptions::for_input("universty"))
            .expect("loanword suggestions should run");

        let first = suggestions
            .first()
            .expect("misspelled loanword should be repaired");
        assert_eq!(first.english, "university");
        assert_eq!(first.bangla, "ইউনিভার্সিটি");
        assert_eq!(first.edit_cost, 1);
        assert_eq!(first.kind, LoanwordSuggestionKind::Fuzzy);
    }

    #[test]
    fn loanword_lookup_normalizes_english_case_at_query_time() {
        let bytes = build_loanword_bytes([
            LoanwordEntry::new("university", "ইউনিভার্সিটি", 16),
            LoanwordEntry::new("api", "এপিআই", 16),
        ])
        .expect("loanword artifact should build");
        let lexicon = LoanwordLexicon::from_bytes(bytes).expect("artifact should load");

        let titlecase = lexicon
            .suggestions("University", LoanwordSearchOptions::for_input("University"))
            .expect("loanword suggestions should run");
        let uppercase = lexicon
            .suggestions("API", LoanwordSearchOptions::for_input("API"))
            .expect("loanword suggestions should run");

        assert_eq!(
            titlecase
                .first()
                .map(|suggestion| suggestion.english.as_str()),
            Some("university")
        );
        assert_eq!(
            titlecase
                .first()
                .map(|suggestion| suggestion.bangla.as_str()),
            Some("ইউনিভার্সিটি")
        );
        assert_eq!(
            titlecase.first().map(|suggestion| suggestion.kind),
            Some(LoanwordSuggestionKind::Exact)
        );
        assert_eq!(
            uppercase
                .first()
                .map(|suggestion| suggestion.english.as_str()),
            Some("api")
        );
        assert_eq!(
            uppercase
                .first()
                .map(|suggestion| suggestion.bangla.as_str()),
            Some("এপিআই")
        );
    }

    #[test]
    fn fuzzy_loanword_lookup_normalizes_mixed_case_typos() {
        let bytes = build_loanword_bytes([LoanwordEntry::new("university", "ইউনিভার্সিটি", 16)])
            .expect("loanword artifact should build");
        let lexicon = LoanwordLexicon::from_bytes(bytes).expect("artifact should load");

        let suggestions = lexicon
            .suggestions("Universty", LoanwordSearchOptions::for_input("Universty"))
            .expect("loanword suggestions should run");

        let first = suggestions
            .first()
            .expect("mixed-case typo should be repaired");
        assert_eq!(first.english, "university");
        assert_eq!(first.bangla, "ইউনিভার্সিটি");
        assert_eq!(first.edit_cost, 1);
        assert_eq!(first.kind, LoanwordSuggestionKind::Fuzzy);
    }

    #[test]
    fn transposition_lookup_repairs_short_english_loanword_typos() {
        let bytes = build_loanword_bytes([
            LoanwordEntry::new("train", "ট্রেন", 16),
            LoanwordEntry::new("team", "টিম", 16),
        ])
        .expect("loanword artifact should build");
        let lexicon = LoanwordLexicon::from_bytes(bytes).expect("artifact should load");

        let suggestions = lexicon
            .suggestions("trian", LoanwordSearchOptions::for_input("trian"))
            .expect("loanword suggestions should run");

        let first = suggestions
            .first()
            .expect("transposed loanword should be repaired");
        assert_eq!(first.english, "train");
        assert_eq!(first.bangla, "ট্রেন");
        assert_eq!(first.edit_cost, 1);
        assert_eq!(first.kind, LoanwordSuggestionKind::Transposition);
    }

    #[test]
    fn short_loanword_keys_stay_exact_only_by_default() {
        let bytes = build_loanword_bytes([
            LoanwordEntry::new("api", "এপিআই", 16),
            LoanwordEntry::new("app", "অ্যাপ", 16),
        ])
        .expect("loanword artifact should build");
        let lexicon = LoanwordLexicon::from_bytes(bytes).expect("artifact should load");

        assert!(lexicon
            .suggestions("apj", LoanwordSearchOptions::for_input("apj"))
            .expect("loanword suggestions should run")
            .is_empty());
        assert_eq!(
            lexicon
                .suggestions("api", LoanwordSearchOptions::for_input("api"))
                .expect("loanword suggestions should run")
                .first()
                .map(|suggestion| suggestion.bangla.as_str()),
            Some("এপিআই")
        );
    }
}
