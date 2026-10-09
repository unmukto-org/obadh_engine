use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

use obadh_engine::{
    key_slip_repaired_outputs, roman_repaired_outputs, FstCandidate, FstLexicon, FstLoanwordMatch,
    FstRepairedBaseline, FstSuggestOptions, LoanwordLexicon, LoanwordSearchOptions, ObadhEngine,
    RomanRepairOptions, RomanRepairedOutput, DEFAULT_FST_MAX_DISTANCE,
    DEFAULT_FST_PREFIX_CANDIDATES, DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
};
use serde::Serialize;

use super::{ensure_parent_dir, read_lexicon_tsvs};

#[cfg(not(target_arch = "wasm32"))]
type FstMapData = memmap2::Mmap;

#[cfg(target_arch = "wasm32")]
type FstMapData = Vec<u8>;

#[derive(Debug, Serialize)]
pub struct FstBuildReport {
    inputs: Vec<String>,
    output: String,
    input_rows: usize,
    duplicate_rows: usize,
    total_frequency: u64,
    max_frequency: u32,
    entries: usize,
    artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct FstInspectReport {
    input: String,
    entries: usize,
    artifact_bytes: u64,
    /// Content fingerprint for the crate ↔ artifact compatibility table. Printing
    /// this is how a release records which `bn.fst` revision a crate version was
    /// built against.
    artifact_fingerprint: u64,
}

#[derive(Debug, Serialize)]
pub struct FstSuggestReport {
    input: String,
    obadh_output: String,
    roman_repairs: Vec<FstRomanRepairReport>,
    exact_frequency: Option<u64>,
    max_distance: u32,
    max_edit_cost: Option<u16>,
    candidate_count: usize,
    returned_candidates: usize,
    truncated: bool,
    candidates: Vec<FstSuggestCandidate>,
}

#[derive(Debug, Serialize)]
struct FstSuggestCandidate {
    text: String,
    source: &'static str,
    edit_cost: u16,
    frequency: u64,
    score: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    roman_repair: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    roman_repair_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    roman_repair_cost: Option<u16>,
}

#[derive(Debug, Serialize)]
struct FstRomanRepairReport {
    input: String,
    output: String,
    kind: &'static str,
    cost: u16,
}

pub fn default_max_distance() -> u32 {
    DEFAULT_FST_MAX_DISTANCE
}

pub fn default_prefix_candidates() -> usize {
    DEFAULT_FST_PREFIX_CANDIDATES
}

pub fn build_fst_lexicon_artifact(
    inputs: &[PathBuf],
    output: &PathBuf,
    allow_non_bangla: bool,
) -> Result<FstBuildReport, Box<dyn std::error::Error>> {
    ensure_parent_dir(output)?;

    let lexicon_input = read_lexicon_tsvs(inputs, allow_non_bangla)?;
    let file = File::create(output)?;
    let writer = BufWriter::new(file);
    let mut builder = fst::MapBuilder::new(writer)?;
    for entry in &lexicon_input.entries {
        builder.insert(entry.word.as_bytes(), entry.frequency as u64)?;
    }
    builder.finish()?;

    Ok(FstBuildReport {
        inputs: inputs
            .iter()
            .map(|input| input.display().to_string())
            .collect(),
        output: output.display().to_string(),
        input_rows: lexicon_input.rows,
        duplicate_rows: lexicon_input.duplicate_rows,
        total_frequency: lexicon_input.total_frequency,
        max_frequency: lexicon_input.max_frequency,
        entries: lexicon_input.entries.len(),
        artifact_bytes: fs::metadata(output)?.len(),
    })
}

pub fn inspect_fst_lexicon(
    input: &PathBuf,
) -> Result<FstInspectReport, Box<dyn std::error::Error>> {
    let map = read_fst_lexicon(input)?;
    Ok(FstInspectReport {
        input: input.display().to_string(),
        entries: map.len(),
        artifact_bytes: fs::metadata(input)?.len(),
        artifact_fingerprint: map.artifact_fingerprint(),
    })
}

pub fn suggest_fst(
    input: &str,
    lexicon_model: &FstLexicon<FstMapData>,
    loanword_model: Option<&LoanwordLexicon<Vec<u8>>>,
    max_distance: u32,
    max_edit_cost: Option<u16>,
    max_candidates: usize,
    max_prefix_candidates: usize,
    response_candidates: usize,
) -> Result<FstSuggestReport, Box<dyn std::error::Error>> {
    if let Some(result) = obadh_engine::emoticon_suggestions(input, response_candidates) {
        return Ok(FstSuggestReport {
            input: input.to_owned(),
            obadh_output: result.baseline,
            roman_repairs: Vec::new(),
            exact_frequency: result.exact_frequency,
            max_distance: result.max_distance,
            max_edit_cost: result.max_edit_cost,
            candidate_count: result.candidate_count,
            returned_candidates: result.returned_candidates,
            truncated: result.truncated,
            candidates: result
                .candidates
                .into_iter()
                .map(fst_suggest_candidate)
                .collect(),
        });
    }
    let obadh = ObadhEngine::new();
    let obadh_output = obadh.transliterate(input);
    let mut repair_records = fst_roman_repairs(input, &obadh, &obadh_output);
    // QWERTY fat-finger (key-slip) repairs, gated to non-word baselines and validated against
    // the lexicon inside the helper — the engine owns every numeric parameter.
    repair_records.extend(key_slip_repaired_outputs(
        input,
        &obadh_output,
        lexicon_model.exact_frequency(&obadh_output),
        |roman| obadh.transliterate(roman),
        |word| lexicon_model.exact_frequency(word).is_some(),
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
    let loanword_suggestions = loanword_model
        .map(|loanwords| loanwords.suggestions(input, LoanwordSearchOptions::for_input(input)))
        .transpose()?
        .unwrap_or_default();
    let fst_loanword_matches = loanword_suggestions
        .iter()
        .map(|entry| FstLoanwordMatch {
            roman_input: input,
            roman_repair: entry.english.as_str(),
            bangla_output: entry.bangla.as_str(),
            frequency: entry.frequency,
            repair_kind: entry.kind.as_str(),
            repair_cost: entry.edit_cost,
        })
        .collect::<Vec<_>>();
    let result = lexicon_model.suggest_with_repaired_baselines_and_loanwords(
        &obadh_output,
        &repaired_baselines,
        &fst_loanword_matches,
        FstSuggestOptions {
            max_distance,
            max_edit_cost,
            max_candidates,
            max_prefix_candidates,
            response_candidates,
        },
    )?;
    let candidates = result
        .candidates
        .into_iter()
        .map(fst_suggest_candidate)
        .collect::<Vec<_>>();

    Ok(FstSuggestReport {
        input: input.to_string(),
        obadh_output,
        roman_repairs: repair_records
            .into_iter()
            .map(|repair| FstRomanRepairReport {
                input: repair.roman_input,
                output: repair.bangla_output,
                kind: repair.repair_kind,
                cost: repair.repair_cost,
            })
            .collect(),
        exact_frequency: result.exact_frequency,
        max_distance: result.max_distance,
        max_edit_cost: result.max_edit_cost,
        candidate_count: result.candidate_count,
        returned_candidates: candidates.len(),
        truncated: result.truncated,
        candidates,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_fst_lexicon(
    input: &PathBuf,
) -> Result<FstLexicon<FstMapData>, Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    // The mapping is read-only and the file handle is kept valid for the map
    // creation call. The returned mmap owns the OS mapping afterwards.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    // The dropped-vowel channel reads skeleton-mates from this same fst at query time (via
    // the skeleton automaton), so there is no separate index to load or attach.
    Ok(FstLexicon::from_map(fst::Map::new(mmap)?))
}

#[cfg(target_arch = "wasm32")]
pub fn read_fst_lexicon(
    input: &PathBuf,
) -> Result<FstLexicon<FstMapData>, Box<dyn std::error::Error>> {
    Ok(FstLexicon::from_bytes(fs::read(input)?)?)
}

fn fst_suggest_candidate(candidate: FstCandidate) -> FstSuggestCandidate {
    FstSuggestCandidate {
        text: candidate.text,
        source: candidate.source.as_str(),
        edit_cost: candidate.edit_cost,
        frequency: candidate.frequency,
        score: candidate.score,
        roman_repair: candidate.roman_repair,
        roman_repair_kind: candidate.roman_repair_kind,
        roman_repair_cost: candidate.roman_repair_cost,
    }
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
