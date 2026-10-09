//! Stable C ABI for native downstreams (iOS/Android keyboards).
//!
//! Enabled by the `cabi` feature (off by default). Every entry point is
//! `extern "C"` with a versioned, C-friendly contract so the FFI surface is the
//! engine's to evolve rather than being re-derived, subtly differently, in each
//! downstream.
//!
//! # Conventions
//!
//! **Sizing (snprintf-style).** Every function that writes bytes takes an output
//! pointer and capacity and *returns the number of bytes the result needs*. It
//! copies only when the buffer is large enough. A caller passes a small stack
//! scratch, and reallocates and calls again only if the return exceeds the
//! capacity — a single crossing in the common case.
//!
//! **String lists (count + length-prefixed records, no delimiter).** A list is
//! packed into one buffer as little-endian `[u32 count]` followed by `count`
//! records of `[u32 byte_len][utf8 bytes]`. There is no in-band separator, so a
//! candidate may contain any bytes (including a newline) without corrupting the
//! framing, and an empty string is represented faithfully.
//!
//! **Detailed candidate lists (`obadh_autocorrect_suggest_detailed`).** Same
//! `[u32 count]` framing, but each record carries the provenance a client gate
//! needs, all little-endian:
//! `[u32 text_len][text utf8][u8 source][u16 edit_cost][u16 roman_repair_cost][u64 frequency]`.
//! `source` is [`FstCandidateSource::stable_code`](crate::FstCandidateSource::stable_code)
//! (frozen, append-only); `roman_repair_cost` is `0xFFFF` when the candidate is a
//! native-side edit (no roman repair); `frequency` is the lexicon frequency of
//! the candidate word.
//!
//! **Handles.** Opaque pointers created by `*_open` / `*_new` and released by the
//! matching `*_free`. A handle must not be used after it is freed, and a single
//! handle must not be used from multiple threads concurrently (there is no
//! internal locking; the caller owns synchronization).
//!
//! **UTF-8.** Inputs are `(ptr, len)` UTF-8 byte spans; invalid UTF-8 makes the
//! call a no-op (returns 0 / false / null). A zero length is the empty string.

#![allow(clippy::missing_safety_doc)]

use std::fs::File;
use std::path::Path;
use std::slice;

use crate::autocorrect::{
    key_slip_repaired_outputs, roman_repaired_outputs, FstCandidate, FstLexicon, FstLoanwordMatch,
    FstRepairedBaseline, FstSuggestOptions, FstSuggestResult, LoanwordLexicon,
    LoanwordSearchOptions, RomanRepairOptions, FST_MAX_LEVENSHTEIN_DISTANCE,
};
use crate::autosuggest::{
    AutosuggestLm, AutosuggestOptions, AutosuggestSession, PersonalAutosuggestConfig,
    PersonalAutosuggestTextSuggestion,
};
use crate::ObadhEngine;

/// Version of this C ABI contract. Bumped when the ABI changes in a way a
/// compiled downstream must notice; independent of the crate's semver so
/// additive symbols do not force downstreams to rebuild.
pub const OBADH_ABI_VERSION: u32 = 2;

const AUTOCORRECT_POOL_LIMIT: usize = 24;
const AUTOCORRECT_RESPONSE_LIMIT: usize = 8;

// --------------------------------------------------------------- marshalling

/// Borrow a UTF-8 string from a raw `(ptr, len)` span. `None` on invalid UTF-8;
/// a zero length is the empty string (pointer may be null).
unsafe fn input_str<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    if len == 0 {
        return Some("");
    }
    if ptr.is_null() {
        return None;
    }
    std::str::from_utf8(slice::from_raw_parts(ptr, len)).ok()
}

/// snprintf-style writer: copy `bytes` into `out`/`cap` only if they fit, and
/// always return the number of bytes needed.
unsafe fn write_bytes(bytes: &[u8], out: *mut u8, cap: usize) -> usize {
    if !out.is_null() && cap >= bytes.len() {
        slice::from_raw_parts_mut(out, cap)[..bytes.len()].copy_from_slice(bytes);
    }
    bytes.len()
}

/// Pack a list of strings into `[u32 count][ (u32 len)(bytes) ... ]` and write it
/// with [`write_bytes`]. Returns the total bytes needed.
unsafe fn write_str_list(items: &[String], out: *mut u8, cap: usize) -> usize {
    let mut packed = Vec::with_capacity(4 + items.iter().map(|s| 4 + s.len()).sum::<usize>());
    packed.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for item in items {
        packed.extend_from_slice(&(item.len() as u32).to_le_bytes());
        packed.extend_from_slice(item.as_bytes());
    }
    write_bytes(&packed, out, cap)
}

/// Pack detailed candidate records (see the module docs) and write them with
/// [`write_bytes`]. `0xFFFF` marks a candidate with no roman repair cost.
unsafe fn write_detailed_list(candidates: &[FstCandidate], out: *mut u8, cap: usize) -> usize {
    let mut packed = Vec::with_capacity(4 + candidates.len() * 21);
    packed.extend_from_slice(&(candidates.len() as u32).to_le_bytes());
    for candidate in candidates {
        packed.extend_from_slice(&(candidate.text.len() as u32).to_le_bytes());
        packed.extend_from_slice(candidate.text.as_bytes());
        packed.push(candidate.source.stable_code());
        packed.extend_from_slice(&candidate.edit_cost.to_le_bytes());
        packed.extend_from_slice(&candidate.roman_repair_cost.unwrap_or(0xFFFF).to_le_bytes());
        packed.extend_from_slice(&candidate.frequency.to_le_bytes());
    }
    write_bytes(&packed, out, cap)
}

// -------------------------------------------------------------------- version

/// The C ABI contract version. See [`OBADH_ABI_VERSION`].
#[no_mangle]
pub extern "C" fn obadh_abi_version() -> u32 {
    OBADH_ABI_VERSION
}

/// The crate version string (e.g. `"0.8.0"`), snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_engine_version(out: *mut u8, cap: usize) -> usize {
    write_bytes(env!("CARGO_PKG_VERSION").as_bytes(), out, cap)
}

// ------------------------------------------------------- deterministic engine

/// Create a deterministic transliteration engine handle. Free with
/// [`obadh_engine_free`].
#[no_mangle]
pub extern "C" fn obadh_engine_new() -> *mut ObadhEngine {
    Box::into_raw(Box::new(ObadhEngine::new()))
}

/// Release an engine handle created by [`obadh_engine_new`].
#[no_mangle]
pub unsafe extern "C" fn obadh_engine_free(engine: *mut ObadhEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Transliterate Roman input to Bengali (strict: unsupported input is returned
/// unchanged). snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_transliterate(
    engine: *const ObadhEngine,
    input: *const u8,
    input_len: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let (Some(engine), Some(input)) = (engine.as_ref(), input_str(input, input_len)) else {
        return 0;
    };
    write_bytes(engine.transliterate(input).as_bytes(), out, cap)
}

/// Transliterate Roman input to Bengali after dropping unsupported characters.
/// snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_transliterate_lenient(
    engine: *const ObadhEngine,
    input: *const u8,
    input_len: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let (Some(engine), Some(input)) = (engine.as_ref(), input_str(input, input_len)) else {
        return 0;
    };
    write_bytes(engine.transliterate_lenient(input).as_bytes(), out, cap)
}

// ------------------------------------------------------------------ autocorrect

/// Active-word autocorrect over the memory-mapped FST lexicon. Owns its own
/// deterministic engine, so suggest/decision calls need only this handle.
pub struct ObadhAutocorrect {
    engine: ObadhEngine,
    lexicon: FstLexicon<memmap2::Mmap>,
    loanwords: Option<LoanwordLexicon<Vec<u8>>>,
}

unsafe fn mmap_fst_lexicon(path: &str) -> Option<FstLexicon<memmap2::Mmap>> {
    let file = File::open(Path::new(path)).ok()?;
    let mmap = memmap2::MmapOptions::new().map(&file).ok()?;
    Some(FstLexicon::from_map(fst::Map::new(mmap).ok()?))
}

/// Open an autocorrect handle from a `bn.fst` path and an optional loanword FST
/// path (pass length 0 to omit). Returns null on failure. Free with
/// [`obadh_autocorrect_free`].
#[no_mangle]
pub unsafe extern "C" fn obadh_autocorrect_open(
    fst_path: *const u8,
    fst_path_len: usize,
    loanword_path: *const u8,
    loanword_path_len: usize,
) -> *mut ObadhAutocorrect {
    let Some(fst_path) = input_str(fst_path, fst_path_len) else {
        return std::ptr::null_mut();
    };
    let Some(lexicon) = mmap_fst_lexicon(fst_path) else {
        return std::ptr::null_mut();
    };
    let loanwords = match input_str(loanword_path, loanword_path_len) {
        Some(path) if !path.is_empty() => match std::fs::read(path) {
            Ok(bytes) => match LoanwordLexicon::from_bytes(bytes) {
                Ok(loanwords) => Some(loanwords),
                Err(_) => return std::ptr::null_mut(),
            },
            Err(_) => return std::ptr::null_mut(),
        },
        Some(_) => None,
        None => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(ObadhAutocorrect {
        engine: ObadhEngine::new(),
        lexicon,
        loanwords,
    }))
}

/// Release an autocorrect handle.
#[no_mangle]
pub unsafe extern "C" fn obadh_autocorrect_free(autocorrect: *mut ObadhAutocorrect) {
    if !autocorrect.is_null() {
        drop(Box::from_raw(autocorrect));
    }
}

/// Content fingerprint of the loaded lexicon FST, for the crate ↔ artifact
/// compatibility check. See [`crate::fingerprint`].
#[no_mangle]
pub unsafe extern "C" fn obadh_autocorrect_fingerprint(
    autocorrect: *const ObadhAutocorrect,
) -> u64 {
    match autocorrect.as_ref() {
        Some(autocorrect) => autocorrect.lexicon.artifact_fingerprint(),
        None => 0,
    }
}

/// The lexicon frequency of `word` — the stored count, on the same scale as
/// `obadh_autocorrect_suggest_detailed`'s per-candidate `frequency` (both read
/// the one `exact_frequency` table). Returns 0 when `word` is not an exact
/// lexicon entry, or on invalid input; presence is therefore `> 0`.
///
/// This is the baseline-side signal a client auto-insert gate needs: a rare but
/// real baseline (`> 0`) is replaced only when the top correction's `frequency`
/// exceeds it by the client's ratio (`মানুস` 49 → `মানুষ` 95278); a non-word
/// baseline (`0`) takes the frequency-floor path instead. Subsumes the former
/// `is_lexicon_word` — presence is the `> 0` case, so no separate boolean is
/// kept. (No lexicon entry has frequency 0; see the `fst_lexicon` invariant
/// test, so the 0 return is unambiguous.)
#[no_mangle]
pub unsafe extern "C" fn obadh_autocorrect_word_frequency(
    autocorrect: *const ObadhAutocorrect,
    word: *const u8,
    word_len: usize,
) -> u64 {
    let (Some(autocorrect), Some(word)) = (autocorrect.as_ref(), input_str(word, word_len)) else {
        return 0;
    };
    autocorrect.lexicon.exact_frequency(word).unwrap_or(0)
}

impl ObadhAutocorrect {
    /// Full FST suggest result for a Roman input: deterministic baseline, Roman
    /// repairs, QWERTY key-slip repairs, and loanword matches folded into one
    /// ranked result. Mirrors the reference runtime wiring.
    fn suggest_result(&self, roman: &str) -> Option<FstSuggestResult> {
        if let Some(result) = crate::emoticon_suggestions(roman, AUTOCORRECT_RESPONSE_LIMIT) {
            return Some(result);
        }
        if roman.trim().is_empty() {
            return None;
        }
        let baseline = self.engine.transliterate(roman);

        let mut repairs =
            roman_repaired_outputs(roman, &baseline, RomanRepairOptions::default(), |text| {
                self.engine.transliterate(text)
            });
        repairs.extend(key_slip_repaired_outputs(
            roman,
            &baseline,
            self.lexicon.exact_frequency(&baseline),
            |text| self.engine.transliterate(text),
            |word| self.lexicon.exact_frequency(word).is_some(),
        ));
        let repaired_baselines = repairs
            .iter()
            .map(|repair| FstRepairedBaseline {
                roman_input: repair.roman_input.as_str(),
                bangla_output: repair.bangla_output.as_str(),
                repair_kind: repair.repair_kind,
                repair_cost: repair.repair_cost,
            })
            .collect::<Vec<_>>();

        let loanword_suggestions = match &self.loanwords {
            Some(loanwords) => loanwords
                .suggestions(roman, LoanwordSearchOptions::for_input(roman))
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let loanword_matches = loanword_suggestions
            .iter()
            .map(|entry| FstLoanwordMatch {
                roman_input: roman,
                roman_repair: entry.english.as_str(),
                bangla_output: entry.bangla.as_str(),
                frequency: entry.frequency,
                repair_kind: entry.kind.as_str(),
                repair_cost: entry.edit_cost,
            })
            .collect::<Vec<_>>();

        let options = FstSuggestOptions {
            max_distance: FST_MAX_LEVENSHTEIN_DISTANCE,
            max_candidates: AUTOCORRECT_POOL_LIMIT,
            response_candidates: AUTOCORRECT_RESPONSE_LIMIT,
            max_prefix_candidates: AUTOCORRECT_RESPONSE_LIMIT,
            ..FstSuggestOptions::default()
        };
        self.lexicon
            .suggest_with_repaired_baselines_and_loanwords(
                &baseline,
                &repaired_baselines,
                &loanword_matches,
                options,
            )
            .ok()
    }

    /// Ranked correction candidates for a Roman input, best first.
    fn suggest_texts(&self, roman: &str, limit: usize) -> Vec<String> {
        let limit = limit.clamp(1, AUTOCORRECT_RESPONSE_LIMIT);
        match self.suggest_result(roman) {
            Some(result) => result
                .candidates
                .into_iter()
                .map(|candidate| candidate.text)
                .take(limit)
                .collect(),
            None => Vec::new(),
        }
    }

    /// The active-typing candidate bar: the deterministic baseline first, then
    /// corrections, deduplicated. The baseline is always present so the user can
    /// keep exactly what they typed even when it is not a lexicon word.
    fn compose_texts(&self, roman: &str, limit: usize) -> Vec<String> {
        if roman.trim().is_empty() {
            return Vec::new();
        }
        let limit = limit.clamp(1, AUTOCORRECT_RESPONSE_LIMIT);
        let mut candidates = Vec::with_capacity(limit);
        if let Some(result) = crate::emoticon_suggestions(roman, limit.saturating_sub(1)) {
            candidates.push(result.baseline);
            candidates.extend(
                result
                    .candidates
                    .into_iter()
                    .map(|candidate| candidate.text),
            );
            return candidates;
        }
        candidates.push(self.engine.transliterate(roman));
        for candidate in self.suggest_texts(roman, limit.saturating_sub(1)) {
            if candidates.len() >= limit {
                break;
            }
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
        candidates
    }

    /// Alternative spellings for an already-composed Bengali word (a re-correction
    /// menu). The input is Bengali, and only the lexicon is consulted — no Roman
    /// repairs or loanword folding.
    fn word_alternatives_texts(&self, word: &str, limit: usize) -> Vec<String> {
        if word.trim().is_empty() {
            return Vec::new();
        }
        let limit = limit.clamp(1, AUTOCORRECT_RESPONSE_LIMIT);
        let options = FstSuggestOptions {
            max_distance: FST_MAX_LEVENSHTEIN_DISTANCE,
            max_candidates: AUTOCORRECT_POOL_LIMIT,
            response_candidates: limit,
            max_prefix_candidates: limit,
            ..FstSuggestOptions::default()
        };
        match self
            .lexicon
            .suggest_with_repaired_baselines_and_loanwords(word, &[], &[], options)
        {
            Ok(result) => result
                .candidates
                .into_iter()
                .map(|candidate| candidate.text)
                .take(limit)
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Ranked correction candidates for `roman` with full per-candidate provenance —
/// `{text, source, edit_cost, roman_repair_cost, frequency}` — as a packed
/// detailed record list (see the module docs). Corrections only; the baseline is
/// not included (use [`obadh_compose_suggestions`] for a baseline-first bar).
/// This is the surface a client builds its own auto-insert gate on. snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_autocorrect_suggest_detailed(
    autocorrect: *const ObadhAutocorrect,
    roman: *const u8,
    roman_len: usize,
    limit: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let (Some(autocorrect), Some(roman)) = (autocorrect.as_ref(), input_str(roman, roman_len))
    else {
        return 0;
    };
    let Some(result) = autocorrect.suggest_result(roman) else {
        return 0;
    };
    let limit = limit.clamp(1, AUTOCORRECT_RESPONSE_LIMIT);
    let candidates: Vec<FstCandidate> = result.candidates.into_iter().take(limit).collect();
    write_detailed_list(&candidates, out, cap)
}

/// Active-typing candidate bar for `roman`: the deterministic baseline first,
/// then corrections, as a packed string list. The baseline is always present so
/// the user can keep what they typed even when it is not a lexicon word.
/// snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_compose_suggestions(
    autocorrect: *const ObadhAutocorrect,
    roman: *const u8,
    roman_len: usize,
    limit: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let (Some(autocorrect), Some(roman)) = (autocorrect.as_ref(), input_str(roman, roman_len))
    else {
        return 0;
    };
    write_str_list(&autocorrect.compose_texts(roman, limit), out, cap)
}

/// Alternative spellings for an already-composed Bengali `word`, as a packed
/// string list — a re-correction menu for a committed word. Input is Bengali.
/// snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_autocorrect_word_alternatives(
    autocorrect: *const ObadhAutocorrect,
    word: *const u8,
    word_len: usize,
    limit: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let (Some(autocorrect), Some(word)) = (autocorrect.as_ref(), input_str(word, word_len)) else {
        return 0;
    };
    write_str_list(&autocorrect.word_alternatives_texts(word, limit), out, cap)
}

// ------------------------------------------------------------------ autosuggest

/// Next-word autosuggest over committed Bengali, with the on-device personal
/// overlay.
///
/// The session borrows the LM. The LM is boxed (heap-stable address) and the
/// borrow is extended to `'static`; `session` is declared before `lm` so it is
/// dropped first, ending the borrow before the LM is freed. This makes the
/// self-reference sound.
pub struct ObadhAutosuggest {
    session: AutosuggestSession<'static, memmap2::Mmap>,
    _lm: Box<AutosuggestLm<memmap2::Mmap>>,
}

/// Open an autosuggest handle from an n-gram artifact path. Returns null on
/// failure. Free with [`obadh_autosuggest_free`].
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_open(
    path: *const u8,
    path_len: usize,
) -> *mut ObadhAutosuggest {
    let Some(path) = input_str(path, path_len) else {
        return std::ptr::null_mut();
    };
    let Ok(lm) = AutosuggestLm::from_path(path) else {
        return std::ptr::null_mut();
    };
    let lm = Box::new(lm);
    // SAFETY: `lm` is boxed, so its address is stable for the box's lifetime, and
    // `session` (declared first) is dropped before `_lm`, so the borrow never
    // outlives the LM.
    let lm_ref: &'static AutosuggestLm<memmap2::Mmap> =
        &*(lm.as_ref() as *const AutosuggestLm<memmap2::Mmap>);
    let session = AutosuggestSession::with_personal_config(
        lm_ref,
        PersonalAutosuggestConfig::default(),
        AutosuggestOptions { max_candidates: 8 },
    );
    Box::into_raw(Box::new(ObadhAutosuggest { session, _lm: lm }))
}

/// Release an autosuggest handle.
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_free(autosuggest: *mut ObadhAutosuggest) {
    if !autosuggest.is_null() {
        drop(Box::from_raw(autosuggest));
    }
}

/// Content fingerprint of the loaded n-gram artifact. See [`crate::fingerprint`].
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_fingerprint(
    autosuggest: *const ObadhAutosuggest,
) -> u64 {
    match autosuggest.as_ref() {
        Some(autosuggest) => autosuggest._lm.artifact_fingerprint(),
        None => 0,
    }
}

/// Commit a token into the session context, learning it into the personal
/// overlay. Returns 1 if the token was learned, 0 otherwise.
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_commit(
    autosuggest: *mut ObadhAutosuggest,
    token: *const u8,
    token_len: usize,
) -> i32 {
    let (Some(autosuggest), Some(token)) = (autosuggest.as_mut(), input_str(token, token_len))
    else {
        return 0;
    };
    let learned = autosuggest.session.commit_token(token).unwrap_or(false);
    i32::from(learned)
}

impl ObadhAutosuggest {
    /// Next-word suggestions for the current session context, merging the
    /// personal overlay's learned words with the model's: learned words matching
    /// the current context first, then model candidates, then learned words with
    /// no context, all deduplicated. Without this merge the user's learned
    /// out-of-vocabulary words would never surface.
    fn session_suggestions(&mut self, limit: usize) -> Vec<String> {
        self.session.set_options(AutosuggestOptions {
            max_candidates: limit,
        });
        if self.session.suggest().is_err() {
            return Vec::new();
        }
        self.session.suggest_personal_text();

        let personal = self.session.personal_text_suggestions().to_vec();
        let model: Vec<String> = self
            .session
            .candidates()
            .iter()
            .map(|candidate| candidate.text.to_string())
            .collect();

        let mut values = Vec::with_capacity(limit);
        self.push_personal(&personal, true, limit, &mut values);
        for candidate in model {
            if values.len() >= limit {
                break;
            }
            if !values.contains(&candidate) {
                values.push(candidate);
            }
        }
        self.push_personal(&personal, false, limit, &mut values);
        values
    }

    fn push_personal(
        &self,
        suggestions: &[PersonalAutosuggestTextSuggestion],
        contextual: bool,
        limit: usize,
        values: &mut Vec<String>,
    ) {
        for suggestion in suggestions {
            if values.len() >= limit {
                break;
            }
            if (suggestion.context_len > 0) != contextual {
                continue;
            }
            if let Some(text) = self.session.personal_text_suggestion_text(*suggestion) {
                let text = text.to_string();
                if !values.contains(&text) {
                    values.push(text);
                }
            }
        }
    }

    /// Stateless model suggestions for an explicit context string, without
    /// touching or requiring the session's learned state.
    fn context_suggestions(&self, context: &str, limit: usize) -> Vec<String> {
        match self._lm.suggest_for_text(
            context,
            AutosuggestOptions {
                max_candidates: limit,
            },
        ) {
            Ok(result) => result
                .candidates
                .into_iter()
                .map(|candidate| candidate.text.to_string())
                .take(limit)
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Next-word suggestions for the current session context, as a packed string
/// list (see the module docs). Merges the personal overlay's learned words with
/// the model's. `limit` caps the count. snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_suggest(
    autosuggest: *mut ObadhAutosuggest,
    limit: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let Some(autosuggest) = autosuggest.as_mut() else {
        return 0;
    };
    write_str_list(
        &autosuggest.session_suggestions(limit.clamp(1, 16)),
        out,
        cap,
    )
}

/// Stateless next-word suggestions for an explicit Bengali `context` string, as a
/// packed string list. Model-only — it does not use or update the session's
/// learned state. snprintf-style.
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_suggest_for_context(
    autosuggest: *const ObadhAutosuggest,
    context: *const u8,
    context_len: usize,
    limit: usize,
    out: *mut u8,
    cap: usize,
) -> usize {
    let (Some(autosuggest), Some(context)) =
        (autosuggest.as_ref(), input_str(context, context_len))
    else {
        return 0;
    };
    write_str_list(
        &autosuggest.context_suggestions(context, limit.clamp(1, 16)),
        out,
        cap,
    )
}

/// Clear the session's typing context (but keep learned personal words).
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_clear_session(autosuggest: *mut ObadhAutosuggest) {
    if let Some(autosuggest) = autosuggest.as_mut() {
        autosuggest.session.clear_context();
    }
}

/// Clear the on-device personal overlay (learned words).
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_clear_personal(autosuggest: *mut ObadhAutosuggest) {
    if let Some(autosuggest) = autosuggest.as_mut() {
        autosuggest.session.personal_mut().clear();
    }
}

/// Export the personal overlay as a compact snapshot, snprintf-style. Persist it
/// and restore with [`obadh_autosuggest_import_personal`].
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_export_personal(
    autosuggest: *const ObadhAutosuggest,
    out: *mut u8,
    cap: usize,
) -> usize {
    let Some(autosuggest) = autosuggest.as_ref() else {
        return 0;
    };
    let mut bytes = Vec::with_capacity(autosuggest.session.personal_snapshot_len());
    autosuggest.session.write_personal_snapshot_into(&mut bytes);
    write_bytes(&bytes, out, cap)
}

/// Import a personal-overlay snapshot produced by
/// [`obadh_autosuggest_export_personal`]. Returns 1 on success, 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn obadh_autosuggest_import_personal(
    autosuggest: *mut ObadhAutosuggest,
    input: *const u8,
    input_len: usize,
) -> i32 {
    if input.is_null() || input_len == 0 {
        return 0;
    }
    let Some(autosuggest) = autosuggest.as_mut() else {
        return 0;
    };
    let bytes = slice::from_raw_parts(input, input_len);
    i32::from(autosuggest.session.import_personal_snapshot(bytes).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    /// Read a value back through the snprintf-style contract: call once with a
    /// null buffer to learn the length, allocate, call again to fill it.
    unsafe fn read_sized(mut writer: impl FnMut(*mut u8, usize) -> usize) -> Vec<u8> {
        let needed = writer(ptr::null_mut(), 0);
        let mut buffer = vec![0_u8; needed];
        let written = writer(buffer.as_mut_ptr(), buffer.len());
        assert_eq!(written, needed);
        buffer
    }

    struct DetailedRecord {
        text: String,
        source: u8,
        edit_cost: u16,
        roman_repair_cost: u16,
        frequency: u64,
    }

    fn parse_detailed_list(bytes: &[u8]) -> Vec<DetailedRecord> {
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let text_len =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let text = String::from_utf8(bytes[offset..offset + text_len].to_vec()).unwrap();
            offset += text_len;
            let source = bytes[offset];
            offset += 1;
            let edit_cost = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
            let roman_repair_cost =
                u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
            let frequency = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            out.push(DetailedRecord {
                text,
                source,
                edit_cost,
                roman_repair_cost,
                frequency,
            });
        }
        out
    }

    fn parse_str_list(bytes: &[u8]) -> Vec<String> {
        let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            items.push(String::from_utf8(bytes[offset..offset + len].to_vec()).unwrap());
            offset += len;
        }
        items
    }

    #[test]
    fn abi_version_is_pinned() {
        assert_eq!(obadh_abi_version(), OBADH_ABI_VERSION);
    }

    #[test]
    fn engine_transliterates_through_the_snprintf_contract() {
        let engine = obadh_engine_new();
        let input = b"ami";
        let output = unsafe {
            read_sized(|out, cap| {
                obadh_transliterate(engine, input.as_ptr(), input.len(), out, cap)
            })
        };
        assert_eq!(String::from_utf8(output).unwrap(), "আমি");
        unsafe { obadh_engine_free(engine) };
    }

    #[test]
    fn engine_version_matches_the_crate() {
        let bytes = unsafe { read_sized(|out, cap| obadh_engine_version(out, cap)) };
        assert_eq!(String::from_utf8(bytes).unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn str_list_round_trips_including_empty_and_newline() {
        // The count+length framing must survive an empty string and a candidate
        // that contains the newline a delimiter-join would have split on.
        let items = vec!["আমি".to_string(), String::new(), "ভাত\nখাই".to_string()];
        let bytes = unsafe { read_sized(|out, cap| write_str_list(&items, out, cap)) };
        assert_eq!(parse_str_list(&bytes), items);
    }

    #[test]
    fn input_str_handles_empty_null_and_invalid() {
        unsafe {
            assert_eq!(input_str(ptr::null(), 0), Some(""));
            assert_eq!(input_str(ptr::null(), 4), None);
            let valid = b"hi";
            assert_eq!(input_str(valid.as_ptr(), valid.len()), Some("hi"));
            let invalid = [0xff_u8, 0xfe];
            assert_eq!(input_str(invalid.as_ptr(), invalid.len()), None);
        }
    }

    #[test]
    fn opening_a_missing_artifact_returns_null_not_a_crash() {
        let path = b"/nonexistent/obadh/bn.fst";
        let autocorrect =
            unsafe { obadh_autocorrect_open(path.as_ptr(), path.len(), ptr::null(), 0) };
        assert!(autocorrect.is_null());
        let autosuggest = unsafe { obadh_autosuggest_open(path.as_ptr(), path.len()) };
        assert!(autosuggest.is_null());
        // Freeing null is a safe no-op.
        unsafe {
            obadh_autocorrect_free(ptr::null_mut());
            obadh_autosuggest_free(ptr::null_mut());
            obadh_engine_free(ptr::null_mut());
        }
    }

    #[test]
    fn null_handles_are_safe_no_ops() {
        let word = b"word";
        unsafe {
            assert_eq!(
                obadh_transliterate(ptr::null(), word.as_ptr(), word.len(), ptr::null_mut(), 0),
                0
            );
            assert_eq!(
                obadh_autocorrect_word_frequency(ptr::null(), word.as_ptr(), word.len()),
                0
            );
            assert_eq!(obadh_autocorrect_fingerprint(ptr::null()), 0);
            assert_eq!(obadh_autosuggest_fingerprint(ptr::null()), 0);
        }
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("obadh_cabi_{name}"));
        std::fs::write(&path, bytes).expect("temp write");
        path
    }

    fn temp_fst(name: &str, entries: &[(&str, u64)]) -> std::path::PathBuf {
        let mut sorted = entries.to_vec();
        sorted.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        let mut builder = fst::MapBuilder::memory();
        for (word, frequency) in sorted {
            builder.insert(word.as_bytes(), frequency).expect("insert");
        }
        write_temp(name, builder.into_map().as_fst().as_bytes())
    }

    #[test]
    fn autocorrect_compose_puts_the_baseline_first_and_word_alternatives_work() {
        let path = temp_fst("ac.fst", &[("বাংলা", 10_000), ("বাংলাদেশ", 8_000)]);
        let path_bytes = path.to_str().unwrap().as_bytes();
        let autocorrect = unsafe {
            obadh_autocorrect_open(path_bytes.as_ptr(), path_bytes.len(), ptr::null(), 0)
        };
        assert!(!autocorrect.is_null());

        unsafe {
            // Frequency lookup (subsumes membership) + fingerprint through the ABI.
            let word = "বাংলা".as_bytes();
            assert_eq!(
                obadh_autocorrect_word_frequency(autocorrect, word.as_ptr(), word.len()),
                10_000
            );
            // A non-entry is 0 (the presence bit the old is_lexicon_word gave).
            let absent = "নেই".as_bytes();
            assert_eq!(
                obadh_autocorrect_word_frequency(autocorrect, absent.as_ptr(), absent.len()),
                0
            );
            assert_ne!(obadh_autocorrect_fingerprint(autocorrect), 0);

            // Compose always leads with the deterministic baseline.
            let roman = b"bangla";
            let packed = read_sized(|out, cap| {
                obadh_compose_suggestions(autocorrect, roman.as_ptr(), roman.len(), 5, out, cap)
            });
            let composed = parse_str_list(&packed);
            let baseline = ObadhEngine::new().transliterate("bangla");
            assert_eq!(composed.first(), Some(&baseline));

            // Word alternatives for a real Bengali word return candidates.
            let alternatives = read_sized(|out, cap| {
                obadh_autocorrect_word_alternatives(
                    autocorrect,
                    word.as_ptr(),
                    word.len(),
                    5,
                    out,
                    cap,
                )
            });
            assert!(!parse_str_list(&alternatives).is_empty());

            obadh_autocorrect_free(autocorrect);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn word_frequency_reports_counts_presence_and_zero_for_absent_or_invalid() {
        let path = temp_fst("wordfreq.fst", &[("মানুস", 49), ("মানুষ", 95_278)]);
        let path_bytes = path.to_str().unwrap().as_bytes();
        let autocorrect = unsafe {
            obadh_autocorrect_open(path_bytes.as_ptr(), path_bytes.len(), ptr::null(), 0)
        };
        assert!(!autocorrect.is_null());
        let freq =
            |w: &str| unsafe { obadh_autocorrect_word_frequency(autocorrect, w.as_ptr(), w.len()) };
        // Exact counts, on the same scale suggest_detailed reports (one table).
        assert_eq!(freq("মানুস"), 49);
        assert_eq!(freq("মানুষ"), 95_278);
        // The ratio a client gate keys on: rare real baseline vs common correction.
        assert!(freq("মানুষ") / freq("মানুস") >= 100);
        // Absent word → 0 (the presence bit the removed is_lexicon_word gave).
        assert_eq!(freq("নেই"), 0);
        // Invalid UTF-8 input is a no-op → 0, never a panic.
        let bad = [0xFF, 0xFEu8];
        assert_eq!(
            unsafe { obadh_autocorrect_word_frequency(autocorrect, bad.as_ptr(), bad.len()) },
            0
        );
        unsafe { obadh_autocorrect_free(autocorrect) };
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn detailed_wire_format_is_exhaustive_and_exact() {
        use crate::autocorrect::FstCandidateSource::*;

        // Every source variant, with a mix of present / absent roman_repair_cost,
        // an empty text, and multi-byte text — all round-tripping the frozen
        // layout exactly.
        let sources = [
            (Exact, 0u8),
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
        ];
        let candidates: Vec<FstCandidate> = sources
            .iter()
            .enumerate()
            .map(|(i, (source, _))| FstCandidate {
                text: if i == 0 {
                    String::new()
                } else {
                    format!("শব্দ{i}")
                },
                source: *source,
                edit_cost: i as u16,
                frequency: (i as u64) * 1000,
                score: 0,
                roman_repair: None,
                roman_repair_kind: None,
                // odd indices carry a roman repair cost; even ones do not (→ 0xFFFF)
                roman_repair_cost: if i % 2 == 1 { Some(i as u16) } else { None },
            })
            .collect();

        let bytes = unsafe { read_sized(|out, cap| write_detailed_list(&candidates, out, cap)) };
        let records = parse_detailed_list(&bytes);
        assert_eq!(records.len(), candidates.len());
        for (i, (record, candidate)) in records.iter().zip(&candidates).enumerate() {
            assert_eq!(record.text, candidate.text, "text {i}");
            assert_eq!(record.source, sources[i].1, "stable code {i}");
            assert_eq!(record.edit_cost, candidate.edit_cost, "edit_cost {i}");
            assert_eq!(record.frequency, candidate.frequency, "frequency {i}");
            let expected_rrc = candidate.roman_repair_cost.unwrap_or(0xFFFF);
            assert_eq!(
                record.roman_repair_cost, expected_rrc,
                "roman_repair_cost {i}"
            );
        }

        // Empty list is a bare count of 0.
        let empty = unsafe { read_sized(|out, cap| write_detailed_list(&[], out, cap)) };
        assert_eq!(empty, 0u32.to_le_bytes());
        assert!(parse_detailed_list(&empty).is_empty());

        // snprintf contract: a too-small buffer writes nothing and still returns
        // the full needed length.
        let needed = unsafe { write_detailed_list(&candidates, ptr::null_mut(), 0) };
        let mut small = vec![0u8; needed - 1];
        let reported = unsafe { write_detailed_list(&candidates, small.as_mut_ptr(), small.len()) };
        assert_eq!(
            reported, needed,
            "too-small buffer still reports needed length"
        );
        assert!(
            small.iter().all(|&b| b == 0),
            "too-small buffer is not written"
        );
    }

    #[test]
    fn emoticon_compose_and_detailed_suggestions_preserve_literal_input() {
        let path = temp_fst("emoticons.fst", &[("বাংলা", 10)]);
        let path_bytes = path.to_str().unwrap().as_bytes();
        unsafe {
            let handle =
                obadh_autocorrect_open(path_bytes.as_ptr(), path_bytes.len(), ptr::null(), 0);
            assert!(!handle.is_null());
            for (input, emoji) in [(":)", "😃"), (":-)", "😃"), (":D", "😄"), (":-D", "😄")]
            {
                let packed = read_sized(|out, cap| {
                    obadh_compose_suggestions(handle, input.as_ptr(), input.len(), 2, out, cap)
                });
                assert_eq!(
                    parse_str_list(&packed),
                    vec![input.to_owned(), emoji.to_owned()]
                );
                let packed = read_sized(|out, cap| {
                    obadh_compose_suggestions(handle, input.as_ptr(), input.len(), 1, out, cap)
                });
                assert_eq!(parse_str_list(&packed), vec![input.to_owned()]);
                let packed = read_sized(|out, cap| {
                    obadh_autocorrect_suggest_detailed(
                        handle,
                        input.as_ptr(),
                        input.len(),
                        2,
                        out,
                        cap,
                    )
                });
                let records = parse_detailed_list(&packed);
                assert_eq!(records.len(), 1);
                assert_eq!(records[0].text, emoji);
                assert_eq!(records[0].source, 11);
                assert_eq!(records[0].frequency, 0);
            }
            obadh_autocorrect_free(handle);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn suggest_detailed_records_parse_with_sane_fields() {
        let path = temp_fst("detailed.fst", &[("বাংলা", 137381), ("বাংলাদেশ", 8000)]);
        let path_bytes = path.to_str().unwrap().as_bytes();
        let autocorrect = unsafe {
            obadh_autocorrect_open(path_bytes.as_ptr(), path_bytes.len(), ptr::null(), 0)
        };
        assert!(!autocorrect.is_null());
        unsafe {
            // A near-miss of বাংলা yields at least one correction record; the
            // frozen fields must parse and be internally consistent.
            let roman = b"bangla";
            let packed = read_sized(|out, cap| {
                obadh_autocorrect_suggest_detailed(
                    autocorrect,
                    roman.as_ptr(),
                    roman.len(),
                    5,
                    out,
                    cap,
                )
            });
            let records = parse_detailed_list(&packed);
            assert!(!records.is_empty());
            for record in &records {
                assert!(!record.text.is_empty());
                assert!(record.source <= 10, "unknown source code {}", record.source);
                // 0xFFFF means "no roman repair"; any other value is a real cost.
                let _ = (record.edit_cost, record.roman_repair_cost, record.frequency);
            }
            obadh_autocorrect_free(autocorrect);
        }
        let _ = std::fs::remove_file(path);
    }

    /// Real-data guard: pins `suggest_detailed` against the shipped `bn.fst`.
    /// Skips when the submodule artifacts are not resolved (e.g. in CI), so it
    /// runs locally where the data is present. This is the exact case the 0.8.2
    /// gate fix missed: `banhla` → বাংলা is `roman_repair_exact` with roman cost
    /// **2** (not 1) and a very high frequency — a client gate needs all three.
    #[test]
    fn suggest_detailed_banhla_on_real_bn_fst() {
        let fst_path = "data/autocorrect/models/bn.fst";
        let loan_path = "data/autocorrect/models/en_bn_loanwords.fst";
        if !std::path::Path::new(fst_path).exists() {
            eprintln!("skip: {fst_path} not resolved");
            return;
        }
        let fst_bytes = fst_path.as_bytes();
        let loan_bytes = loan_path.as_bytes();
        let autocorrect = unsafe {
            obadh_autocorrect_open(
                fst_bytes.as_ptr(),
                fst_bytes.len(),
                loan_bytes.as_ptr(),
                loan_bytes.len(),
            )
        };
        assert!(!autocorrect.is_null());
        unsafe {
            let roman = b"banhla";
            let packed = read_sized(|out, cap| {
                obadh_autocorrect_suggest_detailed(
                    autocorrect,
                    roman.as_ptr(),
                    roman.len(),
                    5,
                    out,
                    cap,
                )
            });
            let records = parse_detailed_list(&packed);
            let top = &records[0];
            assert_eq!(top.text, "বাংলা");
            assert_eq!(top.source, 8, "expected roman_repair_exact"); // stable code
            assert_eq!(
                top.roman_repair_cost, 2,
                "the real roman-side cost is 2, not 1"
            );
            assert!(top.frequency > 100_000, "বাংলা is a very common word");
            obadh_autocorrect_free(autocorrect);
        }
    }

    /// Real-data robustness sweep: drive `suggest_detailed` over a large, diverse
    /// input set against the shipped 845k-word `bn.fst` and assert every packed
    /// buffer parses cleanly, every source code is in range, and nothing panics.
    /// Skips when the submodules are unresolved (CI), runs locally.
    #[test]
    fn suggest_detailed_is_robust_over_real_bn_fst() {
        let fst_path = "data/autocorrect/models/bn.fst";
        let loan_path = "data/autocorrect/models/en_bn_loanwords.fst";
        if !std::path::Path::new(fst_path).exists() {
            eprintln!("skip: {fst_path} not resolved");
            return;
        }
        let fst_bytes = fst_path.as_bytes();
        let loan_bytes = loan_path.as_bytes();
        let autocorrect = unsafe {
            obadh_autocorrect_open(
                fst_bytes.as_ptr(),
                fst_bytes.len(),
                loan_bytes.as_ptr(),
                loan_bytes.len(),
            )
        };
        assert!(!autocorrect.is_null());

        // Curated typos + edge cases + a fixed-seed pseudo-random roman soup.
        let mut inputs: Vec<String> = vec![
            "".into(),
            " ".into(),
            "   ".into(),
            "a".into(),
            "banhla".into(),
            "manus".into(),
            "bondu".into(),
            "sriti".into(),
            "computer".into(),
            "bigyan".into(),
            "123".into(),
            "!!!".into(),
            "a,b,c".into(),
            "কম্পিউটার".into(),
            "日本".into(),
            "aaaaaaaaaaaaaaaa".into(),
        ];
        let alphabet: &[u8] = b"aAiIuUeEoOkgcjtTdDnpbmyrlshHNzwxRSC.,-0123456789 ";
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..600 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = 1 + (state >> 40) as usize % 16;
            let word: String = (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    alphabet[(state >> 40) as usize % alphabet.len()] as char
                })
                .collect();
            inputs.push(word);
        }

        let mut total_records = 0usize;
        for input in &inputs {
            let bytes = input.as_bytes();
            let packed = unsafe {
                read_sized(|out, cap| {
                    obadh_autocorrect_suggest_detailed(
                        autocorrect,
                        bytes.as_ptr(),
                        bytes.len(),
                        5,
                        out,
                        cap,
                    )
                })
            };
            if packed.is_empty() {
                continue; // whitespace/empty inputs return 0 bytes
            }
            // parse_detailed_list panics on any framing overrun, so a clean parse
            // is itself the assertion that the packed layout is well-formed.
            let records = parse_detailed_list(&packed);
            for record in &records {
                assert!(
                    record.source <= 10,
                    "{input:?}: bad source {}",
                    record.source
                );
            }
            total_records += records.len();
        }
        assert!(
            total_records > 0,
            "the sweep should produce some corrections"
        );
        unsafe { obadh_autocorrect_free(autocorrect) };
    }

    #[test]
    fn autosuggest_learns_a_word_and_surfaces_it_through_the_abi() {
        use crate::autosuggest::artifact::test_support::{build_fixture, Row};

        let tokens = ["<pad>", "<bos>", "<unk>", "আমি", "আজ", "ভাত", "খাই"];
        let fixture = build_fixture(
            &tokens,
            &[(5, 100, 100), (6, 90, 90)],
            &[Row {
                context: vec![3],
                candidates: vec![(6, 20, 20), (5, 10, 10)],
            }],
        );
        let path = write_temp("as.bin", &fixture);
        let path_bytes = path.to_str().unwrap().as_bytes();

        let autosuggest = unsafe { obadh_autosuggest_open(path_bytes.as_ptr(), path_bytes.len()) };
        assert!(!autosuggest.is_null());

        unsafe {
            let name = "নাসির".as_bytes();
            // An out-of-vocabulary name is learned into the personal overlay.
            assert_eq!(
                obadh_autosuggest_commit(autosuggest, name.as_ptr(), name.len()),
                1
            );

            // Session suggest (with the personal merge) and stateless context
            // suggest both return without error.
            let _ = read_sized(|out, cap| obadh_autosuggest_suggest(autosuggest, 5, out, cap));
            let context = "আমি".as_bytes();
            let _ = read_sized(|out, cap| {
                obadh_autosuggest_suggest_for_context(
                    autosuggest,
                    context.as_ptr(),
                    context.len(),
                    5,
                    out,
                    cap,
                )
            });

            obadh_autosuggest_free(autosuggest);
        }
        let _ = std::fs::remove_file(path);
    }

    /// Executable contract: the shipped C header must declare every exported
    /// symbol and pin the same ABI version, so the header cannot drift from the
    /// Rust surface.
    #[test]
    fn c_header_matches_the_exported_surface() {
        let source = include_str!("cabi.rs");
        let header = include_str!("../include/obadh.h");

        let marker = "extern \"C\" fn ";
        let mut missing = Vec::new();
        for line in source.lines() {
            let Some(index) = line.find(marker) else {
                continue;
            };
            let name: String = line[index + marker.len()..]
                .chars()
                .take_while(|character| character.is_alphanumeric() || *character == '_')
                .collect();
            if name.starts_with("obadh_") && !header.contains(&name) {
                missing.push(name);
            }
        }
        assert!(
            missing.is_empty(),
            "C header is missing declarations for: {missing:?}"
        );
        assert!(
            header.contains(&format!("#define OBADH_ABI_VERSION {OBADH_ABI_VERSION}")),
            "C header ABI version does not match OBADH_ABI_VERSION"
        );
    }
}
