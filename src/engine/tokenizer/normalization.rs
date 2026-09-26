use super::{move_unit, reph_base_part, PhoneticUnit, PhoneticUnitType};
use crate::definitions::conjuncts::ConjunctDefinitions;

pub(super) fn normalize_reph_and_vocalic_r(units: &mut Vec<PhoneticUnit>) {
    let mut read = 0;
    let mut write = 0;

    while read < units.len() {
        if read + 1 < units.len()
            && units[read].text == "rr"
            && units[read].unit_type == PhoneticUnitType::SpecialForm
            && units[read + 1].text == "i"
            && units[read + 1].unit_type == PhoneticUnitType::Vowel
        {
            let position = units[read].position;
            units[write] = PhoneticUnit {
                text: String::from("rri"),
                unit_type: PhoneticUnitType::Vowel,
                position,
            };
            read += 2;
            write += 1;
            continue;
        }

        if read + 1 < units.len()
            && units[read].text == "rr"
            && units[read].unit_type == PhoneticUnitType::SpecialForm
            && units[read + 1].unit_type == PhoneticUnitType::Consonant
        {
            let position = units[read].position;
            let next_text = units[read + 1].text.as_str();
            let mut reph_text = String::with_capacity(2 + next_text.len());
            reph_text.push_str("rr");
            reph_text.push_str(next_text);

            units[write] = PhoneticUnit {
                text: reph_text,
                unit_type: PhoneticUnitType::RephOverConsonant,
                position,
            };
            read += 2;
            write += 1;
            continue;
        }

        move_unit(units, read, write);

        read += 1;
        write += 1;
    }

    units.truncate(write);
}

pub(super) fn normalize_redundant_reph_hasant(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_redundant_reph_hasant(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_redundant_reph_hasant_at(units, read) {
            move_unit(units, read, write);
            read += 2;
            write += 1;
            continue;
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

pub(super) fn normalize_redundant_khanda_ta_hasant(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_redundant_khanda_ta_hasant(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_redundant_khanda_ta_hasant_at(units, read) {
            move_unit(units, read, write);
            read += 2;
            write += 1;
            continue;
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

pub(super) fn normalize_velar_nasal_conjunct_aliases(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_velar_nasal_conjunct_alias(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_velar_nasal_conjunct_alias_at(units, read) {
            if let Some(canonical_tail) = velar_nasal_conjunct_tail(&units[read + 1].text) {
                let position = units[read].position;
                let mut text = String::with_capacity(4 + canonical_tail.len());
                text.push_str("Ng,,");
                text.push_str(canonical_tail);

                units[write] = PhoneticUnit {
                    text,
                    unit_type: PhoneticUnitType::Conjunct,
                    position,
                };
                read += 2;
                write += 1;
                continue;
            }
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
    }

    units.truncate(write);
}

/// An `ng` anusvar directly before a vowel cannot be valid: an anusvar carries no
/// vowel in Bangla, so the baseline emits a bare independent vowel instead. Render
/// the `ng` as the velar nasal `Ng` so the vowel attaches to it as a kar. Only bare
/// `ng` before a vowel is touched; `ng` before a consonant or at a word end stays
/// anusvar, and `ngg`/`nggh` are already handled as the velar-nasal conjuncts.
pub(super) fn normalize_anusvar_ng_before_vowel(units: &mut [PhoneticUnit]) {
    for index in 0..units.len() {
        if is_anusvar_ng_before_vowel_at(units, index) {
            units[index].text = String::from("Ng");
            units[index].unit_type = PhoneticUnitType::Consonant;
        }
    }
}

fn is_anusvar_ng_before_vowel_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 1 < units.len()
        && units[index].unit_type == PhoneticUnitType::SpecialForm
        && units[index].text == "ng"
        && matches!(
            units[index + 1].unit_type,
            PhoneticUnitType::Vowel | PhoneticUnitType::TerminatingVowel
        )
}

/// A `:` bisarga directly between two numerals is a digit-group separator, not a
/// bisarga: a clock time like `9:45`. Retag it as a plain symbol so it renders as a
/// literal `:` between the Bengali digits, the way `.` stays literal in `3.14`. A
/// `:` next to a letter keeps its bisarga meaning, so `du:kho` stays দুঃখ.
pub(super) fn normalize_colon_between_numerals(units: &mut [PhoneticUnit]) {
    for index in 0..units.len() {
        if is_colon_between_numerals_at(units, index) {
            units[index].unit_type = PhoneticUnitType::Symbol;
        }
    }
}

fn is_colon_between_numerals_at(units: &[PhoneticUnit], index: usize) -> bool {
    index > 0
        && index + 1 < units.len()
        && units[index].unit_type == PhoneticUnitType::SpecialForm
        && units[index].text == ":"
        && is_numeral_text(&units[index - 1].text)
        && is_numeral_text(&units[index + 1].text)
}

/// A unit whose text is all digits, ASCII or already-Bengali. Accepting Bengali
/// digits keeps `9:45` -> `৯:৪৫` stable when its own output is transliterated again.
fn is_numeral_text(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|character| character.is_numeric())
}

pub(super) fn normalize_non_conjunct_ra_ya_zwnj(
    units: &mut Vec<PhoneticUnit>,
    conjuncts: &ConjunctDefinitions,
) {
    // Read-only pass: find every marker start against the original, unmutated
    // units. The leading-র guard inspects preceding units, and the compacting
    // pass below empties source slots as it advances, so detection must happen
    // before any mutation.
    let mut marker_starts: Vec<usize> = Vec::new();
    let mut index = 0;
    while index < units.len() {
        if is_leading_ra_ya_marker_at(units, index, conjuncts) {
            marker_starts.push(index);
            index += 2;
        } else {
            index += 1;
        }
    }
    if marker_starts.is_empty() {
        return;
    }

    // Apply pass: collapse each র + marker (Y/Z) into the reph-ya conjunct. A
    // trailing lowercase `y` is the য় glide and composes after it (রZy → র‍্যয়).
    let mut starts = marker_starts.into_iter().peekable();
    let mut read = 0;
    let mut write = 0;
    while read < units.len() {
        if starts.peek() == Some(&read) {
            starts.next();
            units[write] = PhoneticUnit {
                text: String::from("rZ,,y"),
                unit_type: PhoneticUnitType::Conjunct,
                position: units[read].position,
            };
            read += 2;
            write += 1;
        } else {
            move_unit(units, read, write);
            read += 1;
            write += 1;
        }
    }

    units.truncate(write);
}

/// Fold any `Z` left after the `rZy` marker pass into a plain `z` (য). `Z` is held
/// out of case-folding so the narrow `rZy` non-conjunct ra-ya marker can claim it;
/// a `Z` that no marker consumed has no other meaning, so it becomes `z` rather
/// than leaking a literal Latin glyph into the output.
pub(super) fn normalize_residual_reserved_z(units: &mut [PhoneticUnit]) {
    for unit in units.iter_mut() {
        if unit.unit_type == PhoneticUnitType::Unknown && unit.text == "Z" {
            unit.text = String::from("z");
            unit.unit_type = PhoneticUnitType::Consonant;
        }
    }
}

fn first_redundant_reph_hasant(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_redundant_reph_hasant_at(units, index))
}

fn is_redundant_reph_hasant_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && units[index].unit_type == PhoneticUnitType::SpecialForm
        && units[index].text == "rr"
        && units[index + 1].unit_type == PhoneticUnitType::ConsonantWithHasant
        && is_reph_target_after_redundant_hasant(&units[index + 2])
}

fn is_reph_target_after_redundant_hasant(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant || is_khanda_ta_unit(unit)
}

fn first_redundant_khanda_ta_hasant(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_redundant_khanda_ta_hasant_at(units, index))
}

fn is_redundant_khanda_ta_hasant_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && is_khanda_ta_unit(&units[index])
        && units[index + 1].unit_type == PhoneticUnitType::ConsonantWithHasant
        && units[index + 2].unit_type == PhoneticUnitType::Consonant
}

fn is_khanda_ta_unit(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::SpecialForm && matches!(unit.text.as_str(), "t``" | "T``")
}

fn first_velar_nasal_conjunct_alias(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(1)).find(|&index| {
        is_velar_nasal_conjunct_alias_at(units, index)
            && velar_nasal_conjunct_tail(&units[index + 1].text).is_some()
    })
}

fn is_velar_nasal_conjunct_alias_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 1 < units.len()
        && units[index].unit_type == PhoneticUnitType::SpecialForm
        && units[index].text == "ng"
        && units[index + 1].unit_type == PhoneticUnitType::Consonant
}

fn velar_nasal_conjunct_tail(text: &str) -> Option<&'static str> {
    match text {
        "g" => Some("g"),
        "gh" | "Gh" | "GH" => Some("gh"),
        _ => None,
    }
}

/// A syllable-leading র directly followed by a ya-phola *marker* — capital `Y`
/// (a Consonant unit) or the reserved `Z` (an Unknown unit) — forms the reph-ya
/// র‍্য. Lowercase `y` is the য় glide, not a marker (`ry` → রয়), so a trailing
/// `y` composes after the marker (`rZy` / `rYy` → র‍্যয়). The leading guard keeps
/// an r-phola conjunct tail on its productive ya-phola (`krY` → ক্র্য, `TrYak` →
/// ট্র্যাক) and leaves the glide alone.
fn is_leading_ra_ya_marker_at(
    units: &[PhoneticUnit],
    index: usize,
    conjuncts: &ConjunctDefinitions,
) -> bool {
    if units[index].unit_type != PhoneticUnitType::Consonant || units[index].text != "r" {
        return false;
    }
    let Some(next) = units.get(index + 1) else {
        return false;
    };
    let is_marker = (next.unit_type == PhoneticUnitType::Consonant && next.text == "Y")
        || (next.unit_type == PhoneticUnitType::Unknown && next.text == "Z");
    is_marker && ra_leads_syllable(units, index, conjuncts)
}

/// Whether র at `index` leads its own syllable (so a ya-phola marker forms the
/// reph-ya র‍্য) instead of being the r-phola tail of a conjunct. র does NOT lead
/// only when an explicit hasant binds it to the preceding consonant (T,,rY →
/// ট্র্যাক) or when the whole preceding consonant *run* plus র forms a listed
/// r-phola conjunct (ক্র, ট্র, স্ত্র). After a vowel, numeral, anusvar or any
/// other boundary — or after a consonant run that does not join র (ন্র, or ক্ক·র
/// where র stays standalone) — র starts a fresh syllable.
fn ra_leads_syllable(
    units: &[PhoneticUnit],
    index: usize,
    conjuncts: &ConjunctDefinitions,
) -> bool {
    if index == 0 {
        return true;
    }
    let prev = &units[index - 1];
    match prev.unit_type {
        // Explicit hasant binds র to the preceding consonant (T,,rY → ট্র্যাক).
        PhoneticUnitType::ConsonantWithHasant => false,
        // Reph already sits over a consonant; র joins that consonant as an r-phola
        // when they form a conjunct (rrk·rY → র্ক্র্য).
        PhoneticUnitType::RephOverConsonant => reph_base_part(prev)
            .map_or(true, |base| {
                conjuncts.create_conjunct_from_parts(&[base, "r"]).is_none()
            }),
        // A preceding consonant run: segment it greedily (the same longest-match
        // order the conjunct pass uses) and let র lead only when it starts a fresh
        // segment. র joins the LAST segment when that segment plus র forms a
        // conjunct (ksTr → ক্স + ট্র, so eksTrYak → এক্সট্র্যাক); otherwise the
        // segment is closed and র leads (ক্ক·র → ক্কর‍্য, ন·র → নর‍্য).
        PhoneticUnitType::Consonant => {
            let mut run_start = index;
            while run_start > 0
                && units[run_start - 1].unit_type == PhoneticUnitType::Consonant
            {
                run_start -= 1;
            }
            let last_start = last_conjunct_segment_start(units, run_start, index, conjuncts);
            let mut parts: Vec<&str> =
                units[last_start..index].iter().map(|u| u.text.as_str()).collect();
            parts.push("r");
            conjuncts.create_conjunct_from_parts(&parts).is_none()
        }
        // Vowel, numeral, anusvar/symbol, or any other boundary: র leads.
        _ => true,
    }
}

/// Greedily segment the consonant range `[run_start, end)` into conjuncts (the
/// same longest-match order the conjunct pass uses) and return the start index of
/// the final segment — the consonants a following র could still attach to.
fn last_conjunct_segment_start(
    units: &[PhoneticUnit],
    run_start: usize,
    end: usize,
    conjuncts: &ConjunctDefinitions,
) -> usize {
    let mut seg_start = run_start;
    while seg_start < end {
        let mut best_end = seg_start + 1;
        let mut parts: Vec<&str> = vec![units[seg_start].text.as_str()];
        for next in (seg_start + 1)..end {
            parts.push(units[next].text.as_str());
            if conjuncts.create_conjunct_from_parts(&parts).is_some() {
                best_end = next + 1;
            }
        }
        if best_end >= end {
            break;
        }
        seg_start = best_end;
    }
    seg_start
}
