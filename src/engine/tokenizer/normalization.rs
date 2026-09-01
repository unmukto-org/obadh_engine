use super::{move_unit, PhoneticUnit, PhoneticUnitType};

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

pub(super) fn normalize_non_conjunct_ra_ya_zwnj(units: &mut Vec<PhoneticUnit>) {
    let Some(first_match) = first_non_conjunct_ra_ya_zwnj(units) else {
        return;
    };

    let mut read = first_match;
    let mut write = first_match;

    while read < units.len() {
        if is_non_conjunct_ra_ya_zwnj_at(units, read) {
            units[write] = PhoneticUnit {
                text: String::from("rZ,,y"),
                unit_type: PhoneticUnitType::Conjunct,
                position: units[read].position,
            };
            read += 3;
            write += 1;
            continue;
        }

        move_unit(units, read, write);
        read += 1;
        write += 1;
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

fn first_non_conjunct_ra_ya_zwnj(units: &[PhoneticUnit]) -> Option<usize> {
    (0..units.len().saturating_sub(2)).find(|&index| is_non_conjunct_ra_ya_zwnj_at(units, index))
}

fn is_non_conjunct_ra_ya_zwnj_at(units: &[PhoneticUnit], index: usize) -> bool {
    index + 2 < units.len()
        && units[index].unit_type == PhoneticUnitType::Consonant
        && units[index].text == "r"
        && units[index + 1].unit_type == PhoneticUnitType::Unknown
        && units[index + 1].text == "Z"
        && units[index + 2].unit_type == PhoneticUnitType::Consonant
        && matches!(units[index + 2].text.as_str(), "y" | "Y")
}
