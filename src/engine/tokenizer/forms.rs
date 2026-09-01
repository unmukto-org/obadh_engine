use crate::definitions::conjuncts::conjuncts;

use super::conjunct_runs::{form_conjuncts_in_range, is_conjunct_run_component};
use super::explicit_hasant::collapse_explicit_hasant_chains;
use super::long_iya::normalize_iyw_long_iya_signal;
use super::normalization::{
    normalize_anusvar_ng_before_vowel, normalize_colon_between_numerals,
    normalize_non_conjunct_ra_ya_zwnj, normalize_redundant_khanda_ta_hasant,
    normalize_redundant_reph_hasant, normalize_reph_and_vocalic_r,
    normalize_residual_reserved_z, normalize_velar_nasal_conjunct_aliases,
};
use super::{move_unit, PhoneticUnit, PhoneticUnitType, WordScanHints};

/// Identify complex phonetic forms like conjuncts and consonants with vowel modifiers.
pub(super) fn identify_complex_forms(units: &mut Vec<PhoneticUnit>, scan_hints: WordScanHints) {
    let conjunct_defs = conjuncts();

    if scan_hints.has_redundant_reph_hasant_candidate() {
        normalize_redundant_reph_hasant(units);
    }
    if scan_hints.has_reph_candidate() {
        normalize_reph_and_vocalic_r(units);
    }
    if scan_hints.has_velar_nasal_conjunct_alias_candidate() {
        normalize_velar_nasal_conjunct_aliases(units);
    }
    if scan_hints.has_anusvar_ng_before_vowel_candidate() {
        normalize_anusvar_ng_before_vowel(units);
    }
    if scan_hints.has_colon_between_numerals_candidate() {
        normalize_colon_between_numerals(units);
    }
    if scan_hints.has_non_conjunct_ra_ya_zwnj_candidate() {
        normalize_non_conjunct_ra_ya_zwnj(units);
    }
    if scan_hints.has_reserved_z() {
        normalize_residual_reserved_z(units);
    }
    if scan_hints.has_redundant_khanda_ta_hasant_candidate() {
        normalize_redundant_khanda_ta_hasant(units);
    }

    // Explicit hasant is a user command, so it is preserved even before later
    // valid-conjunct filtering and vowel attachment runs.
    collapse_explicit_hasant_chains(units, conjunct_defs);

    // Process contiguous consonant runs to form conjuncts. Non-consonant units
    // such as anusvar are boundaries, not blockers for subsequent runs.
    let mut run_start = 0;
    while run_start < units.len() {
        while run_start < units.len() && !is_conjunct_run_component(&units[run_start]) {
            run_start += 1;
        }

        let mut run_end = run_start;
        while run_end < units.len() && is_conjunct_run_component(&units[run_end]) {
            run_end += 1;
        }

        form_conjuncts_in_range(units, run_start, run_end, conjunct_defs);
        run_start = run_end;
    }

    compact_units_and_attach_vowels(units);

    // If `iyw` consumes a marker, a following vowel may now be adjacent to
    // `y`/`Y`; run the ordinary attachment pass once more in that case.
    if scan_hints.has_long_iya_marker_candidate() && normalize_iyw_long_iya_signal(units) {
        compact_units_and_attach_vowels(units);
    }
}

fn compact_units_and_attach_vowels(units: &mut Vec<PhoneticUnit>) {
    let mut read = 0;
    let mut write = 0;

    while read < units.len() {
        while read < units.len() && units[read].text.is_empty() {
            read += 1;
        }
        if read >= units.len() {
            break;
        }

        if let Some(next) = next_non_empty_unit_index(units, read + 1) {
            if let Some(combined_type) =
                attached_vowel_unit_type(units[read].unit_type, units[next].unit_type)
            {
                move_unit(units, read, write);
                let vowel_text = std::mem::take(&mut units[next].text);
                units[write].text.push_str(&vowel_text);
                units[write].unit_type = combined_type;
                read = next + 1;
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

fn attached_vowel_unit_type(
    base: PhoneticUnitType,
    vowel: PhoneticUnitType,
) -> Option<PhoneticUnitType> {
    match (base, vowel) {
        (PhoneticUnitType::Consonant, PhoneticUnitType::Vowel) => {
            Some(PhoneticUnitType::ConsonantWithVowel)
        }
        (PhoneticUnitType::Consonant, PhoneticUnitType::TerminatingVowel) => {
            Some(PhoneticUnitType::ConsonantWithTerminator)
        }
        (PhoneticUnitType::Conjunct, PhoneticUnitType::Vowel) => {
            Some(PhoneticUnitType::ConjunctWithVowel)
        }
        (PhoneticUnitType::Conjunct, PhoneticUnitType::TerminatingVowel) => {
            Some(PhoneticUnitType::ConjunctWithTerminator)
        }
        (PhoneticUnitType::RephOverConsonant, PhoneticUnitType::Vowel) => {
            Some(PhoneticUnitType::RephOverConsonantWithVowel)
        }
        (PhoneticUnitType::RephOverConsonant, PhoneticUnitType::TerminatingVowel) => {
            Some(PhoneticUnitType::RephOverConsonantWithTerminator)
        }
        _ => None,
    }
}

fn next_non_empty_unit_index(units: &[PhoneticUnit], start: usize) -> Option<usize> {
    units
        .iter()
        .enumerate()
        .skip(start)
        .find_map(|(index, unit)| (!unit.text.is_empty()).then_some(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(text: &str, unit_type: PhoneticUnitType, position: usize) -> PhoneticUnit {
        PhoneticUnit {
            text: text.to_string(),
            unit_type,
            position,
        }
    }

    #[test]
    fn compact_units_and_attach_vowels_skips_empty_units_in_place() {
        let mut units = vec![
            unit("k", PhoneticUnitType::Consonant, 0),
            unit("", PhoneticUnitType::Consonant, 1),
            unit("A", PhoneticUnitType::Vowel, 2),
            unit("", PhoneticUnitType::Consonant, 3),
            unit("rrk", PhoneticUnitType::RephOverConsonant, 4),
            unit("o", PhoneticUnitType::TerminatingVowel, 7),
            unit("ng", PhoneticUnitType::SpecialForm, 8),
        ];
        let capacity = units.capacity();

        compact_units_and_attach_vowels(&mut units);

        assert_eq!(units.capacity(), capacity);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type, unit.position))
                .collect::<Vec<_>>(),
            vec![
                ("kA", PhoneticUnitType::ConsonantWithVowel, 0),
                ("rrko", PhoneticUnitType::RephOverConsonantWithTerminator, 4),
                ("ng", PhoneticUnitType::SpecialForm, 8),
            ]
        );
    }
}
