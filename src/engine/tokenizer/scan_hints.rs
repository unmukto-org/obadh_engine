use super::{PhoneticUnit, PhoneticUnitType};

#[derive(Default)]
pub(super) struct WordScanHints {
    has_reph_candidate: bool,
    has_redundant_reph_hasant_candidate: bool,
    has_redundant_khanda_ta_hasant_candidate: bool,
    has_velar_nasal_conjunct_alias_candidate: bool,
    has_anusvar_ng_before_vowel_candidate: bool,
    has_long_iya_marker_candidate: bool,
    has_non_conjunct_ra_ya_zwnj_candidate: bool,
    has_reserved_z: bool,
}

impl WordScanHints {
    pub(super) fn observe_unit(&mut self, unit: &PhoneticUnit, previous: Option<&PhoneticUnit>) {
        if is_reph_signal(unit) {
            self.has_reph_candidate = true;
        }

        if is_explicit_hasant_unit(unit) {
            if previous.is_some_and(is_reph_signal) {
                self.has_redundant_reph_hasant_candidate = true;
            } else if previous.is_some_and(is_khanda_ta_signal) {
                self.has_redundant_khanda_ta_hasant_candidate = true;
            }
        }

        if previous.is_some_and(is_anusvar_ng_signal) && is_velar_nasal_conjunct_tail(unit) {
            self.has_velar_nasal_conjunct_alias_candidate = true;
        }

        if previous.is_some_and(is_anusvar_ng_signal) && is_vowel_unit(unit) {
            self.has_anusvar_ng_before_vowel_candidate = true;
        }

        // `iyw` long-ঈয় signal: a `w` consonant directly after a `y`/`Y` consonant.
        // (`w` now tokenizes as a consonant, so this is detected here rather than
        // on the unknown-char path.)
        if is_ba_phola_w_signal(unit) && previous.is_some_and(is_ya_phola_signal) {
            self.has_long_iya_marker_candidate = true;
        }
    }

    pub(super) fn observe_unknown_text(&mut self, text: &str, word: &str, byte_index: usize) {
        if text == "Z" {
            self.has_reserved_z = true;
            if is_non_conjunct_ra_ya_zwnj_marker_at(word, byte_index) {
                self.has_non_conjunct_ra_ya_zwnj_candidate = true;
            }
        }
    }

    pub(super) fn has_reph_candidate(&self) -> bool {
        self.has_reph_candidate
    }

    pub(super) fn has_redundant_reph_hasant_candidate(&self) -> bool {
        self.has_redundant_reph_hasant_candidate
    }

    pub(super) fn has_redundant_khanda_ta_hasant_candidate(&self) -> bool {
        self.has_redundant_khanda_ta_hasant_candidate
    }

    pub(super) fn has_velar_nasal_conjunct_alias_candidate(&self) -> bool {
        self.has_velar_nasal_conjunct_alias_candidate
    }

    pub(super) fn has_anusvar_ng_before_vowel_candidate(&self) -> bool {
        self.has_anusvar_ng_before_vowel_candidate
    }

    pub(super) fn has_long_iya_marker_candidate(&self) -> bool {
        self.has_long_iya_marker_candidate
    }

    pub(super) fn has_non_conjunct_ra_ya_zwnj_candidate(&self) -> bool {
        self.has_non_conjunct_ra_ya_zwnj_candidate
    }

    pub(super) fn has_reserved_z(&self) -> bool {
        self.has_reserved_z
    }
}

fn is_reph_signal(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::SpecialForm && unit.text == "rr"
}

fn is_explicit_hasant_unit(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::ConsonantWithHasant && unit.text == ",,"
}

fn is_khanda_ta_signal(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::SpecialForm && matches!(unit.text.as_str(), "t``" | "T``")
}

fn is_anusvar_ng_signal(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::SpecialForm && unit.text == "ng"
}

fn is_velar_nasal_conjunct_tail(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant
        && matches!(unit.text.as_str(), "g" | "gh" | "Gh" | "GH")
}

fn is_vowel_unit(unit: &PhoneticUnit) -> bool {
    matches!(
        unit.unit_type,
        PhoneticUnitType::Vowel | PhoneticUnitType::TerminatingVowel
    )
}

fn is_ya_phola_signal(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant && matches!(unit.text.as_str(), "y" | "Y")
}

fn is_ba_phola_w_signal(unit: &PhoneticUnit) -> bool {
    unit.unit_type == PhoneticUnitType::Consonant && unit.text == "w"
}

fn is_non_conjunct_ra_ya_zwnj_marker_at(text: &str, byte_index: usize) -> bool {
    let bytes = text.as_bytes();

    byte_index > 0
        && bytes.get(byte_index - 1) == Some(&b'r')
        && matches!(bytes.get(byte_index + 1), Some(b'y') | Some(b'Y'))
}
