use crate::{PhoneticUnit, PhoneticUnitType, Tokenizer};

pub const DEFAULT_ROMAN_REPAIR_BEAM_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct RomanRepairOptions {
    pub max_repairs: usize,
}

impl Default for RomanRepairOptions {
    fn default() -> Self {
        Self {
            max_repairs: DEFAULT_ROMAN_REPAIR_BEAM_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomanRepair {
    pub text: String,
    pub cost: u16,
    pub kind: RomanRepairKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomanRepairedOutput {
    pub roman_input: String,
    pub bangla_output: String,
    pub repair_kind: &'static str,
    pub repair_cost: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RomanRepairKind {
    Original,
    InsertedSeparatorO,
    SplitAspiratedConsonant,
    LowercaseRToFlap,
    PalatalNasalJaFromNg,
    PalatalNasalJaFromNz,
    VelarNasalFromNg,
    JnaConjunctFromGy,
}

impl RomanRepairKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::InsertedSeparatorO => "inserted_separator_o",
            Self::SplitAspiratedConsonant => "split_aspirated_consonant",
            Self::LowercaseRToFlap => "lowercase_r_to_flap",
            Self::PalatalNasalJaFromNg => "palatal_nasal_ja_from_ng",
            Self::PalatalNasalJaFromNz => "palatal_nasal_ja_from_nz",
            Self::VelarNasalFromNg => "velar_nasal_from_ng",
            Self::JnaConjunctFromGy => "jna_conjunct_from_gy",
        }
    }
}

pub fn roman_repair_beam(input: &str, options: RomanRepairOptions) -> Vec<RomanRepair> {
    if input.is_empty() || options.max_repairs == 0 {
        return Vec::new();
    }

    let mut repairs = Vec::with_capacity(options.max_repairs.min(input.len() + 1));

    repairs.push(RomanRepair {
        text: input.to_string(),
        cost: 0,
        kind: RomanRepairKind::Original,
    });

    let tokenizer = Tokenizer::new();
    let separator_repairs = separator_o_repairs(input, &tokenizer);
    for repaired in &separator_repairs {
        push_repair(
            &mut repairs,
            options.max_repairs,
            repaired.text.clone(),
            repaired.cost,
            RomanRepairKind::InsertedSeparatorO,
        );
    }

    for repaired in aspirated_separator_o_repairs(input) {
        push_repair(
            &mut repairs,
            options.max_repairs,
            repaired.text,
            repaired.cost,
            RomanRepairKind::SplitAspiratedConsonant,
        );
    }

    for repaired in lowercase_r_to_flap_repairs(input) {
        push_repair(
            &mut repairs,
            options.max_repairs,
            repaired.text,
            repaired.cost,
            RomanRepairKind::LowercaseRToFlap,
        );
    }

    for repaired in nasal_neighbor_repairs(input) {
        push_repair(
            &mut repairs,
            options.max_repairs,
            repaired.text,
            repaired.cost,
            repaired.kind,
        );
    }

    if let Some(repaired) = jna_conjunct_repair(input) {
        push_repair(
            &mut repairs,
            options.max_repairs,
            repaired,
            2,
            RomanRepairKind::JnaConjunctFromGy,
        );
    }

    for repaired in separator_repairs {
        for second_pass in separator_o_repairs(&repaired.text, &tokenizer) {
            let cost = repaired.cost.saturating_add(second_pass.cost);
            if push_repair(
                &mut repairs,
                options.max_repairs,
                second_pass.text,
                cost,
                RomanRepairKind::InsertedSeparatorO,
            ) {
                continue;
            }
            return repairs;
        }
    }

    repairs
}

pub fn roman_repaired_outputs<F>(
    input: &str,
    baseline_output: &str,
    options: RomanRepairOptions,
    mut transliterate: F,
) -> Vec<RomanRepairedOutput>
where
    F: FnMut(&str) -> String,
{
    roman_repair_beam(input, options)
        .into_iter()
        .filter(|repair| repair.cost > 0)
        .filter_map(|repair| {
            let output = transliterate(&repair.text);
            (output != baseline_output).then_some(RomanRepairedOutput {
                roman_input: repair.text,
                bangla_output: output,
                repair_kind: repair.kind.as_str(),
                repair_cost: repair.cost,
            })
        })
        .collect()
}

fn push_repair(
    repairs: &mut Vec<RomanRepair>,
    max_repairs: usize,
    text: String,
    cost: u16,
    kind: RomanRepairKind,
) -> bool {
    if repairs.iter().any(|repair| repair.text == text) {
        return true;
    }
    if repairs.len() >= max_repairs {
        return false;
    }

    repairs.push(RomanRepair { text, cost, kind });
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SeparatorRepair {
    text: String,
    cost: u16,
}

fn separator_o_repairs(input: &str, tokenizer: &Tokenizer) -> Vec<SeparatorRepair> {
    let units = tokenizer.tokenize_word(input);
    missing_separator_positions(input, &units)
        .into_iter()
        .map(|byte_index| SeparatorRepair {
            text: insert_separator_o(input, byte_index),
            cost: 1,
        })
        .collect()
}

fn aspirated_separator_o_repairs(input: &str) -> Vec<SeparatorRepair> {
    let bytes = input.as_bytes();
    if bytes.len() < 2 {
        return Vec::new();
    }

    let mut repairs = Vec::new();
    let mut index = 0;
    while index + 1 < bytes.len() {
        if is_lowercase_aspirated_digraph(bytes[index], bytes[index + 1]) {
            repairs.push(SeparatorRepair {
                text: insert_separator_o(input, index + 1),
                cost: 1,
            });
            index += 2;
        } else {
            index += 1;
        }
    }
    repairs
}

fn is_lowercase_aspirated_digraph(left: u8, right: u8) -> bool {
    right == b'h' && matches!(left, b'k' | b'g' | b'c' | b'j' | b't' | b'd' | b'p' | b'b')
}

fn lowercase_r_to_flap_repairs(input: &str) -> Vec<SeparatorRepair> {
    let bytes = input.as_bytes();
    let mut repairs = Vec::new();

    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'r' || is_reph_rr_context(bytes, index) {
            continue;
        }

        let mut repaired = String::with_capacity(input.len());
        repaired.push_str(&input[..index]);
        repaired.push('R');
        repaired.push_str(&input[index + 1..]);
        repairs.push(SeparatorRepair {
            text: repaired,
            cost: 1,
        });
    }

    repairs
}

fn is_reph_rr_context(bytes: &[u8], index: usize) -> bool {
    bytes.get(index.wrapping_sub(1)) == Some(&b'r') || bytes.get(index + 1) == Some(&b'r')
}

fn insert_separator_o(input: &str, byte_index: usize) -> String {
    let mut repaired = String::with_capacity(input.len() + 1);
    repaired.push_str(&input[..byte_index]);
    repaired.push('o');
    repaired.push_str(&input[byte_index..]);
    repaired
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NasalNeighborRepair {
    text: String,
    cost: u16,
    kind: RomanRepairKind,
}

fn nasal_neighbor_repairs(input: &str) -> Vec<NasalNeighborRepair> {
    let mut repairs = Vec::new();
    push_ng_neighbor_repairs(input, &mut repairs);
    push_nz_neighbor_repairs(input, &mut repairs);
    repairs
}

/// The জ্ঞ (jna) conjunct is spelled `gg`/`jn`/`NGj` but is commonly pronounced and
/// typed `gy` (`gyan` for জ্ঞান). Offer the `gg` spelling so the FST can surface the
/// real জ্ঞ word; a `gy` that is really a ya-phola (`bhagyo` -> ভাগ্য) yields no
/// lexicon word and is dropped, so this never overrides the deterministic গ্য.
fn jna_conjunct_repair(input: &str) -> Option<String> {
    if input.contains("gy") {
        Some(input.replace("gy", "gg"))
    } else {
        None
    }
}

fn push_ng_neighbor_repairs(input: &str, repairs: &mut Vec<NasalNeighborRepair>) {
    let mut search_start = 0;

    while let Some(relative_start) = input[search_start..].find("ng") {
        let start = search_start + relative_start;
        let end = start + "ng".len();
        if let Some(next_vowel) = front_vowel_at(input, end) {
            push_replaced_range(
                repairs,
                input,
                start,
                end,
                "nj",
                2,
                RomanRepairKind::PalatalNasalJaFromNg,
            );
            push_replaced_range(
                repairs,
                input,
                start,
                end,
                "Ng",
                1,
                RomanRepairKind::VelarNasalFromNg,
            );
            push_replaced_range(
                repairs,
                input,
                start,
                end,
                "ngg",
                1,
                RomanRepairKind::VelarNasalFromNg,
            );
            push_replaced_range(
                repairs,
                input,
                start,
                end,
                "Mg",
                2,
                RomanRepairKind::VelarNasalFromNg,
            );

            if next_vowel == b'i' {
                let vowel_end = end + 1;
                push_replaced_range(
                    repairs,
                    input,
                    start,
                    vowel_end,
                    "NgI",
                    2,
                    RomanRepairKind::VelarNasalFromNg,
                );
                push_replaced_range(
                    repairs,
                    input,
                    start,
                    vowel_end,
                    "nggI",
                    2,
                    RomanRepairKind::VelarNasalFromNg,
                );
                push_replaced_range(
                    repairs,
                    input,
                    start,
                    vowel_end,
                    "MgI",
                    3,
                    RomanRepairKind::VelarNasalFromNg,
                );
            }
        }
        // `ng` immediately before `j`/`J` is the palatal nasal-ja cluster spelled
        // with the velar-nasal `ng`; fold the `g` so `ngj` routes to `nj` (ঞ্জ),
        // the same cluster `nj`/`NGj` produce deterministically.
        if matches!(input.as_bytes().get(end), Some(b'j') | Some(b'J')) {
            push_replaced_range(
                repairs,
                input,
                start,
                end,
                "n",
                2,
                RomanRepairKind::PalatalNasalJaFromNg,
            );
        }

        search_start = end;
    }
}

fn push_nz_neighbor_repairs(input: &str, repairs: &mut Vec<NasalNeighborRepair>) {
    let mut search_start = 0;

    while let Some(relative_start) = input[search_start..].find("nz") {
        let start = search_start + relative_start;
        let end = start + "nz".len();
        push_replaced_range(
            repairs,
            input,
            start,
            end,
            "nj",
            2,
            RomanRepairKind::PalatalNasalJaFromNz,
        );
        search_start = end;
    }
}

fn push_replaced_range(
    repairs: &mut Vec<NasalNeighborRepair>,
    input: &str,
    start: usize,
    end: usize,
    replacement: &str,
    cost: u16,
    kind: RomanRepairKind,
) {
    let mut repaired = String::with_capacity(input.len() + replacement.len());
    repaired.push_str(&input[..start]);
    repaired.push_str(replacement);
    repaired.push_str(&input[end..]);

    if repairs.iter().all(|repair| repair.text != repaired) {
        repairs.push(NasalNeighborRepair {
            text: repaired,
            cost,
            kind,
        });
    }
}

fn front_vowel_at(input: &str, byte_index: usize) -> Option<u8> {
    input
        .as_bytes()
        .get(byte_index)
        .copied()
        .filter(|byte| matches!(byte, b'i' | b'I' | b'e' | b'E'))
}

fn missing_separator_positions(input: &str, units: &[PhoneticUnit]) -> Vec<usize> {
    let mut positions = Vec::new();

    for unit in units {
        if !is_conjunct_like(unit.unit_type) {
            continue;
        }

        append_conjunct_separator_positions(input, unit, &mut positions);
    }

    positions.sort_unstable();
    positions.dedup();
    positions
}

fn append_conjunct_separator_positions(
    input: &str,
    unit: &PhoneticUnit,
    positions: &mut Vec<usize>,
) {
    let mut parts = unit.text.split(",,");
    let Some(mut left) = parts.next() else {
        return;
    };

    let mut offset = 0_usize;
    for right in parts {
        offset += left.len();

        let byte_index = unit.position + offset;
        if is_insertable_boundary(input, byte_index) {
            positions.push(byte_index);
        }

        left = right;
    }
}

fn is_conjunct_like(unit_type: PhoneticUnitType) -> bool {
    matches!(
        unit_type,
        PhoneticUnitType::Conjunct
            | PhoneticUnitType::ConjunctWithVowel
            | PhoneticUnitType::ConjunctWithTerminator
            | PhoneticUnitType::RephOverConsonant
            | PhoneticUnitType::RephOverConsonantWithVowel
            | PhoneticUnitType::RephOverConsonantWithTerminator
    )
}

fn is_insertable_boundary(input: &str, byte_index: usize) -> bool {
    byte_index > 0
        && input.is_char_boundary(byte_index)
        && input
            .as_bytes()
            .get(byte_index)
            .is_some_and(u8::is_ascii_alphabetic)
        && input
            .as_bytes()
            .get(byte_index - 1)
            .is_some_and(u8::is_ascii_alphabetic)
}

#[cfg(test)]
mod tests {
    use super::{roman_repair_beam, RomanRepairKind, RomanRepairOptions};

    #[test]
    fn inserts_separator_o_at_implicit_conjunct_boundary() {
        let repairs = roman_repair_beam("okalpokk", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "okalopokk"
                && repair.cost == 1
                && repair.kind == RomanRepairKind::InsertedSeparatorO
        }));
    }

    #[test]
    fn inserts_separator_o_inside_repeated_consonant_clusters() {
        let repairs = roman_repair_beam("khnn", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "khnon"
                && repair.cost == 1
                && repair.kind == RomanRepairKind::InsertedSeparatorO
        }));
    }

    #[test]
    fn bounded_second_pass_can_restore_omitted_inherent_vowels() {
        let repairs = roman_repair_beam("mnn", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "monon"
                && repair.cost == 2
                && repair.kind == RomanRepairKind::InsertedSeparatorO
        }));
    }

    #[test]
    fn splits_lowercase_aspirated_digraph_when_h_was_intended_as_consonant() {
        let repairs = roman_repair_beam("bhu", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "bohu"
                && repair.cost == 1
                && repair.kind == RomanRepairKind::SplitAspiratedConsonant
        }));
    }

    #[test]
    fn does_not_split_deliberate_uppercase_aspirated_signals() {
        let repairs = roman_repair_beam("bHu", RomanRepairOptions::default());

        assert!(!repairs
            .iter()
            .any(|repair| repair.text == "boHu" || repair.text == "bohu"));
    }

    #[test]
    fn repairs_lowercase_r_to_explicit_flap_signal() {
        let repairs = roman_repair_beam("dariye", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "daRiye"
                && repair.cost == 1
                && repair.kind == RomanRepairKind::LowercaseRToFlap
        }));
    }

    #[test]
    fn does_not_repair_reph_rr_signal_to_flap() {
        let repairs = roman_repair_beam("rram", RomanRepairOptions::default());

        assert!(!repairs
            .iter()
            .any(|repair| repair.text == "Rram" || repair.text == "rRam"));
    }

    #[test]
    fn repairs_lowercase_ng_before_front_vowel_to_palatal_nasal_ja() {
        let repairs = roman_repair_beam("jingira", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "jinjira"
                && repair.cost == 2
                && repair.kind == RomanRepairKind::PalatalNasalJaFromNg
        }));
    }

    #[test]
    fn repairs_lowercase_ng_before_ja_to_palatal_nasal_ja() {
        let repairs = roman_repair_beam("rongjon", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "ronjon" && repair.kind == RomanRepairKind::PalatalNasalJaFromNg
        }));
    }

    #[test]
    fn repairs_lowercase_nz_to_deterministic_palatal_nasal_ja() {
        let repairs = roman_repair_beam("jinzira", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "jinjira"
                && repair.cost == 2
                && repair.kind == RomanRepairKind::PalatalNasalJaFromNz
        }));
    }

    #[test]
    fn repairs_gy_to_the_jna_conjunct_spelling() {
        let repairs = roman_repair_beam("gyan", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "ggan" && repair.kind == RomanRepairKind::JnaConjunctFromGy
        }));
    }

    #[test]
    fn repairs_bare_ng_to_velar_and_anusvar_ga_routes() {
        let repairs = roman_repair_beam("songit", RomanRepairOptions::default());

        assert!(repairs.iter().any(|repair| {
            repair.text == "songgIt"
                && repair.cost == 2
                && repair.kind == RomanRepairKind::VelarNasalFromNg
        }));
        assert!(repairs.iter().any(|repair| {
            repair.text == "soMgIt"
                && repair.cost == 3
                && repair.kind == RomanRepairKind::VelarNasalFromNg
        }));
    }

    #[test]
    fn returns_only_original_for_zero_sized_beam() {
        let repairs = roman_repair_beam("okalpokk", RomanRepairOptions { max_repairs: 0 });

        assert!(repairs.is_empty());
    }
}
