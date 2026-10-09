#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnitClass {
    Other,
    VowelSign,
    Hasant,
    NasalMark,
}

pub(crate) fn bangla_units(text: &str) -> Vec<&str> {
    let mut units = Vec::new();
    for_each_bangla_unit_range(text, |start, end, _| units.push(&text[start..end]));
    units
}

fn for_each_bangla_unit_range(text: &str, mut visit: impl FnMut(usize, usize, bool)) {
    let mut start: Option<usize> = None;
    let mut end = 0;
    let mut join_next = false;

    for (index, ch) in text.char_indices() {
        let class = unit_class(ch);
        if start.is_none() {
            start = Some(index);
            end = index + ch.len_utf8();
            join_next = class == UnitClass::Hasant;
            continue;
        }

        if class != UnitClass::Other || join_next {
            end = index + ch.len_utf8();
            join_next = class == UnitClass::Hasant;
            continue;
        }

        let unit_start = start.expect("unit start should be set");
        visit(unit_start, end, false);
        start = Some(index);
        end = index + ch.len_utf8();
        join_next = false;
    }

    if let Some(unit_start) = start {
        visit(unit_start, end, true);
    }
}

pub(crate) fn unit_class(ch: char) -> UnitClass {
    match ch {
        '\u{0981}'..='\u{0983}' => UnitClass::NasalMark,
        '\u{09BC}' => UnitClass::VowelSign,
        '\u{09BE}'..='\u{09C4}' => UnitClass::VowelSign,
        '\u{09C7}'..='\u{09C8}' => UnitClass::VowelSign,
        '\u{09CB}'..='\u{09CC}' => UnitClass::VowelSign,
        '\u{09CD}' => UnitClass::Hasant,
        '\u{09D7}' => UnitClass::VowelSign,
        '\u{09E2}'..='\u{09E3}' => UnitClass::VowelSign,
        _ => UnitClass::Other,
    }
}

pub(crate) fn unit_similarity(left: &str, right: &str) -> u16 {
    if left == right {
        return 0;
    }

    // Only the attached vowel/nasal marks may change at the discounted cost.
    // Preserve hasants, joiners and nukta: they distinguish consonant bases.
    let is_base = |ch: &char| {
        *ch == '\u{09BC}' || !matches!(unit_class(*ch), UnitClass::VowelSign | UnitClass::NasalMark)
    };
    let left_tail = left.chars().last().map(unit_class);
    let right_tail = right.chars().last().map(unit_class);
    let discounted_marks = (left_tail == Some(UnitClass::VowelSign)
        && right_tail == Some(UnitClass::VowelSign))
        || left_tail == Some(UnitClass::NasalMark)
        || right_tail == Some(UnitClass::NasalMark);
    if discounted_marks
        && left
            .chars()
            .filter(is_base)
            .eq(right.chars().filter(is_base))
    {
        return 1;
    }
    if left.contains('\u{09CD}') || right.contains('\u{09CD}') {
        return 2;
    }

    3
}

pub(crate) fn phonetic_skeleton(text: &str) -> String {
    let mut skeleton = String::with_capacity(text.len());
    let mut previous: Option<char> = None;

    for ch in text.chars() {
        let Some(code) = skeleton_char(ch) else {
            continue;
        };
        if previous == Some(code) {
            continue;
        }
        skeleton.push(code);
        previous = Some(code);
    }

    skeleton
}

pub(crate) fn differs_only_by_nasal_or_breath_mark(left: &str, right: &str) -> bool {
    if left == right {
        return false;
    }

    strip_nasal_and_breath_marks(left).eq(strip_nasal_and_breath_marks(right))
        && (has_nasal_or_breath_mark(left) || has_nasal_or_breath_mark(right))
}

pub(crate) fn differs_only_by_vowel_length(left: &str, right: &str) -> bool {
    if left == right {
        return false;
    }

    let mut changed = false;
    let mut left_chars = left.chars();
    let mut right_chars = right.chars();

    loop {
        match (left_chars.next(), right_chars.next()) {
            (Some(left), Some(right)) if left == right => {}
            (Some(left), Some(right)) if vowel_length_fold(left) == vowel_length_fold(right) => {
                changed = true;
            }
            (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) => return false,
            (None, None) => return changed,
        }
    }
}

pub(crate) fn for_each_chandrabindu_variant(text: &str, mut visit: impl FnMut(&str)) {
    if text.is_empty() || text.chars().any(is_nasal_or_breath_mark) {
        return;
    }

    let mut variant = String::with_capacity(text.len() + CHANDRABINDU.len_utf8());

    for_each_bangla_unit_range(text, |start, end, is_final| {
        if !can_take_autocorrect_chandrabindu(&text[start..end], is_final) {
            return;
        }

        variant.clear();
        variant.push_str(&text[..end]);
        variant.push(CHANDRABINDU);
        variant.push_str(&text[end..]);
        visit(&variant);
    });
}

const CHANDRABINDU: char = '\u{0981}';

fn can_take_autocorrect_chandrabindu(unit: &str, is_final: bool) -> bool {
    if unit.chars().any(is_explicit_vowel_carrier) {
        return true;
    }

    !is_final && unit.chars().all(|ch| unit_class(ch) == UnitClass::Other)
}

fn is_explicit_vowel_carrier(ch: char) -> bool {
    is_independent_vowel(ch)
        || matches!(
            ch,
            '\u{09BE}'..='\u{09C4}'
                | '\u{09C7}'..='\u{09C8}'
                | '\u{09CB}'..='\u{09CC}'
                | '\u{09D7}'
                | '\u{09E2}'..='\u{09E3}'
        )
}

fn strip_nasal_and_breath_marks(text: &str) -> impl Iterator<Item = char> + '_ {
    text.chars().filter(|ch| !is_nasal_or_breath_mark(*ch))
}

fn has_nasal_or_breath_mark(text: &str) -> bool {
    text.chars().any(is_nasal_or_breath_mark)
}

fn is_nasal_or_breath_mark(ch: char) -> bool {
    matches!(ch, '\u{0981}'..='\u{0983}')
}

fn vowel_length_fold(ch: char) -> char {
    match ch {
        'ঈ' => 'ই',
        'ঊ' => 'উ',
        'ী' => 'ি',
        'ূ' => 'ু',
        _ => ch,
    }
}

fn skeleton_char(ch: char) -> Option<char> {
    if is_independent_vowel(ch) {
        return None;
    }

    match ch {
        '\u{0981}'..='\u{0983}' => Some('ং'),
        '\u{0995}'..='\u{09B9}'
        | '\u{09CE}'
        | '\u{09DC}'..='\u{09DD}'
        | '\u{09DF}'..='\u{09E1}' => Some(fold_base_consonant(ch)),
        _ if matches!(unit_class(ch), UnitClass::VowelSign | UnitClass::Hasant) => None,
        _ => None,
    }
}

fn is_independent_vowel(ch: char) -> bool {
    matches!(
        ch,
        '\u{0985}'..='\u{098C}' | '\u{098F}'..='\u{0990}' | '\u{0993}'..='\u{0994}'
    )
}

/// A Bangla *letter*: an independent vowel or a consonant (including ৎ and the
/// phota forms ড়/ঢ়/য়). Deliberately excludes the marks that are not letters on
/// their own — chandrabindu/anusvar/bisarga, vowel signs, hasant — as well as
/// digits and punctuation. This is the "is there a word here" test: a token made
/// only of those non-letters is punctuation/number/symbol, not a misspelled word.
pub(crate) fn is_bengali_letter(ch: char) -> bool {
    is_independent_vowel(ch)
        || matches!(
            ch,
            '\u{0995}'..='\u{09B9}' | '\u{09CE}' | '\u{09DC}'..='\u{09DD}' | '\u{09DF}'..='\u{09E1}'
        )
}

/// Whether `text` contains at least one Bangla letter (see [`is_bengali_letter`]).
pub(crate) fn has_bengali_letter(text: &str) -> bool {
    text.chars().any(is_bengali_letter)
}

fn fold_base_consonant(ch: char) -> char {
    let ch = fold_aspiration(ch).unwrap_or(ch);
    match ch {
        'ঙ' | 'ঞ' | 'ণ' | 'ন' => 'ন',
        'শ' | 'ষ' | 'স' => 'স',
        'য' | 'য়' => 'য',
        'ড়' | 'ঢ়' => 'ড',
        _ => ch,
    }
}

fn fold_aspiration(ch: char) -> Option<char> {
    const VARGA_STARTS: [u32; 5] = [0x0995, 0x099A, 0x099F, 0x09A4, 0x09AA];

    let codepoint = ch as u32;
    for start in VARGA_STARTS {
        if !(start..=start + 4).contains(&codepoint) {
            continue;
        }

        let offset = codepoint - start;
        let folded_offset = match offset {
            1 => 0,
            3 => 2,
            _ => offset,
        };
        return char::from_u32(start + folded_offset);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        bangla_units, differs_only_by_nasal_or_breath_mark, differs_only_by_vowel_length,
        for_each_chandrabindu_variant, has_bengali_letter, phonetic_skeleton,
    };

    #[test]
    fn bangla_units_group_vowel_signs_and_conjuncts() {
        assert_eq!(bangla_units("কিরণ"), vec!["কি", "র", "ণ"]);
        assert_eq!(bangla_units("বিজ্ঞান"), vec!["বি", "জ্ঞা", "ন"]);
    }

    #[test]
    fn has_bengali_letter_separates_words_from_non_letters() {
        // A word has at least one vowel or consonant (incl. ৎ and phota forms).
        for word in ["আমার", "ক", "অ", "য়", "ৎ", "কি?"] {
            assert!(has_bengali_letter(word), "{word:?}");
        }
        // Punctuation, digits, and lone marks (chandrabindu/anusvar/bisarga/hasant/
        // vowel sign) carry no letter — issue #34.
        for non in [",", "।", "?", "১", "২০১১", "ঁ", "ং", "ঃ", "্", "া"] {
            assert!(!has_bengali_letter(non), "{non:?}");
        }
    }

    #[test]
    fn phonetic_skeleton_is_consonant_heavy_and_folded() {
        assert_eq!(phonetic_skeleton("কিরণ"), "করন");
        assert_eq!(phonetic_skeleton("করণ"), "করন");
        assert_eq!(phonetic_skeleton("শাসন"), "সন");
        assert_eq!(phonetic_skeleton("বিজ্ঞান"), "বজন");
        assert_eq!(phonetic_skeleton("অঞ্চল"), phonetic_skeleton("আনছল"));
        assert_eq!(phonetic_skeleton("অক্ষর"), phonetic_skeleton("আক্ষার"));
    }

    #[test]
    fn nasal_or_breath_mark_variants_require_same_base_surface() {
        assert!(differs_only_by_nasal_or_breath_mark("চাদ", "চাঁদ"));
        assert!(differs_only_by_nasal_or_breath_mark("সংগ", "সগ"));
        assert!(!differs_only_by_nasal_or_breath_mark("চাদ", "চাদ"));
        assert!(!differs_only_by_nasal_or_breath_mark("চাদ", "চাদর"));
        assert!(!differs_only_by_nasal_or_breath_mark("চাল", "চাঁদ"));
    }

    #[test]
    fn vowel_length_variants_require_same_surface_except_i_u_length() {
        assert!(differs_only_by_vowel_length("সুশিল", "সুশীল"));
        assert!(differs_only_by_vowel_length("দুর", "দূর"));
        assert!(differs_only_by_vowel_length("ইদ", "ঈদ"));
        assert!(!differs_only_by_vowel_length("সুশিল", "সুনীল"));
        assert!(!differs_only_by_vowel_length("সুশিল", "সুশিল"));
        assert!(!differs_only_by_vowel_length("নদি", "নদীতে"));
    }

    #[test]
    fn chandrabindu_variants_insert_at_bangla_unit_boundaries() {
        let mut variants = Vec::new();
        for_each_chandrabindu_variant("চাদ", |variant| variants.push(variant.to_string()));

        assert!(variants.iter().any(|variant| variant == "চাঁদ"));
        assert!(!variants.iter().any(|variant| variant == "চঁাদ"));
        assert!(!variants.iter().any(|variant| variant == "চাদঁ"));
        assert!(!variants.iter().any(|variant| variant == "চাংদ"));
        assert!(!variants.iter().any(|variant| variant == "চাঃদ"));
    }

    #[test]
    fn chandrabindu_variants_skip_already_marked_words() {
        let mut variants = Vec::new();
        for_each_chandrabindu_variant("চাঁদ", |variant| variants.push(variant.to_string()));
        assert!(variants.is_empty());
    }
}
