use obadh_engine::{
    definitions::{conjuncts, consonants_static},
    ObadhEngine, PhoneticUnitType, Tokenizer,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Tests for valid 2-component conjuncts
#[test]
fn test_two_component_conjuncts() {
    let tokenizer = Tokenizer::new();

    // Test for valid 2-component conjuncts
    let test_cases = vec![
        ("kk", 1),  // Valid 2-component conjunct
        ("st", 1),  // Valid 2-component conjunct
        ("NGj", 1), // Correct form for ঞ্জ (palatalized nya + ja)
        ("nj", 1),  // Ergonomic alias for the same ঞ + জ conjunct
        ("ky", 1),  // Valid 2-component conjunct
        ("kw", 1),  // Valid ba-phola conjunct
        ("bw", 1),  // Valid ba-phola conjunct over regular ba
        ("zy", 1),  // Valid ya-phola conjunct over regular ya
        ("Sw", 1),  // Valid ba-phola conjunct
    ];

    for (input, expected_units) in test_cases {
        let units = tokenizer.tokenize_word(input);

        assert_eq!(
            units.len(),
            expected_units,
            "Expected {} unit(s) for '{}', got {}",
            expected_units,
            input,
            units.len()
        );

        // There should be at least one conjunct in each result
        let has_conjunct = units.iter().any(|u| {
            matches!(
                u.unit_type,
                PhoneticUnitType::Conjunct
                    | PhoneticUnitType::ConjunctWithVowel
                    | PhoneticUnitType::ConjunctWithTerminator
            )
        });

        assert!(
            has_conjunct,
            "No conjunct found in tokenization of '{}'",
            input
        );
    }
}

/// Tests for conjuncts with vowels
#[test]
fn test_conjuncts_with_vowel() {
    let tokenizer = Tokenizer::new();

    // Conjuncts with various vowels
    let test_cases = [
        ("kkA", PhoneticUnitType::ConjunctWithVowel),
        ("ktU", PhoneticUnitType::ConjunctWithVowel),
        ("nti", PhoneticUnitType::ConjunctWithVowel),
        ("kto", PhoneticUnitType::ConjunctWithTerminator), // With terminator vowel
        ("dwa", PhoneticUnitType::ConjunctWithVowel),
        ("Shkwa", PhoneticUnitType::ConjunctWithVowel),
        ("rrwya", PhoneticUnitType::ConjunctWithVowel),
        ("k,,shwa", PhoneticUnitType::ConjunctWithVowel),
        ("r,,rwa", PhoneticUnitType::ConjunctWithVowel),
        ("kShya", PhoneticUnitType::ConjunctWithVowel),
        ("mwa", PhoneticUnitType::ConjunctWithVowel),
        ("mwra", PhoneticUnitType::ConjunctWithVowel),
        ("bwa", PhoneticUnitType::ConjunctWithVowel),
        ("zya", PhoneticUnitType::ConjunctWithVowel),
    ];

    for (input, expected_type) in &test_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify we got at least one unit of the expected type
        assert!(
            units.iter().any(|unit| unit.unit_type == *expected_type),
            "Expected at least one {:?} for '{}'",
            expected_type,
            input
        );
    }
}

#[test]
fn test_source_conjunct_csv_keys_are_implemented() {
    let definitions = conjuncts();
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let key = row.key();
        let actual = definitions.create_conjunct(&key);

        if is_composable_khanda_ta_reph_source_row(&row) {
            assert!(
                actual.is_none(),
                "CSV key '{key}' on row {} should stay composable through khanda-ta rules, not the implicit conjunct table",
                row.line_number
            );
            continue;
        }

        assert!(
            actual.is_some(),
            "Missing conjunct rule for CSV key '{key}' on row {}",
            row.line_number
        );

        let actual = actual.unwrap();
        assert_eq!(
            actual, row.conjunct,
            "Mismatched conjunct rule for CSV key '{key}' on row {}: expected '{}', got '{actual}'",
            row.line_number, row.conjunct
        );
    }
}

#[test]
fn test_source_conjunct_csv_keys_render_through_public_engine() {
    let engine = ObadhEngine::new();
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let key = row.key();
        let actual = engine.transliterate(&key);
        assert_eq!(
            actual, row.conjunct,
            "CSV key '{key}' on row {} rendered as '{actual}', expected '{}'",
            row.line_number, row.conjunct
        );
    }

    assert_eq!(engine.transliterate("rrt`` rrT``"), "র্ৎ র্ৎ");
}

#[test]
fn test_source_conjunct_csv_keys_accept_canonical_dependent_vowels() {
    const VOWEL_CASES: &[(&str, &str)] = &[
        ("A", "া"),
        ("i", "ি"),
        ("I", "ী"),
        ("u", "ু"),
        ("U", "ূ"),
        ("e", "ে"),
        ("O", "ো"),
        ("OI", "ৈ"),
        ("OU", "ৌ"),
    ];

    let engine = ObadhEngine::new();
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let key = row.key();
        let rendered_base = engine.transliterate(&key);

        if is_composable_khanda_ta_reph_source_row(&row) {
            continue;
        }

        for &(vowel_key, dependent_vowel) in VOWEL_CASES {
            let input = format!("{key}{vowel_key}");
            let actual = engine.transliterate(&input);
            let expected = format!("{rendered_base}{dependent_vowel}");
            assert_eq!(
                actual, expected,
                "CSV key '{key}' on row {} did not accept dependent vowel '{vowel_key}'",
                row.line_number
            );
        }
    }
}

#[test]
fn test_source_conjunct_csv_components_render_with_explicit_hasant() {
    let engine = ObadhEngine::new();
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let explicit_key = row.roman_components.join(",,");
        let actual = engine.transliterate(&explicit_key);
        assert_eq!(
            actual, row.conjunct,
            "Explicit key '{explicit_key}' on row {} rendered as '{actual}', expected '{}'",
            row.line_number, row.conjunct
        );
    }
}

#[test]
fn test_source_conjunct_csv_rows_are_structurally_valid() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };
    let consonants = consonants_static();

    for row in source_conjunct_rows(&csv) {
        assert!(
            (2..=4).contains(&row.component_count),
            "Unexpected component count {} on row {}",
            row.component_count,
            row.line_number
        );
        assert_eq!(
            row.bn_components.len(),
            row.component_count,
            "Bengali component count mismatch on row {}",
            row.line_number
        );
        assert_eq!(
            row.roman_components.len(),
            row.component_count,
            "Roman component count mismatch on row {}",
            row.line_number
        );
        assert!(
            !row.example.is_empty(),
            "Missing Bengali example on row {}",
            row.line_number
        );

        for roman in &row.roman_components {
            assert!(
                roman.is_ascii() && !roman.contains(",,"),
                "Invalid Roman component '{roman}' on row {}",
                row.line_number
            );
            assert!(
                consonants.contains_key(roman) || is_conjunct_source_special_component(roman),
                "Unknown Roman component '{roman}' on row {}",
                row.line_number
            );
        }

        for field in std::iter::once(row.conjunct)
            .chain(row.bn_components.iter().copied())
            .chain(std::iter::once(row.example))
        {
            assert!(
                field.chars().all(is_bengali_source_char),
                "Unexpected non-Bengali source character in '{field}' on row {}",
                row.line_number
            );
        }
    }
}

#[test]
fn test_source_conjunct_csv_roman_components_match_bengali_components() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };
    let consonants = consonants_static();

    for row in source_conjunct_rows(&csv) {
        for (roman, bengali) in row.roman_components.iter().zip(&row.bn_components) {
            let expected = source_component_bengali(&row, roman, consonants).unwrap_or_else(|| {
                panic!(
                    "Unknown Roman component '{roman}' on row {}",
                    row.line_number
                )
            });

            assert_eq!(
                *bengali, expected,
                "Roman component '{roman}' maps to '{expected}', not '{}' on row {}",
                bengali, row.line_number
            );
        }
    }
}

#[test]
fn test_source_conjunct_csv_examples_contain_declared_forms() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let normalized_conjunct = row.conjunct.replace('\u{200c}', "");
        let normalized_example = row.example.replace('\u{200c}', "");
        assert!(
            normalized_example.contains(&normalized_conjunct),
            "Example '{}' does not contain declared conjunct '{}' on row {}",
            row.example,
            row.conjunct,
            row.line_number
        );
    }
}

#[test]
fn test_source_conjunct_csv_outputs_match_components_or_declared_exceptions() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    for row in source_conjunct_rows(&csv) {
        let expected = row.bn_components.join("্");
        let without_zwnj = row.conjunct.replace('\u{200c}', "");

        assert!(
            row.conjunct == expected
                || without_zwnj == expected
                || is_contextual_khanda_ta_row(&row),
            "Conjunct output '{}' does not match components '{}' on row {}",
            row.conjunct,
            expected,
            row.line_number
        );
    }
}

#[test]
fn test_source_conjunct_csv_keys_are_unique() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    let mut keys = BTreeSet::new();
    for row in source_conjunct_rows(&csv) {
        let key = row.key();
        assert!(
            keys.insert(key.clone()),
            "Duplicate CSV key '{key}' on row {}",
            row.line_number
        );
    }
}

#[test]
fn test_source_conjunct_csv_uses_canonical_khanda_ta_reph_signal() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    let rows: Vec<_> = source_conjunct_rows(&csv)
        .filter(|row| row.conjunct == "র্ৎ")
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "The র্ৎ source form should have one canonical Roman signal"
    );

    let row = &rows[0];
    assert_eq!(row.roman_components, vec!["rr", "t``"]);
    assert_eq!(row.key(), "rrt``");
    assert!(is_composable_khanda_ta_reph_source_row(row));
}

#[test]
fn test_compiled_conjunct_rules_are_source_owned_or_declared_aliases() {
    let definitions = conjuncts();
    let Some(csv) = source_conjunct_csv() else {
        return;
    };

    let mut allowed_keys = BTreeSet::new();
    for row in source_conjunct_rows(&csv) {
        allowed_keys.insert(row.key());
        allowed_keys.extend(declared_alias_keys_for_source_row(&row));
    }

    let unsourced_keys: Vec<_> = definitions
        .get_all_valid_conjuncts()
        .into_iter()
        .filter(|key| !allowed_keys.contains(*key))
        .collect();

    assert!(
        unsourced_keys.is_empty(),
        "Compiled conjunct keys must come from data/conjuncts.csv or a declared deterministic alias; unsourced keys: {unsourced_keys:?}"
    );
}

#[test]
fn test_source_conjunct_wiki_inventory_is_represented_in_csv() {
    let Some(csv) = source_conjunct_csv() else {
        return;
    };
    let Some(wiki) = source_conjunct_wiki() else {
        return;
    };

    let csv_conjuncts: BTreeSet<_> = source_conjunct_rows(&csv)
        .map(|row| row.conjunct.to_string())
        .collect();

    for row in source_conjunct_wiki_rows(&wiki) {
        assert!(
            csv_conjuncts.contains(row.conjunct),
            "Wiki conjunct '{}' on line {} is missing from data/conjuncts.csv",
            row.conjunct,
            row.line_number
        );
    }
}

#[test]
fn test_source_data_conjunct_aliases_render() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("kShy kShya mw mwa mwr mwra bw bwa zy zY zya"),
        "ক্ষ্য ক্ষ্যা ম্ব ম্বা ম্ব্র ম্ব্রা ব্ব ব্বা য্য য্য য্যা"
    );
}

struct SourceConjunctRow<'a> {
    line_number: usize,
    conjunct: &'a str,
    component_count: usize,
    bn_components: Vec<&'a str>,
    roman_components: Vec<&'a str>,
    example: &'a str,
}

struct SourceConjunctWikiRow<'a> {
    line_number: usize,
    conjunct: &'a str,
}

impl SourceConjunctRow<'_> {
    fn key(&self) -> String {
        self.roman_components.concat()
    }
}

fn source_conjunct_wiki() -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rules/conjunct.wiki");
    match fs::read_to_string(&path) {
        Ok(wiki) => Some(wiki),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("Failed to read {}: {error}", path.display()),
    }
}

fn source_conjunct_csv() -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/conjuncts.csv");
    match fs::read_to_string(&path) {
        Ok(csv) => Some(csv),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => panic!("Failed to read {}: {error}", path.display()),
    }
}

fn source_conjunct_wiki_rows(wiki: &str) -> impl Iterator<Item = SourceConjunctWikiRow<'_>> {
    wiki.lines()
        .enumerate()
        .filter_map(|(index, line)| parse_source_conjunct_wiki_row(index + 1, line))
}

fn parse_source_conjunct_wiki_row(
    line_number: usize,
    line: &str,
) -> Option<SourceConjunctWikiRow<'_>> {
    let declaration = line.strip_prefix("# ")?;
    let (conjunct, _) = declaration.split_once('=')?;

    Some(SourceConjunctWikiRow {
        line_number,
        conjunct: conjunct.trim(),
    })
}

fn source_conjunct_rows(csv: &str) -> impl Iterator<Item = SourceConjunctRow<'_>> {
    csv.lines()
        .enumerate()
        .skip(1)
        .map(|(index, line)| parse_source_conjunct_row(index + 1, line))
}

fn parse_source_conjunct_row(line_number: usize, line: &str) -> SourceConjunctRow<'_> {
    let columns: Vec<&str> = line.split(',').collect();
    assert_eq!(
        columns.len(),
        11,
        "Malformed conjunct CSV row {line_number}: expected 11 columns, got {}",
        columns.len()
    );

    let component_count = columns[1]
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("Invalid component count on row {line_number}: {error}"));

    SourceConjunctRow {
        line_number,
        conjunct: columns[0],
        component_count,
        bn_components: non_empty_components(&columns[2..6]),
        roman_components: non_empty_components(&columns[6..10]),
        example: columns[10],
    }
}

fn non_empty_components<'a>(components: &[&'a str]) -> Vec<&'a str> {
    components
        .iter()
        .copied()
        .filter(|component| !component.is_empty())
        .collect()
}

fn is_bengali_source_char(c: char) -> bool {
    matches!(
        c,
        '\u{0980}'..='\u{09ff}' | '\u{0964}' | '\u{0965}' | '\u{200c}' | '\u{200d}'
    )
}

fn is_contextual_khanda_ta_row(row: &SourceConjunctRow<'_>) -> bool {
    if !matches!(row.key().as_str(), "tk" | "tkh" | "tp" | "ts") {
        return false;
    }

    let Some((first, tail)) = row.bn_components.split_first() else {
        return false;
    };
    if *first != "ত" {
        return false;
    }

    row.conjunct == format!("ৎ{}", tail.join("্"))
}

fn is_composable_khanda_ta_reph_source_row(row: &SourceConjunctRow<'_>) -> bool {
    row.conjunct == "র্ৎ" && row.roman_components == ["rr", "t``"]
}

fn is_conjunct_source_special_component(component: &str) -> bool {
    matches!(component, "rr" | "w" | "t``")
}

fn declared_alias_keys_for_source_row(row: &SourceConjunctRow<'_>) -> BTreeSet<String> {
    let mut keys = expand_component_alias_keys(&row.roman_components);
    keys.extend(ksha_alias_keys(&row.roman_components));
    keys.extend(jna_alias_keys(&row.roman_components));
    keys.extend(palatal_nasal_ja_alias_keys(&row.roman_components));
    keys.extend(nasal_ba_alias_keys(&row.roman_components));
    keys.extend(base_ba_phola_alias_keys(&row.roman_components));
    keys.extend(base_ya_phola_alias_keys(&row.roman_components));
    keys
}

fn expand_component_alias_keys(components: &[&str]) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    push_component_alias_keys(&mut keys, components, 0, &mut String::new());
    keys
}

fn push_component_alias_keys(
    keys: &mut BTreeSet<String>,
    components: &[&str],
    index: usize,
    key: &mut String,
) {
    let Some(component) = components.get(index) else {
        keys.insert(key.clone());
        return;
    };

    let original_len = key.len();
    for alias in component_aliases(component, index) {
        key.push_str(alias);
        push_component_alias_keys(keys, components, index + 1, key);
        key.truncate(original_len);
    }
}

fn component_aliases(component: &str, index: usize) -> Vec<&str> {
    match component {
        "kh" => vec!["kh", "Kh", "KH"],
        "gh" => vec!["gh", "Gh", "GH"],
        "ch" => vec!["ch", "chh", "C", "Ch", "CH", "Chh", "CHH"],
        "jh" => vec!["jh", "Jh", "JH"],
        "Th" => vec!["Th", "TH"],
        "Dh" => vec!["Dh", "DH"],
        "f" | "ph" => vec!["f", "ph", "Ph", "PH"],
        "bh" | "v" => vec!["bh", "Bh", "BH", "v"],
        "j" | "J" => vec!["j", "J"],
        "sh" | "S" => vec!["sh", "S"],
        "Sh" => vec!["Sh", "SH"],
        "y" if index > 0 => vec!["y", "Y"],
        _ => vec![component],
    }
}

fn ksha_alias_keys(components: &[&str]) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();

    for index in 0..components.len().saturating_sub(1) {
        if components[index] == "k" && components[index + 1] == "Sh" {
            keys.extend(replace_component_pair(components, index, "ksh"));

            if index == 0 {
                keys.extend(replace_component_pair(components, index, "kkh"));
            }
        }
    }

    keys
}

fn jna_alias_keys(components: &[&str]) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();

    for index in 0..components.len().saturating_sub(1) {
        if matches!(components[index], "j" | "J") && components[index + 1] == "NG" {
            keys.extend(replace_component_pair(components, index, "jn"));
            keys.extend(replace_component_pair(components, index, "Jn"));
            keys.extend(replace_component_pair(components, index, "gg"));
        }
    }

    keys
}

fn palatal_nasal_ja_alias_keys(components: &[&str]) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();

    for index in 0..components.len().saturating_sub(1) {
        if components[index] == "NG" && matches!(components[index + 1], "j" | "J") {
            keys.extend(replace_component_pair(components, index, "nj"));
            keys.extend(replace_component_pair(components, index, "nJ"));
        }
    }

    keys
}

fn nasal_ba_alias_keys(components: &[&str]) -> BTreeSet<String> {
    if components.first() == Some(&"m") && components.get(1) == Some(&"w") {
        return replace_component_pair(components, 0, "mb");
    }

    BTreeSet::new()
}

fn base_ba_phola_alias_keys(components: &[&str]) -> BTreeSet<String> {
    if components.first() == Some(&"b") && components.get(1) == Some(&"b") {
        return replace_component_pair(components, 0, "bw");
    }

    BTreeSet::new()
}

fn base_ya_phola_alias_keys(components: &[&str]) -> BTreeSet<String> {
    if components.first() == Some(&"y") && components.get(1) == Some(&"y") {
        let mut keys = replace_component_pair(components, 0, "zy");
        keys.extend(replace_component_pair(components, 0, "zY"));
        return keys;
    }

    BTreeSet::new()
}

fn replace_component_pair<'a>(
    components: &[&'a str],
    pair_start: usize,
    replacement: &'a str,
) -> BTreeSet<String> {
    let mut replaced = Vec::with_capacity(components.len() - 1);
    replaced.extend_from_slice(&components[..pair_start]);
    replaced.push(replacement);
    replaced.extend_from_slice(&components[pair_start + 2..]);
    expand_component_alias_keys(&replaced)
}

fn source_component_bengali<'a>(
    row: &SourceConjunctRow<'_>,
    roman: &str,
    consonants: &'a std::collections::HashMap<&'static str, &'static str>,
) -> Option<&'a str> {
    match (row.key().as_str(), row.conjunct, roman) {
        ("rrt``", "র্ৎ", "t``") => Some("ৎ"),
        (_, _, "rr") => Some("র"),
        (_, _, "w") => Some("ব"),
        (_, _, "y" | "Y") => Some("য"),
        _ => consonants.get(roman).copied(),
    }
}

/// Tests for invalid conjuncts (should remain as separate consonants)
#[test]
fn test_invalid_conjuncts() {
    let tokenizer = Tokenizer::new();

    // These combinations are not valid Bengali conjuncts
    let test_cases = [
        "kf",  // ক+ফ is not a standard conjunct
        "pv",  // প+ভ is not a standard conjunct
        "hk",  // হ+ক is not a standard conjunct
        "Rw",  // w is not blindly accepted after every consonant
        "kfw", // only declared full clusters absorb the ba-phola marker
        "pph", // প্ফ is still typeable with explicit `p,,ph`, not implicit source data
        "pPh", // titlecase aliases do not admit unsourced implicit clusters
    ];

    for &input in &test_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify none of the units is a conjunct
        assert!(
            !units.iter().any(|u| {
                matches!(
                    u.unit_type,
                    PhoneticUnitType::Conjunct
                        | PhoneticUnitType::ConjunctWithVowel
                        | PhoneticUnitType::ConjunctWithTerminator
                )
            }),
            "Expected no conjuncts for '{}'",
            input
        );
    }
}

#[test]
fn test_unsourced_cluster_requires_explicit_hasant_signal() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("pph pPh"), "পফ পফ");
    assert_eq!(engine.transliterate("p,,ph p,,Ph"), "প্ফ প্ফ");
}

/// Tests for special cases: reph (রেফ), jo-phola (য-ফলা), and bo-phola (ব-ফলা)
#[test]
fn test_special_consonant_forms() {
    let tokenizer = Tokenizer::new();

    // Test special cases - groups by expected types
    let reph_cases = [
        ("rrk", PhoneticUnitType::RephOverConsonant), // Reph over consonant
        ("rrkA", PhoneticUnitType::RephOverConsonantWithVowel), // Reph with vowel
        ("rrko", PhoneticUnitType::RephOverConsonantWithTerminator), // Reph with terminator
    ];

    let phola_cases = [
        "ky",  // ক্য - ya-phola
        "ty",  // ত্য - ya-phola
        "dhy", // ধ্য - ya-phola
    ];

    // Test reph cases
    for (input, expected_type) in &reph_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify we got the expected reph form
        assert!(
            units.iter().any(|unit| unit.unit_type == *expected_type),
            "Expected at least one {:?} for '{}'",
            expected_type,
            input
        );
    }

    // Test phola cases
    for &input in &phola_cases {
        let units = tokenizer.tokenize_word(input);
        // Verify we got at least one conjunct for phola forms
        assert!(
            units
                .iter()
                .any(|unit| unit.unit_type == PhoneticUnitType::Conjunct),
            "Expected at least one conjunct for '{}'",
            input
        );
    }
}
