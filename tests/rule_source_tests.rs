use std::collections::BTreeSet;

use obadh_engine::definitions::{
    consonant_categories, consonant_value, diacritic_rules, diacritic_value, is_vowel,
    numeral_rules, symbol_rules, symbol_value, vowel_value,
};
use obadh_engine::ObadhEngine;

const CARGO_MANIFEST_DOC: &str = include_str!("../Cargo.toml");
const CONSONANT_RULES_DOC: &str = include_str!("../data/rules/consonants.md");
const README_DOC: &str = include_str!("../README.md");
const CONJUNCT_RULES_DOC: &str = include_str!("../data/rules/conjunct.wiki");
const DELIBERATE_INPUT_CORPUS_DOC: &str = include_str!("../data/rules/deliberate_input_corpus.md");
const SIMPLIFIED_RULES_DOC: &str = include_str!("../data/rules/simplified_rules.md");
const VOWEL_RULES_DOC: &str = include_str!("../data/rules/vowels.md");

#[test]
fn documented_basic_vowel_table_matches_runtime_rules() {
    for row in basic_vowel_table_rows(VOWEL_RULES_DOC) {
        for roman in roman_rule_keys(row.roman_input) {
            let vowel =
                vowel_value(roman).unwrap_or_else(|| panic!("documented vowel {roman:?} missing"));

            assert_eq!(
                vowel.independent, row.independent,
                "documented independent vowel for {roman:?} should match runtime"
            );
            assert_eq!(
                vowel.dependent,
                documented_dependent_vowel(row.dependent),
                "documented dependent vowel for {roman:?} should match runtime"
            );
        }
    }
}

#[test]
fn documented_lowercase_oi_ou_policy_matches_runtime_behavior() {
    assert!(!is_vowel("oi"));
    assert!(!is_vowel("ou"));

    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate("boi bou koi kou"), "বই বউ কই কউ");
    assert_eq!(engine.transliterate("bOI bOU kOI kOU"), "বৈ বৌ কৈ কৌ");
}

#[test]
fn documented_consonant_vowel_examples_match_runtime_behavior() {
    let engine = ObadhEngine::new();

    for row in consonant_vowel_example_rows(VOWEL_RULES_DOC) {
        assert_eq!(
            engine.transliterate(row.roman_input),
            row.bengali_output,
            "documented consonant-vowel example {:?} should match runtime",
            row.roman_input
        );
    }
}

#[test]
fn documented_simplified_rule_signals_match_runtime_behavior() {
    let engine = ObadhEngine::new();

    for example in documented_arrow_examples(SIMPLIFIED_RULES_DOC) {
        assert_eq!(
            engine.transliterate(example.roman_input),
            example.bengali_output,
            "documented simplified rule signal {:?} on line {} should match runtime",
            example.roman_input,
            example.line_number
        );
    }
}

#[test]
fn deliberate_input_rule_probe_corpus_matches_runtime_behavior() {
    let engine = ObadhEngine::new();
    let rows = deliberate_input_corpus_rows(DELIBERATE_INPUT_CORPUS_DOC);
    let mut inputs = BTreeSet::new();
    let mut categories = BTreeSet::new();

    assert_eq!(
        rows.len(),
        25,
        "rule probe corpus should keep the deliberate seed coverage set complete"
    );

    for row in rows {
        assert!(
            categories.insert(row.category),
            "rule probe corpus should not duplicate category {:?}",
            row.category
        );
        assert!(
            inputs.insert(row.roman_input),
            "rule probe corpus should not duplicate Roman input {:?}",
            row.roman_input
        );
        assert_eq!(
            engine.transliterate(row.roman_input),
            row.bengali_output,
            "rule probe corpus row {} in category {:?} should match runtime",
            row.line_number,
            row.category
        );
    }
}

#[test]
fn documented_numeral_table_matches_runtime_rules() {
    let documented_rows = simplified_numeral_table_rows(SIMPLIFIED_RULES_DOC).collect::<Vec<_>>();

    assert_eq!(
        documented_rows.len(),
        numeral_rules().len(),
        "simplified rules doc should list every runtime numeral rule"
    );

    for (&(roman, bengali), row) in numeral_rules().iter().zip(documented_rows) {
        assert_eq!(
            row.latin, roman,
            "documented numeral row should follow runtime numeral order"
        );
        assert_eq!(
            row.bengali, bengali,
            "documented Bengali numeral for {roman:?} should match runtime"
        );
    }
}

#[test]
fn deliberate_input_contract_documents_every_diacritic_rule() {
    let documented_signals = deliberate_input_contract_signal_cells(README_DOC);

    for &(roman, expected) in diacritic_rules() {
        assert_eq!(
            diacritic_value(roman),
            Some(expected),
            "diacritic {roman:?} should be directly renderable"
        );
        assert!(
            documented_signals
                .iter()
                .any(|signal| signal_cell_mentions(signal, roman)),
            "runtime diacritic signal {roman:?} is missing from README deliberate input contract"
        );
    }
}

#[test]
fn deliberate_input_contract_documents_every_symbol_rule() {
    let documented_signals = deliberate_input_contract_signal_cells(README_DOC);
    let engine = ObadhEngine::new();

    for &(roman, expected) in symbol_rules() {
        assert_eq!(
            symbol_value(roman),
            Some(expected),
            "symbol {roman:?} should be directly renderable"
        );
        assert_eq!(
            engine.transliterate(roman),
            expected,
            "symbol {roman:?} should render through the public engine path"
        );
        assert!(
            documented_signals
                .iter()
                .any(|signal| signal_cell_mentions(signal, roman)),
            "runtime symbol signal {roman:?} is missing from README deliberate input contract"
        );
    }

    assert_eq!(
        engine.transliterate("12.34 12.34."),
        "১২.৩৪ ১২.৩৪।",
        "decimal periods should stay ASCII periods between number-bearing tokens"
    );
}

#[test]
fn documented_consonant_table_matches_runtime_rules() {
    let mut documented_keys = BTreeSet::new();

    for row in basic_consonant_table_rows(CONSONANT_RULES_DOC) {
        for roman in roman_rule_keys(row.roman_input) {
            assert!(
                documented_keys.insert(roman),
                "documented consonant key {roman:?} should appear only once"
            );
            assert_eq!(
                consonant_value(roman),
                Some(row.bengali_output),
                "documented consonant output for {roman:?} should match runtime"
            );
        }
    }

    for category in consonant_categories() {
        for &(roman, expected) in category {
            assert_eq!(
                consonant_value(roman),
                Some(expected),
                "consonant {roman:?} should be directly renderable"
            );
            assert!(
                documented_keys.contains(roman),
                "runtime consonant signal {roman:?} is missing from data/rules/consonants.md"
            );
        }
    }
}

#[test]
fn deliberate_input_contract_documents_non_conjunct_ra_ya_zwnj_source_note() {
    let zwnj_ra_ya = "র\u{200D}\u{09CD}য";

    assert!(
        CONJUNCT_RULES_DOC.contains(zwnj_ra_ya),
        "source conjunct notes should retain the non-conjunct ra-ya ZWNJ example"
    );
    assert!(
        README_DOC.contains("`rZy`") && README_DOC.contains(zwnj_ra_ya),
        "README deliberate input contract should document the non-conjunct ra-ya ZWNJ signal"
    );
}

#[test]
fn cargo_package_excludes_non_runtime_audit_and_playground_assets() {
    for excluded_path in [
        "/data/autocorrect/**",
        "/data/autosuggest/**",
        "/docs/**",
        "/tools/**",
        "/www/**",
        "/.gitmodules",
        "/Makefile",
        "/build.sh",
        "/init.sh",
    ] {
        assert!(
            cargo_manifest_exclude_entries(CARGO_MANIFEST_DOC).contains(&excluded_path),
            "Cargo package should exclude {excluded_path} from the shipped crate"
        );
    }
}

struct VowelTableRow<'a> {
    roman_input: &'a str,
    independent: &'a str,
    dependent: &'a str,
}

struct ConsonantTableRow<'a> {
    roman_input: &'a str,
    bengali_output: &'a str,
}

struct ConsonantVowelExampleRow<'a> {
    roman_input: &'a str,
    bengali_output: &'a str,
}

struct DocumentedExample<'a> {
    line_number: usize,
    roman_input: &'a str,
    bengali_output: &'a str,
}

struct NumeralTableRow<'a> {
    latin: &'a str,
    bengali: &'a str,
}

struct RuleProbeRow<'a> {
    line_number: usize,
    category: &'a str,
    roman_input: &'a str,
    bengali_output: &'a str,
}

#[derive(Clone, Copy)]
struct CodeSpan<'a> {
    start: usize,
    end: usize,
    text: &'a str,
}

fn basic_consonant_table_rows(markdown: &str) -> impl Iterator<Item = ConsonantTableRow<'_>> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Roman Input | Bengali Output |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(parse_consonant_table_row)
}

fn parse_consonant_table_row(line: &str) -> Option<ConsonantTableRow<'_>> {
    let mut columns = line.trim_matches('|').split('|').map(str::trim);

    Some(ConsonantTableRow {
        roman_input: columns.next()?,
        bengali_output: columns.next()?,
    })
}

fn consonant_vowel_example_rows(
    markdown: &str,
) -> impl Iterator<Item = ConsonantVowelExampleRow<'_>> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Combination | Roman Input | Bengali Output |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(parse_consonant_vowel_example_row)
}

fn parse_consonant_vowel_example_row(line: &str) -> Option<ConsonantVowelExampleRow<'_>> {
    let mut columns = line.trim_matches('|').split('|').map(str::trim);
    let _combination = columns.next()?;

    Some(ConsonantVowelExampleRow {
        roman_input: columns.next()?,
        bengali_output: columns.next()?,
    })
}

fn basic_vowel_table_rows(markdown: &str) -> impl Iterator<Item = VowelTableRow<'_>> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Roman Input | Independent Vowel |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(parse_vowel_table_row)
}

fn parse_vowel_table_row(line: &str) -> Option<VowelTableRow<'_>> {
    let mut columns = line.trim_matches('|').split('|').map(str::trim);

    Some(VowelTableRow {
        roman_input: columns.next()?,
        independent: columns.next()?,
        dependent: columns.next()?,
    })
}

fn roman_rule_keys(input: &str) -> impl Iterator<Item = &str> {
    input.split('/').map(str::trim)
}

fn documented_dependent_vowel(value: &str) -> Option<&str> {
    if value.starts_with('-') {
        None
    } else {
        Some(value)
    }
}

fn simplified_numeral_table_rows(markdown: &str) -> impl Iterator<Item = NumeralTableRow<'_>> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Latin | Bengali |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(parse_numeral_table_row)
}

fn parse_numeral_table_row(line: &str) -> Option<NumeralTableRow<'_>> {
    let mut columns = line.trim_matches('|').split('|').map(str::trim);

    Some(NumeralTableRow {
        latin: columns.next()?,
        bengali: columns.next()?,
    })
}

fn deliberate_input_corpus_rows(markdown: &str) -> Vec<RuleProbeRow<'_>> {
    let rows = markdown
        .lines()
        .enumerate()
        .skip_while(|(_, line)| !line.starts_with("| Category | Roman Input | Bengali Output |"))
        .skip(2)
        .take_while(|(_, line)| line.starts_with('|'))
        .map(|(line_index, line)| parse_rule_probe_row(line_index + 1, line))
        .collect::<Vec<_>>();

    assert!(
        !rows.is_empty(),
        "rule probe corpus table is missing or empty"
    );

    rows
}

fn parse_rule_probe_row(line_number: usize, line: &str) -> RuleProbeRow<'_> {
    let columns = line
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();

    assert_eq!(
        columns.len(),
        4,
        "Malformed rule probe corpus row {line_number}: expected 4 columns, got {}",
        columns.len()
    );

    assert!(
        !columns[0].is_empty(),
        "Rule probe corpus row {line_number} should have a category"
    );
    assert!(
        !columns[3].is_empty(),
        "Rule probe corpus row {line_number} should explain the contract"
    );

    RuleProbeRow {
        line_number,
        category: columns[0],
        roman_input: single_code_span_text(columns[1]).unwrap_or_else(|| {
            panic!("Rule probe corpus row {line_number} should have one Roman input code span")
        }),
        bengali_output: single_code_span_text(columns[2]).unwrap_or_else(|| {
            panic!("Rule probe corpus row {line_number} should have one Bengali output code span")
        }),
    }
}

fn single_code_span_text(cell: &str) -> Option<&str> {
    let spans = code_spans(cell);
    (spans.len() == 1).then_some(spans[0].text)
}

fn cargo_manifest_exclude_entries(manifest: &str) -> BTreeSet<&str> {
    let mut entries = BTreeSet::new();
    let mut in_exclude = false;

    for line in manifest.lines() {
        let line = line.trim();

        if line.starts_with("exclude = [") {
            in_exclude = true;
            continue;
        }

        if in_exclude && line == "]" {
            break;
        }

        if in_exclude {
            let Some(entry) = line
                .strip_prefix('"')
                .and_then(|line| line.split_once('"').map(|(entry, _)| entry))
            else {
                continue;
            };
            entries.insert(entry);
        }
    }

    entries
}

fn deliberate_input_contract_signal_cells(markdown: &str) -> Vec<&str> {
    markdown
        .lines()
        .skip_while(|line| !line.starts_with("| Roman Signal | Bengali Rule Intent |"))
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .filter_map(|line| line.trim_matches('|').split('|').next().map(str::trim))
        .collect()
}

fn signal_cell_mentions(signal: &str, roman: &str) -> bool {
    signal.contains(&format!("`{roman}`")) || signal.contains(&format!("<code>{roman}</code>"))
}

fn documented_arrow_examples(markdown: &str) -> Vec<DocumentedExample<'_>> {
    let mut examples = Vec::new();

    for (line_index, line) in markdown.lines().enumerate() {
        let spans = code_spans(line);
        for (arrow_index, arrow) in line.match_indices('→') {
            let Some(output_span) = spans.iter().find(|span| span.start > arrow_index) else {
                continue;
            };

            for input_span in connected_input_spans_before_arrow(line, &spans, arrow_index) {
                examples.push(DocumentedExample {
                    line_number: line_index + 1,
                    roman_input: input_span.text,
                    bengali_output: output_span.text,
                });
            }

            debug_assert_eq!(arrow, "→");
        }
    }

    examples
}

fn connected_input_spans_before_arrow<'line, 'spans>(
    line: &'line str,
    spans: &'spans [CodeSpan<'line>],
    arrow_index: usize,
) -> &'spans [CodeSpan<'line>] {
    let Some(last_input_index) = spans
        .iter()
        .rposition(|span| span.end <= arrow_index && connector_text(line, span.end, arrow_index))
    else {
        return &[];
    };

    let mut first_input_index = last_input_index;
    while first_input_index > 0 {
        let previous = spans[first_input_index - 1];
        let current = spans[first_input_index];
        if !connector_text(line, previous.end, current.start) {
            break;
        }
        first_input_index -= 1;
    }

    &spans[first_input_index..=last_input_index]
}

fn connector_text(line: &str, start: usize, end: usize) -> bool {
    line[start..end]
        .chars()
        .all(|character| character.is_whitespace() || character == '/')
}

fn code_spans(line: &str) -> Vec<CodeSpan<'_>> {
    let mut spans = Vec::new();
    let mut cursor = 0;

    while cursor < line.len() {
        let next_backtick = line[cursor..].find('`').map(|offset| cursor + offset);
        let next_html = line[cursor..].find("<code>").map(|offset| cursor + offset);

        if let Some(content_start) = next_html.filter(|&html_start| {
            next_backtick.is_none_or(|backtick_start| html_start < backtick_start)
        }) {
            if let Some(content_end) = line[content_start + 6..]
                .find("</code>")
                .map(|offset| content_start + 6 + offset)
            {
                spans.push(CodeSpan {
                    start: content_start,
                    end: content_end + 7,
                    text: &line[content_start + 6..content_end],
                });
                cursor = content_end + 7;
                continue;
            }
        }

        let Some(content_start) = next_backtick else {
            break;
        };
        let Some(content_end) = line[content_start + 1..]
            .find('`')
            .map(|offset| content_start + 1 + offset)
        else {
            break;
        };

        spans.push(CodeSpan {
            start: content_start,
            end: content_end + 1,
            text: &line[content_start + 1..content_end],
        });
        cursor = content_end + 1;
    }

    spans.sort_unstable_by_key(|span| span.start);
    spans
}
