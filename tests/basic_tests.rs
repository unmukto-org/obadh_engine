use obadh_engine::definitions::{
    bengali_digit, conjuncts, consonant_categories, consonant_system, consonant_value,
    consonants_static, diacritic_rules, diacritic_value, diacritics_static, is_consonant, is_vowel,
    numeral_rules, numerals, numerals_static, symbol_rules, symbol_value, symbols_static,
    vowel_rules, vowel_value, vowels_static,
};
use obadh_engine::{ObadhEngine, PhoneticUnit, PhoneticUnitType, Tokenizer};

fn phonetic_shapes(units: &[PhoneticUnit]) -> Vec<(&str, PhoneticUnitType)> {
    units
        .iter()
        .map(|unit| (unit.text.as_str(), unit.unit_type))
        .collect()
}

#[test]
fn test_engine_creation() {
    let engine = ObadhEngine::new();
    assert_eq!(engine.transliterate(""), "");
}

#[test]
fn test_shared_definition_maps_expose_core_rules() {
    assert_eq!(consonants_static().get("kh"), Some(&"খ"));
    assert_eq!(consonant_value("kh"), Some("খ"));
    assert!(is_consonant("kh"));
    assert!(!is_consonant("q"));
    assert!(consonant_system().velars.contains(&("kh", "খ")));
    assert!(consonant_system().retroflexes.contains(&("T", "ট")));
    assert_eq!(consonant_categories()[0][0], ("k", "ক"));
    assert_eq!(
        vowels_static().get("a").map(|vowel| vowel.independent),
        Some("আ")
    );
    assert_eq!(vowel_value("A").map(|vowel| vowel.independent), Some("আ"));
    assert_eq!(
        vowel_value("aY").map(|vowel| (vowel.independent, vowel.dependent)),
        Some(("অ্যা", Some("্যা")))
    );
    assert_eq!(
        vowel_value("AY").map(|vowel| (vowel.independent, vowel.dependent)),
        Some(("অ্যা", Some("্যা")))
    );
    assert!(is_vowel("A"));
    assert!(!is_vowel("kh"));
    assert_eq!(vowel_rules()[0].0, "o");
    assert_eq!(
        vowels_static().get("u").map(|vowel| vowel.independent),
        Some("উ")
    );
    for repeated_vowel in ["aa", "ee", "ii", "oo", "uu"] {
        assert!(!is_vowel(repeated_vowel));
        assert!(vowels_static().get(repeated_vowel).is_none());
    }
    assert_eq!(diacritics_static().get(",,").copied(), Some("্"));
    assert_eq!(diacritic_value(",,"), Some("্"));
    assert_eq!(diacritic_rules()[0], (",,", "্"));
    assert_eq!(diacritics_static().get("t``").copied(), Some("ৎ"));
    assert_eq!(diacritics_static().get("T``").copied(), Some("ৎ"));
    assert_eq!(diacritics_static().get("M").copied(), Some("ং"));
    assert_eq!(diacritic_value("M"), Some("ং"));
    assert_eq!(symbols_static().get(".").copied(), Some("।"));
    assert_eq!(symbol_value("."), Some("।"));
    assert_eq!(symbol_rules()[0], (".", "।"));
    assert_eq!(numerals_static().get("9").copied(), Some("\u{09ef}"));
    assert_eq!(numeral_rules()[0], ("0", "\u{09e6}"));
    assert_eq!(numerals().get("9").copied(), Some("\u{09ef}"));
    assert_eq!(conjuncts().create_conjunct("kkh"), Some("ক্ষ"));
    assert_eq!(conjuncts().create_conjunct("rrg"), Some("র্গ"));
    assert_eq!(conjuncts().create_conjunct("rrt"), Some("র্ত"));
}

#[test]
fn test_conjunct_definition_helpers_use_compiled_rule_table() {
    let definitions = conjuncts();
    let valid = definitions.get_all_valid_conjuncts();

    assert!(definitions.can_form_conjunct("kk"));
    assert!(valid.contains("kk"));
    assert!(valid.contains("rrt"));
    assert!(!definitions.can_form_conjunct("kf"));
    assert!(definitions.is_special_form("rr"));
    assert!(definitions.is_special_form("w"));
    assert!(!definitions.is_special_form("z"));
    assert_eq!(
        definitions.get_components("ক্ষ"),
        Some(vec!["k".to_string(), "Sh".to_string()])
    );
    assert_eq!(
        definitions.get_components("ঙ্ক্ষ"),
        Some(vec!["Ng".to_string(), "k".to_string(), "Sh".to_string()])
    );
    assert_eq!(
        definitions.get_components("দ্ভ"),
        Some(vec!["d".to_string(), "bh".to_string()])
    );
    assert_eq!(
        definitions.get_components("ম্ব"),
        Some(vec!["m".to_string(), "w".to_string()])
    );
}

#[test]
fn test_ascii_digits_render_as_bengali_digits() {
    const BENGALI_DIGITS: &str =
        "\u{09e6}\u{09e7}\u{09e8}\u{09e9}\u{09ea}\u{09eb}\u{09ec}\u{09ed}\u{09ee}\u{09ef}";

    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("0123456789"), BENGALI_DIGITS);
    assert_eq!(engine.transliterate("k2 k20 a1b2"), "ক২ ক২০ আ১ব২");
}

#[test]
fn test_direct_digit_lookup_matches_ordered_numeral_table() {
    for &(roman, bengali) in numeral_rules() {
        let digit = roman
            .chars()
            .next()
            .expect("numeral rule keys are one ASCII digit");
        assert_eq!(bengali_digit(digit), Some(bengali));
    }

    assert_eq!(bengali_digit('৯'), None);
    assert_eq!(bengali_digit('x'), None);
}

#[test]
fn test_basic_transliteration() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("ami"), "আমি");

    // Further verify that sanitization is working
    let sanitized = engine.sanitize("ami").unwrap();
    assert_eq!(sanitized, "ami");

    // Verify that tokenization is working
    let tokens = engine.tokenize("ami");
    assert_eq!(tokens.len(), 1); // Should be a single word token
    assert_eq!(tokens[0].content, "ami");
}

#[test]
fn test_transliterate_invalid_input_returns_original_text() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("ami😀"), "ami😀");
    assert_eq!(engine.transliterate("ami 12.34😀"), "ami 12.34😀");
    assert_eq!(
        engine.sanitize("ami😀").unwrap_err(),
        "Invalid characters found: 😀"
    );
}

#[test]
fn test_lenient_transliteration_cleans_then_uses_direct_rendering() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate_lenient("ami😀"), "আমি");
    assert_eq!(
        engine.transliterate_lenient("ami 12.34😀 Taka."),
        "আমি ১২.৩৪ টাকা।"
    );
    assert_eq!(
        engine.transliterate_lenient("rYab😀 rrkSh 1.a2"),
        "র‍্যাব র্ক্ষ ১।আ২"
    );

    let cleaned = engine.sanitize("ami 12.34 Taka.").unwrap();
    assert_eq!(
        engine.transliterate_lenient("ami😀 12.34 Taka."),
        engine.transliterate(&cleaned)
    );
}

#[test]
fn test_tokenization() {
    let engine = ObadhEngine::new();

    // Test that the tokenizer breaks text into appropriate tokens
    let tokens = engine.tokenize("Hello, world!");

    // Should have 5 tokens: "Hello", ",", " ", "world", "!"
    assert_eq!(tokens.len(), 5);

    // Check that tokens have the correct content
    assert_eq!(tokens[0].content, "Hello");
    assert_eq!(tokens[1].content, ",");
    assert_eq!(tokens[2].content, " ");
    assert_eq!(tokens[3].content, "world");
    assert_eq!(tokens[4].content, "!");
}

#[test]
fn test_structural_whitespace_is_preserved() {
    let engine = ObadhEngine::new();

    assert_eq!(
        engine.transliterate("ami\nbangla\tlekhi\r\n12.34"),
        "আমি\nবাংলা\tলেখি\r\n১২.৩৪"
    );
    assert_eq!(
        engine.transliterate("ami\u{00a0}bangla\u{2003}lekhi"),
        "আমি\u{00a0}বাংলা\u{2003}লেখি"
    );
}

#[test]
fn test_decimal_point_is_not_dari_between_numbers() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("12.34"), "১২.৩৪");
    assert_eq!(engine.transliterate("12.34."), "১২.৩৪।");
    assert_eq!(engine.transliterate("ami 12.34 Taka."), "আমি ১২.৩৪ টাকা।");
    assert_eq!(engine.transliterate("k12.34 a1.b2"), "ক১২.৩৪ আ১।ব২");
}

#[test]
fn test_ascii_ellipsis_is_preserved_while_single_period_becomes_dari() {
    let engine = ObadhEngine::new();

    assert_eq!(engine.transliterate("ami."), "আমি।");
    assert_eq!(engine.transliterate("ami..."), "আমি...");
    assert_eq!(engine.transliterate("ami... 12.34..."), "আমি... ১২.৩৪...");
}

#[test]
fn test_contextual_token_transliteration_matches_tokenized_render() {
    let engine = ObadhEngine::new();
    let tokens = engine.tokenize("12.34 a1.b2. ami...");

    let rendered_tokens = (0..tokens.len())
        .map(|index| {
            engine
                .transliterate_token_at(&tokens, index)
                .expect("token index should render")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        rendered_tokens,
        vec![
            "১২",
            ".",
            "৩৪",
            " ",
            "আ১",
            "।",
            "ব২",
            "।",
            " ",
            "আমি",
            "..."
        ]
    );
    assert_eq!(
        rendered_tokens.concat(),
        engine.transliterate_tokens(&tokens)
    );
    assert_eq!(engine.transliterate_token_at(&tokens, tokens.len()), None);
}

#[test]
fn test_reusable_phonetic_tokenization_matches_allocating_api() {
    let engine = ObadhEngine::new();
    let mut units = Vec::with_capacity(64);

    engine.tokenize_phonetic_into("rrkSh", &mut units);
    assert_eq!(
        phonetic_shapes(&units),
        phonetic_shapes(&engine.tokenize_phonetic("rrkSh"))
    );
    assert_eq!(units.capacity(), 64);

    engine.tokenize_phonetic_into("kOU^:", &mut units);
    assert_eq!(
        phonetic_shapes(&units),
        vec![
            ("kOU", PhoneticUnitType::ConsonantWithVowel),
            ("^", PhoneticUnitType::SpecialForm),
            (":", PhoneticUnitType::SpecialForm),
        ]
    );
    assert_eq!(units.capacity(), 64);

    let tokenizer = Tokenizer::new();
    tokenizer.tokenize_word_into("kkA", &mut units);
    assert_eq!(
        phonetic_shapes(&units),
        vec![("k,,kA", PhoneticUnitType::ConjunctWithVowel)]
    );
}
