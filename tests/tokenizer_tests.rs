use obadh_engine::{ObadhEngine, PhoneticUnitType, Token, TokenType, Tokenizer};

fn token_shapes(tokens: &[Token]) -> Vec<(&str, TokenType)> {
    tokens
        .iter()
        .map(|token| (token.content.as_str(), token.token_type.clone()))
        .collect()
}

#[test]
fn test_text_tokenization() {
    let tokenizer = Tokenizer::new();

    let tokens = tokenizer.tokenize_text("Hello World!");
    assert_eq!(
        token_shapes(&tokens),
        vec![
            ("Hello", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("World", TokenType::Word),
            ("!", TokenType::Punctuation),
        ]
    );

    let tokens = tokenizer.tokenize_text("Amar nam, 1234.");
    assert_eq!(
        token_shapes(&tokens),
        vec![
            ("Amar", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("nam", TokenType::Word),
            (",", TokenType::Punctuation),
            (" ", TokenType::Whitespace),
            ("1234", TokenType::Number),
            (".", TokenType::Punctuation),
        ]
    );

    let tokens = tokenizer.tokenize_text("ami... 12.34...");
    assert_eq!(
        token_shapes(&tokens),
        vec![
            ("ami", TokenType::Word),
            ("...", TokenType::Punctuation),
            (" ", TokenType::Whitespace),
            ("12", TokenType::Number),
            (".", TokenType::Punctuation),
            ("34", TokenType::Number),
            ("...", TokenType::Punctuation),
        ]
    );
}

#[test]
fn test_non_ascii_punctuation_tokenization_is_context_free() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize_text("। ami।bangla ।। 123।");

    let symbol_positions: Vec<_> = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (token.content == "।").then_some(index))
        .collect();

    assert_eq!(symbol_positions, vec![0, 3, 6, 7, 10]);
    for index in symbol_positions {
        assert_eq!(tokens[index].token_type, TokenType::Symbol);
    }
}

#[test]
fn test_standalone_diacritic_markers_are_phonetic_tokens() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize_text("^ : ^: :^");

    assert_eq!(
        token_shapes(&tokens),
        vec![
            ("^", TokenType::Word),
            (" ", TokenType::Whitespace),
            (":", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("^:", TokenType::Word),
            (" ", TokenType::Whitespace),
            (":^", TokenType::Word),
        ]
    );
}

#[test]
fn test_standalone_hasant_marker_is_a_phonetic_token() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize_text(",, k,, k,,k ,,");

    assert_eq!(
        token_shapes(&tokens),
        vec![
            (",,", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("k,,", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("k,,k", TokenType::Word),
            (" ", TokenType::Whitespace),
            (",,", TokenType::Word),
        ]
    );
}

#[test]
fn test_text_tokenization_tracks_numeric_words_incrementally() {
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize_text("123 ১২৩ a1 1a ^12 12^ k,, 1,,");

    assert_eq!(
        token_shapes(&tokens),
        vec![
            ("123", TokenType::Number),
            (" ", TokenType::Whitespace),
            ("১২৩", TokenType::Number),
            (" ", TokenType::Whitespace),
            ("a1", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("1a", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("^12", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("12^", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("k,,", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("1,,", TokenType::Word),
        ]
    );
}

#[test]
fn test_empty_phonetic_tokenization_is_safe() {
    let tokenizer = Tokenizer::new();
    assert!(tokenizer.tokenize_word("").is_empty());

    let engine = ObadhEngine::new();
    assert!(engine.tokenize_phonetic("").is_empty());
}

#[test]
fn test_phonetic_tokenization_uses_definition_rules() {
    let tokenizer = Tokenizer::new();

    let cases = [
        ("k", "k", PhoneticUnitType::Consonant),
        ("kh", "kh", PhoneticUnitType::Consonant),
        ("g", "g", PhoneticUnitType::Consonant),
        ("Gh", "Gh", PhoneticUnitType::Consonant),
        ("i", "i", PhoneticUnitType::Vowel),
        ("I", "I", PhoneticUnitType::Vowel),
        ("e", "e", PhoneticUnitType::Vowel),
        ("E", "E", PhoneticUnitType::Vowel),
        ("rr", "rr", PhoneticUnitType::SpecialForm),
        (",,", ",,", PhoneticUnitType::ConsonantWithHasant),
        ("M", "M", PhoneticUnitType::SpecialForm),
        (".", ".", PhoneticUnitType::Symbol),
        ("$", "$", PhoneticUnitType::Symbol),
        ("kha", "kha", PhoneticUnitType::ConsonantWithVowel),
        ("gha", "gha", PhoneticUnitType::ConsonantWithVowel),
    ];

    for (input, text, unit_type) in cases {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(units.len(), 1, "{input} should produce one unit");
        assert_eq!(units[0].text, text);
        assert_eq!(units[0].unit_type, unit_type);
    }

    // `ng` before a vowel folds to the velar nasal ঙ, so it tokenizes as one
    // consonant-with-vowel unit, identical to `Ng` + vowel.
    let units = tokenizer.tokenize_word("nga");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].text, "Nga");
    assert_eq!(units[0].unit_type, PhoneticUnitType::ConsonantWithVowel);

    // Before a consonant, `ng` stays the anusvar special form.
    let units = tokenizer.tokenize_word("ngla");
    assert_eq!(units[0].text, "ng");
    assert_eq!(units[0].unit_type, PhoneticUnitType::SpecialForm);

    let units = tokenizer.tokenize_word("k2");
    assert_eq!(
        units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("k", PhoneticUnitType::Consonant),
            ("2", PhoneticUnitType::Numeral),
        ]
    );
}

#[test]
fn test_phonetic_tokenization_canonicalizes_safe_case_fallbacks() {
    let tokenizer = Tokenizer::new();

    for (input, expected) in [
        ("B", "b"),
        ("G", "g"),
        ("P", "p"),
        ("F", "f"),
        ("K", "k"),
        ("L", "l"),
        ("V", "v"),
        ("H", "h"),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type))
                .collect::<Vec<_>>(),
            vec![(expected, PhoneticUnitType::Consonant)],
            "{input} should canonicalize to {expected}"
        );
    }

    for (input, expected_type) in [
        ("T", PhoneticUnitType::Consonant),
        ("D", PhoneticUnitType::Consonant),
        ("N", PhoneticUnitType::Consonant),
        ("S", PhoneticUnitType::Consonant),
        ("I", PhoneticUnitType::Vowel),
        ("U", PhoneticUnitType::Vowel),
        ("O", PhoneticUnitType::Vowel),
        ("Y", PhoneticUnitType::Consonant),
        ("M", PhoneticUnitType::SpecialForm),
    ] {
        let units = tokenizer.tokenize_word(input);
        assert_eq!(
            units
                .iter()
                .map(|unit| (unit.text.as_str(), unit.unit_type))
                .collect::<Vec<_>>(),
            vec![(input, expected_type)],
            "{input} should keep its exact protected behavior"
        );
    }

    // `Z` is not a trie case-fallback: it stays reserved so the narrow `rZy` marker
    // can claim it, and a `Z` that no marker consumes is folded to `z` by a later
    // normalization rather than leaking a literal glyph.
    let z_units = tokenizer.tokenize_word("Z");
    assert_eq!(
        z_units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![("z", PhoneticUnitType::Consonant)],
        "a stray Z folds to z"
    );
}

#[test]
fn test_integration_with_engine() {
    let engine = ObadhEngine::new();

    let tokens = engine.tokenize("Amar nam, 1234.");
    assert_eq!(
        token_shapes(&tokens),
        vec![
            ("Amar", TokenType::Word),
            (" ", TokenType::Whitespace),
            ("nam", TokenType::Word),
            (",", TokenType::Punctuation),
            (" ", TokenType::Whitespace),
            ("1234", TokenType::Number),
            (".", TokenType::Punctuation),
        ]
    );

    let units = engine.tokenize_phonetic("Amar");
    assert_eq!(units.len(), 3);
    assert_eq!(units[0].text, "A");
    assert_eq!(units[0].unit_type, PhoneticUnitType::Vowel);
    assert_eq!(units[1].text, "ma");
    assert_eq!(units[1].unit_type, PhoneticUnitType::ConsonantWithVowel);
    assert_eq!(units[2].text, "r");
    assert_eq!(units[2].unit_type, PhoneticUnitType::Consonant);
}

#[test]
fn test_phonetic_matching_uses_deterministic_longest_prefixes() {
    let tokenizer = Tokenizer::new();

    for input in ["t``", "T``"] {
        let khanda_ta = tokenizer.tokenize_word(input);
        assert_eq!(khanda_ta.len(), 1);
        assert_eq!(khanda_ta[0].text, input);
        assert_eq!(khanda_ta[0].unit_type, PhoneticUnitType::SpecialForm);
    }

    let vocalic_r = tokenizer.tokenize_word("rria");
    assert_eq!(
        vocalic_r
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rri", PhoneticUnitType::Vowel),
            ("a", PhoneticUnitType::Vowel),
        ]
    );

    let aspirated = tokenizer.tokenize_word("kha");
    assert_eq!(aspirated.len(), 1);
    assert_eq!(aspirated[0].text, "kha");
    assert_eq!(aspirated[0].unit_type, PhoneticUnitType::ConsonantWithVowel);

    let terminal_fallback = tokenizer.tokenize_word("ka");
    assert_eq!(terminal_fallback.len(), 1);
    assert_eq!(terminal_fallback[0].text, "ka");
    assert_eq!(
        terminal_fallback[0].unit_type,
        PhoneticUnitType::ConsonantWithVowel
    );

    let diphthong = tokenizer.tokenize_word("kOU");
    assert_eq!(diphthong.len(), 1);
    assert_eq!(diphthong[0].text, "kOU");
    assert_eq!(diphthong[0].unit_type, PhoneticUnitType::ConsonantWithVowel);
}

#[test]
fn test_adjacent_rr_normalization_is_left_to_right() {
    let tokenizer = Tokenizer::new();

    let units = tokenizer.tokenize_word("rrrrka");
    assert_eq!(
        units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rr", PhoneticUnitType::SpecialForm),
            ("rrka", PhoneticUnitType::RephOverConsonantWithVowel),
        ]
    );

    let units = tokenizer.tokenize_word("rrirrka");
    assert_eq!(
        units
            .iter()
            .map(|unit| (unit.text.as_str(), unit.unit_type))
            .collect::<Vec<_>>(),
        vec![
            ("rri", PhoneticUnitType::Vowel),
            ("rrka", PhoneticUnitType::RephOverConsonantWithVowel),
        ]
    );
}
