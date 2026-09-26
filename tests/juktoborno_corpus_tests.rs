//! Exhaustive juktoborno (যুক্তবর্ণ) corpus.
//!
//! Guards the two halves of the conjunct contract:
//!
//!  * **Completeness** — every conjunct in the source inventory forms from its
//!    romanized key in word-initial, medial and final position, and accepts the
//!    dependent vowels.
//!  * **Legality** — a consonant pair outside the inventory never joins on its
//!    own. Bangla does not license arbitrary `C + ্ + C`, so silent
//!    over-production is as much a defect as a missing conjunct.
//!
//! Both halves are driven from `data/conjuncts.csv`, the source of truth, so the
//! expectations are derived rather than hand-transcribed. On top of that sits a
//! well-formedness sweep asserting the engine never emits a malformed cluster
//! (dotted circle, dangling hasant, hasant before a kar, doubled hasant), and
//! a small curated table of real Bangla words.

use obadh_engine::{
    definitions::{conjuncts, consonant_value},
    ObadhEngine,
};
use std::fs;
use std::path::Path;

const HASANT: char = '\u{09CD}';
const KHANDA_TA: char = '\u{09CE}';
const DOTTED_CIRCLE: char = '\u{25CC}';

/// One canonical roman key per Bengali consonant letter. `y`/`Y` (ya-phola),
/// `w` (ba-phola), `rr` (reph) and the phota letters are excluded: they are
/// productive or positional forms, tested separately.
const CONSONANTS: &[&str] = &[
    "k", "kh", "g", "gh", "Ng", "c", "ch", "j", "jh", "NG", "T", "Th", "D", "Dh", "N", "t", "th",
    "d", "dh", "n", "p", "ph", "b", "bh", "m", "z", "r", "l", "sh", "Sh", "s", "h",
];

/// Dependent vowel signs, keyed by their roman trigger.
const VOWELS: &[(&str, &str)] = &[
    ("A", "\u{09BE}"),
    ("i", "\u{09BF}"),
    ("I", "\u{09C0}"),
    ("u", "\u{09C1}"),
    ("U", "\u{09C2}"),
    ("e", "\u{09C7}"),
    ("O", "\u{09CB}"),
    ("OI", "\u{09C8}"),
    ("OU", "\u{09CC}"),
];

const KARS: &[char] = &[
    '\u{09BE}', '\u{09BF}', '\u{09C0}', '\u{09C1}', '\u{09C2}', '\u{09C3}', '\u{09C7}', '\u{09C8}',
    '\u{09CB}', '\u{09CC}',
];

/// Signs that can never carry a hasant.
const NON_JOINERS: &[char] = &[KHANDA_TA, '\u{0982}', '\u{0983}', '\u{0981}'];

struct Row {
    line: usize,
    conjunct: String,
    roman: Vec<String>,
    example: String,
}

impl Row {
    fn key(&self) -> String {
        self.roman.concat()
    }

    /// `র্ৎ` composes a reph over khanda ta and cannot take a kar.
    fn is_khanda_ta_reph(&self) -> bool {
        self.roman.iter().any(|c| c == "t``")
    }

    /// Rows whose members are all plain consonants — no reph, phola, or khanda-ta
    /// signal. Only these can be mechanically broken apart with the inherent vowel.
    fn is_plain(&self) -> bool {
        self.roman
            .iter()
            .all(|c| !matches!(c.as_str(), "rr" | "w" | "y" | "Y" | "t``"))
    }
}

fn rows() -> Vec<Row> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/conjuncts.csv");
    let csv = fs::read_to_string(&path).expect("data/conjuncts.csv must be readable");

    csv.lines()
        .enumerate()
        .skip(1)
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(idx, line)| {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 11, "row {} has {} fields", idx + 1, f.len());
            Row {
                line: idx + 1,
                conjunct: f[0].to_string(),
                roman: f[6..10]
                    .iter()
                    .filter(|c| !c.is_empty())
                    .map(|c| c.to_string())
                    .collect(),
                example: f[10].to_string(),
            }
        })
        .collect()
}

fn joins(text: &str) -> bool {
    text.contains(HASANT) || text.contains(KHANDA_TA)
}

// ---------------------------------------------------------------- completeness

#[test]
fn every_source_conjunct_forms_in_every_word_position() {
    let engine = ObadhEngine::new();
    let mut checked = 0usize;

    for row in rows() {
        let key = row.key();

        // Isolated.
        assert_eq!(
            engine.transliterate(&key),
            row.conjunct,
            "row {}: '{key}' did not render '{}' in isolation",
            row.line,
            row.conjunct
        );

        // Word-final, after a vowel-bearing syllable.
        let final_input = format!("bA{key}");
        assert_eq!(
            engine.transliterate(&final_input),
            format!("বা{}", row.conjunct),
            "row {}: '{final_input}' broke '{}' word-finally",
            row.line,
            row.conjunct
        );

        if row.is_khanda_ta_reph() {
            checked += 2;
            continue;
        }

        // Word-initial and medial, before each dependent vowel.
        for (vk, kar) in VOWELS {
            let initial = format!("{key}{vk}");
            assert_eq!(
                engine.transliterate(&initial),
                format!("{}{kar}", row.conjunct),
                "row {}: '{initial}' broke '{}' word-initially",
                row.line,
                row.conjunct
            );

            let medial = format!("bA{key}{vk}");
            assert_eq!(
                engine.transliterate(&medial),
                format!("বা{}{kar}", row.conjunct),
                "row {}: '{medial}' broke '{}' medially",
                row.line,
                row.conjunct
            );
            checked += 2;
        }
        checked += 2;
    }

    assert!(checked > 5_000, "expected a large corpus, ran {checked}");
}

#[test]
fn every_source_conjunct_also_forms_via_explicit_hasant() {
    let engine = ObadhEngine::new();

    for row in rows() {
        let explicit = row.roman.join(",,");
        assert_eq!(
            engine.transliterate(&explicit),
            row.conjunct,
            "row {}: explicit '{explicit}' did not render '{}'",
            row.line,
            row.conjunct
        );
    }
}

/// The engine's cluster semantics: adjacency joins, and the inherent vowel `o`
/// breaks. `kta` → ক্তা but `kota` → কতা; `nsa` → ন্সা but `nosa` → নসা.
#[test]
fn inherent_vowel_breaks_every_plain_two_member_conjunct() {
    let engine = ObadhEngine::new();
    let mut checked = 0usize;

    for row in rows() {
        if row.roman.len() != 2 || !row.is_plain() {
            continue;
        }
        let (c1, c2) = (&row.roman[0], &row.roman[1]);
        let (Some(b1), Some(b2)) = (consonant_value(c1), consonant_value(c2)) else {
            continue;
        };

        let input = format!("{c1}o{c2}A");
        let actual = engine.transliterate(&input);
        assert_eq!(
            actual,
            format!("{b1}{b2}\u{09BE}"),
            "row {}: inherent vowel must break '{}' — '{input}' gave '{actual}'",
            row.line,
            row.conjunct
        );
        checked += 1;
    }

    assert!(checked > 150, "expected many plain pairs, ran {checked}");
}

// -------------------------------------------------------------------- legality

/// Bangla licenses a specific inventory, not arbitrary `C + ্ + C`. A pair
/// outside the inventory must render as two independent syllables.
#[test]
fn unattested_consonant_pairs_never_join_on_their_own() {
    let engine = ObadhEngine::new();
    let defs = conjuncts();
    let (mut joined, mut separate) = (0usize, 0usize);

    for c1 in CONSONANTS {
        for c2 in CONSONANTS {
            // `rr` is the reph signal, not the pair র+র.
            if format!("{c1}{c2}") == "rr" {
                continue;
            }

            let input = format!("{c1}{c2}A");
            let actual = engine.transliterate(&input);

            if defs.create_conjunct_from_parts(&[c1, c2]).is_some() {
                assert!(
                    joins(&actual),
                    "'{input}' is in the inventory but did not join: '{actual}'"
                );
                joined += 1;
            } else {
                assert!(
                    !joins(&actual),
                    "'{input}' is NOT in the inventory but produced a conjunct: '{actual}'"
                );
                separate += 1;
            }
        }
    }

    assert_eq!(joined + separate, CONSONANTS.len() * CONSONANTS.len() - 1);
    assert!(separate > 800, "legality guard too weak: {separate} negatives");
}

/// Explicit `,,` is the escape hatch: it joins any pair, attested or not.
#[test]
fn explicit_hasant_joins_every_consonant_pair() {
    let engine = ObadhEngine::new();

    for c1 in CONSONANTS {
        for c2 in CONSONANTS {
            let input = format!("{c1},,{c2}A");
            let actual = engine.transliterate(&input);
            assert!(
                joins(&actual),
                "explicit '{input}' must join, got '{actual}'"
            );
        }
    }
}

/// Dead ত ligates before ত, থ, ন, ব, ম, য, র; before ক, খ, প, স it renders as
/// খণ্ড-ত (উৎকর্ষ, উৎপল, বৎসর). ত + ল is *not* a licensed conjunct — single-
/// morpheme words use a live ত (কাতলা, পাতলা, মতলব) — so `tl` renders standalone,
/// never as ৎল; খণ্ড-ত before ল only crosses a morpheme boundary (সৎ+লোক) via the
/// explicit `t,,l` signal.
#[test]
fn khanda_ta_follows_the_unicode_ligature_rule() {
    let engine = ObadhEngine::new();

    for c2 in ["t", "th", "n", "w", "m", "y", "r"] {
        let actual = engine.transliterate(&format!("t{c2}A"));
        assert!(
            actual.starts_with('\u{09A4}') && actual.contains(HASANT),
            "ত + {c2} must ligate, not use khanda ta: '{actual}'"
        );
        assert!(!actual.contains(KHANDA_TA), "ত + {c2} must not use ৎ");
    }

    for c2 in ["k", "kh", "p", "s"] {
        let actual = engine.transliterate(&format!("t{c2}A"));
        assert!(
            actual.starts_with(KHANDA_TA),
            "ত + {c2} must use khanda ta: '{actual}'"
        );
    }

    // ত + ল is not a licensed conjunct: it renders standalone (তলা). The
    // dedicated খণ্ড-ত signal `t``` still produces ৎ before ল for morpheme
    // boundaries (সৎ+লোক), while the general explicit-hasant `,,` gives an
    // ordinary visible-hasant conjunct.
    assert_eq!(
        engine.transliterate("tlA"),
        "তলা",
        "ত + ল must render standalone, not as ৎল"
    );
    assert_eq!(
        engine.transliterate("t``lA"),
        "ৎলা",
        "the khanda-ta signal t`` must still produce ৎ before ল"
    );
    assert_eq!(
        engine.transliterate("t,,lA"),
        "ত্লা",
        "the general explicit-hasant ,, yields a visible-hasant conjunct"
    );
}

// ------------------------------------------------------------ well-formedness

/// Nothing the engine emits may be a malformed cluster.
#[test]
fn engine_never_emits_a_malformed_cluster() {
    let engine = ObadhEngine::new();
    let mut inputs: Vec<String> = Vec::new();

    for row in rows() {
        let key = row.key();
        inputs.push(key.clone());
        inputs.push(format!("bA{key}"));
        for (vk, _) in VOWELS {
            inputs.push(format!("{key}{vk}"));
            inputs.push(format!("bA{key}{vk}o"));
        }
    }
    for c1 in CONSONANTS {
        for c2 in CONSONANTS {
            // `rr` is a pending reph signal, not a pair; with no consonant to sit
            // on it legitimately renders as a bare র্.
            if format!("{c1}{c2}") == "rr" {
                continue;
            }
            inputs.push(format!("{c1}{c2}A"));
            inputs.push(format!("bA{c1}{c2}"));
        }
    }

    assert!(inputs.len() > 5_000, "corpus too small: {}", inputs.len());

    for input in &inputs {
        let out = engine.transliterate(input);
        let chars: Vec<char> = out.chars().collect();

        assert!(
            !out.contains(DOTTED_CIRCLE),
            "'{input}' produced a dotted circle: '{out}'"
        );
        assert!(
            !out.ends_with(HASANT),
            "'{input}' produced a dangling hasant: '{out}'"
        );

        for (i, &ch) in chars.iter().enumerate() {
            if ch != HASANT {
                continue;
            }
            if let Some(&next) = chars.get(i + 1) {
                assert!(
                    !KARS.contains(&next),
                    "'{input}': hasant before kar in '{out}'"
                );
                assert!(
                    next != HASANT,
                    "'{input}': doubled hasant in '{out}'"
                );
            }
            if let Some(prev) = i.checked_sub(1).and_then(|p| chars.get(p)) {
                assert!(
                    !NON_JOINERS.contains(prev),
                    "'{input}': hasant after a non-joining sign in '{out}'"
                );
            }
        }
    }
}

// ------------------------------------------------------------------ real words

/// Every CSV row ships an attested example word; its conjunct must survive.
/// Examples may spell the cluster with an explicit ZWNJ (খড়্‌গ), which is a
/// rendering choice over the same conjunct, so join controls are stripped first.
#[test]
fn source_example_words_contain_their_conjunct() {
    let strip = |s: &str| s.replace(['\u{200C}', '\u{200D}'], "");

    for row in rows() {
        if row.conjunct.contains(KHANDA_TA) || row.is_khanda_ta_reph() {
            continue;
        }
        assert!(
            strip(&row.example).contains(&strip(&row.conjunct)),
            "row {}: example '{}' does not contain '{}'",
            row.line,
            row.example,
            row.conjunct
        );
    }
}

#[test]
fn real_bangla_words_transliterate_correctly() {
    let engine = ObadhEngine::new();

    // Native / tatsama.
    let native = [
        ("montro", "মন্ত্র"),
        ("bakY", "বাক্য"),
        ("rokto", "রক্ত"),
        ("cokro", "চক্র"),
        ("bidyaloy", "বিদ্যালয়"),
        ("utsob", "উৎসব"),
        ("utpadon", "উৎপাদন"),
        // Word-final ত is live (ভাত, রাত); a *dead* final ত is খণ্ড ত and needs
        // the explicit `t\`\`` signal.
        ("bidyut``", "বিদ্যুৎ"),
        ("proshno", "প্রশ্ন"),
        ("bishw", "বিশ্ব"),
        ("nobanno", "নবান্ন"),
        ("condro", "চন্দ্র"),
        ("shokti", "শক্তি"),
        ("pokkho", "পক্ষ"),
        ("biggan", "বিজ্ঞান"),
        ("Daktar", "ডাক্তার"),
    ];

    // Native /n+s/ is written with anusvar, never ন্স.
    let anusvar = [("hongso", "হংস"), ("bongsho", "বংশ")];

    // Loanwords: ন্স is a loanword-only cluster.
    let loan = [
        ("laisens", "লাইসেন্স"),
        ("byalens", "ব্যালেন্স"),
        ("Diphens", "ডিফেন্স"),
        ("sens", "সেন্স"),
        ("TrAnsphar", "ট্রান্সফার"),
        ("kansa", "কান্সা"),
        ("eksTra", "এক্সট্রা"),
    ];

    // The inherent vowel `o` is the cluster break, engine-wide: `amra` → আম্রা but
    // `amora` → আমরা. ন্স follows the same rule as every other conjunct, so a word
    // that spells ন and স separately must type the `o`.
    let broken = [
        ("kanosa", "কানসা"),
        ("nosa", "নসা"),
        ("amra", "আম্রা"),
        ("amora", "আমরা"),
        ("inosTol", "ইনস্টল"),
        ("inospekTor", "ইনস্পেক্টর"),
        ("inosTiTiuT", "ইনস্টিটিউট"),
        ("konosarrT", "কনসার্ট"),
        ("anosar", "আনসার"),
        ("monosur", "মনসুর"),
    ];

    for (input, expected) in native
        .iter()
        .chain(&anusvar)
        .chain(&loan)
        .chain(&broken)
    {
        assert_eq!(
            &engine.transliterate(input),
            expected,
            "'{input}' should transliterate to '{expected}'"
        );
    }
}
