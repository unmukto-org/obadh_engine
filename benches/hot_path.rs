use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use obadh_engine::{
    AutocorrectEngine, AutosuggestContext, AutosuggestLm, AutosuggestOptions, AutosuggestSession,
    CorrectionRequest, FstLexicon, FstSuggestOptions, LexiconEntry, LoanwordLexicon,
    LoanwordSearchOptions, ObadhEngine, PersonalAutosuggest, PersonalAutosuggestConfig, Tokenizer,
};
use std::fs;
use std::time::Duration;

const RULE_STRESS_TEXT: &str =
    "kha gha kkA kko rrka rrko kShya k,,Ya n,,d,,rA songskrriti bidyuT`` rrT``sa 123.45";
const CONJUNCT_STRESS_WORD: &str = "rrkShkShmyntrngghya";
const MIXED_RULE_TEXT: &str =
    "kA khA gA. rrkSh rrT``sa; k,,y k,,w m,,w,,ra\nngga ngghAt jNG jn 123.45";
const LENIENT_MIXED_TEXT: &str = "ami😀 12.34 Taka. rZyab🔥 rrkSh 1.a2 songskrriti🚫";
const AUTOCORRECT_INPUT: &str = "কীরন";
const SHIPPED_AUTOCORRECT_FST_PATHS: &[&str] = &[
    "www/assets/autocorrect/bn.fst",
    "data/autocorrect/models/bn.fst",
];
const SHIPPED_AUTOSUGGEST_NGRAM_PATHS: &[&str] = &[
    "www/assets/autosuggest/autosuggest-ngram.bin",
    "data/autosuggest/models/ngram/autosuggest-ngram.bin",
];
const AUTOSUGGEST_CONTEXT_TEXT: &str = "আমি আজ";

fn bench_tokenizer(c: &mut Criterion) {
    let tokenizer = Tokenizer::new();

    let mut group = c.benchmark_group("tokenizer");
    group.throughput(Throughput::Bytes(RULE_STRESS_TEXT.len() as u64));
    group.bench_function("tokenize_text_rule_stress", |b| {
        b.iter(|| tokenizer.tokenize_text(black_box(RULE_STRESS_TEXT)));
    });

    group.throughput(Throughput::Bytes(CONJUNCT_STRESS_WORD.len() as u64));
    group.bench_function("tokenize_word_conjunct_stress", |b| {
        b.iter(|| tokenizer.tokenize_word(black_box(CONJUNCT_STRESS_WORD)));
    });
    group.finish();
}

fn bench_transliterator(c: &mut Criterion) {
    let engine = ObadhEngine::new();
    let tokens = engine.tokenize(MIXED_RULE_TEXT);

    let mut group = c.benchmark_group("transliterator");
    group.throughput(Throughput::Bytes(MIXED_RULE_TEXT.len() as u64));
    group.bench_function("transliterate_mixed_rule_text", |b| {
        b.iter(|| engine.transliterate(black_box(MIXED_RULE_TEXT)));
    });

    group.bench_function("render_pre_tokenized_mixed_rule_text", |b| {
        b.iter(|| engine.transliterate_tokens(black_box(&tokens)));
    });

    group.throughput(Throughput::Bytes(LENIENT_MIXED_TEXT.len() as u64));
    group.bench_function("transliterate_lenient_mixed_invalid_text", |b| {
        b.iter(|| engine.transliterate_lenient(black_box(LENIENT_MIXED_TEXT)));
    });
    group.finish();
}

fn bench_autocorrect(c: &mut Criterion) {
    let obadh = ObadhEngine::new();
    let autocorrect = AutocorrectEngine::from_entries(stress_lexicon_entries());
    let request = CorrectionRequest::new(AUTOCORRECT_INPUT);

    let mut group = c.benchmark_group("autocorrect");
    group.throughput(Throughput::Elements(1));
    group.bench_function("decide_stress_lexicon_input", |b| {
        b.iter(|| autocorrect.decide(black_box(request.clone())));
    });

    group.bench_function("obadh_request_then_decide_stress_lexicon", |b| {
        b.iter(|| {
            let request = obadh.autocorrect_request(black_box("biggan"));
            autocorrect.decide(black_box(request))
        });
    });
    group.finish();

    let Some(shipped_fst_bytes) = read_first_existing(SHIPPED_AUTOCORRECT_FST_PATHS) else {
        return;
    };

    let mut init_group = c.benchmark_group("autocorrect_init");
    init_group.throughput(Throughput::Bytes(shipped_fst_bytes.len() as u64));
    init_group.bench_function("shipped_fst_map_from_bytes", |b| {
        b.iter(|| {
            FstLexicon::from_bytes(black_box(shipped_fst_bytes.clone()))
                .expect("shipped FST lexicon should load")
        });
    });
    init_group.finish();

    let shipped_fst =
        FstLexicon::from_bytes(shipped_fst_bytes).expect("shipped FST lexicon should load");
    let fst_options = FstSuggestOptions {
        max_distance: 2,
        max_candidates: 512,
        max_prefix_candidates: 24,
        response_candidates: 12,
        ..FstSuggestOptions::default()
    };
    let mut group = c.benchmark_group("autocorrect_shipped");
    group.throughput(Throughput::Elements(1));
    group.bench_function("shipped_fst_suggest_sushil_512", |b| {
        b.iter(|| {
            shipped_fst
                .suggest(black_box("সুশিল"), black_box(fst_options))
                .expect("shipped FST suggestion should succeed")
        });
    });
    for (name, input) in [
        ("shipped_fst_suggest_kolom_512", "কলম"),
        ("shipped_fst_suggest_k_512", "ক"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                shipped_fst
                    .suggest(black_box(input), black_box(fst_options))
                    .expect("shipped FST suggestion should succeed")
            });
        });
    }
    group.finish();
}

fn bench_loanword_lookup(c: &mut Criterion) {
    let Some(bytes) = read_first_existing(&[
        "www/assets/autocorrect/en_bn_loanwords.fst",
        "data/autocorrect/models/en_bn_loanwords.fst",
    ]) else {
        return;
    };
    let lexicon = LoanwordLexicon::from_bytes(bytes).expect("shipped loanwords should load");
    let mut group = c.benchmark_group("loanword_shipped");
    for (name, input) in [
        ("exact", "keyboard"),
        ("single_swap", "ekyboard"),
        ("two_swaps", "ekyborad"),
        ("swap_and_substitution", "ekyboarx"),
    ] {
        let options = LoanwordSearchOptions::for_input(input);
        group.bench_function(name, |b| {
            b.iter(|| {
                lexicon
                    .suggestions(black_box(input), black_box(options))
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_autosuggest(c: &mut Criterion) {
    let Some(ngram_bytes) = read_first_existing(SHIPPED_AUTOSUGGEST_NGRAM_PATHS) else {
        return;
    };
    let lm =
        AutosuggestLm::from_bytes(&ngram_bytes).expect("shipped autosuggest model should load");
    let options = AutosuggestOptions { max_candidates: 5 };
    let context = autosuggest_context(&lm, AUTOSUGGEST_CONTEXT_TEXT);
    let token_cycle = autosuggest_token_cycle(&lm);
    let mut candidates = Vec::with_capacity(options.max_candidates);
    let mut candidate_ids = Vec::with_capacity(options.max_candidates);
    let mut session = autosuggest_session(&lm, options, &token_cycle);
    let mut id_session = autosuggest_session(&lm, options, &token_cycle);
    let mut commit_id_session = autosuggest_session(&lm, options, &token_cycle);
    let mut full_personal = full_personal_autosuggest();
    let mut cycle_index = 0_usize;
    let mut id_cycle_index = 0_usize;
    let mut rejected_token_id = 50_000_u32;

    let mut init_group = c.benchmark_group("autosuggest_init");
    init_group.throughput(Throughput::Bytes(ngram_bytes.len() as u64));
    init_group.bench_function("shipped_ngram_from_bytes", |b| {
        b.iter(|| {
            AutosuggestLm::from_bytes(black_box(ngram_bytes.as_slice()))
                .expect("shipped autosuggest model should load")
        });
    });
    init_group.finish();

    let mut group = c.benchmark_group("autosuggest");
    group.throughput(Throughput::Elements(1));
    group.bench_function("suggest_for_text_ngram", |b| {
        b.iter(|| {
            lm.suggest_for_text_into(
                black_box(AUTOSUGGEST_CONTEXT_TEXT),
                black_box(options),
                black_box(&mut candidates),
            )
            .expect("shipped autosuggest text suggestion should succeed")
        });
    });

    group.bench_function("suggest_for_context_ngram", |b| {
        b.iter(|| {
            lm.suggest_for_context_into(black_box(context), black_box(options), &mut candidates)
                .expect("shipped autosuggest context suggestion should succeed")
        });
    });

    group.bench_function("suggest_ids_for_context_ngram", |b| {
        b.iter(|| {
            lm.suggest_ids_for_context_into(
                black_box(context),
                black_box(options),
                &mut candidate_ids,
            )
            .expect("shipped autosuggest ID suggestion should succeed")
        });
    });

    group.bench_function("session_suggest_personal_overlay", |b| {
        b.iter(|| {
            session
                .suggest()
                .expect("shipped autosuggest session suggestion should succeed")
        });
    });

    group.bench_function("session_suggest_ids_personal_overlay", |b| {
        b.iter(|| {
            id_session
                .suggest_ids()
                .expect("shipped autosuggest session ID suggestion should succeed")
        });
    });

    group.bench_function("session_commit_token_id_then_suggest", |b| {
        b.iter(|| {
            let token_id = token_cycle[cycle_index % token_cycle.len()];
            cycle_index = cycle_index.wrapping_add(1);
            session
                .commit_token_id(Some(black_box(token_id)), false)
                .expect("known autosuggest token ID should be accepted");
            session
                .suggest()
                .expect("shipped autosuggest session suggestion should succeed")
        });
    });

    group.bench_function("session_commit_token_id_then_suggest_ids", |b| {
        b.iter(|| {
            let token_id = token_cycle[id_cycle_index % token_cycle.len()];
            id_cycle_index = id_cycle_index.wrapping_add(1);
            commit_id_session
                .commit_token_id(Some(black_box(token_id)), false)
                .expect("known autosuggest token ID should be accepted");
            commit_id_session
                .suggest_ids()
                .expect("shipped autosuggest session ID suggestion should succeed")
        });
    });

    group.bench_function("personal_full_store_reject_singleton", |b| {
        b.iter(|| {
            full_personal.observe_context_ids_target(&[], black_box(rejected_token_id));
            rejected_token_id = rejected_token_id.wrapping_add(1);
        });
    });
    group.finish();
}

fn autosuggest_context<D: AsRef<[u8]>>(lm: &AutosuggestLm<D>, text: &str) -> AutosuggestContext {
    let mut context = AutosuggestContext::new();
    lm.push_context_text(&mut context, text)
        .expect("benchmark context should use known tokens");
    context
}

fn autosuggest_token_cycle<D: AsRef<[u8]>>(lm: &AutosuggestLm<D>) -> [u32; 4] {
    ["আমি", "আজ", "বাংলা", "মানুষ"].map(|token| {
        lm.token_id(token)
            .expect("benchmark token lookup should succeed")
            .expect("benchmark token should exist in shipped autosuggest vocab")
    })
}

fn autosuggest_session<'lm, D: AsRef<[u8]>>(
    lm: &'lm AutosuggestLm<D>,
    options: AutosuggestOptions,
    token_cycle: &[u32],
) -> AutosuggestSession<'lm, D> {
    let mut session =
        AutosuggestSession::with_personal_config(lm, PersonalAutosuggestConfig::default(), options);
    for _ in 0..PersonalAutosuggestConfig::default().min_count {
        session.clear_context();
        for token_id in token_cycle {
            session
                .commit_token_id(Some(*token_id), false)
                .expect("known autosuggest token ID should be accepted");
        }
    }
    session.clear_context();
    for token_id in token_cycle.iter().take(2) {
        session
            .commit_token_id(Some(*token_id), false)
            .expect("known autosuggest token ID should be accepted");
    }
    session
}

fn full_personal_autosuggest() -> PersonalAutosuggest {
    let config = PersonalAutosuggestConfig {
        max_entries: 4096,
        min_count: 1,
    };
    let mut personal = PersonalAutosuggest::new(config);
    for token_id in 3..(3 + config.max_entries as u32) {
        personal.observe_context_ids_target(&[], token_id);
        personal.observe_context_ids_target(&[], token_id);
    }
    personal.observe_context_ids_target(&[], 49_999);
    personal
}

fn read_first_existing(paths: &[&str]) -> Option<Vec<u8>> {
    for path in paths {
        match fs::read(path) {
            Ok(bytes) => return Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!("skipping benchmark artifact {path}: {error}"),
        }
    }

    eprintln!(
        "skipping shipped-artifact benchmarks; none of these paths exist: {}",
        paths.join(", ")
    );
    None
}

fn stress_lexicon_entries() -> Vec<LexiconEntry> {
    let heads = [
        "ক", "খ", "গ", "ঘ", "চ", "জ", "ট", "ড", "ত", "দ", "ন", "প", "ব", "ম", "য", "র",
    ];
    let tails = [
        "ান", "িন", "ীন", "ুন", "েন", "োন", "ার", "ির", "ের", "াল", "িল", "ুল", "েল", "াম", "িম", "ুম",
    ];

    heads
        .iter()
        .flat_map(|head| tails.iter().map(move |tail| format!("{head}{tail}")))
        .enumerate()
        .map(|(index, word)| LexiconEntry::new(word, 10_000_u32.saturating_sub(index as u32)))
        .collect()
}

criterion_group! {
    name = hot_path;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets = bench_tokenizer, bench_transliterator, bench_autocorrect, bench_loanword_lookup, bench_autosuggest
}
criterion_main!(hot_path);
