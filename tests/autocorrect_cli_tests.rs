#![cfg(feature = "cli")]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[test]
fn emoticons_bypass_spelling_correction_in_both_cli_backends() {
    let workspace = TempWorkspace::new("obadh-emoticon-cli");
    let source = workspace.path("lexicon.tsv");
    fs::write(&source, "বাংলা\t100\n").unwrap();
    for (build_command, suggest_command) in [
        ("build-lexicon", "suggest"),
        ("build-fst-lexicon", "suggest-fst"),
    ] {
        let artifact = workspace.path(build_command);
        let output = run_obadh_autocorrect([
            build_command,
            "--input",
            path_str(&source),
            "--output",
            path_str(&artifact),
        ]);
        assert!(output.status.success(), "{}", stderr(&output));
        for (input, emoji) in [(":)", "😃"), (":-)", "😃"), (":D", "😄"), (":-D", "😄")] {
            let output = run_obadh_autocorrect([
                suggest_command,
                "--lexicon",
                path_str(&artifact),
                "--input",
                input,
            ]);
            assert!(output.status.success(), "{}", stderr(&output));
            let result = json_stdout(&output);
            assert_eq!(result["obadh_output"], input);
            assert_eq!(result["candidates"][0]["text"], emoji);
            assert_eq!(result["candidates"][0]["source"], "emoticon_exact");
            assert_eq!(result["candidates"][0]["frequency"], 0);
            assert_eq!(result["returned_candidates"], 1);
            assert!(result.get("replacement").is_none_or(Value::is_null));
        }
    }
}

#[test]
fn autocorrect_cli_builds_inspects_and_evaluates_artifacts() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-roundtrip");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let bangla_eval = workspace.path("bangla-eval.tsv");
    let roman_eval = workspace.path("roman-eval.tsv");

    fs::write(&lexicon_tsv, "আমি\t100\nআমার\t80\nবিজ্ঞান\t70\nকিরণ\t60\n")
        .expect("lexicon fixture should write");
    fs::write(&bangla_eval, "আমী\tআমি\nবিজান\tবিজ্ঞান\n").expect("bangla eval fixture should write");
    fs::write(&roman_eval, "biggan\tবিজ্ঞান\nami\tআমি\n").expect("roman eval fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));
    let build_json = json_stdout(&build);
    assert_eq!(build_json["entries"], 4);
    assert_eq!(build_json["input_rows"], 4);
    assert_eq!(build_json["duplicate_rows"], 0);
    assert_eq!(build_json["total_frequency"], 310);
    assert_eq!(build_json["max_frequency"], 100);
    assert!(build_json["trie_nodes"].as_u64().unwrap() >= 4);
    assert_eq!(build_json["skeleton_keys"], 4);
    assert!(build_json["skeleton_delete_keys"].as_u64().unwrap() >= 4);
    assert!(artifact.exists());

    let inspect = run_obadh_autocorrect(["inspect-lexicon", "--input", path_str(&artifact)]);
    assert!(inspect.status.success(), "stderr: {}", stderr(&inspect));
    let inspect_json = json_stdout(&inspect);
    assert_eq!(inspect_json["entries"], 4);
    assert_eq!(inspect_json["artifact_bytes"], build_json["artifact_bytes"]);
    assert_eq!(inspect_json["skeleton_keys"], build_json["skeleton_keys"]);
    assert_eq!(
        inspect_json["skeleton_delete_keys"],
        build_json["skeleton_delete_keys"]
    );

    let bangla_report = run_obadh_autocorrect([
        "eval",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&bangla_eval),
        "--input-kind",
        "bangla",
    ]);
    assert!(
        bangla_report.status.success(),
        "stderr: {}",
        stderr(&bangla_report)
    );
    let bangla_json = json_stdout(&bangla_report);
    assert_eq!(bangla_json["total"], 2);
    assert_eq!(bangla_json["target_in_lexicon"], 2);
    assert_eq!(bangla_json["target_in_candidates"], 2);
    assert_eq!(bangla_json["target_lexicon_coverage"], 1.0);
    assert_eq!(bangla_json["candidate_recall_given_target_in_lexicon"], 1.0);
    assert_eq!(bangla_json["suggestion_recall"], 2);
    assert!(
        bangla_json["mean_reciprocal_rank"].as_f64().unwrap() > 0.0,
        "expected candidates to be ranked for typo pairs: {bangla_json}"
    );

    let roman_report = run_obadh_autocorrect([
        "eval",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&roman_eval),
        "--input-kind",
        "roman",
    ]);
    assert!(
        roman_report.status.success(),
        "stderr: {}",
        stderr(&roman_report)
    );
    let roman_json = json_stdout(&roman_report);
    assert_eq!(roman_json["total"], 2);
    assert_eq!(roman_json["target_in_lexicon"], 2);
    assert_eq!(roman_json["final_output_correct"], 2);
    assert_eq!(roman_json["baseline_accuracy"], 1.0);
}

#[test]
fn autocorrect_cli_builds_and_queries_fst_lexicon() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(
        &lexicon_tsv,
        "কেমন\t100\nকমন\t20\nযেমন\t30\nবিজ্ঞান\t70\nঅকালপক্ক\t5\n",
    )
    .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));
    let build_json = json_stdout(&build);
    assert_eq!(build_json["entries"], 5);
    assert_eq!(build_json["input_rows"], 5);
    assert_eq!(build_json["duplicate_rows"], 0);
    assert_eq!(build_json["total_frequency"], 225);
    assert_eq!(build_json["max_frequency"], 100);
    assert!(artifact.exists());

    let inspect = run_obadh_autocorrect(["inspect-fst-lexicon", "--input", path_str(&artifact)]);
    assert!(inspect.status.success(), "stderr: {}", stderr(&inspect));
    let inspect_json = json_stdout(&inspect);
    assert_eq!(inspect_json["entries"], 5);
    assert_eq!(inspect_json["artifact_bytes"], build_json["artifact_bytes"]);

    let exact = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "biggan",
        "--response-candidates",
        "8",
    ]);
    assert!(exact.status.success(), "stderr: {}", stderr(&exact));
    let exact_json = json_stdout(&exact);
    assert_eq!(exact_json["obadh_output"], "বিজ্ঞান");
    assert_eq!(exact_json["exact_frequency"], 70);
    let exact_candidate = exact_json["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "বিজ্ঞান")
        .expect("exact Obadh output should be returned");
    assert_eq!(exact_candidate["source"], "fst_exact");
    assert_eq!(exact_candidate["edit_cost"], 0);

    let edit = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "kmn",
        "--max-distance",
        "1",
        "--max-candidates",
        "64",
        "--response-candidates",
        "8",
    ]);
    assert!(edit.status.success(), "stderr: {}", stderr(&edit));
    let edit_json = json_stdout(&edit);
    assert_eq!(edit_json["obadh_output"], "ক্মন");
    let kemon = edit_json["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "কেমন")
        .expect("কেমন should be present through bounded FST edit search");
    assert_eq!(kemon["source"], "fst_edit_distance");

    let prefix = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "kem",
        "--max-distance",
        "0",
        "--max-prefix-candidates",
        "8",
        "--response-candidates",
        "8",
    ]);
    assert!(prefix.status.success(), "stderr: {}", stderr(&prefix));
    let prefix_json = json_stdout(&prefix);
    assert_eq!(prefix_json["obadh_output"], "কেম");
    let prefix_kemon = prefix_json["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "কেমন")
        .expect("কেমন should be present through FST prefix search");
    assert_eq!(prefix_kemon["source"], "fst_prefix_completion");
}

#[test]
fn autocorrect_cli_prefers_one_step_roman_repair_over_expensive_surface_edit() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-roman-repair");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "অকালপক্ক\t18\nঅকাল্পক্কো\t1\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "okalpokk",
        "--max-distance",
        "2",
        "--max-candidates",
        "64",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "অকাল্পক্ক");
    assert!(json["roman_repairs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|repair| repair["input"] == "okalopokk" && repair["output"] == "অকালপক্ক"));

    let first = &json["candidates"].as_array().unwrap()[0];
    assert_eq!(first["text"], "অকালপক্ক");
    assert_eq!(first["source"], "fst_roman_repair_exact");
    assert_eq!(first["roman_repair"], "okalopokk");
    assert_eq!(first["roman_repair_kind"], "inserted_separator_o");
    assert_eq!(first["roman_repair_cost"], 1);
}

#[test]
fn autocorrect_cli_surfaces_split_aspirated_roman_repair() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-aspirated-split-roman-repair");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "ভু\t125\nবহু\t24144\nভূ\t3348\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "bhu",
        "--max-distance",
        "2",
        "--max-candidates",
        "64",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "ভু");
    assert!(json["roman_repairs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|repair| repair["input"] == "bohu" && repair["output"] == "বহু"));

    let candidates = json["candidates"].as_array().unwrap();
    let exact = candidates
        .iter()
        .find(|candidate| candidate["text"] == "ভু")
        .expect("deterministic exact candidate should remain visible");
    assert_eq!(exact["source"], "fst_exact");

    let target = candidates
        .iter()
        .find(|candidate| candidate["text"] == "বহু")
        .expect("বহু should be surfaced through aspirated split repair");
    assert_eq!(target["source"], "fst_roman_repair_exact");
    assert_eq!(target["roman_repair"], "bohu");
    assert_eq!(target["roman_repair_kind"], "split_aspirated_consonant");
    assert_eq!(target["roman_repair_cost"], 1);
}

#[test]
fn autocorrect_cli_promotes_lowercase_r_to_flap_repair_with_mark_variant() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-flap-mark-roman-repair");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "হারিয়ে\t27441\nদাড়িয়ে\t203\nদাঁড়িয়ে\t18053\n")
        .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "dariye",
        "--max-distance",
        "1",
        "--max-candidates",
        "64",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "দারিয়ে");
    assert!(json["roman_repairs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|repair| repair["input"] == "daRiye" && repair["output"] == "দাড়িয়ে"));

    let first = &json["candidates"].as_array().unwrap()[0];
    assert_eq!(first["text"], "দাঁড়িয়ে");
    assert_eq!(first["source"], "fst_roman_repair_exact");
    assert_eq!(first["roman_repair"], "daRiye");
    assert_eq!(first["roman_repair_kind"], "lowercase_r_to_flap");
    assert_eq!(first["roman_repair_cost"], 1);
}

#[test]
fn autocorrect_cli_surfaces_stem_suffix_completions() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-suffix");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "নদী\t100\nনদীটি\t30\nনদীকথা\t200\n")
        .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "nodI",
        "--max-distance",
        "0",
        "--max-prefix-candidates",
        "8",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "নদী");
    let suffix = json["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "নদীটি")
        .expect("নদীটি should be present through stem suffix completion");
    assert_eq!(suffix["source"], "fst_stem_suffix_completion");

    let compound = json["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "নদীকথা")
        .expect("ordinary lexical prefix completion should remain present");
    assert_eq!(compound["source"], "fst_prefix_completion");
}

#[test]
fn autocorrect_cli_surfaces_missing_chandrabindu_as_diacritic_edit() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-diacritic");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "চাদ\t301\nচাদর\t443\nচাদের\t136\nচাঁদ\t2294\n")
        .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "cad",
        "--max-distance",
        "2",
        "--max-prefix-candidates",
        "8",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "চাদ");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "চাদ");
    assert_eq!(candidates[0]["source"], "fst_exact");

    let chandrabindu = candidates
        .iter()
        .find(|candidate| candidate["text"] == "চাঁদ")
        .expect("চাঁদ should be present through diacritic edit");
    assert_eq!(chandrabindu["source"], "fst_diacritic_edit");

    let suffix = candidates
        .iter()
        .find(|candidate| candidate["text"] == "চাদর")
        .expect("suffix candidate should remain visible");
    assert!(
        chandrabindu["score"].as_i64().unwrap() > suffix["score"].as_i64().unwrap(),
        "missing-mark candidate should beat suffix completion noise"
    );
}

#[test]
fn autocorrect_cli_promotes_second_pass_roman_separator_repair() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-second-pass-roman-repair");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "মান\t10116\nমন\t3852\nমনন\t142\n")
        .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "mnn",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "ম্নন");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "মনন");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "monon");
    assert_eq!(candidates[0]["roman_repair_kind"], "inserted_separator_o");
    assert_eq!(candidates[0]["roman_repair_cost"], 2);
}

#[test]
fn autocorrect_cli_promotes_repeated_cluster_separator_repair() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-repeated-separator-roman-repair");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(
        &lexicon_tsv,
        "খণ্ড\t3640\nখনি\t1660\nখনন\t2286\nজন্য\t338242\n",
    )
    .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "khnn",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "খন্ন");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "খনন");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "khnon");
    assert_eq!(candidates[0]["roman_repair_kind"], "inserted_separator_o");
    assert_eq!(candidates[0]["roman_repair_cost"], 1);
}

#[test]
fn autocorrect_cli_promotes_palatal_nasal_ja_roman_repair() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-palatal-nasal-roman-repair");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "জিরা\t288\nজিঞ্জিরা\t61\nজিঙ্গিরা\t1\n")
        .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "jingira",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "জিঙিরা");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "জিঞ্জিরা");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "jinjira");
    assert_eq!(
        candidates[0]["roman_repair_kind"],
        "palatal_nasal_ja_from_ng"
    );
    assert_eq!(candidates[0]["roman_repair_cost"], 2);
}

#[test]
fn autocorrect_cli_promotes_nz_to_deterministic_palatal_nasal_ja() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-nz-palatal-nasal");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "জিঞ্জিরা\t61\nজিনজিরা\t29\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "jinzira",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "জিনযিরা");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "জিঞ্জিরা");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "jinjira");
    assert_eq!(
        candidates[0]["roman_repair_kind"],
        "palatal_nasal_ja_from_nz"
    );
    assert_eq!(candidates[0]["roman_repair_cost"], 2);
}

#[test]
fn autocorrect_cli_promotes_nz_to_corpus_strong_nj_word() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-nz-panjabi");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "পাঞ্জাবি\t1890\nপানজাবি\t1\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "panzabi",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "পানযাবি");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "পাঞ্জাবি");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "panjabi");
    assert_eq!(
        candidates[0]["roman_repair_kind"],
        "palatal_nasal_ja_from_nz"
    );
}

#[test]
fn autocorrect_cli_promotes_strong_velar_ng_over_low_frequency_exact_artifact() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-ng-velar-exact-noise");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "জঙি\t1\nজঙ্গি\t938\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "jongi",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "জঙি");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "জঙ্গি");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "jonggi");
    assert_eq!(candidates[0]["roman_repair_kind"], "velar_nasal_from_ng");
}

#[test]
fn autocorrect_cli_prefers_corpus_strong_ng_candidate_over_rare_velar_variant() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-ng-rongin");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(
        &lexicon_tsv,
        "রঙিন\t1763\nরঙ্গিন\t177\nরঙীন\t49\nরঙ্গীন\t51\n",
    )
    .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "rongin",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "রঙিন");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "রঙিন");
    // রঙিন is now the baseline itself, so it wins as an exact corpus hit rather
    // than via a velar-nasal repair — and still outranks the rare velar variant.
    assert_eq!(candidates[0]["source"], "fst_exact");
    let rongin_rank = candidate_rank(candidates, "রঙিন").expect("রঙিন should be returned");
    let ronggin_rank = candidate_rank(candidates, "রঙ্গিন").expect("রঙ্গিন should be returned");
    assert!(
        rongin_rank < ronggin_rank,
        "the strong corpus রঙিন should outrank the rare velar variant রঙ্গিন"
    );
}

#[test]
fn autocorrect_cli_promotes_high_frequency_velar_ng_with_long_i_variant() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-ng-songit");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(
        &lexicon_tsv,
        "সঙ্গীত\t16122\nসংগীত\t6166\nসঙ্গিত\t49\nসংগিত\t26\nসঞ্জিত\t24\n",
    )
    .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "songit",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "সঙিত");

    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "সঙ্গীত");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "songgIt");
    assert_eq!(candidates[0]["roman_repair_kind"], "velar_nasal_from_ng");

    let songit_rank = candidate_rank(candidates, "সঙ্গীত").expect("সঙ্গীত should be returned");
    let sonjit_rank = candidate_rank(candidates, "সঞ্জিত").expect("সঞ্জিত should be returned");
    assert!(
        songit_rank < sonjit_rank,
        "velar ng repair should outrank the competing nj repair when the corpus strongly supports it"
    );
}

#[test]
fn autocorrect_cli_keeps_palatal_nasal_repair_lower_trust_than_stronger_edit() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-fst-palatal-nasal-lower-trust");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");

    fs::write(&lexicon_tsv, "সংগীত\t6166\nসঞ্জিত\t24\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "songit",
        "--max-distance",
        "2",
        "--response-candidates",
        "8",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "সঙিত");

    let candidates = json["candidates"].as_array().unwrap();
    let songit_rank = candidate_rank(candidates, "সংগীত").expect("সংগীত should be returned");
    let sonjit_rank = candidate_rank(candidates, "সঞ্জিত").expect("সঞ্জিত should be returned");

    assert!(
        songit_rank < sonjit_rank,
        "lower-trust ng->nj repair should not outrank the stronger সংগীত candidate"
    );
}

#[test]
fn autocorrect_cli_builds_lexicon_from_multiple_sources() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-merge-lexicons");
    let primary = workspace.path("primary.tsv");
    let secondary = workspace.path("secondary.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let pairs = workspace.path("pairs.tsv");
    let output = workspace.path("candidates.jsonl");

    fs::write(&primary, "আমি\t100\nবিজ্ঞান\t70\n").expect("primary lexicon fixture should write");
    fs::write(&secondary, "আমি\t50\nকিরণ\t40\n").expect("secondary lexicon fixture should write");
    fs::write(&pairs, "আমী\tআমি\n").expect("pair fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&primary),
        "--input",
        path_str(&secondary),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));
    let build_json = json_stdout(&build);
    assert_eq!(build_json["inputs"].as_array().unwrap().len(), 2);
    assert_eq!(build_json["input_rows"], 4);
    assert_eq!(build_json["duplicate_rows"], 1);
    assert_eq!(build_json["entries"], 3);
    assert_eq!(build_json["total_frequency"], 260);
    assert_eq!(build_json["max_frequency"], 150);

    let export = run_obadh_autocorrect([
        "export-candidates",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&pairs),
        "--output",
        path_str(&output),
        "--input-kind",
        "bangla",
    ]);
    assert!(export.status.success(), "stderr: {}", stderr(&export));

    let jsonl = fs::read_to_string(&output).expect("candidate export should read");
    let record = serde_json::from_str::<Value>(jsonl.trim()).expect("JSONL line should parse");
    let merged = record["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "আমি")
        .expect("merged target candidate should be present");
    assert_eq!(merged["frequency"], 150);
}

#[test]
fn autocorrect_cli_exports_compact_lexicon_to_mergeable_tsv() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-export-lexicon");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let exported = workspace.path("exported.tsv");

    fs::write(&lexicon_tsv, "আমি\t100\nবিজ্ঞান\t70\n").expect("lexicon fixture should write");
    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let export = run_obadh_autocorrect([
        "export-lexicon",
        "--input",
        path_str(&artifact),
        "--output",
        path_str(&exported),
    ]);
    assert!(export.status.success(), "stderr: {}", stderr(&export));
    let export_json = json_stdout(&export);
    assert_eq!(export_json["entries"], 2);
    assert_eq!(export_json["total_frequency"], 170);
    assert_eq!(export_json["max_frequency"], 100);

    let exported_tsv = fs::read_to_string(&exported).expect("exported TSV should read");
    assert!(exported_tsv.contains("আমি\t100"));
    assert!(exported_tsv.contains("বিজ্ঞান\t70"));

    let audit = run_obadh_autocorrect(["audit-lexicon", "--input", path_str(&exported)]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);
    assert_eq!(audit_json["accepted_rows"], 2);
    assert_eq!(audit_json["non_bangla_rows"], 0);
}

#[test]
fn autocorrect_cli_rejects_non_bangla_lexicon_words_by_default() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-rejects-pollution");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    fs::write(&lexicon_tsv, "আমি\t100\nhello\t1\n").expect("lexicon fixture should write");

    let output = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);

    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("non-Bangla lexicon word"),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn autocorrect_cli_normalizes_bangla_lexicon_tsv_words() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-normalizes-lexicon");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    fs::write(&lexicon_tsv, "বড়\t3\nবড\u{09BC}\t4\n").expect("lexicon fixture should write");

    let audit = run_obadh_autocorrect(["audit-lexicon", "--input", path_str(&lexicon_tsv)]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);
    assert_eq!(audit_json["rows"], 2);
    assert_eq!(audit_json["accepted_rows"], 2);
    assert_eq!(audit_json["unique_words"], 1);
    assert_eq!(audit_json["duplicate_rows"], 1);

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));
    let build_json = json_stdout(&build);
    assert_eq!(build_json["input_rows"], 2);
    assert_eq!(build_json["duplicate_rows"], 1);
    assert_eq!(build_json["entries"], 1);
    assert_eq!(build_json["total_frequency"], 7);
}

#[test]
fn autocorrect_cli_extracts_and_audits_clean_bangla_word_frequencies() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-audit");
    let corpus = workspace.path("corpus.txt");
    let extracted = workspace.path("extracted.tsv");
    fs::write(
        &corpus,
        "আমি আমি তুমি। hello ১২৩। র‌্যাব র‌্যাব ্ ামি ক্ । কিরণ বড় বড\u{09BC}",
    )
    .expect("corpus fixture should write");

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&corpus),
        "--output",
        path_str(&extracted),
        "--min-frequency",
        "2",
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["text_inputs"], 1);
    assert_eq!(extract_json["html_inputs"], 0);
    assert_eq!(extract_json["epub_inputs"], 0);
    assert_eq!(extract_json["epub_spine_items"], 0);
    assert_eq!(extract_json["epub_fallback_inputs"], 0);
    assert_eq!(extract_json["epub_fallback_items"], 0);
    assert_eq!(extract_json["token_count"], 8);
    assert_eq!(extract_json["unique_words"], 5);
    assert_eq!(extract_json["emitted_entries"], 3);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("আমি\t2"));
    assert!(extracted_tsv.contains("র‌্যাব\t2"));
    assert!(extracted_tsv.contains("বড়\t2"));
    assert_eq!(
        extracted_tsv
            .lines()
            .filter(|line| line.starts_with("বড়\t"))
            .count(),
        1
    );
    assert!(!extracted_tsv.contains("১২৩"));
    assert!(!extracted_tsv.contains("hello"));

    let audit = run_obadh_autocorrect(["audit-lexicon", "--input", path_str(&extracted)]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);
    assert_eq!(audit_json["rows"], 3);
    assert_eq!(audit_json["accepted_rows"], 3);
    assert_eq!(audit_json["unique_words"], 3);
    assert_eq!(audit_json["non_bangla_rows"], 0);
}

#[test]
fn autocorrect_cli_extracts_bangla_words_from_epub_body_text() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-epub");
    let epub = workspace.path("book.epub");
    let extracted = workspace.path("extracted.tsv");
    write_minimal_epub(
        &epub,
        r#"<?xml version="1.0" encoding="utf-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml">
          <body>
            <p>আমি তুমি তুমি hello &amp; &#x995;&#x9BF;&#x9B0;&#x9A3;</p>
            <span title="রাখবে না">আমি</span>
          </body>
        </html>"#,
    );

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&epub),
        "--output",
        path_str(&extracted),
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["text_inputs"], 0);
    assert_eq!(extract_json["html_inputs"], 0);
    assert_eq!(extract_json["epub_inputs"], 1);
    assert_eq!(extract_json["epub_spine_items"], 0);
    assert_eq!(extract_json["epub_fallback_inputs"], 1);
    assert_eq!(extract_json["epub_fallback_items"], 1);
    assert_eq!(extract_json["token_count"], 5);
    assert_eq!(extract_json["unique_words"], 3);
    assert_eq!(extract_json["emitted_entries"], 3);
    assert!(extract_json["corpus_bytes"].as_u64().unwrap() > 0);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("আমি\t2"));
    assert!(extracted_tsv.contains("তুমি\t2"));
    assert!(extracted_tsv.contains("কিরণ\t1"));
    assert!(!extracted_tsv.contains("রাখবে"));
    assert!(!extracted_tsv.contains("hello"));
}

#[test]
fn autocorrect_cli_extracts_standalone_html_without_markup_noise() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-html");
    let html = workspace.path("chapter.xhtml");
    let extracted = workspace.path("extracted.tsv");
    fs::write(
        &html,
        r#"<html>
          <head>
            <style>.hidden::after { content: "দূষণ"; }</style>
            <script>const ignored = "দূষণ";</script>
          </head>
          <body>
            <p>আমি বই</p>
            <span title="রাখবে না">নদী</span>
            <p>&#x995;&#x9BF;&#x9B0;&#x9A3;</p>
          </body>
        </html>"#,
    )
    .expect("HTML corpus should write");

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&html),
        "--output",
        path_str(&extracted),
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["text_inputs"], 0);
    assert_eq!(extract_json["html_inputs"], 1);
    assert_eq!(extract_json["epub_inputs"], 0);
    assert_eq!(extract_json["token_count"], 4);
    assert_eq!(extract_json["unique_words"], 4);
    assert_eq!(extract_json["emitted_entries"], 4);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("আমি\t1"));
    assert!(extracted_tsv.contains("বই\t1"));
    assert!(extracted_tsv.contains("নদী\t1"));
    assert!(extracted_tsv.contains("কিরণ\t1"));
    assert!(!extracted_tsv.contains("রাখবে"));
    assert!(!extracted_tsv.contains("দূষণ"));
}

#[test]
fn autocorrect_cli_prefers_epub_spine_over_nav_and_unreferenced_xhtml() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-epub-spine");
    let epub = workspace.path("book.epub");
    let extracted = workspace.path("extracted.tsv");
    write_epub(
        &epub,
        &[
            (
                "META-INF/container.xml",
                r#"<?xml version="1.0"?>
                <container>
                  <rootfiles>
                    <rootfile full-path="OEBPS/content.opf"/>
                  </rootfiles>
                </container>"#,
            ),
            (
                "OEBPS/content.opf",
                r#"<package>
                  <manifest>
                    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
                    <item id="appendix" href="appendix.xhtml" media-type="application/xhtml+xml"/>
                  </manifest>
                  <spine>
                    <itemref idref="nav"/>
                    <itemref idref="chapter"/>
                    <itemref idref="appendix" linear="no"/>
                  </spine>
                </package>"#,
            ),
            ("OEBPS/nav.xhtml", "<html><body>সূচি দূষণ</body></html>"),
            ("OEBPS/chapter.xhtml", "<html><body>মূল মূল বই</body></html>"),
            (
                "OEBPS/appendix.xhtml",
                "<html><body>পরিশিষ্ট দূষণ</body></html>",
            ),
            (
                "OEBPS/unreferenced.xhtml",
                "<html><body>অতল দূষণ</body></html>",
            ),
        ],
    );

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&epub),
        "--output",
        path_str(&extracted),
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["text_inputs"], 0);
    assert_eq!(extract_json["html_inputs"], 0);
    assert_eq!(extract_json["epub_inputs"], 1);
    assert_eq!(extract_json["epub_spine_items"], 1);
    assert_eq!(extract_json["epub_fallback_inputs"], 0);
    assert_eq!(extract_json["epub_fallback_items"], 0);
    assert_eq!(extract_json["token_count"], 3);
    assert_eq!(extract_json["unique_words"], 2);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("মূল\t2"));
    assert!(extracted_tsv.contains("বই\t1"));
    assert!(!extracted_tsv.contains("সূচি"));
    assert!(!extracted_tsv.contains("পরিশিষ্ট"));
    assert!(!extracted_tsv.contains("অতল"));
    assert!(!extracted_tsv.contains("দূষণ"));
}

#[test]
fn autocorrect_cli_extracts_corpus_directories_deterministically() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-dir");
    let corpus_dir = workspace.path("corpus");
    let nested_dir = corpus_dir.join("nested");
    fs::create_dir_all(&nested_dir).expect("corpus directories should create");
    let extracted = workspace.path("extracted.tsv");
    fs::write(corpus_dir.join("b.txt"), "আমি বই").expect("text corpus should write");
    fs::write(nested_dir.join("a.md"), "আমি নদী").expect("markdown corpus should write");
    fs::write(
        nested_dir.join("d.xhtml"),
        r#"<html><body><p>আমি</p><span title="দূষণ">বই</span></body></html>"#,
    )
    .expect("HTML corpus should write");
    fs::write(corpus_dir.join("ignore.css"), "দূষণ দূষণ").expect("ignored corpus should write");
    write_minimal_epub(
        &nested_dir.join("c.epub"),
        "<html><body>বই নদী নদী</body></html>",
    );

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&corpus_dir),
        "--output",
        path_str(&extracted),
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["inputs"].as_array().unwrap().len(), 1);
    assert_eq!(extract_json["expanded_inputs"], 4);
    assert_eq!(extract_json["text_inputs"], 2);
    assert_eq!(extract_json["html_inputs"], 1);
    assert_eq!(extract_json["json_inputs"], 0);
    assert_eq!(extract_json["epub_inputs"], 1);
    assert_eq!(extract_json["epub_fallback_inputs"], 1);
    assert_eq!(extract_json["token_count"], 9);
    assert_eq!(extract_json["unique_words"], 3);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("নদী\t3"));
    assert!(extracted_tsv.contains("আমি\t3"));
    assert!(extracted_tsv.contains("বই\t3"));
    assert!(!extracted_tsv.contains("দূষণ"));
}

#[test]
fn autocorrect_cli_extracts_json_prose_and_rejects_assamese_only_letters() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-extract-json");
    let corpus = workspace.path("article.json");
    let extracted = workspace.path("extracted.tsv");
    fs::write(
        &corpus,
        r#"{"title":"বাংলা ভাষা","url":"https://example.test/দূষণ","content":"আমি বাংলা লিখি। প্ৰথম ৱেবছাইট"}"#,
    )
    .expect("json corpus should write");

    let extract = run_obadh_autocorrect([
        "extract-lexicon",
        "--input",
        path_str(&corpus),
        "--output",
        path_str(&extracted),
    ]);
    assert!(extract.status.success(), "stderr: {}", stderr(&extract));
    let extract_json = json_stdout(&extract);
    assert_eq!(extract_json["expanded_inputs"], 1);
    assert_eq!(extract_json["json_inputs"], 1);
    assert_eq!(extract_json["token_count"], 5);
    assert_eq!(extract_json["unique_words"], 4);

    let extracted_tsv = fs::read_to_string(&extracted).expect("extracted TSV should read");
    assert!(extracted_tsv.contains("বাংলা\t2"));
    assert!(extracted_tsv.contains("ভাষা\t1"));
    assert!(extracted_tsv.contains("লিখি\t1"));
    assert!(!extracted_tsv.contains("দূষণ"));
    assert!(!extracted_tsv.contains("প্ৰথম"));
    assert!(!extracted_tsv.contains("ৱেবছাইট"));
}

#[test]
fn autocorrect_cli_prepares_local_corpus_lexicon_in_one_pass() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-prepare-lexicon");
    let corpus_dir = workspace.path("corpus");
    fs::create_dir_all(&corpus_dir).expect("corpus directory should create");
    fs::write(corpus_dir.join("a.txt"), "আমি বই").expect("text corpus should write");
    write_minimal_epub(
        &corpus_dir.join("b.epub"),
        "<html><body>বই নদী নদী</body></html>",
    );
    let words = workspace.path("prepared/words.tsv");
    let artifact = workspace.path("prepared/obadh.bn.lex");

    let prepare = run_obadh_autocorrect([
        "prepare-lexicon",
        "--input",
        path_str(&corpus_dir),
        "--words-output",
        path_str(&words),
        "--lexicon-output",
        path_str(&artifact),
        "--min-frequency",
        "2",
        "--pretty",
    ]);
    assert!(prepare.status.success(), "stderr: {}", stderr(&prepare));
    let prepare_json = json_stdout(&prepare);
    assert_eq!(
        prepare_json["extract"]["inputs"].as_array().unwrap().len(),
        1
    );
    assert_eq!(prepare_json["extract"]["expanded_inputs"], 2);
    assert_eq!(prepare_json["extract"]["text_inputs"], 1);
    assert_eq!(prepare_json["extract"]["json_inputs"], 0);
    assert_eq!(prepare_json["extract"]["epub_inputs"], 1);
    assert_eq!(prepare_json["extract"]["token_count"], 5);
    assert_eq!(prepare_json["extract"]["unique_words"], 3);
    assert_eq!(prepare_json["extract"]["emitted_entries"], 2);
    assert_eq!(prepare_json["audit"]["accepted_rows"], 2);
    assert_eq!(prepare_json["audit"]["non_bangla_rows"], 0);
    assert_eq!(prepare_json["lexicon"]["entries"], 2);
    assert!(words.exists());
    assert!(artifact.exists());

    let prepared_tsv = fs::read_to_string(&words).expect("prepared TSV should read");
    assert!(prepared_tsv.contains("বই\t2"));
    assert!(prepared_tsv.contains("নদী\t2"));
    assert!(!prepared_tsv.contains("আমি\t1"));

    let inspect = run_obadh_autocorrect([
        "inspect-lexicon",
        "--input",
        path_str(&artifact),
        "--pretty",
    ]);
    assert!(inspect.status.success(), "stderr: {}", stderr(&inspect));
    let inspect_json = json_stdout(&inspect);
    assert_eq!(inspect_json["entries"], 2);
}

#[test]
fn autocorrect_cli_merges_lexicon_sources_with_explicit_invalid_drops() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-merge-clean");
    let primary = workspace.path("primary.tsv");
    let secondary = workspace.path("secondary.tsv");
    let merged = workspace.path("merged.tsv");

    fs::write(&primary, "আমি\t100\nhello\t10\nবিজ্ঞান\tbad\n")
        .expect("primary lexicon fixture should write");
    fs::write(&secondary, "আমি\t50\nকিরণ\t40\n").expect("secondary lexicon fixture should write");

    let strict = run_obadh_autocorrect([
        "merge-lexicon",
        "--input",
        path_str(&primary),
        "--input",
        path_str(&secondary),
        "--output",
        path_str(&merged),
    ]);
    assert!(!strict.status.success());
    assert!(
        stderr(&strict).contains("non-Bangla lexicon word"),
        "stderr: {}",
        stderr(&strict)
    );

    let merge = run_obadh_autocorrect([
        "merge-lexicon",
        "--input",
        path_str(&primary),
        "--input",
        path_str(&secondary),
        "--output",
        path_str(&merged),
        "--drop-invalid",
    ]);
    assert!(merge.status.success(), "stderr: {}", stderr(&merge));
    let merge_json = json_stdout(&merge);
    assert_eq!(merge_json["rows"], 5);
    assert_eq!(merge_json["accepted_rows"], 3);
    assert_eq!(merge_json["duplicate_rows"], 1);
    assert_eq!(merge_json["dropped_rows"], 2);
    assert_eq!(merge_json["non_bangla_rows"], 1);
    assert_eq!(merge_json["invalid_frequency_rows"], 1);
    assert_eq!(merge_json["unique_words"], 2);
    assert_eq!(merge_json["emitted_entries"], 2);
    assert_eq!(merge_json["total_frequency"], 190);

    let merged_tsv = fs::read_to_string(&merged).expect("merged TSV should read");
    assert!(merged_tsv.contains("আমি\t150"));
    assert!(merged_tsv.contains("কিরণ\t40"));
    assert!(!merged_tsv.contains("hello"));
    assert!(!merged_tsv.contains("বিজ্ঞান"));
}

#[test]
fn autocorrect_cli_audits_bangla_pair_quality() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-audit-bangla-pairs");
    let pairs = workspace.path("bangla-pairs.tsv");
    fs::write(
        &pairs,
        "আমী\tআমি\nআমী\tআমি\nআমি\tআমি\nhello\tআমি\nআমী\thello\nআমী\tআমি\textra\n",
    )
    .expect("pair fixture should write");

    let audit = run_obadh_autocorrect([
        "audit-pairs",
        "--input",
        path_str(&pairs),
        "--input-kind",
        "bangla",
        "--pretty",
    ]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);

    assert_eq!(audit_json["rows"], 6);
    assert_eq!(audit_json["accepted_rows"], 3);
    assert_eq!(audit_json["unique_pairs"], 2);
    assert_eq!(audit_json["duplicate_rows"], 1);
    assert_eq!(audit_json["identity_rows"], 1);
    assert_eq!(audit_json["baseline_exact_rows"], 1);
    assert_eq!(audit_json["non_bangla_source_rows"], 1);
    assert_eq!(audit_json["non_bangla_expected_rows"], 1);
    assert_eq!(audit_json["malformed_rows"], 1);
    assert!(
        audit_json["baseline_mean_edit_cost"].as_f64().unwrap() > 0.0,
        "expected typo rows to produce a non-zero mean edit cost: {audit_json}"
    );
}

#[test]
fn autocorrect_cli_audits_clean_roman_pair_quality() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-audit-roman-pairs");
    let pairs = workspace.path("roman-pairs.tsv");
    fs::write(&pairs, "biggan\tবিজ্ঞান\nami\tআমি\nআমি\tআমি\nami!\tআমি\n")
        .expect("pair fixture should write");

    let audit = run_obadh_autocorrect([
        "audit-pairs",
        "--input",
        path_str(&pairs),
        "--input-kind",
        "roman",
    ]);
    assert!(audit.status.success(), "stderr: {}", stderr(&audit));
    let audit_json = json_stdout(&audit);

    assert_eq!(audit_json["rows"], 4);
    assert_eq!(audit_json["accepted_rows"], 2);
    assert_eq!(audit_json["unique_pairs"], 2);
    assert_eq!(audit_json["baseline_exact_rows"], 2);
    assert_eq!(audit_json["non_roman_source_rows"], 2);
    assert_eq!(audit_json["non_bangla_expected_rows"], 0);
    assert_eq!(audit_json["baseline_exact_rate"], 1.0);
}

#[test]
fn autocorrect_cli_suggests_runtime_candidates_for_roman_input() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-suggest");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");

    fs::write(
        &lexicon_tsv,
        "কেমন\t100\nকোন\t80\nকেন\t60\nএখন\t40\nতোমার\t30\nকোথায়\t20\nযেমন\t10\nযাবো\t9\nকরবো\t8\n",
    )
    .expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "kmn",
        "--max-candidates",
        "64",
        "--max-skeleton-candidates",
        "64",
        "--response-candidates",
        "64",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));
    let suggest_json = json_stdout(&suggest);

    assert_eq!(suggest_json["input"], "kmn");
    assert_eq!(suggest_json["obadh_output"], "ক্মন");
    assert!(suggest_json["candidate_count"].as_u64().unwrap() >= 2);
    assert_eq!(
        suggest_json["returned_candidates"].as_u64().unwrap(),
        suggest_json["candidates"].as_array().unwrap().len() as u64
    );
    let candidate = suggest_json["candidates"]
        .as_array()
        .unwrap()
        .first()
        .expect("suggestion response should include at least one candidate");
    assert!(candidate["text"].as_str().is_some());
    assert!(candidate["source"].as_str().is_some());
    assert_eq!(candidate["features"].as_array().unwrap().len(), 9);

    let suggest = run_obadh_autocorrect([
        "suggest",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "kem",
        "--max-candidates",
        "64",
        "--max-prefix-candidates",
        "8",
        "--max-skeleton-candidates",
        "0",
        "--response-candidates",
        "64",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));
    let suggest_json = json_stdout(&suggest);
    assert_eq!(suggest_json["obadh_output"], "কেম");
    let kemon = suggest_json["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["text"] == "কেমন")
        .expect("কেমন should be present as a prefix completion");
    assert_eq!(kemon["source"], "prefix_completion");
    assert_eq!(kemon["features"][0], 4);
    assert_eq!(kemon["edit_cost"], 0);
}

#[test]
fn autocorrect_cli_exports_labeled_candidate_jsonl() {
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-export-candidates");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.lex");
    let pairs = workspace.path("pairs.tsv");
    let output = workspace.path("candidates.jsonl");

    fs::write(&lexicon_tsv, "আমি\t1000\nবিজ্ঞান\t900\n").expect("lexicon fixture should write");
    fs::write(&pairs, "আমী\tআমি\nবিজ্ঞান\tবিজ্ঞান\n").expect("pair fixture should write");

    let build = run_obadh_autocorrect([
        "build-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let export = run_obadh_autocorrect([
        "export-candidates",
        "--lexicon",
        path_str(&artifact),
        "--input",
        path_str(&pairs),
        "--output",
        path_str(&output),
        "--input-kind",
        "bangla",
        "--max-candidates",
        "1",
    ]);
    assert!(export.status.success(), "stderr: {}", stderr(&export));
    let export_json = json_stdout(&export);
    assert_eq!(export_json["rows"], 2);
    assert_eq!(export_json["target_in_lexicon_rows"], 2);
    assert_eq!(export_json["target_present_rows"], 2);
    assert!(export_json["candidate_rows"].as_u64().unwrap() >= 2);

    let jsonl = fs::read_to_string(&output).expect("candidate export should read");
    let records = jsonl
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("JSONL line should parse"))
        .collect::<Vec<_>>();

    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["candidates"].as_array().unwrap().len(), 1);
    assert_eq!(records[0]["baseline"], "আমী");
    assert_eq!(records[0]["baseline_exact"], false);
    assert_eq!(records[0]["target_in_lexicon"], true);
    assert!(records[0]["target_rank"].as_u64().unwrap() >= 1);
    let labeled = records[0]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["label"] == true)
        .expect("expected target candidate should be labeled");
    assert_eq!(labeled["text"], "আমি");
    assert_eq!(labeled["source"], "lexicon_edit");
    assert_eq!(labeled["features"].as_array().unwrap().len(), 9);

    assert_eq!(records[1]["baseline"], "বিজ্ঞান");
    assert_eq!(records[1]["baseline_exact"], true);
}

#[test]
fn autocorrect_cli_key_slip_recovers_adjacent_finger_error() {
    // "desj" is a QWERTY fat-finger for "desh" (j is the key next to h). Its baseline is a
    // non-word, so the key-slip channel rewrites j→h, transliterates "desh" → দেশ, and
    // promotes the real word.
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-key-slip-recover");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");
    fs::write(&lexicon_tsv, "দেশ\t55873\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "desj",
        "--max-distance",
        "2",
        "--response-candidates",
        "5",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "দেশ");
    assert_eq!(candidates[0]["source"], "fst_roman_repair_exact");
    assert_eq!(candidates[0]["roman_repair"], "desh");
    assert_eq!(candidates[0]["roman_repair_kind"], "qwerty_key_slip");
}

#[test]
fn autocorrect_cli_key_slip_leaves_correctly_typed_word_untouched() {
    // A correctly-typed word transliterates to a lexicon word, so the key-slip gate stays
    // shut: the exact word stays #1 and no qwerty_key_slip repair is even attempted.
    let workspace = TempWorkspace::new("obadh-autocorrect-cli-key-slip-precision");
    let lexicon_tsv = workspace.path("lexicon.tsv");
    let artifact = workspace.path("obadh.bn.fst");
    fs::write(&lexicon_tsv, "দেশ\t55873\nদেব\t9000\n").expect("lexicon fixture should write");

    let build = run_obadh_autocorrect([
        "build-fst-lexicon",
        "--input",
        path_str(&lexicon_tsv),
        "--output",
        path_str(&artifact),
    ]);
    assert!(build.status.success(), "stderr: {}", stderr(&build));

    let suggest = run_obadh_autocorrect([
        "suggest-fst",
        "--lexicon",
        path_str(&artifact),
        "--input",
        "desh",
        "--max-distance",
        "2",
        "--response-candidates",
        "5",
    ]);
    assert!(suggest.status.success(), "stderr: {}", stderr(&suggest));

    let json = json_stdout(&suggest);
    assert_eq!(json["obadh_output"], "দেশ");
    let candidates = json["candidates"].as_array().unwrap();
    assert_eq!(candidates[0]["text"], "দেশ");
    assert_eq!(candidates[0]["source"], "fst_exact");
    assert!(
        json["roman_repairs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|repair| repair["kind"] != "qwerty_key_slip"),
        "no key-slip repair should be attempted for a correctly-typed word"
    );
}

struct TempWorkspace {
    root: PathBuf,
}

impl TempWorkspace {
    fn new(label: &str) -> Self {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{label}-{}-{suffix}", std::process::id()));
        fs::create_dir_all(&root).expect("temp workspace should be created");
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn run_obadh_autocorrect<const N: usize>(args: [&str; N]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_obadh-autocorrect"))
        .args(args)
        .output()
        .expect("obadh-autocorrect should run")
}

fn write_minimal_epub(path: &Path, chapter: &str) {
    write_epub(path, &[("OEBPS/chapter.xhtml", chapter)]);
}

fn write_epub(path: &Path, members: &[(&str, &str)]) {
    let file = fs::File::create(path).expect("epub fixture should create");
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    zip.start_file("mimetype", options)
        .expect("epub mimetype member should start");
    zip.write_all(b"application/epub+zip")
        .expect("epub mimetype member should write");

    for (name, content) in members {
        zip.start_file(*name, options)
            .expect("epub member should start");
        zip.write_all(content.as_bytes())
            .expect("epub member should write");
    }

    zip.finish().expect("epub fixture should finish");
}

fn json_stdout(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("stdout should be JSON: {error}; stdout={}", stdout(output)))
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("test path should be valid UTF-8")
}

fn candidate_rank(candidates: &[Value], text: &str) -> Option<usize> {
    candidates
        .iter()
        .position(|candidate| candidate["text"] == text)
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
