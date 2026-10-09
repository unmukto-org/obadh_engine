# Changelog

All notable changes to `obadh_engine` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning (with the `0.x` caveat that the minor version carries breaking changes).

Releases before `0.7.0` predate this file; see the git history and tags for those.

## [Unreleased]

### Added

- `ok` and `okay` loanwords mapping to ওকে, with rebuilt lexicon artifacts.
- Exact ASCII emoticon suggestions across Rust, CLI, WASM, and the C ABI, using
  322 aliases from the pinned, MIT-licensed `wooorm/emoticon` dataset. Both
  noseless and nosed forms work (`:)` / `:-)` → 😃, `:D` / `:-D` → 😄).
  Compose returns the literal first and its emoji as an explicit alternative;
  the deterministic transliteration core is unchanged. Emoticon alternatives
  have the append-only candidate source code `11` (`emoticon_exact`).
- Playground Copy, Save, reversible Clear, text-size settings, and an optional
  Roman history view.

### Changed

- Simplified the playground around its editor with shared Obadh branding,
  locally hosted fonts, consistent light and dark themes, and responsive sizing.
- The candidate ribbon follows the macOS client: compact cells, wrapping arrow
  and Tab navigation, and Space or Enter to commit the selected candidate.
- Reworked the collapsible Inspector with expandable ranking and phonetic-unit
  details, timing bars, runtime resources, and JSON syntax highlighting.

### Fixed

- Playground suggestions for the previous input are hidden when the active
  input changes. Committing an emoji preserves it in the composer.
- Long inspector content scrolls within bounded panels; whitespace tokens have
  visible markers rather than blank conversion rows.

## [0.9.4] - 2026-09-25

### Changed

- The reph-ya loanword form র‍্য is now encoded with the Unicode-prescribed ZWJ
  (U+200D) instead of ZWNJ. The marker scheme is orthogonal: after a leading র,
  `Y` and `Z` are the ya-phola marker (`rY`/`rZ` → র‍্য, `rYab`/`rZab` → র‍্যাব),
  while lowercase `y` stays the য় glide and composes after the marker
  (`rZy` → র‍্যয়). This changes the shipped output of র‍্য words. The public API
  and the C ABI are unchanged, so downstream consumers upgrade without recompiling.
- The গ্ণ, ঙ্ক্ত and স্প্ল clusters are encoded as plain, joiner-free conjuncts.

### Fixed

- Productive ya-phola stacks on an r-phola conjunct tail, so loanword clusters
  render correctly (`TrYak` → ট্র্যাক, `eksTrYak` → এক্সট্র্যাক). The leading-র
  guard follows real conjunct segmentation, so `krY` stays ক্র্য.
- `t` + `l` renders a live ত (কাতলা, পাতলা, মতলব, বোতল) instead of খণ্ড-ত. খণ্ড-ত
  before a consonant stays reachable through the explicit khanda-ta signal.
- The autosuggest open-vocabulary validator no longer lets a joiner slip a
  hasant-before-kar sequence through.

### Added

- A curated English→Bangla loanword lexicon in the autocorrect dataset.

## [0.9.3] - 2026-09-01

### Removed

- Reverted the `gy` → জ্ঞ autocorrect route from 0.9.2. The জ্ঞ candidate was
  generated for `gyan` but ranked below the runtime's 8-candidate response cap, so
  the C ABI and WASM compose never surfaced it (it appeared only in the wider
  `suggest-fst` dev pool). The জ্ঞ cluster stays reachable by the common spelling
  (`biggan` → বিজ্ঞান). Surfacing it for the bare `gyan` spelling needs conjunct
  inventory completion plus an invalid-conjunct response filter, which is deferred.

## [0.9.2] - 2026-09-01

### Added

- Autocorrect routes a bare `ng` before `j` to the palatal nasal-ja cluster ঞ্জ, so a
  word spelled with the velar-nasal `ng` (`rongjon` for রঞ্জন) reaches ঞ্জ the way
  `nj` / `NGj` do deterministically. It is FST-filtered, so it only surfaces a real ঞ্জ
  word and never invents one.
- Autocorrect offers the জ্ঞ (jna) conjunct spelling for `gy` input, since জ্ঞ is
  commonly typed and pronounced "gy": `gyan` now surfaces জ্ঞান beside গ্যান. It is
  FST-filtered, so a `gy` that is really a ya-phola (`bhagyo` → ভাগ্য) keeps its
  deterministic গ্য first and the জ্ঞ candidate ranks far below the correct word.
- A `:` between two digits stays a literal separator instead of the bisarga, for clock
  times: `9:45` → ৯:৪৫, `12:34:56` → ১২:৩৪:৫৬. A `:` next to a letter or at a word edge
  keeps its bisarga meaning (`du:kho` → দুঃখ). The render stays idempotent on its own
  Bengali-digit output.

### Fixed

- A bare `ng` directly before a vowel now renders the velar nasal ঙ with the vowel as a
  kar, matching `Ng` + vowel: `bhanga` → ভাঙা, `bangali` → বাঙালি, `rongin` → রঙিন. An
  anusvar cannot carry a vowel, so the baseline previously emitted a loose independent
  vowel (`nga` → ংআ). Anusvar before a consonant or at a word end (`bangla` → বাংলা,
  `rong` → রং) and the `ngg` / `nggh` velar conjuncts are unchanged.
- A stray `Z` now folds to য instead of leaking a literal Latin glyph (`namaZ` →
  নামায). `Z` stays reserved only inside the narrow `rZy` non-conjunct ra-ya marker
  (`rZy` → র‌্য).

## [0.9.1]

### Fixed

- Autocorrect no longer offers unrelated short lexicon entries as "corrections" for a token
  with no Bangla letter. Punctuation, digits, and bare symbols (`,` → ও/এ/অং, a lone `1`, `()`,
  `^`) fall within edit distance of the shortest, most frequent words, so the edit-distance
  channel was surfacing them as suggestions behind the correct baseline. It now runs only on a
  word-like baseline, so a non-word token yields just its deterministic baseline through
  `obadh_compose_suggestions` / `obadh_autocorrect_suggest_detailed`. Reported from obadh-macos
  ([#34](https://github.com/nsssayom/obadh_engine/issues/34)). No ABI change; classification of a
  token as word vs. punctuation/number stays the client's to make.

### Documentation

- README + crate-root docs now name the [obadh-ios](https://github.com/nsssayom/obadh-ios) keyboard
  as the reference C ABI downstream (SDK Shape, C ABI, Runtime Data); the crate-root doc was
  rewritten to describe the full layered SDK (deterministic core + autocorrect + autosuggest +
  `cabi`) rather than only transliteration; a CHANGELOG pointer was added to the README. No API or
  behavior change.
- The `।` sentence-ender is now named **dari** (its Bangla name) rather than "danda", in the README
  symbol table and the sanitizer comments. Terminology only; no behavior change.
- Added `RELEASING.md` documenting the branch and release process; removed em-dashes from README
  prose (house style). No API or behavior change.

## [0.9.0]

**Breaking (crate + C ABI, `cabi` version → 2).** A clean-up driven by the iOS keyboard downstream:
the engine now exposes only what a client uses and what the engine can test. Auto-insert *policy*
moves to the client; the engine keeps candidate generation, truthful provenance, and ranking. The
deterministic transliteration core is unchanged.

Why: two shipped auto-insert gates (0.8.0 heap-path, 0.8.2 roman-cost) were both wrong on the real
`bn.fst` — a data-dependent decision the test suite structurally cannot validate (it runs without
the artifacts). Validated against the real artifact, the discriminator turned out to be *frequency*,
not a structural gate: the wrong corrections are low-frequency words and the missed-but-correct ones
are high-frequency. That belongs in a client that has the data, not an engine that can't test it.

### Removed

- **The FST auto-insert gate.** `FstSuggestResult::auto_replacement`,
  `FstCandidateSource::is_auto_replace_eligible`, `FstCandidate::auto_replace_cost`, and the C ABI
  `obadh_autocorrect_should_replace`. (The heap `AutocorrectEngine::decide` / `AutocorrectDecision`
  path — not the mmap runtime, not the C ABI — is retained as a non-runtime convenience.)
- **The personal membership / commit-strength apparatus.** `CommitStrength`, `committed_text_weight`,
  `committed_token_weight`, `is_text_established`, session `established_weight` / `is_word_established`,
  `observe_committed_token_with_strength`, `commit_token_with_strength`; the C ABI
  `obadh_autosuggest_is_word_established` / `_established_weight` and the `strength` parameter on
  `obadh_autosuggest_commit`. The keyboard owns learned-word protection client-side, so these had no
  consumer.
- **`obadh_autocorrect_is_lexicon_word`** — subsumed by the new `obadh_autocorrect_word_frequency`
  (presence is `> 0`), which also returns the count the client gate's frequency ratio needs. Keeping
  both would leave a redundant boolean with no unique consumer.
- **`obadh_autocorrect_suggest`** (the plain string-list corrections) — no consumer. The typing bar
  uses `obadh_compose_suggestions` (baseline-first); the auto-insert gate uses
  `obadh_autocorrect_suggest_detailed` (the same corrections, plus provenance). The bare-strings
  variant sat between the two with no caller. The C ABI is now **23 symbols**.

### Added

- **`obadh_autocorrect_suggest_detailed`** — ranked corrections with full per-candidate provenance
  (`text`, `source`, `edit_cost`, `roman_repair_cost`, `frequency`) as a packed record list. This is
  the surface a client builds its own auto-insert gate on — `frequency` is the field that lets it
  reject the low-frequency false positives and override a rare-word baseline. `source` is a **frozen,
  append-only** numeric code (`FstCandidateSource::stable_code`); an unknown code must be treated as
  not auto-replaceable.
- **`obadh_autocorrect_word_frequency`** — the lexicon frequency of any word (`0` = not an entry),
  on the same scale as `suggest_detailed`'s per-candidate `frequency` (both read one table). It
  supplies the *baseline* side of the ratio the gate needs: a rare real-word baseline (`মানুস`, 49)
  is replaced only when the top correction (`মানুষ`, 95278) is far more frequent — the স/ষ, দ/ধ
  consonant-confusion class where the typo is itself a rare dictionary word. It **replaces**
  `is_lexicon_word`: presence is the `> 0` case, so no redundant boolean remains.
- A **reference client-gate policy** in the README (prose, not a shipped symbol), so downstreams copy
  a sound gate instead of inventing one — the anti-fragmentation cost of client-owned policy without
  the engine carrying untestable code.
- A **real-`bn.fst` test** (`suggest_detailed_banhla_on_real_bn_fst`) that pins `banhla` → বাংলা at
  `roman_repair_exact`, roman cost 2, high frequency against the shipped artifact; it skips when the
  submodules are not resolved, so it runs locally and is a no-op in CI. This is the verification the
  0.8.2 fix lacked.

### Migration

- **Auto-insert gate.** Replace `obadh_autocorrect_should_replace` with
  `obadh_autocorrect_suggest_detailed` and apply a client policy over the returned candidates — see
  *Auto-insert policy* in the README. The detailed records now carry `frequency`, which is what makes
  the decision correct; the removed gate had no access to it at the boundary.
- **Membership check.** Replace `obadh_autocorrect_is_lexicon_word(w) == 1` with
  `obadh_autocorrect_word_frequency(w) > 0`. The accessor also returns the count, which the
  rare-real-word override (`word_frequency(baseline)` vs the correction's `frequency`) needs.
- **Plain corrections.** `obadh_autocorrect_suggest` is gone; take the `text` fields from
  `obadh_autocorrect_suggest_detailed`, or use `obadh_compose_suggestions` for a baseline-first bar.
- **Learned-word protection.** `obadh_autosuggest_is_word_established` / `_established_weight` are
  gone; track established words in the client. `obadh_autosuggest_suggest` continues to surface
  learned words in its own suggestions.
- **Commit.** `obadh_autosuggest_commit` no longer takes a `strength` argument — call
  `obadh_autosuggest_commit(handle, token, token_len)`.
- **Rebuild against ABI version 2** (`obadh_abi_version()` returns `2`). Removed symbols fail to
  link by design.

### Notes

- Ranking quality (e.g. `sriti → ক্ষিতি`, `dinn → দিনন` ranking a wrong word first) is the engine's
  own workstream, measured against a labeled recall@1 set — the real lever for reliable auto-insert,
  and it improves every client's gate.

## [0.8.2]

A one-fix patch to the FST auto-insert gate, from the iOS keyboard downstream wiring their opt-in
auto-insert onto the engine's own gate. Confined to the opt-in gate; `suggest` / `compose` and the
deterministic core are unaffected.

### Fixed

- **`FstSuggestResult::auto_replacement()` now auto-applies a confident roman key-slip.** The gate
  excluded the most common correction on a transliteration keyboard — a one-key roman slip that
  resolves to an exact lexicon word (`banhla` → বাংলা). For a `RomanRepairExact` /
  `EnglishLoanwordExact` candidate, `edit_cost` carries the *Bangla-side* distance from the original
  baseline to the repaired output (বানহ্লা → বাংলা is 4 edits), while the *roman-side* cost sits in
  `roman_repair_cost`; the `edit_cost ≤ 1` bar was measuring the wrong dimension. New
  `FstCandidate::auto_replace_cost()` gates roman-repair channels on `roman_repair_cost` and
  native-script channels on `edit_cost`, so a confident single roman key-slip to an exact word
  qualifies by default — as confident as the Bangla one-edit the gate already accepted — while
  fuzzy recovery is still never auto-applied and a real word the user typed is still never
  overridden. No new config; `obadh_autocorrect_should_replace` picks it up automatically. ABI
  version unchanged.

## [0.8.1]

A patch on the `0.8.0` C ABI, from a gap analysis against the iOS keyboard's `ObadhBridge` surface —
one bug fix and three additive functions. The ABI version stays `1` (no signature changes), and the
deterministic core is untouched.

### Fixed

- **`obadh_autosuggest_suggest` now surfaces the user's learned words.** It returned only the model's
  candidates and never consulted the personal overlay, so learned out-of-vocabulary words — the whole
  point of the overlay — never appeared. It now merges like the reference bridge: learned words
  matching the current context first, then model candidates, then learned words with no context, all
  deduplicated.

### Added

- **`obadh_compose_suggestions`** — the active-typing candidate bar: the deterministic baseline first,
  then corrections. The baseline is always present so the user can keep exactly what they typed even
  when it is not a lexicon word. (`obadh_autocorrect_suggest` returns corrections only.)
- **`obadh_autocorrect_word_alternatives`** — alternative spellings for an already-composed *Bengali*
  word, a re-correction menu; lexicon-only, no Roman repairs.
- **`obadh_autosuggest_suggest_for_context`** — stateless model suggestions for an explicit context
  string, without using or updating the session's learned state.

The C ABI now exports 26 symbols. Autosuggest's internal `artifact` module became `pub(crate)` so the
C-ABI end-to-end tests can build fixtures; nothing is re-exported at the crate root, so the Rust
public API is unchanged.

## [0.8.0]

Downstream-integration release, driven by feedback from an iOS keyboard built on the engine. Four
additions that expose what the layered architecture already computes, at the boundary — plus a
stable C ABI so native integrators stop hand-rolling one each. **Every change is additive; the
existing Rust public API and every artifact format are unchanged**, and the deterministic
transliteration core is byte-for-byte identical.

### Added

- **A stable, versioned C ABI (`cabi` feature, off by default).** Native downstreams (iOS/Android
  keyboards) had no C ABI and each hand-rolled one, re-deriving the buffer and lifetime edges
  differently. The `cabi` feature exposes a header (`include/obadh.h`) the engine owns: deterministic
  transliteration, the autocorrect decision/suggest/membership/fingerprint surface, and the
  autosuggest commit/suggest/membership/snapshot/fingerprint surface, behind opaque handles. Writers
  are snprintf-style (return the needed length, copy only if it fits); string lists are
  count + length-prefixed records with no in-band delimiter (so a candidate may contain any bytes and
  an empty string is faithful). The ABI version is independent of crate semver. An executable test
  keeps the header in sync with the exports. The autosuggest neural/scorer handoff is deliberately
  left out until it settles. ([#23](https://github.com/nsssayom/obadh_engine/issues/23))

- **Personal-model membership query and commit-strength hint.** `PersonalAutosuggest::committed_text_weight`
  / `is_text_established` and the session-level `established_weight` / `is_word_established` answer
  "has the user established this word?" from the personal overlay, so a downstream can stop
  maintaining a parallel on-device store to protect names and slang from auto-correction. The measure
  is **post-decay** (counts halve in `decay_counts`), the honest signal rather than a raw ever-seen
  flag. `CommitStrength` (Committed / CorrectionRejected / ManuallyAdded) lets the caller classify a
  commit's intent; the engine owns the class → weight mapping.
  ([#21](https://github.com/nsssayom/obadh_engine/issues/21))

- **Artifact content fingerprints.** `FstLexicon` / `LoanwordLexicon` / `AutosuggestLm` expose
  `artifact_fingerprint()`, and `verify_artifact_fingerprint(bytes, expected)` fails loudly with a
  `FingerprintMismatch` so a stale pinned artifact fails fast at load instead of degrading silently. A
  content hash, not a per-format header field, so it is uniform across three unrelated binary formats
  and needs no format change. `obadh-autocorrect inspect-fst-lexicon` now prints the fingerprint for
  the crate ↔ artifact compatibility table below.
  ([#22](https://github.com/nsssayom/obadh_engine/issues/22))

- **An engine-owned auto-insert gate on the FST runtime path.** `FstSuggestResult::auto_replacement()`
  returns the correction confident enough to apply without asking — the baseline is not itself an
  exact word, the top candidate is an auto-replace-eligible channel
  (`FstCandidateSource::is_auto_replace_eligible`: a confident edit or exact repair, never a
  completion or heuristic guess), and its edit cost is at most one. Structural, not a score threshold,
  so it stays explainable. This is the mmap-path equivalent of the heap
  `AutocorrectEngine::decide` gate, which the keyboard could not reach.
  ([#23](https://github.com/nsssayom/obadh_engine/issues/23),
  [#20](https://github.com/nsssayom/obadh_engine/issues/20))

- **Reference-free property tests, structural sweeps, and CI** — carried in from 0.7.1 and extended;
  the `--features cabi` suite now runs in CI as well.

### Documentation

- **`AutocorrectEngine::decide` and the auto-insert gate are now documented**, with a doctest showing
  `replacement.is_some()` as the "should replace" signal and `suggest` as the lossy wrapper that
  discards it. The `roman_input` / `auto_replace_roman_input` interaction — which suppresses
  auto-replacement by default — is spelled out, since it is the usual surprise when migrating an
  app-side gate onto the engine's. ([#20](https://github.com/nsssayom/obadh_engine/issues/20))

### Notes

- `transliterate` (strict) and `transliterate_tokens` (best-effort) still disagree on **unsupported**
  input; unchanged here and pinned by a test, tracked in
  [#16](https://github.com/nsssayom/obadh_engine/issues/16).
- **Artifact compatibility.** `0.8.0` is built and tested against the `data/autocorrect` and
  `data/autosuggest` submodule revisions pinned in this commit. Downstreams pinning artifacts should
  verify with `artifact_fingerprint()`; run `obadh-autocorrect inspect-fst-lexicon --input <bn.fst>`
  against the resolved submodule to record the exact fingerprints for a private compatibility table.

## [0.7.1]

A patch release: three orthography fixes in the deterministic core, a terminology sweep, and the
test and CI infrastructure that found the bugs. **The public API is unchanged.** Every output that
changes was previously a cluster Bangla does not permit.

### Fixed

- **A vowel after a conjunct, reph or numeral could render as a second kar.**
  `transliterate_word_units_into` carries one flag, `previous_unit_accepts_dependent_vowel`, that
  decides whether the next vowel is a dependent sign or an independent letter. Seven of its sixteen
  match arms never assigned it, so a stale `true` leaked across them from a preceding bare
  consonant: `pxiE` → পক্সি**ে** instead of পক্সিএ, `krrkie` → কর্কি**ে** instead of কর্কিএ, and
  `k1i` → ক১**ি** — a vowel sign hanging off a Bengali digit. The suite already asserted this
  contract for `,,`, `^`, `:`, `ng`, <code>t``</code> and `rr`; conjuncts, reph and numerals were
  exempt only because their arms forgot.
  ([#9](https://github.com/nsssayom/obadh_engine/issues/9))

- **An explicit hasant with no consonant to sit on is now dropped.** A hasant suppresses a
  consonant's *inherent* vowel; after an explicit kar there is nothing left to suppress, so `ka,,`
  emitted কা**্**. The same hole let it land on chandrabindu (`k^,,`), anusvar (`kng,,`), bisarga
  (`k:,,`), khanda ta (<code>kt``,,</code> — already a dead consonant), a numeral (`k1,,`), and
  even another hasant (`rr,,` → র্**্**). The rule is now one predicate: **a hasant attaches only
  to a consonant.** As a standalone marker `,,` still renders on its own, and everything the rule
  sources document keeps working — `k,,` → ক্, `ko,,` → ক্, `kk,,` → ক্ক্, `rr,,ka` → র্কা,
  <code>rrk,,Sh</code> → র্ক্ষ, `kR,,` → কড়্.
  ([#10](https://github.com/nsssayom/obadh_engine/issues/10))

- **`ন্স` (`ns`) is now a conjunct.** It was absent from the inventory, so `laisens` produced
  লাইসেনস instead of লাইসেন্স, and `kansa` gave কানসা instead of কান্সা. The whole `-ence`/
  `-ance`/`-ense` loanword family was affected (ডিফেন্স, ব্যালেন্স, সেন্স, রেসপন্স,
  ফাইন্যান্স), as was `ট্রান্সফার`. It behaves like every other conjunct: adjacency joins and
  the inherent vowel breaks, so `nsa` → ন্সা while `nosa` → নসা, exactly as `kta` → ক্তা and
  `kota` → কতা.

  `ন্স` is a **loanword-only** cluster. Native and tatsama Bangla writes the same sequence with
  anusvar — ধ্বংস, হংস, বংশ — and those are unaffected.

### Changed

- **Scope of the change: roman input where `n` is immediately followed by `s`. Nothing else
  moves.** No other conjunct, cluster, or romanization is affected.

  Within that scope, `ns` now behaves like every other conjunct rather than like a pair the
  engine could not spell. Adjacency joins, and the inherent vowel breaks — the same rule that
  has always governed `amra` → আম্রা vs `amora` → আমরা. So words that take the cluster now come
  out right on their own (`respons` → রেস্পন্স, `TrAnsphar` → ট্রান্সফার), and words that spell
  ন and স separately type the `o` (`anosar` → আনসার, `monosur` → মনসুর, `inosTol` → ইনস্টল,
  `konosarrT` → কনসার্ট).

  Previously ন্স was simply unreachable, so `ansar` → আনসার fell out by default. That default is
  gone, not because the convention changed, but because `ns` finally has a conjunct to form.

- **Bangla terminology replaces Sanskrit, Devanagari and Arabic terms** throughout the source,
  tests and rule docs: `virama` → `hasant` (হসন্ত), `visarga` → `bisarga` (বিসর্গ), `anusvara` →
  `anusvar` (অনুস্বার), `matra` → `kar` (কার), `nukta` → `phota` (ফোটা). In Bangla, মাত্রা is the
  headstroke on a letter, not the vowel sign — that is a কার; and নুক্তা is Arabic (نقطة), while
  the Bangla name for the subscript dot is ফোটা. `chandrabindu`, `khanda ta`, `reph`, `phola` and
  `juktoborno` were already Bangla and are untouched.

  The renamed constants are `pub(crate)` or private, so **the public API does not change**.
  Verified behavior-neutral: 19,110 generated inputs transliterate byte-identically before and
  after. ([#11](https://github.com/nsssayom/obadh_engine/issues/11))

### Added

- **`tests/property_tests.rs`** — seven reference-free properties over a fixed-seed corpus of ~45k
  supported inputs and ~14k inputs that deliberately leave the sanitizer's allowed set. None of
  them needs to know the expected Bengali for a given Roman input, so they hold for inputs nobody
  tabulated: `transliterate` is idempotent on its own output; `transliterate(x)` equals
  `transliterate_tokens(tokenize(x))` on supported input; `transliterate_lenient(x)` equals
  `transliterate(clean(x))`; tokenizer spans reassemble the input; no render path emits a dotted
  circle. Each property was mutation-tested — an injected bug in the token render path, in
  `transliterate_lenient`, and in word-internal numeral rendering each failed exactly one property
  and no others.

- **Structural sweeps in `tests/orthography_tests.rs` and `tests/hasant_tests.rs`.** Over 4,320 and
  7,072 generated inputs respectively, no output may stack two vowel signs, hang one off a numeral,
  or place a hasant on a sign that cannot carry it. Both fail on the pre-fix engine (839/4,320 and
  6,051/7,072), so they are regression nets rather than rubber stamps.

- **CI (`.github/workflows/ci.yml`).** `cargo test`, `cargo test --features cli`, the `wasm32`
  target check and the benchmark compile now run on every push and pull request, with
  `RUSTFLAGS: -D warnings`. Submodules are deliberately not checked out, so CI enforces rather than
  assumes the claim that the suite passes without the Git-LFS data.

- **`tests/juktoborno_corpus_tests.rs`** — an exhaustive conjunct corpus (~20k assertions)
  driven from `data/conjuncts.csv` rather than hand-transcribed. It guards both directions of
  the contract: every inventory conjunct must form word-initially, medially and finally and
  accept every dependent vowel; and every consonant pair *outside* the inventory must **not**
  join on its own (1,023 negative cases), since silent over-production is as much a defect as a
  missing conjunct. It also pins the inherent-vowel break, the `,,` explicit-hasant escape for
  all 1,024 pairs, the Unicode khanda-ta ligature rule (ত renders as ৎ except before
  ত/থ/ন/ব/ম/য/র), and a well-formedness sweep asserting the engine never emits a dotted circle,
  a dangling hasant, a hasant before a kar, a doubled hasant, or a hasant on a non-joining sign.

### Notes

- `data/rules/conjunct.wiki` mirrors the Bengali Wikipedia যুক্তবর্ণ list, which claims its 306
  entries are exhaustive (*"এর বাইরে কোন যুক্তবর্ণ সম্ভবত বাংলায় প্রচলিত নয়"*). That claim does not
  hold for modern loanwords — bn.wikipedia's own article title লাইসেন্স uses `ন্স`. The mirror is
  left byte-identical to upstream; `data/conjuncts.csv` is the engine's inventory and now
  intentionally carries one conjunct the wiki does not. The `wiki ⊆ csv` contract test still holds.

- `transliterate` and `transliterate_tokens` disagree on **unsupported** input: the former is
  strict and returns the whole input unchanged, the latter renders the supported tokens and passes
  the rest through (`ké` → `ké` vs `কé`). The behavior is unchanged in this release and is now
  pinned by a test; resolving it is a public-API decision tracked in
  [#16](https://github.com/nsssayom/obadh_engine/issues/16).

## [0.7.0]

Three grounded recovery channels are added to the **active-word autocorrect layer**
(`FstLexicon` suggest path). Each is data-grounded rather than hand-tuned, runs only when the
typed word is a weak signal, and is precision-capped so it can never demote a word the input
actually spells. The deterministic transliteration hot path is unchanged.

### Added

- **Dropped-vowel ("skeleton") channel.** Recovers words typed with vowels omitted
  (e.g. `krlm` → করলাম) by matching the consonant skeleton. It reads skeleton-mates directly
  out of the existing lexicon FST via a custom `fst::Automaton` — **no second index, no extra
  artifact, no added memory or bundle size**, and it always stays consistent with the lexicon.
  Only identical modern-Bengali phonemes are folded (শ/ষ, জ/য, ণ/ন, ড়/ঢ়), grounded in the
  documented grapheme→IPA inventory. Verified over the full 845k-word vocabulary:
  **72% recall@1 / 95% recall@5** (frequency-weighted).
- **Consonant-confusion channel.** Fixes same/near-sound spelling slips (e.g. মানুশ → মানুষ)
  by substituting a baseline consonant with a near phoneme, cost **graded by Bengali IPA
  articulatory-feature distance** (`src/autocorrect/phoneme.rs`) — not a hand-listed confusion
  set.
- **QWERTY key-slip ("fat-finger") channel.** Corrects a physically adjacent-key slip in the
  roman input before transliteration (e.g. `banhla` → বাংলা, `desj` → দেশ). Grounded in
  **FFitts law** (Bi, Li & Zhai, CHI 2013): a finger's touch lands as a 2-D Gaussian around
  the key, so slip cost is quadratic in key distance with σ ≈ 0.5 key-widths. It fires **only
  when the untouched baseline is not itself a lexicon word**, so a validly-typed word is never
  second-guessed. Measured recall when the channel acts (non-word typo): **94% @1 / 99% @5**,
  at **100% precision** on correctly-typed inputs.
- **`key_slip_repaired_outputs(input, baseline, baseline_frequency, transliterate, is_word)`**
  — public helper that produces lexicon-validated key-slip repaired baselines. It mirrors the
  existing `roman_repaired_outputs`: the caller passes only plumbing closures; every numeric
  parameter (touch σ, neighbour radius, variant cap, scoring, gate) is baked into the engine
  and is **not** a client tunable. Wire it into a suggest pipeline exactly where you already
  build `roman_repaired_outputs` (see `src/bin/obadh_autocorrect/fst_cli.rs` and
  `src/wasm/mod.rs`); the WASM `ObadhAutocorrectWasm.suggest` path already includes it.

### Changed

- **`FstCandidateSource` is now `#[non_exhaustive]`.** Future suggestion channels can add
  variants without a breaking change. Downstream `match` statements on this enum must now
  include a wildcard arm.
- **Recovery channels are subordinated to confident readings.** When a word is produced by
  more than one channel it now ranks by its best score but is *labelled* by the most
  authoritative channel that found it, and the skeleton/confusion channels are capped just
  below the best *confident* candidate (exact word, cheap roman-repair, or exact loanword) —
  so a heuristic guess can never relabel or outrank a word the input actually spells.

### API

- **Added variants** to `FstCandidateSource`: `SkeletonVowelDrop` (`"fst_skeleton_vowel_drop"`)
  and `ConsonantConfusion` (`"fst_consonant_confusion"`). The QWERTY channel surfaces as the
  existing `RomanRepairExact` source with `roman_repair_kind = "qwerty_key_slip"`.
- **Added function**: `key_slip_repaired_outputs` (re-exported from the crate root).
- No existing function signatures were removed or changed.

### Migration

- Adding a wildcard arm (`_ => …`) to any exhaustive `match` on `FstCandidateSource` is the
  only source change downstreams may need. Everything else is additive.
