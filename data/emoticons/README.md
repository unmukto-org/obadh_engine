# Emoticon mapping reference

This is reference data for Obadh's exact active-input emoticon candidate
channel. `tools/emoticons/generate_table.py` generates the sorted static Rust
table; runtime lookup uses binary search and never parses this JSON.

Check the source table with `python3 tools/emoticons/generate_table.py --check`.
After `cargo build --features cli` and `./build.sh wasm`, run
`node tools/emoticons/test_browser.mjs` to exercise every alias through both
browser backends and verify composer recognition and commit behavior.

## Source and reproducibility

- Source: [wooorm/emoticon](https://github.com/wooorm/emoticon), version 4.1.0.
- Pinned commit: `5b43e4c3e6fe43601790cc888a5e655fa9c1ffc4`.
- Original data: `upstream-index.js`, preserved verbatim from that commit.
- Original SHA-256: `f701675bd5a88cce1a7a671332b917af0effc2fa071a1ede6b4502dfa4bdf5c7`.
- License: MIT; upstream attribution and terms are preserved in `LICENSE`.
- `emoticons.json` contains the same description, emoji, emoticons, name, and
  tags fields, converted to JSON without changing mappings.

The snapshot contains 29 emoji groups and 322 distinct ASCII aliases. All 29
targets were checked against the fully-qualified entries in Unicode's
[Emoji 17.0 test data](https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt).

Both nose-less and nose-bearing forms are present, including `:)` / `:-)` and
`:D` / `:-D`. Upstream maps the former pair to 😃 and the latter pair to 😄.
Obadh uses these pinned source choices as its preferred alternatives.

## Authority and coverage

[Unicode UTS #51, section 1.1](https://www.unicode.org/reports/tr51/#Emoticons)
defines emoticons and gives an example of emoticon-to-emoji input. It does not
provide an exhaustive normative ASCII-to-emoji mapping table. This snapshot
is an attributable community reference, not a Unicode standard or a complete
inventory of every historical emoticon. It does not cover all kaomoji.

Cross-check: [iamcal/emoji-data](https://github.com/iamcal/emoji-data), commit
`13ee711e222ea17fe537bfea953c687866f16411`, lists `:)` for multiple emoji,
including 😃, 😄, 😊, and 🙂. Therefore matching an alias exactly does not make
its intended emoji uniquely determined. Obadh must explicitly choose preferred
targets and may offer alternatives.

## Integration

Preserve the literal emoticon as candidate one; offer its preferred emoji as
candidate two. Match the complete active input exactly, preserving case and
punctuation. Do not use fuzzy spelling repair or word-frequency scoring for
this channel. Recognize the full emoticon before ordinary punctuation splits
it; `:D` must not become a colon plus a transliterated `D`.

This belongs above deterministic transliteration alongside active-input
candidate generation. The native compose API already presents the baseline
first. Recognized emoticons need a literal baseline override in this higher
layer. Automatic emoji insertion should be an explicit client preference.

Clients must also account for existing Unicode emoji: strict core
transliteration currently returns input containing unsupported characters
unchanged, while lenient transliteration removes those characters. Emoji
candidates should be committed directly rather than passed back through the
lenient core.
