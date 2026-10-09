# Contextual typing: shared data foundation

This is the first implementation slice of the contextual typing upgrade. Corpus
partitioning, artifact provenance, evaluation policy, and model development live
in the engine repository so iOS, macOS, and future clients use the same evidence.
It does not yet change keyboard suggestions or enable a neural model in an app.

## Architecture and ownership

The shared Rust engine owns token/context identity, candidate retrieval, ranking
policy, correction confidence, and versioned model contracts. Platform adapters
own inference runtime integration, host text access, scheduling, and presentation.
Core ML is an Apple runtime adapter, not the definition of the language system.
Training tools produce portable, identified artifacts; every platform must test
the same input/output fixtures before shipping its adapter.

The existing FST lexicon remains the fast dictionary/prefix/spelling candidate
index. The compact n-gram model supplies cheap contextual retrieval. A separate
small neural model scores longer context; the initial experiment uses a GRU,
not a replacement FST. The current neural vocabulary is limited to 32,768 entries:
unknown words still require dictionary/fallback handling, and these word-level
scores do not constitute a general grammar corrector.

The intended product has three separately evaluated operations: completion and
next-word prediction, conservative automatic spelling correction, and explicit
acceptance of sentence-level grammar edits. Automatic replacement needs its own
precision and false-correction measurements; an improvement in next-word top-5
accuracy cannot justify a more aggressive correction policy.

The first gate is trustworthy data. Previously, sentence offsets in the same
corpus were used for neural training and evaluation, while vocabulary/retrieval
construction could see that corpus. Those results remain useful for development
but do not demonstrate end-to-end performance on unseen documents.

## Version 1 corpus contract

Input is a directory containing `sentences/*.tsv.gz`. Each UTF-8 TSV has the
existing five columns plus an optional `group_id`:

| Column | Contract |
| --- | --- |
| `source` | Stable ASCII identifier, 1–64 letters/digits/underscores/hyphens, beginning with a letter/digit. Windows reserved names and case-insensitive collisions are rejected. |
| `document_id` | Nonempty stable source-local document identity. A book is one document; a chapter is not an independent split unit. |
| `sentence_id` | Nonnegative signed-64-bit integer, unique within source/document. |
| `token_count` | Positive number of space-separated tokens. |
| `tokens` | Existing tokenized text with single spaces and no control characters. Original Unicode text is preserved. |
| `group_id` | Optional global identity joining related documents across sources. Use for a conversation, duplicate cluster, or original/corrupted sentence family. |

An explicit group overrides the source/document split key. All rows of one
document must agree on their group. Importers must preserve stable identities;
assigning a fresh document ID to every sentence defeats document isolation.
Unknown columns, duplicate sentence IDs, inconsistent groups, and malformed rows
fail the build. Source-specific metadata belongs in the importer manifest rather
than extra sentence columns.

The builder hashes the group with a declared seed into integer basis points:
90% train, 5% validation, 5% test by default. These are expected proportions of
groups, not guaranteed token ratios. Large books can make small-source splits
uneven. Assignment is independent of file iteration order and Python hash seeds.

An exact sentence occurring across split boundaries is quarantined from **all**
splits. Identity uses NFC solely for duplicate detection. Same-split repetition
remains; importers must handle excessive boilerplate separately. This policy
does not detect paraphrases, near duplicates, mistranscriptions, or undeclared
relationships between documents. Set `group_id` after source-level duplicate
clustering before making release-quality claims.

SQLite stages rows and duplicate identities on disk, with a bounded Python cache.
Output order and gzip headers are deterministic. SHA-256 receipts identify input
bytes, output bytes, the split policy, and optional input-manifest bytes. Identical
inputs/policy produce the same dataset ID with the same Unicode database and
compression implementation;
changing compressed input bytes intentionally changes that ID even when text is
equivalent. Membership remains stable across compression implementations.

Publication refuses an existing destination. A staging directory and `.building`
marker keep incomplete output from being consumed as a valid partition set.
Ordinary failures remove staging output; a killed process can leave an unpublished
staging directory or a marked destination. Inspect and remove those explicitly
before retrying. Manifests detect accidental changes, not malicious forgery.

## Build and verify

Run from the engine repository using Python 3.11 or newer; the partitioner and
retrieval evaluation need only the standard library. Allocate temporary disk
space for the uncompressed SQLite staging database and its indexes as well as
the compressed output. Published datasets must remain immutable during consumers.

```sh
python3 -m tools.corpus.partition build \
  --corpus-dir data/autosuggest/corpus \
  --output target/contextual-corpus-v1

python3 -m tools.corpus.partition verify \
  --dataset target/contextual-corpus-v1

python3 -m tools.autosuggest.build_vocab \
  --corpus-dir target/contextual-corpus-v1/train \
  --output target/contextual-v1/vocab.tsv

python3 -m tools.autosuggest.build_ngram_lm \
  --corpus-dir target/contextual-corpus-v1/train \
  --vocab target/contextual-v1/vocab.tsv \
  --output target/contextual-v1/ngram.bin \
  --backend sqlite --sqlite-path target/contextual-v1/counts.sqlite \
  --max-context-order 3 --max-candidates-per-prefix 64 --compact-count-records

python3 -m tools.autosuggest.eval_ngram_lm \
  --model target/contextual-v1/ngram.bin \
  --corpus-dir target/contextual-corpus-v1/validation
```

Keep the dataset directory together: partition receipts reference its root
manifest. `verify` validates receipts and bytes; semantic isolation was established
by `build`, not reconstructed by the verifier. Preserve vocabulary/model sidecars
alongside their artifacts. Training builders reject validation/test directories.
The n-gram builder requires vocabulary lineage from the same train partition.
Its verified SQLite cache also binds source selection, weights, ingestion limits,
and counting completion, so interrupted or mismatched counts cannot be reused.

Evaluation requires a model trained on the corresponding train partition and
checks its bytes. Reports explicitly distinguish `partition_verified` from
`unverified` legacy evaluations. The verified status covers document groups and
exact NFC sentence overlap only. Unknown target tokens remain in the all-target
denominator; report vocabulary coverage and candidate recall alongside rank metrics.

Both neural training entry points (`train_next_word_lm` and
`train_candidate_reranker`) accept `--validation-corpus-dir`. For partitioned
training it is required, must name `validation` from the same dataset, and defaults
the validation sentence offset to zero. The next-word trainer checks resumed and
distillation-teacher checkpoints against the dataset and retrieval/vocabulary
identity. Historical checkpoints without lineage cannot silently become verified
models. Legacy corpus workflows remain supported and are labeled unverified.

Validation is for model selection, early stopping, and threshold calibration.
Reserve test for a frozen final comparison; never tune a policy from test results.
The older policy-sweep/package tools still need end-to-end provenance integration
before they can form a production neural release pipeline. No new neural artifact
is approved for app distribution by this change.

## Additional Bangla sources

Source review date: 2026-10-05. The existing corpus is dominated by news and
Wikipedia; adding more formal text alone does not address everyday typing.
The table distinguishes the acquired book collection from other acquisition candidates.

| Source | Value for this project | Ingestion decision |
| --- | --- | --- |
| [eBanglaLibrary](https://www.ebanglalibrary.com/) | Additional books and literary dialogue; provided and authorized by the project owner. | Eight books acquired in v2; four fixed training works, two validation works, and two test works. Chapters share a global work identity. See the acquisition record below. |
| [SHONGLAP paper](https://aclanthology.org/2022.lrec-1.623/) | 7.7k+ annotated dialogues drawn from podcasts and talk shows. | Investigate native conversational context; group by original programme/episode. Confirm dataset access and dataset terms at acquisition; the paper alone is not a data receipt. |
| [SentNoB authors' repository](https://github.com/KhondokerIslam/SentNoB) | Noisy public comments across 13 domains, including dialect and grammatical variation. | Candidate robustness benchmark/domain adaptation. Preserve publisher splits and exclude their held-out examples from training. Sentiment labels are not grammatical corrections. |
| [SentiGOLD dataset card](https://huggingface.co/datasets/SayedShaun/sentigold) | 70,000 sentiment-labeled examples with domain metadata. | Inspect source identities, terms, duplicate relationships, and native-language quality before import. No explicit dataset license was located on the reviewed card. |
| [BanglishRev authors' dataset](https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset) | 1.74M written product reviews, including Bengali, English, and romanized/code-mixed Bengali. | Useful typing noise and mixing patterns; published CC-BY-NC-SA-4.0, so keep out of the default distributable training corpus unless the intended use is covered. Import text only, retain review/reply grouping, and audit copied reviews and seller boilerplate. |
| [Bengali DailyDialogue card](https://huggingface.co/datasets/csebuetnlp/dailydialogue_bn) | Translated dialogue, useful as a controlled comparison. | Lower priority than native dialogue; published noncommercial terms. Do not treat translated English dialogue as a native Bangla typing benchmark. |

For a new acquisition, retain the source snapshot/revision, retrieval URL, content
hashes, extraction version, domain, and applicable usage terms. Remove account
identifiers from model text. Source IDs and grouping metadata must survive text
cleaning; retain the original snapshot for reproducibility. The current partitioner
records the supplied input manifest digest but does not implement or certify those
source-specific acquisition steps. Existing publisher-held-out sets require an
import policy that preserves their exclusion; do not feed them through the general
hash partitioner and accidentally turn them into training examples.

## Initial full-corpus validation

The 2026-10-05 local build processed all 13,344,360 sentences / 161,638,268 tokens
and passed receipt verification. Dataset ID:
`01ba7230f39d974de4997912838942bcab180b5fc2adfb7eb8be304e8e1e688c`.
Output is `target/contextual-corpus-v1`, with the verification receipt at
`target/contextual-corpus-v1-verification.json`.

| Partition | Documents with emitted rows | Sentences | Tokens |
| --- | ---: | ---: | ---: |
| Train | 515,266 | 12,010,180 | 145,460,399 |
| Validation | 28,450 | 660,115 | 8,000,794 |
| Test | 28,773 | 674,065 | 8,177,075 |

No additional cross-split exact NFC duplicates were found. The input corpus
manifest already records SQLite sentence deduplication; this result does not
establish absence of near duplicates.

All 13 existing EPUB documents landed in train. In this initial v1 dataset, validation and test
contain news and Wikipedia only. This is a reproducible formal-text baseline,
**not a domain-balanced keyboard benchmark**. Before claiming literary or everyday
typing quality, construct a separately identified benchmark with deliberate
held-out book/conversation coverage and preserve its exclusions when rebuilding
training artifacts. Do not choose another split seed based on model scores.

## Expanded book collection and v2 partitions

`acquire_books.py` uses the public-page extraction functions from the existing
`epub-exporter` project. It performs bounded, serial HTTPS acquisition, retains
content-addressed HTML snapshots, and records URLs, hashes, timestamps, extraction
code hashes, book identities, and the owner's source authorization. Its cache
supports offline replay. It neither needs nor reads browser/account credentials.

The checked-in plan is `tools/corpus/sources/ebanglalibrary-v1.json`. Before model
scoring, its eight new works were assigned four train, two validation,
and two test roles. Explicit global-group assignments preserve these roles through
partitioning. This is work-level isolation, not an author-held-out claim: the
older EPUB collection can contain other works by those authors. Unknown or stale
assignment keys fail the build. Related editions
or translated copies still require a source-level near-duplicate audit.

```sh
uv run --with ../epub-exporter python -m tools.corpus.acquire_books \
  --plan tools/corpus/sources/ebanglalibrary-v1.json \
  --base-corpus data/autosuggest/corpus \
  --output target/contextual-corpus-raw-v2 \
  --cache "$HOME/.cache/obadh/corpus/ebanglalibrary"

python3 -m tools.corpus.partition build \
  --corpus-dir target/contextual-corpus-raw-v2 \
  --output target/contextual-corpus-v2 \
  --group-assignments target/contextual-corpus-raw-v2/group-assignments.json

python3 -m tools.corpus.partition verify --dataset target/contextual-corpus-v2
```

The completed acquisition contains **8 books / 81 chapters / 13,054 sentence
chunks / 114,302 tokens**. The raw chapter JSONL preserves cleaned paragraphs,
including punctuation and English; the existing Bangla-only token view intentionally
drops those for compatibility with the current vocabulary. Do not use that token
view to claim code-mixed language coverage.

One attempted book, *ফুটিডাঙায় ফাটাফাটি*, was rejected because every lesson URL
redirected to the book introduction. The importer now rejects chapter-to-book
redirects and identical extracted chapter bodies instead of silently multiplying
introductory text. No rows from that book entered v2.

Acquisition ID:
`c7461ffdd2edf30da04b594336128e170b823a847541451d8aff61a3266b0bb0`.
Partition dataset ID:
`2a9a614b2abb6675be0bc70b2cca183a479f45dc887c774c2aefe203e1819ff1`.
The expanded build quarantined 509 cross-split sentence occurrences representing
222 distinct NFC sentence identities (1,309 tokens).

| Partition | Documents | Sentences | Tokens | New books |
| --- | ---: | ---: | ---: | ---: |
| Train | 515,270 | 12,014,063 | 145,494,563 | 4 |
| Validation | 28,452 | 665,137 | 8,043,239 | 2 |
| Test | 28,775 | 677,705 | 8,213,459 | 2 |

The 32,768-entry vocabulary uses only v2 train and covers 91.94% of its tokens;
that is training coverage, not held-out accuracy. Artifacts live under
`target/contextual-v2`; raw and partitioned corpora remain outside Git. See
[training and recovery](../autosuggest/TRAINING.md) for the checkpoint contract
and the bounded A100 feasibility experiment.

## Acceptance and next milestones

```sh
python3 -m unittest discover -s tools/tests -v
```

CI runs the standard-library corpus and real vocabulary/retrieval pipeline tests
on Linux, macOS, and Windows. They cover repeatability, document/global grouping,
Unicode-equivalent overlap, malformed input, interrupted publication, file and
manifest tampering, artifact/cache lineage, held-out/OOV evaluation, and checkpoint
provenance. Runtime training and device performance require separate validation.

With the training dependencies available, the same test command also runs CPU
smoke tests through both neural CLIs, including resume, distillation, and rejection
of test data for model selection. These tests skip when PyTorch is absent;
the standard-library CI matrix does not claim neural runtime coverage.

Local checks passed on macOS: 41 Python tests with the optional EPUB extractor and
PyTorch installed, including eleven neural/checkpoint/export tests, plus 147 existing Rust
autosuggest tests. The corpus suite was also exercised on Python 3.11. The neural
checks cover exact CPU mid-epoch resume, atomic-save failure, and unknown-only
source denominators. CUDA save/reload/resume passed on the allocated A100 with
the CHPC PyTorch 2.6 container.
The updated Python builder's compact v3 output was also loaded and queried by the
Rust CLI. Linux/Windows execution is configured in CI and was not run locally.

Next: freeze a domain-balanced native Bangla benchmark with reviewed typo/correction
pairs and no-edit examples; audit near duplicates; rebuild retrieval on train only;
measure candidate recall and false corrections; then compare compact contextual
rankers. Any shipping model must pass per-domain quality gates plus real keyboard
extension latency, memory, cancellation, Unicode, and fallback tests on supported
devices. These are release requirements, not claims already established here.
