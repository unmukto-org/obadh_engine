# Contextual model training and recovery

**Parked on 8 October 2026.** Read [MODEL_HANDOFF.md](MODEL_HANDOFF.md) first
for the current selected checkpoint, final v9/v10 outcomes, private archive,
known preservation gaps and deliberate recovery procedure. Active-run commands
below are historical; their allocation and absolute cutoffs have expired.

The shared engine contains the data, retrieval, neural training, and evaluation
contracts. Next-word/context experiments compare a GRU and compact transformers.
`tools/correction` now also trains a transformer encoder with a copy decoder and
a parallel token-edit model for spelling, grammar and punctuation. These are
experimental training/evaluation implementations; automatic replacements remain
disabled in the apps. Runtime quality and keyboard integration remain separate
release gates.

The DeltaAI v3 run adds a fresh 20.45M-parameter transformer, a train-only 16K
Unicode byte-BPE tokenizer and 252M punctuation-preserving target tokens.
`document_data.py` creates document-bounded blocks and balances expected token
exposure across news, Wikipedia and contributor-balanced books. The frozen run
and recovery state are archived at
`/Users/nsssayom/Dev/obadh-model-runs/2026-10-05-deltaai`.

`evaluate_cached_words.py` compares complete-word predictions from the unchanged
16-beam/12-piece search using PyTorch versus exported ONNX or INT8 Core ML cache
graphs. `cached_word_scorer.py` owns request-local, bounded beam caches and resets
on every request. Branch reordering, recycled backend buffers, capacity limits
and full-word score parity are tested. This Python host measures desktop decode
latency including tokenization and search; it is not the native keyboard runtime.
Export graphs use document-bounded validation fixtures and verify tokenizers,
checkpoints and data receipts before use.

## Correction experiments

Correction pairs preserve the parent corpus's work-level partitions and tokenizer
identity. The expanded training set contains 1,385,320 sentence/prefix pairs from
98,625 source sentences across books, news and Wikipedia. Synthetic corruption is
a restoration task, not a measurement of natural typing accuracy. The six requested
grammar categories in teacher generation are prompts, not verified annotations.

`teacher_roundtrip.py` creates minimal errors, asks a fresh source-only pass to
recover the reference, and requests a separate conservative critique. Results are
fsynced after each batch and resumed against an exact bank/code/model contract.
`roundtrip_data.py` requires the completed run and matching hashes before admitting
labels. It additionally excludes ambiguous question/emphasis punctuation changes
and third-person honorific substitutions. Same-model agreement remains fallible.
Teacher training records must be disjoint from held-out work and text identities;
`distillation_sampling.py` verifies this and balances the admitted error categories.

The 21.05M-parameter parallel editor predicts KEEP, DELETE, REPLACE and up to three
appended subword tokens. Unsupported alignments are counted, never relabeled as
identity. `edit_policy.py` admits related token edits atomically: a rejected
replacement cannot leave its accompanying deletions applied. Classification and
payload probabilities are uncalibrated; the numerical threshold is an experiment
setting, not a production confidence estimate.

`evaluate.py` separates raw outputs from guarded fallbacks, reports unchanged-text
preservation and synthetic restoration by error type, and keeps IndiGEC development
results separate. `compare.py` requires identical evaluation banks and uses the
upstream corpus GLEU scorer with paired bootstrap intervals. Reused development
probes are diagnostics, not an independent shipping benchmark. No correction model
has passed native typing accuracy or physical iPhone 11 memory/latency qualification.

Frozen correction projects, checkpoint hashes, evaluation rows, and recovery
commands are in the DeltaAI archive's `correction-state.json`. Resume using each
project's frozen code, since changes to architecture/data/sampling/code invalidate
the checkpoint contract. The grammar fine-tuning experiment retains checkpoints
every 500 steps to expose quality regressions hidden by average validation loss.

## Verified inputs

Use the immutable partitions described in [the corpus contract](../corpus/README.md).
Build the vocabulary and retrieval artifact from `train` only, and retain their
`.manifest.json` sidecars. Neural training requires `--validation-corpus-dir` from
the same partition set. Dataset, vocabulary/retrieval, and checkpoint mismatches
fail before training. A test partition cannot be supplied for model selection.

`train_next_word_lm` reports unknown target words as misses in global and per-source
all-target metrics. Always distinguish these from in-vocabulary metrics. Current
examples use clean prior tokens (teacher forcing), not noisy real keyboard
histories, and the token view is Bangla-only. These scores do not measure spelling
correction precision, grammar correction, or user acceptance.

## Checkpoint contract

Supply `--output-checkpoint checkpoints/latest.pt` for recoverable training.
`--checkpoint-seconds` defaults to 300 and may not exceed 900 seconds. Checkpoints
are written to a temporary file in the destination directory, flushed and fsynced,
then atomically replaced. A failed save leaves the previous checkpoint intact.

A checkpoint includes model and AdamW state, completed epochs, the next batch,
optimizer steps, partial-epoch loss accounting, Python/PyTorch device random state,
and a training contract. Each epoch's batch permutation is reproducible from an
independent generator. Resume must preserve architecture, sampling, optimizer,
data, seed, and trainer code identity; incompatible mid-run resumes fail closed.
CPU interruption/resume is tested for identical final weights. Evaluation-only
invocations cannot overwrite a recovery checkpoint. CUDA operators may
be nondeterministic; the saved state does not promise bitwise cross-device or
cross-runtime reproduction. Preserve the container/runtime for continuation.

`--best-checkpoint` retains the best completed epoch by the unweighted mean of
per-source validation top-5 all-target accuracy. Its path must differ from latest.
Latest is the recovery artifact, and best is the selection artifact. Neither is a
shipping model package. Saving before validation protects completed training if
validation is interrupted; the retained best may then precede the newest epoch.

`--max-steps` and `--max-training-seconds` bound an invocation. `--stop-at-utc` is an
absolute timezone-qualified cutoff checked between optimizer steps. A stop saves
partial progress, then runs final evaluation. Leave time for evaluation and result
transfer; the cutoff does not kill a process inside an individual operation.

Resume with the same command plus `--input-checkpoint checkpoints/latest.pt`.
`--epochs N` means N epochs starting after the saved completed-epoch count; a
partially trained first epoch is finished as one of those N. For example, after
three complete epochs of a six-epoch plan, use `--epochs 3`. After two complete
and part of epoch three, use `--epochs 4`. Set a new absolute cutoff only when
compute has actually been authorized and allocated. Do not resume a completed
run accidentally: that starts additional epochs.

## CHPC experiment, 2026-10-05

Use only the project owner's `~/Dev/chpc-agentic-llm/chpc-gpu` wrapper for remote
access. Read `~/Dev/chpc-agentic-llm/docs/training-gpu.md` before using it. No
allocation acquisition/release or Slurm cancellation is part of this workflow.
A missing SSH master requires the owner's password/Duo login; stop remote work
instead of trying another route.

Remote experiment root:
`/scratch/general/vast/garcia/u1523034/train/obadh-contextual-20261005`.
Local code snapshot: `target/chpc-contextual`.
Local downloaded results: `target/chpc-results-20261005`.

The allocation ends at 05:39 MDT (11:39 UTC). This experiment's training cutoff is
05:00 MDT, leaving time for evaluation and transfer before the owner's 05:25 pull
limit. It also has a two-hour training budget. Its wrapper loads
`deeplearning/24.12.torch` and invokes Python inside the supplied container with
one A100 80GB, eight CPU threads, and no background data workers.

The smoke test used a tiny verified fixture on CUDA, saved after one optimizer
step, then reloaded and finished the epoch in a second invocation. Synthetic
fixture scores are excluded from model-quality results.

The experiment uses v2 dataset
`2a9a614b2abb6675be0bc70b2cca183a479f45dc887c774c2aefe203e1819ff1`
and a 32,768-entry vocabulary built from all 145,494,563 training tokens. Retrieval
counts are bounded to 1,000,000 sentences per source, retaining all training books;
this is a declared feasibility budget, not a rebuilt full-corpus shipping model.
Retrieval uses three preceding tokens, up to 64 candidates per prefix, and compact
v3 count records. The neural model uses 16 tokens of context, a single 320-wide
GRU, tied 320-wide embeddings, token-weighted cross entropy, and AdamW. The
six-epoch plan caps each source at 1,000,000 sentences / 12,000,000 examples, with
batch size 1,024. Validation caps are 6,000 sentences / 40,000 examples per source.
The exact command and code hashes are saved in `train_contextual.sh` and
`run-manifest.json` in the local and remote snapshots.

```sh
G=~/Dev/chpc-agentic-llm/chpc-gpu
$G info
$G ps
$G logs obadh-v2-gru320
$G pull obadh-contextual-20261005/checkpoints target/chpc-results-20261005/
$G pull obadh-contextual-20261005/reports target/chpc-results-20261005/
$G pull runs/obadh-v2-gru320.log target/chpc-results-20261005/
```

Use `start`, not `run`, for long training. A recovery command, after inspecting
latest's completed-epoch count and updating the budget, has this form:

```sh
$G start obadh-v2-gru320-resume -- bash obadh-contextual-20261005/train_contextual.sh \
  --input-checkpoint checkpoints/latest.pt --epochs REMAINING \
  --stop-at-utc NEW_AUTHORIZED_CUTOFF
```

`start` uses the archived trainer snapshot under the remote experiment root.
Keep `target/chpc-contextual` intact locally: its trainer hash is the one bound
to these checkpoints, and later working-tree guard changes intentionally have a
different hash. Inspect logs after any
failure before relaunching. Stop only this experiment's own detached run if
needed; never release the owner's allocation.

## Frozen evaluation

`eval_next_word_lm` loads one checkpoint and one fixed ranking policy. It has no
optimizer, sweep, or selection step. It verifies train/validation lineage and
allows validation or test from the corresponding dataset, rejecting train.

```sh
python -m tools.autosuggest.eval_next_word_lm \
  --model artifacts/ngram.bin --checkpoint checkpoints/best.pt \
  --corpus-dir data/corpus/test --device cuda \
  --pool-k 64 --rank-penalty 0.5 \
  --output-report reports/test.json
```

Freeze the checkpoint, source limits, and policy before looking at test results.
Report n-gram, neural, and hybrid scores for each source, all-target vocabulary
coverage, and candidate recall. A poor test result remains a result; improve a
subsequent version using training/validation evidence, not test-set tuning.

The ONNX/Core ML exporter verifies checkpoint and corpus lineage and permits only
validation for its policy comparisons. Legacy workflows are labeled unverified.
The separate package/policy tools still need lineage integration.

Before distribution, complete that package integration, verify
portable inference parity, and measure real supported-device keyboard latency,
peak memory, cancellation, and fallback behavior. Export success or a fast GPU
training run is not evidence of those properties.

## Transformer comparison

The second run is `obadh-v2-transformer256`, rooted remotely at
`/scratch/general/vast/garcia/u1523034/train/obadh-contextual-transformer-20261005`.
Its local frozen trainer snapshot is `target/chpc-transformer`, and downloaded
results belong in `target/chpc-transformer-results-20261005`. It references the
first experiment's immutable dataset and retrieval files rather than duplicating
or changing them.

The model has two transformer encoder layers, four attention heads, 256-wide tied
embeddings, a 1,024-wide feed-forward block, and 10,005,504 parameters. It encodes
only known preceding tokens; bidirectional attention within that prefix exposes
no target/future words. The 16-token context is deliberately held equal to the
GRU and current shared-engine contract. A longer window requires a separately
versioned runtime/model comparison.

Training retains the same source limits, six epochs, batch size 1,024, validation
selection metric, and frozen test policy. AdamW uses learning rate 0.0005 and
maximum gradient norm 1.0; non-finite gradients fail before an optimizer update.
This is a comparison of declared training profiles, not a claim that every
hyperparameter has been optimized for either architecture.

Padding behavior is part of checkpoint config: `masked-v1` masks padding keys,
with one visible final slot for an empty context to avoid fully masked attention.
`unmasked-v0` preserves historical checkpoints. Loaders retain the recorded mode;
a training resume cannot silently change it. ONNX/Core ML export uses portable
attention operations and restores PyTorch's prior backend settings afterward.
Empty and nonempty context parity is covered by ONNX tests and a full-size Core ML
conversion smoke test. Trained-model export must still be checked separately.

The GRU's archived trainer predates this transformer padding version and later
CLI guard improvements. Continue it using its preserved snapshot, not the newer
working-tree trainer: checkpoint contracts deliberately reject code drift.

## Local Apple export environment

Use the tested lock file on macOS arm64 with Python 3.12:

```sh
uv run --python 3.12 \
  --with-requirements tools/autosuggest/requirements-export-macos.lock \
  python -m tools.autosuggest.export_next_word_lm --help
```

This pins PyTorch 2.6.0, Core ML Tools 9.0, and NumPy 1.26.4, plus their tested
export dependencies. The A100 training environment is the separate CHPC container.
Core ML Tools 9.0 hit the documented [NumPy scalar conversion issue](https://github.com/apple/coremltools/issues/2633)
in transformer attention export with the newer NumPy environment; the pinned
export environment passed conversion without modifying third-party packages.

Export reports record checkpoint identity, validation lineage, numerical error,
artifact size, and warm mean prediction time. These are model-call measurements,
not end-to-end keyboard latency, cold-start measurements, memory ceilings, or
physical-iPhone results. Quantized graphs require their own quality evaluation;
small numerical errors alone do not certify unchanged candidate rankings.

## Native Apple runtime measurements

`apple/benchmark.swift` loads only Core ML and system frameworks. Compilation and
prediction are separate processes. This avoids reporting the Python/PyTorch
export process's memory as deployment memory. The exporter can create numeric
fixtures alongside its graph with `--native-fixtures PATH`; they carry vocabulary,
checkpoint, and retrieval identities. Use the compiled graph from that export.
The benchmark records fixture identity and the supplied compiled-model path;
these fields alone are not a cryptographic assertion that an arbitrary supplied
graph belongs to the fixture's checkpoint.

```sh
swiftc -warnings-as-errors -O tools/autosuggest/apple/benchmark.swift \
  -o target/coreml-native-benchmark

target/coreml-native-benchmark compile PATH/scorer.mlpackage PATH/scorer.mlmodelc

target/coreml-native-benchmark benchmark \
  PATH/scorer.mlmodelc PATH/native-fixtures.json all 2000

target/coreml-native-benchmark benchmark \
  PATH/scorer.mlmodelc PATH/native-fixtures.json cpu 2000
```

The compiler refuses an existing destination. Benchmark input dimensions, integer
types, token bounds, output dimensions, and finite scores are checked. Prediction
latencies exclude feature preparation and output validation. The JSON reports
p50/p95/p99, model initialization, first prediction, resident memory, physical
footprint, and kernel lifetime peaks. The process baseline and post-feature
snapshots help distinguish model/framework costs from fixture allocations.

A fresh process is not necessarily a cold OS/runtime cache. Neither caches nor
system model services are reset by this tool. The first GRU process runs showed
higher resident-memory peaks than later repeats; retain the initial observations
and the repeat reports, and do not select only the smallest memory result.
Physical footprint and resident memory are different measures, and neither a
macOS process nor an isolated graph is the actual iOS keyboard extension.

## Controlled transformer initialization comparison

A second transformer profile uses `independent-v1` initialization. Inspection of
the inherited profile found initial position-embedding standard deviation 0.992
versus 0.020 for token embeddings, and identical initial attention-layer weights.
That observation motivates an ablation, not a claim that initialization alone
causes any measured quality difference.

`independent-v1` initializes position embeddings and each layer's attention and
feed-forward weights independently at standard deviation 0.02, with zero linear
biases. Architecture, data, token order, optimizer settings, epoch count, validation
metric, and fixed ranking policy otherwise remain the same. Constructors/loaders
preserve `legacy-v0` for older checkpoints; the initialization ID is part of new
training contracts. The two profiles have the same inference graph and parameter
count once their weights are loaded.

Its run is `obadh-v2-transformer256-init`, remote project root
`/scratch/general/vast/garcia/u1523034/train/obadh-contextual-transformer-init-20261005`,
local source snapshot `target/chpc-transformer-init`, and result destination
`target/chpc-transformer-init-results-20261005`. Select the transformer profile
and epoch by validation macro top-5 all-target accuracy before running its frozen
test evaluation. Do not run additional tuning from test results.


## Expanded iPhone 11 investigation

The product floor is iPhone 11 (A13), iOS 18 in the current app. Newer Apple
Intelligence hardware is not a baseline dependency. The data and portable model
logic remain in this repository; Apple hosts own Core ML execution and lifecycle.

`subword_lm.py prepare` builds a byte-level BPE tokenizer using verified training
partitions only, and writes uint16 token streams with SHA-256 receipts. Validation
never trains the tokenizer. Current configuration: 8,192 tokens, NFC, explicit
BOS/EOS between packed sentences. Corpus preparation retains the existing
Bangla-only text view; UTF-8 coverage does not imply English/code-mixed training.
`subword_lm.py train` implements a small causal transformer with rotary positions,
gated feed-forward blocks and tied embeddings. It uses bf16 CUDA training,
AdamW, warmup/cosine decay, norm clipping, deterministic sampling, atomic five-minute
checkpoints, a fixed validation sample and strict resume contracts. Model selection
uses validation token loss; word-level evaluation is separate. The schedule's
step count is immutable on resume. `--max-steps-this-run` caps additional steps for
smoke tests; an absolute `--stop-epoch` is mandatory. Save/reload and GPU-resume
smokes must precede each architecture's full run.

`teacher_candidates.py` produces immutable candidate banks from train or validation
only. It chooses one target per sentence deterministically. Training can insert a
missing gold candidate; validation never does. Score caches bind the bank, model
files, revision label, scoring code and runtime. They commit each context to SQLite
and skip completed IDs on resume. Complete-word scoring adds the probability of
a following word boundary to the candidate's subword log likelihood. The raw and
length-normalized scores remain available for diagnosis; do not assume a larger
teacher is better before evaluating it. A teacher's external pretraining overlap
with public evaluation text remains unknown. No user typing is sent to a model
service: these commands process offline public corpus fixtures on the allocated
GPU.

The CHPC container's existing Transformers 4.51.3 supports Qwen3. Installing
4.57.6 into a separate environment exposed an incompatibility with this container's
pre-release PyTorch 2.6 (`TransformGetItemToIndex` import failure). Teacher inference
therefore uses the original container Python. The separate download environment
is used only for fetching pinned Hugging Face snapshots. Never change the shared
container or the user's allocation to fix an experiment.

`compress_coreml.py` compares immutable candidate-scorer packages using exported
validation fixtures. Linear 8-bit quantization and 4/8-bit k-means palettization
are independent experiments. Reports record model inventories, fixture hash,
ordered ranking agreement and, when labels are supplied, subset accuracy. File
size is not resident memory. A label-bearing export fixture remains a selected
known-vocabulary validation subset, not a full all-target benchmark.

The iOS DEBUG-only `ContextualModelProbe` can load a compiled model and exported
fixtures in the actual keyboard process. Its command is `modelprobe:cpu` or
`modelprobe:all`; `modelprobe:unload` releases the retained model. Put
`model.mlmodelc` and `fixtures.json` under `obadh-debug/model-probe` beside the
existing debug channel. Compile with `xcrun coremlcompiler compile PACKAGE OUT
--platform iOS --deployment-target 18.0`. Reports identify Simulator explicitly.
No model probe or command is included in Release. Model-call timings exclude
input preparation; the probe does not replace suggestions or change user text.

## Frozen external typing evaluation and cache exports

The Unicode-aware tokenizer profile is `unicode-marks-v1`. It groups Unicode
letters, combining marks and joiners before byte BPE. The legacy
`gpt2-regex-v0` profile remains explicit for reproducibility. On the same bounded
training text, the corrected profile reduced token count from 125,529,312 to
44,540,499; this is a tokenizer efficiency result, not a word-accuracy claim.

`eval_subword_candidates.py` evaluates complete-word probability on a verified
validation bank. `decode_subword_words.py` adds bounded open-vocabulary beam
search with strict UTF-8 lexical output and an explicit word-boundary probability.
It is a reference evaluator, not a shipping keyboard decoder. The default is
16 beams and at most 12 word pieces; the current reference recomputes context.

`external_typing_benchmark.py prepare` builds a test-only comment bank after
normalized deduplication and exact whole-comment overlap checks against verified
training sentences. It preserves Latin words and never treats noisy comment
references as correction labels. `evaluate` requires a frozen selection receipt
binding both checkpoints, tokenizer, test bank, and both retrieval artifacts.
It refuses to overwrite a report and checks training lineage. Word-vocabulary
OOVs and retrieval misses stay in the denominator. Six comparison methods and
the subword decoder budget were recorded before this experiment's evaluation.
Do not use its scores for tuning and call the same data a fresh test again.

`export_subword_lm.py` exports a last-valid-position scorer; it avoids projecting
all context positions through the vocabulary head. `subword_cache.py` and
`export_subword_cache.py` provide explicit-cache prefill and single-token graphs
for ONNX and Core ML. Their host contract is:

- Cache identity includes the exact token prefix and model version. Reset after
  edits, cursor discontinuities, field changes or model replacement.
- Validate `0 <= position < capacity` before calling the graph. Do not wrap
  positions; prefill a truncated context when full.
- Mask future slots even when buffers contain stale values. Each beam owns an
  independent logical cache; shared-prefix storage is a host optimization.
- Account for cache buffers, both graph packages, transient activations and the
  rest of the keyboard process. Weight-file size alone is not a memory budget.

The final large-model cache export verified 288 prefix/step outputs. ONNX FP32
had full top-one agreement; Core ML INT8 with FP16 external caches had 98.26%.
That does not establish compressed word-decoder quality. Generic dynamic ONNX
INT8 lost substantial agreement and was not admitted for deployment. Use the
verified FP32 graph as the portable reference while investigating backend-specific
quantization.

All completed runs, frozen source snapshots, data receipts, checkpoints, exports,
reports and logs are preserved outside `target` at
`/Users/nsssayom/Dev/obadh-model-runs/2026-10-05-contextual-v2`.
Its README contains exact resume commands. Source formatting after evaluation
changes strict trainer code hashes: resume with the frozen per-run source,
not an arbitrary current checkout. `evaluated-tooling` preserves the exact
unformatted evaluator bytes named by the selection receipt; this repository
contains the formatted equivalent, with tests rerun.

Current evidence supports further contextual prediction development. A separate
reviewed spelling/grammar benchmark, calibrated no-edit policy, production host
integration and physical iPhone 11 performance tests remain required before
shipping. The fresh comments are a narrow external check, not a representative
measurement of every typing domain.

## October 5: diversified literary adaptation on DeltaAI

The next corpus adds 74 successfully acquired books from an 80-book plan.
Six books failed source-integrity checks and were not admitted. Together with
the original eight website books and 12 usable EPUBs, this produces 94 works.
`Prajapati.epub` is excluded from this new view because its embedded notice
identifies unreviewed OCR and it lacks creator metadata. Earlier datasets remain
immutable. Source plans, author URLs, aliases, and exclusion reasons live in
`tools/corpus/sources/`.

`tools.corpus.contextual_books` preserves NFC text, punctuation, paragraph
breaks, mixed scripts, joiners, digits and short replies. It retains original
work assignments and reserves five additional authors each for validation and
test before training. Reviewed name aliases and shared contributor lists are
checked against those holdouts. Long matching passages across partitions are
quarantined; this is not semantic duplicate detection. Common short replies
are retained. Every text window carries its source reference and byte digest.

`tools.autosuggest.literary_data` prepares document-bounded sequences with
masked padding. Coauthors and translators form connected author groups. Training
uses square-root group sizes with a 5% ceiling on expected valid-token exposure
within the literary branch, compensating for short final blocks. The current
training view has 66 groups and 9,223,549 nonpadding target tokens; the largest
group is 4.0255%. These are sampling expectations, not hard per-minibatch quotas.
The test partition is never tokenized for training.

The controlled adaptation experiments retain the 17.3M architecture and 8K
tokenizer, initialize from the selected v2 weights, and start a fresh optimizer.
They compare 25% and 50% literary examples against continued training on the
original stream, each with 18,000 steps. The remaining examples retain the old
corpus view; punctuation preservation currently applies to the literary branch.
`--initialize-from` is distinct from exact `--resume`. Resume requires the same
initialization argument, data receipts, settings and frozen source bytes.

`compare_subword_models.py` evaluates the fixed decoding budget. Reuse of the
already-observed comment bank is explicitly a development diagnostic, not a new
test. `literary_validation_bank.py` freezes punctuation-preserving continuation
examples from validation books. `evaluate_literary.py` reports teacher-forced
token and punctuation-token metrics; neither is grammar or autocorrect accuracy.

Current artifacts and frozen commands are outside build output at
`/Users/nsssayom/Dev/obadh-model-runs/2026-10-05-deltaai`.
DeltaAI commands must set `CHPC_GPU_SITE=deltaai`. The ARM container needs an
explicit bind of `/work/nvme/bicv/nsayom/train` when a run reads sibling project
directories. The allocated GPU must never be acquired or released by these
training scripts. Record completion, back up checkpoints and logs, and notify
the owner as soon as it is idle.

### Byte-level accuracy reference

`tools.correction.byte_data` and `tools.correction.byte_train` provide an offline
ByT5-small comparison before further mobile optimization. The approximately
300M-parameter reference is not an admitted keyboard artifact. It combines
existing work-separated sentence pairs, filtered BanglaSEC lexical pairs, and
separately identified Gemma-approved grammar examples. Sentence, typing-prefix,
and isolated-word tasks use explicit prefixes; correct inputs remain supervised
identity examples.

External spelling admission excludes source ambiguity, valid-word collisions,
run-on/deletion and split-word categories, and evaluation vocabulary. Lexical
validation groups variants by their target word; this does not assert that those
words were absent from multilingual pretraining or the sentence corpus. Existing
sentence validation remains separate. Every input file and builder is hashed;
capacity exclusions and filtering counts are recorded. The byte limit includes
the task prefix and EOS, and overlong data are excluded rather than truncated.

The trainer uses FP32 master weights with BF16 CUDA autocast, finite-gradient
checks, atomic five-minute checkpoints, and optimizer/RNG recovery. Resume rejects
changed data, model files, code, or schedule. Inference decodes UTF-8 strictly,
requires EOS, and reports raw outputs separately from protected-span fallbacks.
Evaluations use the existing case identities and comparison code; synthetic
repair rates are not native typing accuracy. No Core ML export or production
admission is implied by an accuracy-reference run.

`--edit-weight` makes correction supervision an explicit experimental factor.
The default is ordinary cross-entropy. A higher value weights all UTF-8 bytes of
edited Unicode characters; deletions weight the following retained character or
EOS. Unchanged characters and identity examples retain unit weight. This changes
the training objective and its validation loss scale, so compare decoded repair
and preservation metrics rather than comparing cross-entropy between objectives.

### Contextual spelling and acceptance experiments

`contextual_data.py` inserts filtered lexical errors into existing training
sentences and prefixes, preserving punctuation and parent identities. It excludes
known development error pairs, held lexical words, conflicting targets, and
validation texts. Clean controls include source sentences and explicitly authored
preservation examples. This relaxes the earlier blanket vocabulary exclusion:
ordinary words may appear in other training contexts, while declared validation
partitions and known benchmark error pairs remain reserved.

`--initialize-from` starts a new, recorded training schedule from compatible
weights. It differs from `--resume`, which restores the exact schedule, optimizer
and RNG state. `--family-weights`, `--initial-identity` and `--final-identity` are
checkpoint-contract fields. Compare a matched continuation control before
attributing gains to contextual augmentation and its sampling mix.

`contextual_bank.py` constructs an additional diagnostic from held sentence
contexts and the external source's held word-hash partition. Each spelling error
has a clean control. Reference/source texts used by other evaluation or acceptance
banks are excluded. These are synthetic contextual errors, not natural typing
records or a test of pretraining vocabulary exposure.

`acceptance.py` learns a small logistic acceptance model from generator
likelihoods and edit features. It is not a second semantic language model. Its
features never read reference text; references supply exact-match labels only.
Generator checkpoint hashes bind all features. Training, calibration and
evaluation must have disjoint source/reference texts. A separate calibration set
selects a proposal F0.5 threshold; its scores are not certified user-facing
probabilities. Policy evaluations retain the original generator proposal and
report the text actually accepted. This stage can reject harmful edits but cannot
recover corrections the generator never proposed.

`acceptance_lexical.py` optionally adds edited-word frequency and vocabulary
features. Its vocabulary uses deduplicated training sentence identities and
lexical training words; validation/evaluation references are excluded. Vocabulary
hashes and feature schemas must match across fitting, calibration and evaluation.
Fitted policy identity records training/calibration evidence separately from
evaluation receipts, so applying it to another bank does not change the model.

Grammar expansion uses `teacher_roundtrip --source-offset` to continue with new
distinct training references. `byte_teacher_data.py` checks source lineage, blind
recovery, critique, conservative admission, held-out texts and conflicting labels
before adding those pairs to the contextual byte dataset. Validation files remain
byte-for-byte unchanged. Source offsets and all generation/admission evidence are
recorded. Frozen projects and automatic generation/training/evaluation/backup
queues are archived under `2026-10-05-deltaai`; their status JSON files distinguish
queued, running, completed and blocked work. They never release the allocation.

`select_initialization.py` compares candidate generators on the same calibration
bank before a continuation run. Its exact-reference correction F0.5 counts missed
errors as well as wrong edits, including changes to clean inputs. The selection
receipt records the input/checkpoint hashes and scores; evaluation rows are not
passed to the selector. This criterion is distinct from the acceptance model's
proposal-only calibration objective.

### Complementary proposal selection and the next grammar policy

`tools.correction.acceptance_union` fits an offline two-generator selector from
paired scored training proposals. It merges duplicate candidates, records which
generator supports each candidate, and calibrates a sentence-level exact-reference
F0.5 threshold on a separate bank. Errors with no available repair remain in the
recall denominator. Its `evaluate` command applies a frozen artifact, verifies
checkpoint and vocabulary identities, rejects training/calibration text overlap,
and never refits from evaluation labels. This is an accuracy diagnostic for
possible future distillation, not a two-model mobile deployment proposal.

The frozen `obadh-acceptance-union-v1-20261006` experiment improved synthetic
restoration from 530 to 544 of 985 relative to the contextual lexical gate, but
reduced natural-text GLEU from 59.9587 to 59.6933. On the separate contextual bank,
it repaired 57/171 errors and preserved 168/171 clean inputs, versus 61 and 167
for that single-generator gate. The tradeoff does not justify promotion.

`acceptance fit --calibration-objective sentence-f05` now counts every erroneous
calibration sentence, including errors the generator never proposes to fix.
`proposal-f05` remains the default for reproduction of earlier experiments.
Neither objective measures real-user accuracy or establishes a safe automatic
replacement threshold under actual keyboard traffic prevalence.

`obadh-grammar-acceptance-v4-20261006` is queued after the grammar run and verified
checkpoint backup. Its predeclared candidate is step 48000; it does not pick a
checkpoint by evaluation score. It runs a CUDA feature smoke, generates proposals
for the existing disjoint 5000/1500 training/calibration banks, uses the frozen
training-only lexical vocabulary, fits sentence-level acceptance, and evaluates
both existing development banks. Local `queue-grammar-acceptance-v4.py` handles
serial execution, milestone pulls, comparison, and a local completion notification.
The original grammar run still evaluates and saves all five scheduled checkpoints.
The pipeline does not release the user's GPU allocation.

### Persistent GPU monitoring

`tools/correction/training_monitor.py` is a read-only, standard-library monitor
scheduled every 60 seconds by the local LaunchAgent
`com.obadh.training-monitor.deltaai`. Its frozen operational copy and config live
in `obadh-model-runs/2026-10-05-deltaai/training-monitor/`. `status.json` contains
an atomic heartbeat, active Slurm steps, latest progress, pipeline status and a
timestamped GPU utilization reading. `events.jsonl` records status transitions.
Successful checks exit normally; launchd starts the next check independently of
terminals. Add future pipeline status files to `config.json`; active jobs whose
names begin with `obadh-` are discovered through the owner's `chpc-gpu ps` wrapper.

Local macOS notifications cover completion/idle allocation, failed pipelines,
monitor errors, and 15 minutes of unchanged progress or idle unfinished work.
Unresolved alerts repeat every 30 minutes. Notification display depends on local
macOS settings. A missing SSH master latches the monitor without retrying login;
after the owner restores password/Duo authentication, invoke the monitor with
`--reset-connection --config PATH` to clear the latch. No monitor operation starts
training or acquires/releases the allocation. The paired `.awake` LaunchAgent
prevents idle sleep only until the configured allocation deadline. Shutdown or
loss of connectivity can still interrupt observation; the checked timestamp is
part of the status contract. Remote polling stops at the deadline.

The preservation-v5 pipeline completed both 6000-update arms and verified six
checkpoint backups. Calibration selected `mined-6000`. Against the previous
48k grammar checkpoint, held contextual repairs changed 95→87/171 while clean
preservation improved 162→166/171. Primary synthetic repairs changed 710→704/985,
clean preservation improved 427→434/460, and guarded natural-text GLEU changed
59.2110→59.2946. This is a precision/recall tradeoff, not a release admission.

### Preference supervision and label-quality audit

The v7 experiment trained matched SFT and edit-weighted preference arms from the
v5 checkpoint on 5,903 training-only pairs. Each arm saved steps 100, 300 and 600;
all six checkpoint backups were hash-verified. Calibration selected preference
step 100. Relative to v5, primary synthetic repairs fell 704→677/985 and clean
preservation improved 434→437/460. Contextual repairs fell 87→79/171 and clean
preservation improved 166→169/171. Natural-text guarded GLEU rose 59.2946→59.7786,
still below unchanged input at 60.0927. These reused development banks do not
establish real-user accuracy. No model was promoted to the keyboard.

`tools.correction.preference_audit` audits the original training preferences with
Gemma 4 31B IT. Each pair receives two blinded judgments with reversed candidate
ordering; the initial ordering is hash-randomized. Mode-specific instructions
preserve unfinished prefixes, dialect, facts and optional punctuation. Outcomes
separate confirmed preferences, contradictions, acceptable alternatives, neither
acceptable, uncertainty, invalid responses and ordering disagreements. Teacher
agreement is a label-noise diagnostic, not native-speaker gold. The audit does not
invert labels, train on evaluation references or change benchmark scoring.

The frozen v8 project is `obadh-preference-audit-v8-20261006` under the shared
model-run archive and remote training workdir. It fsyncs rows and atomically
publishes a hashed report every batch. Resume verifies code, model revision,
input identity, ordering and parsed decisions; only an incomplete JSONL tail is
truncated. The local watcher backs up results every minute. Resume after reading
the failed log and checking for competing GPU workloads with:

```sh
export CHPC_GPU_SITE=deltaai
~/Dev/chpc-agentic-llm/chpc-gpu start obadh-preference-audit-v8 -- \
  bash obadh-preference-audit-v8-20261006/python.sh \
  -m tools.correction.preference_audit --data data \
  --model-root ../obadh-gemma4-20261005 --output audit --batch 8 \
  --stop-epoch 1791388800
```

### Accuracy work: experimental and release rules

The v9 experiment tests a specific hypothesis: removing disputed training
preferences improves repair coverage without worsening preservation. Its frozen
`experiment-plan.json` predates the completed v8 audit. The filtered arm keeps
confirmed repairs, and keeps both confirmed and both-acceptable preservation
preferences. Keeping valid input unchanged remains desirable even when an
alternative rewrite is also grammatical. No rejected label becomes a new target.
A hash-random control has identical repair/preservation counts. Both arms start
from v5 step 6000, use the same seed, effective batch 32, 600 updates, optimizer,
learning rate and decoding. Step 600 is the predeclared comparison endpoint;
steps 100/300/600 are separately available for calibration-only selection.

`select_initialization --baseline NAME` requires at least as many correct repairs
and preserved clean inputs, and no more incorrect edits, than that baseline on
the common calibration bank. It then maximizes repairs, preservation, and fewer
incorrect edits, retaining the baseline on exact ties. Legacy selection without
`--baseline` retains the previous F0.5 behavior for reproducibility. This is an
experimental admission rule, not statistical proof or a deployment gate. A single
training seed and random subset cannot isolate every difficulty/coverage effect.

The local `queue-audited-preference-v9.py` waits for the completed, hash-verified
audit backup, builds both datasets, runs short CUDA save/resume checks, starts
detached training and verifies all six milestone backups. It evaluates both
predeclared endpoints regardless of selection, plus the selected checkpoint.
The monitor observes queue and training failures. The queue never releases the
allocation. Read a failed log before restarting; do not overlap GPU workloads.
Resume the frozen remote pipeline after successful smoke checks using
`chpc-gpu start obadh-audited-preference-v9 -- bash
obadh-audited-preference-v9-20261006/python.sh pipeline.py`, with
`CHPC_GPU_SITE=deltaai`. Restore its local backup watcher as well. Re-running the
local queue verifies existing preparation and checkpoints; do not run it while
its named GPU workload is already active.

The progression toward production has four distinct evidence requirements:

1. **Training validity.** Keep source/work partitions and frozen manifests;
   separate required repairs, valid variants, and optional rewrites. Teacher
   outputs and synthetic corruptions remain explicitly labeled. Repeat a
   successful matched experiment with another seed before attributing a robust
   gain to the filtering rule. Do not expand books merely to increase token count
   while label quality remains unresolved.
2. **Independent linguistic evaluation.** Collect consented real typing or
   explicitly authored typing tasks covering dialogue, literary and contemporary
   prose, punctuation, prefixes, spelling, agreement, mixed Bangla/Latin text,
   names, numbers and dialect/register. Record provenance and context. Have two
   native Bangla reviewers independently mark required repairs, valid alternatives
   and ambiguous cases, with adjudication of disagreements. Annotators must not
   see model identities or teacher judgments. Store multiple acceptable outputs
   where appropriate. Do not label model-generated or existing synthetic examples
   as natural user gold. Training/calibration and final-test source groups must
   remain disjoint. Freeze the final set and admission criteria before scoring a
   release candidate; after inspecting failures, retire that set to development.
3. **Keyboard utility and safety.** Evaluate automatic replacement separately
   from suggestions. Report correct repairs, incorrect changes, clean-input
   changes, suggestion top-k coverage, and punctuation/register-specific results;
   include unchanged input and the current shipping engine. Use paired source-
   group uncertainty estimates. The intended initial automatic-correction target
   is a one-sided 95% upper bound of 0.1% for unwanted changes on independently
   reviewed clean inputs, with a positive repair gain over the shipping engine.
   This is a proposed product target, not an achieved result or a literature
   standard. Zero failures would require roughly 3,000 independent clean examples
   to support that bound; repeated variants of the same source do not qualify as
   independent. Suggestions may use a different calibrated operating point.
4. **Deployment parity.** Distill/compress only after a repeatable accuracy gain.
   Re-run linguistic and protected-span checks on the actual exported runtime.
   Measure peak memory, cold/warm latency, cancellation, sustained typing and
   thermal behavior in the iPhone 11 keyboard process. Set a measured memory and
   latency budget from that host before choosing the final model architecture.
   Simulator checks cover functionality; they cannot establish A13 performance.

The independent human-reviewed release bank and physical-device performance
evidence do not yet exist in these experiment artifacts. All current numerical
accuracy comparisons remain development diagnostics. No v7/v8/v9 artifact is a
production model admission.

### v9 outcome and semantic candidate selection diagnostic

The matched v9 experiment completed and all six milestone checkpoints were
backed up with verified hashes. Both arms contained 4,695 pairs (1,949 repair,
2,746 preservation). Filtering improved repairs relative to its matched random
control: primary synthetic 682 vs 672/985 and contextual 82 vs 75/171. Both still
trailed the v5 baseline at 704 and 87. Filtered clean preservation was 437/460
and 169/171, compared with baseline 434 and 166. No checkpoint passed the
predeclared calibration nonregression rule; the baseline remains selected.
The evidence does not support further repeating this preference schedule as the
main route to more repair coverage.

The v10 diagnostic instead tests reference-blind selection among the already
generated v5 candidates. On the existing calibration bank, beam top-1 repairs
497/732 errors, preserves 745/768 clean inputs and makes 74 incorrect changes.
The reference appears somewhere in the candidate set for 626/732 errors; that
number is an oracle ceiling, not deployable accuracy. `teacher_rerank.py` presents
only source text, mode and fixed candidates to Gemma 4 31B IT, in a hash-random
ordering and its reverse. Reference targets and case labels never enter prompts.
The two preregistered policies use either the first judgment or require agreement
between orderings, otherwise preserving the original. Invalid choices also
preserve input, and unsafe protected-span edits are excluded before judging.

The frozen `obadh-teacher-rerank-v10-20261006/pipeline.py` first evaluates all 1,500
calibration cases. Only a policy with strictly more repairs than beam top-1, no
fewer preserved clean inputs and no more incorrect changes triggers collection
of teacher decisions on the disjoint 5,000 training cases. No model weights are
trained in this diagnostic, and calibration decisions cannot become training
labels. The offline teacher is not an iPhone deployment proposal. Its purpose is
to establish whether semantic selection provides a useful distillation target.

Rows are fsynced every batch, contracts pin inputs/model/code, and local backups
run each minute via `watch-teacher-rerank-v10.py`. After inspecting any failure,
resume with `CHPC_GPU_SITE=deltaai` and `chpc-gpu start obadh-teacher-rerank-v10 --
bash obadh-teacher-rerank-v10-20261006/python.sh pipeline.py`, and restart the local
watcher if needed. Neither script releases the GPU allocation.
