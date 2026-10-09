# Parked Bangla model research and recovery

Parked on 8 October 2026 (Denver time) at the owner's request. Training and
monitoring are inactive. The DeltaAI allocation ended on 7 October; no allocation
was acquired, released or cancelled during parking. No experiment has passed
production admission. This document supersedes historical active-run instructions
in [TRAINING.md](TRAINING.md) and the archived logs.

## Branches and archive

All corpus, neural completion, correction, evaluation and Apple probe work belongs
on `feature/contextual-typing` in the engine and iOS repositories. The engine's
normal checkout returns to `main`; the iOS checkout returns to
`feature/voice-typing`, whose voice implementation was already committed at
`42701b7606cf5fb02f9f0da35d7b8835b0e22ead`. No model work is merged into those branches.
The engine's pre-research base is `96a3bcc5839c51f75f5010283e60120af7ec9e03`.

The private archive is [nsssayom/obadh-model-runs](https://github.com/nsssayom/obadh-model-runs).
Its `manifest.json` is authoritative: `status: complete` means every compressed
Git LFS part was downloaded from GitHub and SHA-256 verified. Until that status,
the archive is in progress and local originals must be retained. After completion
and an independent restore check, the owner requests removal of local model-run
payloads to recover disk space. The small backup checkout and this documentation
remain. Do not assume `~/Dev/obadh-model-runs` still exists when resuming.

The archive covers all three original folders:

- `2026-10-05-contextual-v2`: A100 completion runs, checkpoints, input corpora,
  frozen code, Core ML exports, parity fixtures and native/simulator measurements.
- `2026-10-05-improvement-audit`: coverage diagnostics.
- `2026-10-05-deltaai`: diversified books, punctuation-preserving data, completion
  and correction experiments through v10, teacher decisions, manifests, logs,
  full frozen project sources, resumable optimizer/RNG checkpoints and monitors.

Restore with Git LFS, Python 3.12+ and `zstd` installed:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nsssayom/obadh-model-runs.git
cd obadh-model-runs
python3 restore.py --fetch --destination /path/with/space/obadh-model-runs
```

The restore utility checks compressed-part and source-file hashes, reconstructs
original paths and hard links, and deletes only its downloaded cache after each
batch passes. Use `--batch ID` to restore selected independent batches; inspect
`manifest.json` to find their file lists. A full restore needs the original tree's
space plus roughly 4 GiB of staging, rather than a second complete LFS copy.
Archive construction is resumable using its `archive.py`; its README gives the
exact command and failure recovery route. Never delete unverified originals.

## Selected artifacts

Paths below are relative to restored `2026-10-05-deltaai` unless stated otherwise.
All 37 receipt-pinned local checkpoint files passed the parking hash audit.

| Purpose | Artifact | SHA-256 |
| --- | --- | --- |
| Current accuracy reference, v5 mined step 6000 | `obadh-preservation-v5-20261006/mined/milestones/step-6000/latest.pt` | `163eeb164db1ef72dc935b2750a9a4895802970a5d3f75d455045c4086f150af` |
| Its v4 initialization, step 48000 | `obadh-byte-grammar-v4-20261006/milestones/step-48000/latest.pt` | `c49901d479adb7e3f89906b43dc85b56025ef86936b0525a7e63a56bc6a201cb` |
| Earlier subword completion best, in `2026-10-05-contextual-v2` | `obadh-subword-v2-20261005/checkpoints-large/best.pt` | `18c0aac663d056f5d38284a17694acdab3cfc03f3b516aa60d7eb2c3df6de4bd` |
| Earlier word completion best, same folder | `obadh-contextual-transformer-init-20261005/checkpoints/best.pt` | `688ed6b973d1ecfcb127f3855f45e64b8038a19880a3340092d59b67aaa55c4f` |

The v5 checkpoint is 3,595,894,699 bytes and includes optimizer and RNG state.
Its data is `obadh-preservation-v5-20261006/data`, identity
`19d039a1b82df7c6196aea3a2e4fe24166daa6154e3a85f8537582a93385dbaa`.
The v4 parent data identity is
`71952cde2c68d76604d5b7fbb0fafc04fed829793d4e3872b13c659660154796`.
The corpus, pair data, calibration/evaluation banks and source receipts remain in
the archive. The complete frozen v5 `tools` tree and `arguments.json` are required
for strict continuation; current branch sources are not interchangeable with them.

ByT5 pretrained weights and tokenizer/config files are archived in
`obadh-byte-accuracy-v1-20261005/pretrained`; upstream is `google/byt5-small` at
revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`. Gemma teacher weights were
remote-only and were not copied to this Mac. Re-download `google/gemma-4-31B-it`
at revision `842da3794eaa0b77d5f08bae87a17459d91ff475` with repository access
when a future experiment actually needs the teacher. Teacher predictions,
contracts and judgments are preserved locally/in the archive.

## Outcomes and limits

These are reused development diagnostics, mostly synthetic, not independent
human-reviewed typing accuracy. Completion and correction banks measure different
tasks; their percentages cannot be compared as a single accuracy score.

| Experiment | Measured outcome | Decision |
| --- | --- | --- |
| Earlier completion, 1000 topic-skewed native comments | top-3 exact next word: shipped n-gram 9.5%, word Transformer 12.3%, 17.3M subword Transformer 14.0% | Research improvement; insufficient for release |
| v4 grammar, 48000 steps | synthetic restoration 710/985; clean preservation 427/460; contextual repair 95/171, clean context 162/171 | Initialization retained |
| Selected v5 mined, 6000 steps | synthetic 704/985; clean 434/460; contextual repair 87/171, clean context 166/171; natural development GLEU 59.29458 vs unchanged input 60.0927 | Current accuracy reference |
| v6 candidate retrieval | calibration top-1 repairs 497/732, preserves 745/768 clean inputs, 74 incorrect changes; candidate oracle 626/732 | Oracle is a ceiling, not achieved accuracy |
| v7 preference and v8 audit | 5903 preference pairs; 4125 teacher-confirmed, 802 both acceptable, 339 contradicted, 581 order disagreements, 54 neither, 2 uncertain | Labels not automatically flipped; no production gain established |
| v9 matched filtered/control | 4695 pairs per arm; filtered synthetic 682 vs control 672, both below v5 704; filtered contextual 82 vs control 75, both below v5 87 | No checkpoint passed nonregression; baseline retained |
| v10 reference-blind Gemma reranking | single choice: 540 repairs, 704 clean preserved, 120 incorrect; ordering agreement: 520 repairs, 731 preserved, 64 incorrect; baseline: 497, 745, 74 | Both policies failed clean preservation; no training labels collected and no weights trained |

v10 completed all 1500 calibration cases. References were excluded from teacher
prompts. The decision and complete reports are in
`obadh-teacher-rerank-v10-20261006`; its rejected gate prevented the proposed
5000-case training-label collection. Repeating the same preference schedule is
not supported as the main accuracy improvement route.

The preservation audit verified 260/261 receipt checks and all 37 checkpoints.
One historical file was already absent: `grammar-v3-control/rows.jsonl`, expected
SHA-256 `df33a7bf6d46c87336bb0e80d205937a4294e45c6d5d5725e74af818d9a8c435`.
No duplicate was found locally. Its report survives, but that particular
comparison cannot be reproduced from its original predictions. Some older
per-project `data/manifest.json` paths are also absent; those projects used shared
inputs. Use the actual frozen arguments and manifests, not guessed directory
names. These gaps do not affect the selected v5 data and checkpoint recovery.

## Resume deliberately, after the owner resumes the project

1. Restore the archive and run its hash checks. Read `parking-status.json`, the
   selected project's arguments, receipts and completed `pipeline-status.json`.
   Treat archived PIDs and queue-status timestamps as historical, not live state.
2. A future allocation and authentication are required. Read
   `~/Dev/chpc-agentic-llm/docs/training-gpu.md`, set `CHPC_GPU_SITE=deltaai` in
   every shell, and use only `chpc-gpu` for remote operations. First run `info`
   and `ps`. Stop on a missing SSH master; the owner must authenticate with Duo.
   Never acquire/release/cancel the owner's allocation.
3. Restore project directories as siblings under `/work/nvme/bicv/nsayom/train`
   using `push`, including v5, v4 initialization, pretrained ByT5 and the required
   banks/data. The old ARM container was
   `/sw/user/NGC_containers/pytorch_26.07-py3.sif`, torch
   `2.13.0a0+9186a08b2c.nv26.07`; archived model contracts record Transformers
   `5.18.0`. The remote `obadh-teacher-venv` is not in this archive: recreate its
   ARM-compatible dependencies inside the container and verify imports. Do not
   reuse x86 wheels. Verify trainer/checkpoint contracts before claiming exact
   recovery with a changed runtime.
4. Re-evaluate selected weights before any new training. From the v5 project,
   construct the byte trainer invocation from `arguments.json`, replace output
   with a **new** directory, and append `--evaluate-checkpoint
   mined/milestones/step-6000/latest.pt --evaluate-output NEW_DIRECTORY`.
   Evaluation must not overwrite checkpoints or the historical reports. CUDA is
   required by this evaluator. The old wrapper points to a remote venv that must
   exist before use. Archive hashes permit inspecting weights/data without CUDA.
5. Completed schedules do not have unfinished steps. For a new accuracy
   experiment, declare a new output, schedule, data contract and cutoff, using
   `--initialize-from` for fresh optimizer state. Use `--resume` only for an
   interrupted run with the same schedule/code/data contract; do not silently
   increase `--steps`. Old pipeline scripts embed expired absolute deadlines.
   Preserve them as history and write a new coordinator rather than running old
   queues blindly.
6. Smoke-test GPU data loading, actual updates, checkpoint save/reload and resume
   before a detached `start` run. Checkpoint every 15 minutes, set a cutoff with
   time for evaluation/transfers, and pull and hash-check milestones. Only after
   new compute exists, restore/update a monitor for the new job and deadline.
   The old launch agents were unloaded and their plists archived under
   `training-monitor/parked-launch-agents`; do not re-enable them automatically.

The next useful accuracy stage needs an independently reviewed native Bangla
bank (including dialogue, punctuation, dialect, protected spans and mixed script)
and evidence that expanded proposals or selection improve repairs without clean
regression. Separate suggestion and automatic-replacement policies. Distillation
and compression follow a repeatable accuracy gain. Shipping then requires export
parity and physical iPhone 11/A13 keyboard memory, latency and sustained typing
measurements; simulator checks cannot establish those device limits. The detailed
proposed admission criteria are retained in [TRAINING.md](TRAINING.md).

Parking validation: engine ML tests ran 148 cases, 142 passed and 6 optional
backend tests skipped; iOS Swift packages ran 9 metrics and 177 keyboard-core
cases with zero failures. These verify software contracts, not model quality.
