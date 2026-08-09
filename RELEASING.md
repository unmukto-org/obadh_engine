# Releasing obadh_engine

The release and branching process for `obadh_engine`. Follow it exactly. It exists
because the one time it was not followed, a feature branch cut from an out-of-date
local `main` swept two unrelated commits into a squash merge and left `main`
diverged from `origin`.

## Golden rules

1. **Branch only from an up-to-date `origin/main`.** Run `git fetch origin` and cut
   work branches from `origin/main`, never from a local `main` that may carry
   unpushed commits. A squash merge diffs your branch against `origin/main`, so
   anything on your local `main` that origin lacks gets bundled into the merge.
2. **Never strand a release-bound change.** A change meant for the next release
   lives in exactly one of two places: a commit pushed to `origin/main`, or an
   entry under `## [Unreleased]` in `CHANGELOG.md` backed by a pushed commit. Never
   leave it as a local-only commit or an uncommitted working-tree edit.
3. **Start every release, branch, and merge from a clean working tree.** `git
   status` is empty first.

## Two kinds of change

### A. Ship-now fix or feature (goes through a PR)

1. `git fetch origin && git switch -c <type>/<slug> origin/main`
2. Implement, add tests, add a `## [Unreleased]` entry to `CHANGELOG.md`.
3. Push, open a PR that closes the issue, let CI pass.
4. Squash-merge to `main` (admin override only if branch protection blocks a
   green PR).

### B. Change waiting for the next release

Small docs, terminology, or polish that should ride the next version:

1. `git fetch origin && git switch main && git pull --ff-only`.
2. Commit it straight to `main` with a `## [Unreleased]` CHANGELOG entry, and
   **push immediately**. Do not keep it local-only.

`## [Unreleased]` is the ledger of everything on `main` that has not shipped yet.
When a release is cut, that whole section becomes the new version's entry.

## Cutting a release

SemVer with the `0.x` caveat: a `0.x` minor may carry breaking changes; a patch
(`0.9.0` to `0.9.1`) is a bug fix or docs-only release.

1. `git fetch origin && git switch main && git reset --hard origin/main`, with a
   clean working tree, so you are exactly at origin.
2. Run the checklist from the README's Release Checklist section: `cargo test`,
   `cargo test --features cli`, `cargo test --features cabi`, the wasm32 check,
   `cargo bench --bench hot_path --no-run`, `cargo publish --dry-run`.
3. Bump the version together in `Cargo.toml`, `Cargo.lock` (run `cargo build` to
   refresh it), and `www/package.json`.
4. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z]` and add a fresh
   empty `## [Unreleased]` above it.
5. Regenerate the browser bundle with `./build.sh dist` (updates `docs/`), so the
   playground and the crate version agree.
6. Commit all of the above as a single commit titled `Release vX.Y.Z`.
7. `cargo publish` (crates.io versions are immutable; there is no undo, only yank).
8. Tag the exact published commit and push both:
   `git tag -a vX.Y.Z -m "..."` then `git push origin main vX.Y.Z`.
9. `gh release create vX.Y.Z --title "..." --notes-file <notes>`, notes drawn from
   the CHANGELOG entry.

Do not publish a behavior change that is data-dependent (frequency, ranking) until
the downstream keyboard (`obadh-ios`) has validated it on the real `bn.fst`. CI
runs without the artifacts and cannot test that class of change.

## The failure this prevents

`main` carried two unpushed local commits. A branch cut from that local `main` was
squash-merged onto `origin/main`, which did not have them, so the squash bundled
the two unrelated commits, and `main` then diverged from `origin`. Fetching first,
branching from `origin/main`, and never stranding release-bound work each avoid it
on their own; do all three.
