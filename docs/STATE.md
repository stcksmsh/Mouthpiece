# MOUTHPIECE state

Updated 2026-11-25 (evening). Plan: `docs/PLAN.md`. Coordinator keeps this file current; run `nonslop status` first.

## Goal
Album instrument: speech collage with the patchwork feel (frequent source changes) but legible. Today's output is "meh"; only `--diction 1` was somewhat legible.

## Done (merged)
- P0: S0.1–S0.12 (#2–#13). Attribution backfill applied to real DB (0 email speakers).
- S1.1 bench `tools/bench.py` (#14). Baseline `renders/bench/ed5ab62-before/`: 286 words, 248 source switches, 14 stitched, 1 gap.
- S1.2 pure cost fns (#15). Phrase-swap fix (#16). S1.2b acoustic features + `mouthpiece features` (#17).
- S1.3 Viterbi `--selector viterbi --smooth --continuity` (#18, 05f7288); greedy still default.
- S3.5a LibriVox `clean`, new `voa` adapter, RSS size/entity guard (#19, 97af8c3).

## In flight
- S1.6a tier + source filter: PR #20, `claude/s1.6a-source-filter` @7d38587. Verify ok, review clean. **Waiting for owner "merge 20".**

## Next, in order (each heavy step needs owner go)
1. After #20 merges: apply tiers to real DB: `tools/backfill_attribution.py --db db/mouthpiece.db --csv corpus/attribution.csv --dry-run`, show owner, then without `--dry-run`. Light, but writes the DB: ask.
2. Obama harvester fix (`harvest.py`): archive links are relative (`/videos/YYYY/Month/…mp3`), pagination is `index__q_page=N.html`; adapter regex wants absolute URLs and `?page=N` → finds 0 items. Code-only builder slice.
3. Resume feature backfill (`.venv/bin/python -u -m mouthpiece features`, resumable): paused at 61,132/165,425 words, ~1 h CPU, ~2.4 GB RAM peak. **Owner go only.**
4. Listening comparison (after backfill): bench greedy / greedy `--diction 1` / viterbi / viterbi `--diction 1`, labels via `--label`; send 3–4 clips each + index.txt. Owner picks; default flips to viterbi only if he prefers it.
5. Then S1.4 edge context, S1.5 voice modes, S1.6b picker UI, S2.1 render persistence, S0.11 README.
6. Corpus growth: batch A agreed in size (≈20 Obama + 30 LibriVox + 20 VOA, "A or more") but **no harvest/ingest until owner says go**: disk `/secondary` 99% full (8.6 GB free), RAM/CPU tight.

## Waiting on owner
- merge #20; OK to apply tiers to real DB; go for backfill / listening / harvest when resources allow.
- Churchill licence evidence (stays `restricted` until then).

## Decisions
- MIT, copyright "Kosta Vukićević". No CI; pytest gate (`.nonslop.toml` test via `sh -c '"$(git rev-parse --git-common-dir)/../.venv/bin/python" -m pytest -q'`).
- Patchwork wanted: selection never penalises source/speaker change; penalise pitch/colour/tempo/effort jumps and bad cuts.
- `licence_url` blank in attribution until S3.1 verifies (A). Tiers: clean obama, librivox, nasa_reels, apollo11, voa; unverified nasa collection, scotus, speeches; restricted churchill, trump.
- Renders exclude `restricted` by default; `--tier all` to include (A).
- Every manifest records `controls.sources` (the filter), including default.
- Old `corpus/batch.csv` renamed to `corpus/batch_2026-11-24.csv` (old header, has an email; excluded via .git/info/exclude).

## Hard-won facts
- Builders cannot run the `sh -c … git rev-parse` gate (worktree guard): run `/secondary/Programming/Github/Mouthpiece/.venv/bin/python -m pytest -q` directly. `nonslop slice verify` runs the real gate fine. Suite ≈45 s at 7d38587.
- Verify needs branch ⊇ origin/master; have the builder `git merge origin/master` (no tests needed if disjoint).
- `gh pr ready` sometimes fails during GitHub incidents ("Something went wrong"); `gh pr edit` always fails (classic Projects): use `gh api -X PATCH repos/stcksmsh/Mouthpiece/pulls/<n>`.
- CLI DB path: env `MOUTHPIECE_DB`. Main checkout is the owner's working copy; `git pull --ff-only` after merges.
- Features: f0 pyin (Hz, voiced), f0_var semitones², rms dBFS (−100 floor, `rms IS NULL` = not analysed), tilt dB/oct. pyin has octave errors.
- Viterbi tempo = phones/s both sides (CMUdict); temperature 0.5 ⇒ smooth 0 samples p², sharper than greedy.
- Stale DIRTY worktrees from merged slices (S0.9, S0.10, S1.3 old) hold only caches/superseded edits; leave them.
