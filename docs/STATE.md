# MOUTHPIECE state

Updated 2026-11-25. Plan: `docs/PLAN.md` (slice IDs S0.x–S5.x). Coordinator keeps this file current.

## Done

- S0.1 snapshot branch `wip/snapshot-2026-11-24`; DB and corpus CSVs backed up off-repo.
- S0.2a–d + S0.4 tests + S0.12 normalised words: #2, #5. S0.12 applied to the real DB.
- S0.3 MIT licence: #3. S0.5: `allow_no_ci = true`; pytest gate on (owner, 2026-11-25).
- Website without Churchill, licence wording fixed: #6.
- S0.6 seeded reroll: #7. S0.7 attribution fields + `speaker_label`: #8.
- S0.9 manifest v2 + `mouthpiece replay`: #11 (57d8bde).
- S0.8 attribution backfill tool + `corpus/attribution.csv`: #10 (9c46313). **Applied to the real DB 2026-11-25**: 34 rows; 0 email speakers, 0 speaker=author, French source `lang='fr'`.
- S0.10 harvester: per-adapter licence + tier, no default, yt-dlp removed, no email written: #12 (86cbaac).

## In flight

- none

## Next, in order

1. S0.11 README rewrite (needs S0.9: done).
2. Fix: swapping one member of a phrase via `/swap` is silently ignored (members keep `phrase_id`/`phrase_len`; `_group_phrase_units` still treats the run as one cut). Found in S0.9.
3. P1 (bench S1.1 first, then Viterbi, voice policy, picker, bias controls) in parallel with P3 (corpus: S3.1 register, S3.2 language check).

## Waiting on owner

- Churchill licence evidence, if Churchill should stay public (S3.1).
- `git pull` in the main checkout: it is on old code (its `mouthpiece sources` still prints "unknown speaker").

## Decisions

- Licence: MIT, copyright "Kosta Vukićević" (owner, 2026-11-24).
- No CI by owner decision; the local pytest gate replaces it.
- Owner-local untracked files (`corpus/batch*.csv` with an uploader email, `demo/`, `.devlog/`, `package-lock.json`) are excluded via `.git/info/exclude`. Never commit them.
- S0.8: NASA uploader email dropped, not moved. LibriVox reader only from archive.org metadata. `licence_url` blank everywhere until S3.1 verifies (owner chose A, 2026-11-25).
- S0.10 tiers: obama, nasa_reels, apollo11 `clean`; nasa collection, scotus, speeches, librivox `unverified`; trump, churchill `restricted`. nasa_reels/apollo11 kept clean: item metadata names NASA JSC as archivist (owner, 2026-11-25). librivox can go clean once its PD page is confirmed (it returned 522).

## Facts for builders

- Test gate: `sh -c '"$(git rev-parse --git-common-dir)/../.venv/bin/python" -m pytest -q'` (worktrees have no `.venv`). The worktree guard refuses that form inside builder sessions: run `<main>/.venv/bin/python -m pytest -q` directly. 72 pass at 86cbaac.
- Real DB `db/mouthpiece.db` is on the S0.7 schema now.
- CLI DB path: env `MOUTHPIECE_DB` (default `db/mouthpiece.db`).
- Indexed words are normalised (`index.normalize_word`) at every insert. Syllable/phone index code goes through `_UNITS` in index.py.
- pyrubberband `rate` = our `speed` (>1 faster). Output 44.1 kHz. StaticFiles mount stays last in `server.py`.
- `find_phrase` (match.py) ignores its `rng` argument. Candidate for S1.x.
- `harvest.py` runs `os.makedirs(OUT)` at import (creates `corpus/`).
- `.nonslop.toml` is hook-protected: only the owner edits it.
- `gh pr edit` fails (Projects classic GraphQL); use `gh api -X PATCH repos/stcksmsh/Mouthpiece/pulls/<n>`.
- A background coordinator cannot write in the main checkout; briefs go in the job scratch dir.
