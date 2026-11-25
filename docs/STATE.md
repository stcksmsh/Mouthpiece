# MOUTHPIECE state

Updated 2026-11-25. Plan: `docs/PLAN.md` (slice IDs S0.x–S5.x). Coordinator keeps this file current.

## Done

- S0.1 snapshot branch `wip/snapshot-2026-11-24`; DB and corpus CSVs backed up off-repo.
- S0.2a–d + S0.4 tests + S0.12 normalised words: #2, #5. S0.12 already applied to the real DB.
- S0.3 MIT licence: #3. S0.5 partial: `allow_no_ci = true`.
- Website without Churchill, licence wording fixed: #6.
- S0.6 seeded reroll: #7. S0.7 attribution fields + `speaker_label`: #8 (1dbd3a7).
- pytest gate on: `.nonslop.toml` `run = ["lint", "test"]` (owner, 2026-11-25).

## In flight

- S0.8 backfill attribution: `claude/s0.8-backfill-attribution` (builder). Tool + `corpus/attribution.csv`; never writes the real DB.
- S0.9 manifest v2 + `replay`: `claude/s0.9-manifest-v2` (builder).

## Next, in order

1. After S0.8 merges: owner approves the CSV, coordinator applies the tool to the real `db/mouthpiece.db` (backup exists).
2. S0.10 harvester licence honesty; S0.11 README (needs S0.9).
3. P1 (bench, Viterbi, voice policy, picker, bias controls) in parallel with P3 (corpus).

## Waiting on owner

- Churchill licence evidence, if Churchill should stay public (S3.1).

## Decisions

- Licence: MIT, copyright "Kosta Vukićević" (owner, 2026-11-24).
- Stacked chain #2→#4→#5 landed as one PR (owner, 2026-11-24).
- No CI by owner decision; the local pytest gate replaces it.
- Owner-local untracked files (`corpus/batch*.csv` with an uploader email, `demo/`, `.devlog/`, `package-lock.json`) are excluded via `.git/info/exclude`. Never commit them.
- S0.8: NASA uploader email is dropped, not moved to `uploader`. LibriVox reader only from archive.org metadata, never guessed (coordinator default per PLAN, 2026-11-25).

## Facts for builders

- Run tests: `.venv/bin/python -m pytest -q` (45 pass at 1dbd3a7). Fixture DB from `tests/conftest.py`.
- Real DB `db/mouthpiece.db` was still old-schema at 1dbd3a7; `index.get_connection` migrates in place on open.
- Indexed words are normalised (`index.normalize_word`) at every insert; `'hello'` → `hello`, `'em`/`goin'` kept.
- Syllable/phone index code goes through `_UNITS` in index.py; do not add per-kind copies.
- pyrubberband `rate` = our `speed` (>1 faster). Output 44.1 kHz. StaticFiles mount stays last in `server.py`.
- `find_phrase` (match.py) ignores its `rng` argument: phrase choice does not vary with seed. Candidate for S1.x.
- `.nonslop.toml` is hook-protected: only the owner edits it.
- `gh pr edit --base` fails (Projects classic GraphQL); use `gh api -X PATCH repos/stcksmsh/Mouthpiece/pulls/<n> -f base=master`.
- A background coordinator cannot write in the main checkout; briefs go in the job scratch dir.
