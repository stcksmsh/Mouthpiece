# MOUTHPIECE state

Updated 2026-11-24. Plan: `docs/PLAN.md` (slice IDs S0.x–S5.x). Coordinator keeps this file current.

## Open PRs (all draft, owner merges)

| PR | Slice | Status |
|---|---|---|
| #5 | S0.2a–d + S0.4 + S0.12, landed as one PR (owner decision) | in verify/review |
| #3 | S0.3 MIT licence + NOTICE | verified, review clean; waiting for owner merge |
| #2, #4 | superseded by #5 | close once #5 merges |

## Done

- S0.1: snapshot branch `wip/snapshot-2026-11-24` pushed. DB and corpus CSVs backed up off-repo (other drive).
- S0.5 (partial): `allow_no_ci = true` on master (1463725).
- S0.12 already applied to the real DB (`mouthpiece resyllabify`): 20,953 words normalised, +36,192 syllables, +94,361 phones.

## Waiting on owner

1. Merge #3, then #5 once it is verified and reviewed.
2. After #5 merges: add the pytest gate to `.nonslop.toml` (agents cannot edit it):
   `[commands] test = ".venv/bin/python -m pytest -q"`, `[done] run = ["lint", "test"]`.
3. Churchill licence evidence, if Churchill should stay public (S3.1).

## Next slices, in order

1. **Website (urgent):** the live bank serves all 10 Churchill broadcasts labelled public domain. PR on `website` branch: rebuild bank excluding Churchill, fix licence wording. (Early part of S4.5.)
2. S0.6 seeded reroll (`server.py` reroll uses an unseeded RNG).
3. S0.7 attribution model → S0.8 backfill real DB (NASA uploader email in `speaker`, LibriVox author-as-speaker).
4. S0.9 manifest v2 + `replay`; S0.10 harvester licence honesty; S0.11 README.
5. Then P1 (bench, Viterbi, voice policy, picker, bias controls) in parallel with P3 (corpus).

## Decisions

- Licence: MIT, copyright "Kosta Vukićević" (owner, 2026-11-24).
- Stacked chain #2→#4→#5 lands as one PR #5 (owner chose A, 2026-11-24): no duplicated fixes.
- No CI by owner decision; the local pytest gate replaces it.
- Owner-local untracked files (`corpus/batch*.csv` with an uploader email, `demo/`, `.devlog/`, `package-lock.json`) are excluded via `.git/info/exclude`. Never commit them.

## Facts for builders

- Run tests: `.venv/bin/python -m pytest -q` (29 pass at 3ae5e27). Fixture DB builds under `.claude/tmp`.
- Indexed words are normalised (`index.normalize_word`) at every insert; `'hello'` → `hello`, `'em`/`goin'` kept.
- Syllable/phone index code goes through `_UNITS` in index.py; do not add per-kind copies.
- pyrubberband `rate` = our `speed` (>1 faster). Output 44.1 kHz. StaticFiles mount stays last in `server.py`.
- `find_phrase` (match.py) ignores its `rng` argument: phrase choice does not vary with seed. Candidate for S1.x.
- 29 advisory lint findings remain (prints in CLI, `zip()` without `strict=`). Not blocking.
- `.nonslop.toml` is hook-protected: only the owner edits it (web editor on master works).
- `gh pr edit --base` fails (Projects classic GraphQL); use `gh api -X PATCH repos/stcksmsh/Mouthpiece/pulls/<n> -f base=master`.
