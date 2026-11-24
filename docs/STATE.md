# MOUTHPIECE state

Updated 2026-11-24. Plan: `docs/PLAN.md` (slice IDs S0.x–S5.x). Coordinator keeps this file current.

## Open PRs (all draft, owner merges)

| PR | Slice | Base | Status |
|---|---|---|---|
| #2 | S0.2a–d land uncommitted engine work (+ reroll route fix) | master | ready for owner review |
| #4 | S0.4 fixture corpus + pytest (23 tests, ~5 s, no torch) | #2 | merge after #2 |
| #5 | S0.12 strip punctuation from indexed words; dedupe syllable/phone index helpers; reference.py error handling | #4 | merge after #4 |
| #3 | S0.3 MIT licence + NOTICE | master | **owner must confirm MIT** |

Merge order: #2 → #4 → #5. GitHub retargets each to master as its base merges. #3 is independent.

## Done

- S0.1: snapshot branch `wip/snapshot-2026-11-24` pushed. DB and corpus CSVs backed up to `/home/stcksmsh/Backups/mouthpiece-2026-11-24/` (other drive).
- S0.12 already applied to the real DB (`mouthpiece resyllabify`): 20,953 words normalised, +36,192 syllables, +94,361 phones.

## Owner actions pending

1. Merge #2, #4, #5; decide #3.
2. **S0.5**: edit `.nonslop.toml` (hook-protected; agents cannot):
   ```toml
   [commands]
   test = "python -m pytest -q"
   [done]
   run = ["lint", "test"]
   [merge]
   require = ["verify", "review"]
   allow_no_ci = true
   ```
3. Untracked files block the done-check: `corpus/batch*.csv` (contain an uploader email: never commit), `demo/`, `.devlog/`, `package-lock.json` (empty). Exclude locally (`.git/info/exclude`) or delete.
4. Churchill licence evidence, if Churchill should stay public (S3.1).

## Next slices, in order

1. **Website (urgent):** the live bank serves all 10 Churchill broadcasts labelled public domain. PR on `website` branch: rebuild bank excluding Churchill, fix licence wording. (Early part of S4.5.)
2. S0.6 seeded reroll (`server.py` reroll uses an unseeded RNG).
3. S0.7 attribution model → S0.8 backfill real DB (NASA uploader email in `speaker`, LibriVox author-as-speaker).
4. S0.9 manifest v2 + `replay`; S0.10 harvester licence honesty; S0.11 README.
5. Then P1 (bench, Viterbi, voice policy, picker, bias controls) in parallel with P3 (corpus).

## Facts for builders

- Run tests: `.venv/bin/python -m pytest -q`. Fixture DB is built under `.claude/tmp` (system `/tmp` fsync is ~0.4 s here).
- Indexed words are normalised (`index.normalize_word`); keep it that way at every insert.
- pyrubberband `rate` = our `speed` (>1 faster). Output 44.1 kHz. StaticFiles mount stays last in `server.py`.
- `find_phrase` (match.py) ignores its `rng` argument: phrase choice does not vary with seed. Candidate for S1.x.
- 29 advisory lint findings remain (prints in CLI, `zip()` without `strict=`). Not blocking.
- No CI by owner decision. The local pytest gate takes CI's place once S0.5 lands.
