# MOUTHPIECE plan

Written 2026-11-24 from `mouthpiece-handover-2026-11-24.zip` plus owner answers. Planning only. Items marked **(proposal)** are planner's, not owner decisions.

## 0. Decisions taken

| # | Decision | Source |
|---|---|---|
| D1 | Open source + showcase site. Album is main use. Repo already public | owner |
| D1b | Code licence MIT. Corpus and bank carry per-source licence, not MIT | **proposal** |
| D2 | Must "work well" on wide variety of text. Feature order left to planner | owner |
| D3 | Everything legally clean, many sources, with picker | owner |
| D4 | GUI wanted. GUI = existing local web UI, launched by one command. No Tauri/Electron | owner + **proposal** |
| D5 | Site = demo + download | owner |
| D6 | Named officials stay in public demo if legal. Legality not confirmed by planner (see R2) | owner, open |
| 7 | Source-level attribution enough for SCOTUS/NASA/Apollo | owner |
| 8 | Selection-exact reproducibility. No bit-exact WAV guarantee | owner |
| 9 | Sub-word stitching automatic, judged by ear. Tier stamped per fragment in manifest | owner + **proposal** |
| 10 | Diarization off by default. No MFA. Real sub-word boundaries from WhisperX's own aligner instead (S1.8) | **proposal** |
| 11 | No CI. Local pytest gate enforced by nonslop | owner + **proposal** |
| C | Controls: Diction, Intensity, Pitch bias, Tempo bias (all search bias), gaps, voice policy, source picker, presets. Per-word speed/pitch lanes stay | **proposal** |

Owner still owes: link to Churchill database and its licence statement (S3.1). Confirm MIT. Confirm D6 after reading R2.

## 1. Phases

Order: P0 → (P1 and P3 in parallel) → P2 → P4 → P5. No dates.

| Phase | Goal | Exit criteria | Brief principles |
|---|---|---|---|
| P0 Stabilise | Nothing can be lost, nothing lies, every change testable | master = disk state; `pytest -q` green in <30 s without torch; zero emails or author-as-speaker in DB; manifest replays to same clip list; README matches reality | 2, 6, 7 |
| P1 Quality core | Sentences sound like speech, not a word playlist | Owner prefers new selector on ≥14 of 20 bench sentences; voice policy, picker and four bias controls work in CLI and UI | 3, 4, 1 |
| P2 Album instrument | Renders live in Reaper as playable material | Import action works in Reaper; renders survive restart; quantised render lands on grid; RS5k map playable from keyboard | 5, 6, 1 |
| P3 Corpus | Bigger, clean, licence proven per source | Every source has licence record with evidence URL; no `unverified` source in published bank; coverage of top-10k word list reported and rising | 2 |
| P4 Website | Honest demo that sounds like the real engine, with download | First paint shows a render; first click audible <2 s on cable; JS and Python agree on golden vectors; licence table generated from register | 1, 2 |
| P5 Distribution | Stranger on Linux gets first render without ML-ops | Fresh Ubuntu: install script → `mouthpiece doctor` green → render from starter pack, no torch needed | 1 |

## 2. Slices

One slice = one PR on a `claude/*` branch unless marked OWNER. Sizes S/M/L.

### P0 Stabilise

| ID | Title | Why | Touches | Acceptance | Deps | Size |
|---|---|---|---|---|---|---|
| S0.1 | Snapshot and backup (OWNER, today) | 2,488 uncommitted lines and a 62 MB DB that costs hours of WhisperX exist on one disk | git only; external disk | `git status --short` reviewed for **untracked** files (patch is `git diff`, so untracked files are not in it); branch `wip/snapshot-2026-11-24` pushed with everything; `sqlite3 <backup>.db "select count(*) from word_instances"` → 165425; `corpus/batch.csv` copied too | none | S |
| S0.2a | Land schema, phonemes, ingest, pyproject | Bottom layer of uncommitted work | `index.py`, `phonemes.py`, `ingest.py`, `pyproject.toml` | `python3 -m compileall -q mouthpiece` clean; `mouthpiece coverage "hello"` runs on real DB | S0.1 | S |
| S0.2b | Land matching and CLI | Syllable/phone fallback, phrase reuse, sliders | `match.py`, `cli.py` | `mouthpiece render "we choose to go to the moon" --seed 1 -o /tmp/a.wav` writes WAV + manifest; same command twice → identical manifest `words` | S0.2a | M |
| S0.2c | Land assembly | Crossfades, per-word speed/pitch | `assemble.py` | `soxi /tmp/a.wav` shows 44100 Hz; listen: no clicks, short words present | S0.2b | S |
| S0.2d | Land server and UI | Advertised features finally on master | `server.py`, `static/index.html` | `mouthpiece serve`, render in UI, swap one word, export Reaper Lua; `git diff wip/snapshot-2026-11-24 master --stat` empty | S0.2c | M |
| S0.3 | LICENSE and NOTICE | Public repo with no licence = nobody may use it | `LICENSE`, `NOTICE.md` | MIT text present; NOTICE says corpus/bank audio is not MIT, see licence register | none | S |
| S0.4 | Fixture corpus and pytest | Nothing is testable today | `tests/`, `tests/fixtures/` (≤10 s PD audio, `build_fixture_db.py` with hand-set timestamps, no WhisperX), dev extra in `pyproject.toml` | `pytest -q` passes in venv **without torch**; tests cover: seeded render twice → equal manifests; unknown word → gap + warning + exit 0; output 44.1 kHz; crossfade never exceeds ¼ of shorter clip; speed 2.0 halves duration | S0.2d | M |
| S0.5 | Local gate instead of CI | Owner wants no CI; today coordinator can never merge | `.nonslop.toml` (`test = "pytest -q"`, `[done].run` adds test, `[merge]` drops `ci` / allows no CI), `docs/STATE.md` | Builder slice with failing test cannot finish; coordinator merge of green slice works | S0.4 | S |
| S0.6 | Seeded reroll | `api_reroll` uses unseeded RNG: breaks determinism constraint | `server.py`, test | Reroll RNG derived from (render seed, word index, reroll count); test: same seed + same reroll sequence → same clips; `grep -n "default_rng()" mouthpiece/` shows only fresh-seed generation | S0.4 | S |
| S0.7 | Attribution model | Speaker field holds emails, book authors, group labels | `index.py` migration, `manifest.py`, `match.py` row shape, UI tile label, tests | `sources` gains `speaker` (person or NULL), `author`, `reader`, `uploader`, `source_label`, `licence`, `licence_url`, `lang`; manifest emits all; display rule when speaker NULL: "unidentified speaker, {source_label}, {date}"; migration test on fixture DB | S0.4 | M |
| S0.8 | Backfill real DB | Fix stored data, remove personal email | `tools/backfill_attribution.py`, `corpus/attribution.csv` (reviewed by owner, no emails) | `select count(*) from sources where speaker like '%@%'` → 0; no source where speaker = book author; LibriVox reader filled only where source metadata names one, else NULL (never guessed); French source `lang='fr'` | S0.7 | M |
| S0.9 | Manifest v2 and replay | Manifest stores seed + text only; cannot reproduce | `manifest.py`, `cli.py` (`replay`), `server.py`, tests | Manifest has `schema_version`, engine version, all controls, `nth`, locks/swaps, per-word speed/pitch/gap, per-fragment tier (phrase/word/syllable/phone) + instance id, corpus hash; `mouthpiece replay x.manifest.json -o y.wav` → same instance ids; corpus hash mismatch → warning, still renders | S0.6, S0.7 | M |
| S0.10 | Harvester licence honesty | Every item defaults to "public domain": fabricated claim | `harvest.py` | No default licence; each adapter declares licence + tier; missing → `unverified`; yt-dlp path removed; Trump/C-SPAN adapter tier `restricted` | S0.7 | S |
| S0.11 | README rewrite | README describes v0.1 | `README.md` | Fresh venv, README only → first render on fixture corpus; lists every CLI command that exists; states 44.1 kHz, Reaper export, no CI | S0.9 | S |

### P1 Quality core

| ID | Title | Why | Touches | Acceptance | Deps | Size |
|---|---|---|---|---|---|---|
| S1.1 | Listening bench | "Sounds good" needs fixed ear test and numbers | `tools/bench.py`, `bench/sentences.txt` (20 lines: function-word heavy, rare words, names, numbers, one 30-word sentence) | `python tools/bench.py` → `renders/bench/<sha>/NN.wav` + `summary.csv` (source switches, sub-word count, gaps, runtime per line) | S0.9 | S |
| S1.2 | Split cost from selection | Viterbi needs unit cost as pure function | `match.py`, tests | No behaviour change: golden manifests identical before/after | S0.4 | M |
| S1.3 | Viterbi selector | Brief principle 4. Biggest quality lever | `match.py`, `cli.py` (`--selector greedy\|viterbi`), tests | DP over top-K per token; transition cost: adjacent-in-source = 0 (phrase reuse falls out), same source, same speaker, f0 jump, loudness jump, rate jump; seeded tie-break; bench: switches per sentence down vs greedy; 30 words <1 s; **listen: owner picks viterbi on ≥14/20**, then default flips | S1.1, S1.2 | L |
| S1.4 | Edge-context cost | "me", "the" cut from mid-phrase sound clipped | `match.py`, `index.py` (gap to neighbour words) | Unit cost prefers clips whose position in source (after pause / before pause / mid-run) matches position in target; listen to bench lines 1–5: fewer clipped function words | S1.3 | M |
| S1.5 | Voice policy | Album doctrine needs single clean speaker and maximum fragmentation | `match.py`, `cli.py` (`--voice free\|single\|ransom`), UI select | `single`: all words from one source where possible, missing words fall back + warning (fail soft); `ransom`: no two adjacent words share source when alternatives exist; tests for both | S1.3 | M |
| S1.6a | Source filter (engine) | Picker; also keeps restricted sources out of chosen renders | `match.py`, `cli.py` (`--only`, `--exclude`, `--tier`), manifest | Render with `--only "Barack Obama"` uses only that speaker or gaps with warning; filter stored in manifest | S0.7, S1.2 | S |
| S1.6b | Source picker (UI) | Same, visible | `server.py`, `static/index.html` | Panel lists sources: label, words, licence badge, checkbox; selection drives render and persists per session | S1.6a | M |
| S1.7 | Bias controls | Affect by choosing clips (principle 3) | `match.py`, `cli.py`, UI sliders | Diction, Intensity (replaces "emotion"), Pitch bias, Tempo bias; f0 normalised per source so Pitch bias does not just pick high-voiced speakers; `tools/bench.py --pitch -1` vs `+1`: mean normalised f0 of chosen clips ordered; audible difference same text same seed | S1.2 | M |
| S1.8 | Real sub-word boundaries | Stitched words use proportional guesses. MFA setup too heavy | `transcribe.py`, `index.py`, `phonemes.py`, `cli.py` (`realign`) | **Spike first:** confirm installed WhisperX returns character-level alignments (unverified). If yes: store them, derive syllable/phone bounds, `mouthpiece realign` updates existing DB without re-transcribing; bench rare-word lines sound less smeared; if no: report, slice closes, proportional stays | S0.2a | L |
| S1.9 | Presets | Simple surface over many controls | `server.py`, UI, `presets/*.json` | Save/load named control sets; three shipped: Plain, Ransom note, One voice | S1.5, S1.7 | S |

### P2 Album instrument

| ID | Title | Why | Touches | Acceptance | Deps | Size |
|---|---|---|---|---|---|---|
| S2.1 | Render persistence | Renders die on restart | `server.py`, `renders/<id>/` | Restart server → history lists earlier renders, each reopens with words and lanes | S0.9 | M |
| S2.2 | Reaper "import last render" | Brief v0.2, still missing | `reaper/mouthpiece_import_last.lua`, `GET /api/renders/latest` + file fallback `renders/latest/` | Run action in Reaper: items at edit cursor on selected track; option one item per word, take named `word, speaker/source` | S2.1 | M |
| S2.3 | Gaps and room tone | Pacing control | `assemble.py`, `cli.py`, UI lane | Global gap, per-word gap lane, punctuation pauses; fill = silence or room tone cut from non-speech parts of same source (real audio, recorded in manifest) | S0.9 | M |
| S2.4 | Preview vs final render; cached models | UI edits slow; reference match cold-loads WhisperX each call | `assemble.py`, `server.py`, `reference.py` | Slider change re-renders <1 s for 12 words in preview; final export uses full quality; second reference-match call skips model load | S2.1 | M |
| S2.5 | Batch render | Flooding sections need hundreds of lines | `cli.py` (`render --file lines.txt --seeds N`) | 10 lines × 3 seeds → 30 WAV + manifests in one folder, deterministic names | S0.9 | S |
| S2.6 | Quantise and stutter | Speech on grid | `assemble.py`, `cli.py` (`--bpm`, `--grid`, `--stutter`) | Test: 120 BPM, 1/8 → word onsets in manifest multiples of 0.25 s ±5 ms; listen against Reaper click | S2.3 | M |
| S2.7 | RS5k sample map | Speech playable from keyboard | `cli.py` (`export-map`), `reaper/mouthpiece_load_map.lua` | Folder of word WAVs + note map text + Lua that builds one ReaSamplOmatic5000 per note; KeyLab plays words. RS5k parameter scripting unverified: builder checks in Reaper first | S2.1 | M |

### P3 Corpus (parallel with P1)

| ID | Title | Why | Touches | Acceptance | Deps | Size |
|---|---|---|---|---|---|---|
| S3.1 | Licence register (OWNER + builder) | "Legally clean" needs evidence per source | `corpus/LICENCES.md`, licence fields from S0.7 | Each of 34 sources: licence, evidence URL, date checked, tier `clean` / `restricted` / `unverified`. Churchill and C-SPAN start `restricted` (see R1) until owner supplies proof | S0.8 | S |
| S3.2 | Language check at ingest | French book indexed as English | `transcribe.py`, `ingest.py`, `match.py` | Detected language stored; non-English excluded from matching by default; existing French source excluded without deleting it | S0.7 | S |
| S3.3 | Coverage report | Tells which harvest helps | `cli.py` (`coverage --wordlist`), `data/top10k.txt` | Prints % covered as whole word, missing list, per-speaker counts, per-tier counts | S0.7 | S |
| S3.4 | Harvest runs (OWNER, compute) | 521 words from an elected official | none (runs existing adapters) | Obama weekly addresses fully ingested; coverage report rerun and committed to `docs/STATE.md` | S0.10, S3.2 | M |
| S3.5 | More clean adapters | Variety of voices | `harvest.py` | One adapter per PR: House/Senate floor, further White House archives, presidential libraries. Each PR cites licence evidence URL in adapter and register. Endpoints unverified: builder confirms source exists before coding | S0.10 | M each |

### P4 Website

| ID | Title | Why | Touches | Acceptance | Deps | Size |
|---|---|---|---|---|---|---|
| S4.1 | Pre-rendered first state | First impression is black box | `website/index.html`, `src/main.js`, static WAV + manifest | Page load shows waveform + fragment cards for default sentence before any click | none | S |
| S4.2 | Cold start | First render >7 s | `tools/export_bank.py`, `src/render.js` | First sprite holds most frequent words + default sentence; visible progress; first click audible <2 s on 20 Mbit throttle | none | M |
| S4.3 | Golden vectors | JS port drifts from Python silently | `tests/golden/*.json`, pytest, `website` node test | Same fixture bank + texts + seeds → same instance ids in both engines; both test commands fail on mismatch | S0.4 | M |
| S4.4 | Port selector to JS | Demo should sound like real engine | `src/engine.js` | Golden vectors from S1.3–S1.5 pass in JS | S1.5, S4.3 | M |
| S4.5 | Bank regen, honest attribution | Site says all public domain; names authors as speakers | `tools/export_bank.py`, `index.html` | Bank built only from tier `clean`; cards use S0.7 fields; LibriVox footnote gone; licence table generated from register | S0.8, S3.1 | M |
| S4.6 | Provenance and use page | Tool makes named people "say" text | `website/` page | States: real clips only, collage not quotation, manifest per render, licence per source, contact for removal; downloaded WAV always comes with manifest | S4.5 | S |
| S4.7 | Album context (OWNER supplies audio) | Site never says what it is for | `index.html` | Section with album name, 2–3 finished excerpts, devlog link | none | S |
| S4.8 | Download section | D5 | `index.html` | Version, release link, install one-liner, changelog link | S5.4 | S |
| S4.9 | Share card and cleanup | Unchecked og tags; stale `page/` bundle on `io-page` | `index.html`, `io-page` branch | og:title/image/audio render in a link preview checker; `page/` removed after owner confirms unused | none | S |

### P5 Distribution

| ID | Title | Why | Touches | Acceptance | Deps | Size |
|---|---|---|---|---|---|---|
| S5.1 | Split dependencies | Rendering needs no torch; only ingest does | `pyproject.toml` (`[ingest]` extra), lazy imports in `transcribe.py`, `reference.py` | `pip install .` in clean venv without torch → `render` and `serve` work on fixture DB; reference matching reports "needs ingest extra" instead of crashing | S0.4 | M |
| S5.2 | Install script and doctor | Sane setup | `scripts/install.sh`, `cli.py` (`doctor`) | Fresh Ubuntu container: script creates venv, installs, prints missing apt packages (`rubberband-cli`, `ffmpeg`); `mouthpiece doctor` lists each dependency OK/missing with fix line | S5.1 | M |
| S5.3 | Starter corpus pack | Download without corpus = empty tool | `tools/export_pack.py`, release asset | Pack = DB + audio for tier `clean` only, lossless, each file under GitHub's release asset limit; `mouthpiece pack install <file>` → render works | S3.1, S5.1 | M |
| S5.4 | Launcher and release | GUI in one step; versioned download | `cli.py` (`serve` opens browser), `.desktop` file, `CHANGELOG.md`, tag | `mouthpiece` with no args opens UI; tag `v0.5.0`; release page has install line + pack | S5.2, S5.3 | S |

## 3. Risks

| ID | Risk | Handling |
|---|---|---|
| R1 | Churchill and C-SPAN not clean. Churchill estate asserts copyright over his literary works and audio recordings, licensed through Curtis Brown; UK term for literary works is life + 70, Churchill died 1965. "Free database" proves access, not licence. C-SPAN allows non-commercial use with attribution for federal events; an album for sale is commercial. Only House/Senate floor video is public domain | Tier `restricted`: usable locally at owner's risk, never in public bank or starter pack. Owner supplies database licence statement to change tier. Planner is not a lawyer |
| R2 | D6 "if legal": copyright-clean audio does not settle other claims (false statements attributed to living people, use of a person's voice/identity). Rules differ per country. Not verified | Owner decides. Plan keeps named officials in demo, with manifest always shown and "collage, not quotation" stated. If owner wants certainty before P4: one hour with a lawyer |
| R3 | Viterbi sounds no better than greedy | S1.1 bench first; default flips only on listening result; greedy stays selectable |
| R4 | WhisperX character alignment unavailable or poor | S1.8 starts with spike; fallback = keep proportional slicing |
| R5 | No CI: gate only as strong as local runs | nonslop done-gate runs pytest; tests need no torch so run anywhere in <30 s |
| R6 | Corpus/DB live on one machine | S0.1 backup; S5.3 pack doubles as off-site copy of clean tier |
| R7 | JS and Python engines drift | S4.3 golden vectors; no JS feature without a vector |
| R8 | Scope creep (UI already past brief) | No new UI features outside slices above until P1 exit |
| R9 | Attribution backfill guesses a reader | Rule in S0.8: unknown = NULL, shown as unidentified |

## 4. Won't build

| Item | Reason |
|---|---|
| TTS, voice cloning, any generated speech | Identity of project |
| GitHub Actions CI | Owner decision; local gate covers it |
| MFA | Conda-grade setup; S1.8 tries same gain with existing dependency |
| Speaker identification inside SCOTUS/NASA | Source-level attribution accepted (answer 7); naming people from diarization invites fabricated attribution |
| VST/CLAP plugin | Brief principle 5 |
| Tauri/Electron app, Docker image | Local web UI already is the GUI; packaging cost without album benefit |
| Degrade chain | Done better in Reaper |
| Audio watermarking | Manifest is the provenance mechanism; watermark would alter clips |
| Bit-exact WAV reproduction | Answer 8; would pin rubberband/librosa forever |
| Hosted render service, accounts, paid tier | D1 = open source + showcase |
| yt-dlp / YouTube harvesting | ToS risk; removed in S0.10 |
| Letter-rule sounding-out in Python engine | Web-only today; keep engine's fail-soft gap. Revisit only if bench shows need |

## 5. Handover corrections

- Patch is 2,488 lines in 9 files, not ~1,800. It is a `git diff`, so untracked files are not in it: check `git status` in S0.1.
- Server reroll is not seeded (`server.py`, `api_reroll`).
- Manifest stores only seed, text, words: no controls, no versions.
- Attribution problem covers SCOTUS ("SCOTUS oral argument"), NASA ("NASA"), Apollo ("Apollo 11") too.
- `harvest.py` defaults licence to "public domain" for every item.
- Trump adapter is C-SPAN via archive.org, not the White House archive.
- Site states all recordings are public domain; not established for Churchill. Check whether Churchill is in the exported bank.
- No LICENSE file.

## Sources

- Churchill copyright: https://winstonchurchill.org/resources/reference/the-churchill-copyright/ and https://churchillarchive.com/copyright-and-third-party-copyright-agreements
- UK terms: https://www.gov.uk/government/publications/copyright-notice-duration-of-copyright-term/copyright-notice-duration-of-copyright-term
- C-SPAN policy (read via mirror and secondary reports, not c-span.org directly): https://www.c-span.org/about/copyrightsAndLicensing
