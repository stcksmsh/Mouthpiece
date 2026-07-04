# MOUTHPIECE — Project Brief

You are helping me build a tool called **MOUTHPIECE**. Read this whole brief before writing any code. Ask me questions where something is genuinely ambiguous; otherwise make reasonable decisions and note them.

## What it is

MOUTHPIECE is a **concatenative speech-collage engine**. I type text, and it assembles audio of real government officials / public figures speaking that text, word by word, by cutting and splicing from a corpus of real speech recordings. The output is a WAV I can drop into my DAW (Reaper, on Linux) and process further (vocode, degrade, etc.).

It is **NOT** a text-to-speech or voice-cloning tool. Every word in the output is a real, unmodified-in-content clip of someone actually saying that word, with full provenance. That constraint is core to the concept and must never be violated — do not add any TTS/synthesis of new speech. Assembly, cutting, time-stretching, and EQ/degradation of *real* clips is fine; generating new speech is not.

This is for an art album, but I want it built to a sellable standard: clean architecture, good UX, nice-looking UI, documented.

## Who I am / environment

- Linux, Reaper as DAW, comfortable with Python.
- No assumption of ML-ops experience — keep dependencies sane and setup documented.
- Prioritize a working first result over completeness. I want to *hear something* early.

## Design principles (in priority order)

1. **Simple surface, deep interior.** Typing text and getting a usable WAV must be trivial. Everything advanced (affect, cadence, per-word swapping) is layered on top and optional.
2. **Real clips only, always attributable.** Every render emits a per-word provenance manifest (speaker, date, source, timestamp). This is a first-class feature, not a debug log. Never fabricate attribution.
3. **Search bias, not effects, for affect.** "Rage/calm" etc. work by biasing *which real clip* is chosen (using prosodic features extracted from the audio), not by DSP-faking emotion. Same words, different authentic performances.
4. **Sequence-level selection, not per-word-best.** Choosing the best *sequence* of clips (with continuity costs) is what separates coherent speech from a word playlist.
5. **Offline engine, thin DAW bridge.** No VST. A standalone Python engine does the work; a small ReaScript imports results into Reaper.
6. **Deterministic & reproducible.** Every render is seeded; the seed is shown and stored. I must be able to reproduce an exact render months later.
7. **Fail soft.** Missing words never crash a render — degrade gracefully (fuzzy/lemma match → gap with warning), and make coverage visible before rendering.

## Architecture

Two processes:

**A. Engine (Python).** Does ingest, indexing, matching, assembly. Exposes both a CLI and (later) a local HTTP API (FastAPI).

Pipeline stages:
1. **Ingest** — take source audio files + metadata (speaker, date, event, source URL), normalize to mono WAV at a fixed sample rate, store originals + metadata.
2. **Transcribe + align** — use WhisperX for transcription with **word-level timestamps** and speaker diarization. (Phoneme-level alignment via Montreal Forced Aligner is a *later* optional refinement for fragment fallback — design for it, don't build it in v0.1.)
3. **Feature extraction** — per word instance, extract prosodic features: f0 mean/variance, RMS/loudness, duration, spectral tilt. Use parselmouth (Praat) and/or librosa. These power the affect sliders later.
4. **Index** — store everything in **SQLite** (single file, no server). One row per word instance: word, lemma, phonemes(nullable), source_file, start, end, speaker, date, source_url, and the prosodic features. Use FTS5 for fast text lookup. Design the schema so adding new sources never requires reprocessing existing ones.
5. **Match (query time)** — tokenize input text → retrieve candidates per word → score by (affect-slider distance, audio quality, duration sanity) → then run a **sequence selection** pass (cheap Viterbi/DP) with transition costs: reward same-speaker and same-source continuity, penalize large pitch/loudness jumps. Voice-policy modes bias these costs (see below).
6. **Assemble** — cut with small handles, crossfade (configurable length), apply timing (word-gap control; later: quantize onsets to a grid/tempo), optional degrade chain (bandwidth-limit / phone / PA character), render WAV + JSON manifest.
7. **Serve (later)** — FastAPI local server; a local web UI (HTML/CSS/JS) for the visual surface.

**B. Reaper bridge (Lua ReaScript).** A small action, "MOUTHPIECE: import last render," that fetches the latest render from the engine (HTTP endpoint, or just reads a known output path in v0.1) and inserts it at the edit cursor on the selected track. Keep all intelligence in the engine; the bridge is dumb.

## Feature roadmap — BUILD IN THIS ORDER

Do **v0.1 first and only v0.1** until I confirm it works. Understand the later versions only so your early schema/structure choices don't block them.

**v0.1 — Prove it (build this now):**
- Ingest a handful of source WAVs + a simple metadata format (JSON or CSV per source).
- WhisperX transcription + word-level timestamps → SQLite index.
- CLI: `mouthpiece "text to say" -o out.wav`
- Naive per-word-best matching, hard cuts (no crossfade yet is OK, but a tiny fade to avoid clicks is fine).
- Emit `out.wav` + `out.manifest.json` (per-word provenance).
- Missing word → warn and insert a short gap; don't crash.
- README with exact setup + a tiny sample-corpus walkthrough.

**v0.2 — Usable on the album:**
- Crossfades, configurable word-gap, room-tone/silence gap option.
- Deterministic seeds + `--seed`; "reroll" = next seed.
- Reaper import ReaScript.
- Preview-quality vs HQ render split.

**v0.3 — Depth:**
- Prosodic feature extraction + affect sliders (Intensity / Pitch bias / Tempo bias) as *search bias*.
- Viterbi sequence selection with continuity costs.
- Voice-policy modes: Single-speaker / Ransom-note (maximize speaker changes) / Free.
- FastAPI server + web UI with a **word strip**: per-word tiles showing source, lock/swap/reroll individual words, coverage warnings visible pre-render.

**v0.4 — Instrument:**
- Quantize word onsets to project tempo/grid; rhythmic patterns; stutter-repeat.
- **Sliced sample-map export**: one word per MIDI note (e.g. RS5k-loadable) so speech becomes playable from a MIDI keyboard.
- Degrade chain.
- Phoneme-level fragment fallback (needs MFA).

## UI design language (for v0.3+ web UI)

Near-black background, fine line work, restrained, clinical/institutional. The **word strip** is the signature UI: each word is a tile visibly "torn" from a different document (ransom-note aesthetic) showing speaker + date + tiny waveform. This is the concept made visible. Don't build the UI in v0.1 — but when you do, art-direct it, don't ship a default-looking form.

## Tech constraints & preferences

- Python 3.11+. Keep the dependency list documented and justified. Prefer a venv; provide `requirements.txt` or `pyproject.toml`.
- SQLite for the index (with FTS5). No external DB servers.
- WhisperX for ASR+alignment+diarization. Note the model-download/first-run cost in the README.
- rubberband (via CLI or pyrubberband) for any time-stretching — but v0.1 needs no stretching.
- FastAPI + a plain-HTML/JS front end later (no heavy SPA framework unless you make a strong case).
- Everything Linux-first. Flag anything platform-specific.
- Determinism: any randomness goes through a seeded RNG; store the seed in the manifest.

## Corpus note

I'll supply source audio. Corpus curation is the real bottleneck, so v0.1 should work with just a few hours of a single source. Favor US federal / public-domain material as the core (e.g. congressional/committee recordings, .gov sources) for legal cleanliness — design the metadata schema to record source + license per file, but you don't need to fetch anything; I'll provide files and metadata.

## What I want from you right now

1. Confirm you've understood the above and note any decisions/assumptions you're making.
2. Propose the repo structure and the v0.1 SQLite schema (forward-compatible with the roadmap).
3. Set up the project scaffold and implement **v0.1** end to end.
4. Give me a dead-simple way to test it with a tiny corpus, and tell me exactly what to run.

Do not skip ahead past v0.1 in implementation. Ask before adding any dependency that pulls in something heavy or platform-fragile.
