---
title: Started Mouthpiece, a speech collage engine
date: 2026-07-04
tags: [devlog, mouthpiece, python, audio]
---

The idea: type text, get back a WAV of real people actually saying it — cut word by word from a corpus of real speech, not synthesized. It's for an album, and the "real clip only, always attributable" part is the actual point, not a nice-to-have. If a word isn't in the corpus, it should fail soft (skip with a gap, not fabricate a TTS word).

[First pass](https://github.com/stcksmsh/Mouthpiece/commit/18a5b8efd3fab72a82eb90f12101f6b5aa4707a3) has the full shape already: WhisperX for ingest/transcription, SQLite FTS5 for word search, assembly with crossfades + RMS normalization, per-word speed/pitch stretching via rubberband, and a candidate browser in a FastAPI + vanilla-JS UI. No affect/emotion search yet (that's supposed to work by biasing which real clip gets picked, not DSP faking it) — right now it's just "find the word, splice it in."
