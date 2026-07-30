---
title: Making renders match a target voice
date: 2026-07-20
tags: [devlog, mouthpiece, python, audio]
---

[Two phases here](https://github.com/stcksmsh/Mouthpiece/commit/47c25864eb6c8fabb0017e87a535028f45dc3826). Phase 0 was just filling in schema columns that existed but were never populated at ingest — lemma via WordNet, plus `f0_var` and a rough spectral-tilt voice-quality proxy alongside the existing f0_mean/RMS, and exposing all of it through the search/candidates APIs.

Phase 1 is the actual feature: `POST /api/render/{id}/auto_pitch` shifts every word's pitch toward a target center Hz while preserving a scaled version of the render's own intonation contour — so it's not flattening everything to one pitch, just recentering the performance. New "pitch match" panel with center-pitch/intonation-range/speaking-rate controls, and the candidates panel's pitch-sort now uses that target when the panel's open instead of sorting against nothing in particular.
