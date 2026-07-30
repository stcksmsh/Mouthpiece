---
title: "Phonemes, real F0, undo, and not losing my mind to the mouse"
date: 2026-07-08
tags: [devlog, mouthpiece, python, audio]
---

[Three things landed together](https://github.com/stcksmsh/Mouthpiece/commit/b3909cc086fe4dde65d0ac3086fd6fb07f48e259), all kind of in service of "make word selection actually good":

Speaking rate was normalized by character count, which is bad for short function words like "the" or "we" — added a proper `phonemes.py` (CMUdict lookup, syllable-count fallback) and switched `normalizeSpeed()` to phoneme-count-per-duration instead.

Also started pulling real per-word F0 (`librosa.yin`) and RMS at ingest time instead of guessing, and wired a "match pitch" sort so the candidate browser can re-order clips by how close their pitch is to the words next to them — which matters a lot for concatenative synthesis actually sounding coherent instead of like a ransom note.

And: keyboard shortcuts, finally. Space to play/pause, arrows to navigate word tiles, up/down to nudge speed, shift+up/down for pitch in semitones, L/R/N for lock/reroll/normalize, and proper undo/redo (`markDirty()` snapshots state before a drag, normalize, or reset). Doing all of this by mouse-only was miserable.
