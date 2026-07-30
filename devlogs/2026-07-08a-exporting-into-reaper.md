---
title: Exporting straight into Reaper
date: 2026-07-08
tags: [devlog, mouthpiece, python, reaper]
---

Added a [`/api/render/{id}/reaper` endpoint](https://github.com/stcksmsh/Mouthpiece/commit/d6e118ab41e9ae97a31a06f24f93c75680f6ce11) that hands back a ready-to-run Lua ReaScript — drop it into Reaper and it imports the assembled WAV onto a new track, then adds a named timeline region per word (token + speaker), color-coded by speaker from a small muted palette. Way more useful than staring at a flat WAV when you're trying to see which speaker said what, where.

The annoying part was `compute_output_times()` — had to re-derive, server-side in Python, the exact same per-word position math the client JS already does (crossfade padding, speed scaling, gaps for missing words) so the regions land in the right place. Two implementations of the same timing logic in two languages isn't great, but there wasn't a clean way to share it between a browser and a Lua export.
