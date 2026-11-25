"""Cut clips from corpus, crossfade joins, concatenate to output WAV."""

from pathlib import Path

import numpy as np
import soundfile as sf

from . import index

SAMPLE_RATE = 16_000      # internal / corpus rate (WhisperX requires 16 kHz)
OUTPUT_RATE = 44_100      # output WAV rate (standard for DAWs)
CLIP_PAD    = 0.04        # 40 ms handle each side of word timestamp
CROSSFADE   = int(0.025 * SAMPLE_RATE)   # 25 ms crossfade, capped per clip below
FADE_LEN    = int(0.005 * SAMPLE_RATE)   # 5 ms anti-click fade
WORD_GAP    = int(0.10 * SAMPLE_RATE)    # 100 ms silence between words
MISSING_GAP = int(0.40 * SAMPLE_RATE)   # 400 ms silence for missing words
SYLL_CROSSFADE = int(0.012 * SAMPLE_RATE)  # 12 ms tight join between syllables of one word

TARGET_RMS  = 0.08   # normalise every clip to ~-22 dBFS
PEAK_LIMIT  = 0.95   # hard ceiling after normalisation


def _apply_fades(clip: np.ndarray) -> np.ndarray:
    # Clip-relative cap: a fixed 5ms fade eats a huge fraction of a very short
    # clip (e.g. 'a' at ~20ms) — scale down for anything under ~100ms instead.
    fade_len = min(FADE_LEN, max(len(clip) // 10, 2))
    if len(clip) < 2 * fade_len:
        return clip
    out = clip.copy()
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    out[:fade_len]  *= ramp
    out[-fade_len:] *= ramp[::-1]
    return out


def _normalize_clip(clip: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(clip ** 2))
    if rms < 1e-6:
        return clip
    clip = clip * (TARGET_RMS / rms)
    peak = np.max(np.abs(clip))
    if peak > PEAK_LIMIT:
        clip = clip * (PEAK_LIMIT / peak)
    return clip


def _apply_rubberband(clip: np.ndarray, speed: float = 1.0, pitch: float = 0.0) -> np.ndarray:
    """
    Time-stretch and pitch-shift via rubberband.
    speed > 1 = faster (shorter), speed < 1 = slower (longer); pitch in semitones.
    pyrubberband rate convention: rate > 1 = faster — same as our speed, pass directly.
    """
    if abs(speed - 1.0) < 0.02 and abs(pitch) < 0.05:
        return clip
    import pyrubberband as pyrb
    if abs(speed - 1.0) >= 0.02:
        clip = pyrb.time_stretch(clip, SAMPLE_RATE, float(speed))
    if abs(pitch) >= 0.05:
        clip = pyrb.pitch_shift(clip, SAMPLE_RATE, float(pitch))
    return clip.astype(np.float32)


def _crossfade_join(a: np.ndarray, b: np.ndarray, xf_cap: int = CROSSFADE) -> np.ndarray:
    # //8 rather than //4: a fixed 25ms crossfade already caps out at 25% of a
    # short clip's length per side (50% total) — tighten so short words like
    # 'is'/'a' keep more of their own audio instead of mostly being blended
    # into neighbors. Only bites for clips under ~200ms; normal words are
    # already well under either cap via the absolute xf_cap.
    xf = min(xf_cap, len(a) // 8, len(b) // 8)
    if xf < 2:
        return np.concatenate([a, b])
    ramp = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    overlap = a[-xf:] * ramp + b[:xf] * ramp[::-1]
    return np.concatenate([a[:-xf], overlap, b[xf:]])


def _cut_clip(
    source_path: Path,
    start_s: float,
    end_s: float,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> np.ndarray:
    import librosa
    duration = end_s - start_s
    pad = min(CLIP_PAD, max(duration * 0.25, 0.010))
    padded_start = max(0.0, start_s - pad)
    padded_dur   = (end_s + pad) - padded_start
    clip, _ = librosa.load(
        str(source_path),
        sr=SAMPLE_RATE,
        mono=True,
        offset=padded_start,
        duration=max(padded_dur, 0.05),
    )
    clip = _normalize_clip(clip)
    clip = _apply_rubberband(clip, speed=speed, pitch=pitch)
    return _apply_fades(clip)


def _cut_syllable_clip(source_path: Path, start_s: float, end_s: float) -> np.ndarray:
    """Like _cut_clip, but for a sub-word syllable span: smaller pad, no
    rubberband/outer fades yet — those are applied once to the stitched whole."""
    import librosa
    duration = end_s - start_s
    pad = min(0.01, max(duration * 0.25, 0.005))
    padded_start = max(0.0, start_s - pad)
    padded_dur   = (end_s + pad) - padded_start
    clip, _ = librosa.load(
        str(source_path),
        sr=SAMPLE_RATE,
        mono=True,
        offset=padded_start,
        duration=max(padded_dur, 0.03),
    )
    clip = _normalize_clip(clip)
    return _apply_fades(clip)


def _group_syllable_runs(rows: list[dict]) -> list[list[dict]]:
    """Collapse consecutive syllable rows sharing a non-null run_id (a natural
    multi-syllable span found by match.find_syllable_run — real continuous
    corpus audio) into one segment. Singles (run_id=None) stay 1-row segments."""
    segments: list[list[dict]] = []
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        if r.get("run_id") is not None and r.get("run_pos") == 0:
            rlen = r.get("run_len", 1)
            segments.append(rows[i:i + rlen])
            i += rlen
        else:
            segments.append([r])
            i += 1
    return segments


def _row_phones(r: dict) -> list[str]:
    """A row's phone list, whether it's a syllable_instances row ('phones',
    space-joined) or a phone_instances row ('phone', single symbol)."""
    if "phones" in r:
        return r["phones"].split()
    return [r["phone"]]


def _segment_rate(seg: list[dict], phonemes) -> tuple[float, float]:
    """(phone-weight-sum, natural duration) for a whole segment — a run's
    'own rate' is computed across its full span, not per-member, so its
    already-correct internal pacing isn't disturbed by the tempo-normalize
    step below, only the segment's pace relative to its neighbors."""
    w   = sum(phonemes.phone_weight_sum(_row_phones(r)) for r in seg)
    dur = seg[-1]["end_s"] - seg[0]["start_s"]
    return w, dur


def _cut_stitched_clip(
    rows: list[dict],
    corpus_dir: Path,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Cut each segment (a single syllable, or a natural multi-syllable run —
    see _group_syllable_runs) as one clip, time-stretch it toward the whole
    word's median segment speaking rate (segments are pulled from unrelated
    source words/recordings — without this, a segment spoken quickly next to
    one spoken slowly makes the stitched word audibly change tempo mid-word),
    tightly join them, then apply speed/pitch and anti-click fades exactly
    like a normal word clip."""
    from . import phonemes

    segments = _group_syllable_runs(rows)

    rates = []
    for seg in segments:
        w, dur = _segment_rate(seg, phonemes)
        if dur > 0 and w > 0:
            rates.append(w / dur)
    target_rate = float(np.median(rates)) if rates else None

    clips = []
    for seg in segments:
        first, last = seg[0], seg[-1]
        clip = _cut_syllable_clip(corpus_dir / first["filename"], first["start_s"], last["end_s"])
        w, dur = _segment_rate(seg, phonemes)
        if target_rate and dur > 0 and w > 0:
            own_rate    = w / dur
            local_speed = max(0.5, min(2.0, target_rate / own_rate))
            if abs(local_speed - 1.0) >= 0.05:
                import pyrubberband as pyrb
                clip = pyrb.time_stretch(clip, SAMPLE_RATE, float(local_speed)).astype(np.float32)
        clips.append(clip)

    combined = clips[0]
    for c in clips[1:]:
        combined = _crossfade_join(combined, c, xf_cap=SYLL_CROSSFADE)
    combined = _apply_rubberband(combined, speed=speed, pitch=pitch)
    return _apply_fades(combined)


def _group_phrase_units(
    matches: list[tuple[str, dict | list[dict] | None]],
) -> list[tuple[int, list[str], list]]:
    """Collapse consecutive entries sharing a phrase_id into one assembly unit.
    Returns (first_index, tokens, rows) triples — len(rows)==1 for a normal
    single-token entry (row is dict|list[dict]|None), or the full phrase run
    (each element a plain word_instance dict) for a phrase unit."""
    units: list[tuple[int, list[str], list]] = []
    i, n = 0, len(matches)
    while i < n:
        token, row = matches[i]
        if isinstance(row, dict) and row.get("phrase_id") is not None and row.get("phrase_pos") == 0:
            plen = row.get("phrase_len", 1)
            group = matches[i:i + plen]
            units.append((i, [t for t, _ in group], [r for _, r in group]))
            i += plen
        else:
            units.append((i, [token], [row]))
            i += 1
    return units


def assemble(
    matches: list[tuple[str, dict | list[dict] | None]],
    corpus_dir: Path,
    adjustments: dict[int, dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build audio from match results.
    adjustments: {word_index: {speed, pitch, gap, gap_s}}: per-word rubberband processing,
    and seconds of silence before the word (gap) or in place of a missing word (gap_s).
    Defaults are WORD_GAP between units and MISSING_GAP for a missing word.
    Returns (audio_array_float32, manifest_words).
    """
    adj = adjustments or {}

    parts: list[np.ndarray] = []
    manifest_words: list[dict] = []

    units = _group_phrase_units(matches)

    for u_idx, (i, tokens_u, rows_u) in enumerate(units):
        default_gap = WORD_GAP / SAMPLE_RATE if u_idx > 0 else 0.0
        gap_n = round(adj.get(i, {}).get("gap", default_gap) * SAMPLE_RATE)
        if gap_n > 0:
            parts.append(np.zeros(gap_n, dtype=np.float32))
        gap = round(gap_n / SAMPLE_RATE, 4)

        if len(rows_u) > 1:
            token    = " ".join(tokens_u)
            row0     = rows_u[0]
            row_last = rows_u[-1]
            word_adj = adj.get(i, {})
            speed    = word_adj.get("speed", 1.0)
            pitch    = word_adj.get("pitch", 0.0)
            src      = corpus_dir / row0["filename"]
            clip     = _cut_clip(
                src, row0["start_s"], row_last["end_s"],
                speed=speed, pitch=pitch,
            )
            parts.append(clip)
            manifest_words.append(
                {
                    "token":            token,
                    "found":            True,
                    "phrase":           True,
                    "tokens":           tokens_u,
                    "words": [
                        {
                            "token":            t,
                            "word_instance_id": r["id"],
                            "start_s":          round(r["start_s"], 4),
                            "end_s":            round(r["end_s"], 4),
                            "confidence":       r.get("confidence"),
                        }
                        for t, r in zip(tokens_u, rows_u)
                    ],
                    "word_instance_id": row0["id"],
                    "source_file":      row0["filename"],
                    **index.attribution(row0),
                    "date":             row0.get("date"),
                    "source_url":       row0.get("source_url"),
                    "start_s":          round(row0["start_s"], 4),
                    "end_s":            round(row_last["end_s"], 4),
                    "confidence":       min((r.get("confidence") or 0.0) for r in rows_u),
                    "duration_s":       round(row_last["end_s"] - row0["start_s"], 3),
                    "speed":            speed,
                    "pitch":            pitch,
                    "gap":              gap,
                    "fragments":        [{"tier": "phrase", "instance_id": r["id"]} for r in rows_u],
                }
            )
            continue

        token, row = tokens_u[0], rows_u[0]

        if row is None:
            print(f"  [MISSING] '{token}' — gap inserted")
            word_adj  = adj.get(i, {})
            missing_n = round(word_adj.get("gap_s", MISSING_GAP / SAMPLE_RATE) * SAMPLE_RATE)
            parts.append(np.zeros(missing_n, dtype=np.float32))
            manifest_words.append(
                {"token": token, "found": False, "gap_s": round(missing_n / SAMPLE_RATE, 3),
                 "speed": word_adj.get("speed", 1.0), "pitch": word_adj.get("pitch", 0.0),
                 "gap": gap, "fragments": []}
            )
        elif isinstance(row, list):
            word_adj = adj.get(i, {})
            speed    = word_adj.get("speed", 1.0)
            pitch    = word_adj.get("pitch", 0.0)
            clip     = _cut_stitched_clip(row, corpus_dir, speed=speed, pitch=pitch)
            parts.append(clip)

            durations = [r["end_s"] - r["start_s"] for r in row]
            total_dur = sum(durations)
            f0_pairs  = [(r["f0_mean"], d) for r, d in zip(row, durations) if r.get("f0_mean")]
            f0_hz     = (sum(f * d for f, d in f0_pairs) / sum(d for _, d in f0_pairs)) if f0_pairs else None
            manifest_words.append(
                {
                    "token":       token,
                    "found":       True,
                    "stitched":    True,
                    "syllables": [
                        {
                            "phones":               " ".join(_row_phones(r)),
                            "phone_key":            r["phone_key"],
                            "parent_word":          r["parent_word"],
                            "syllable_instance_id": r["id"],
                            "source_file":          r["filename"],
                            **index.attribution(r),
                            "date":                 r.get("date"),
                            "start_s":              round(r["start_s"], 4),
                            "end_s":                round(r["end_s"], 4),
                            "run_id":               r.get("run_id"),
                        }
                        for r in row
                    ],
                    **index.stitched_attribution(row),
                    "date":       row[0].get("date"),
                    "source_url": row[0].get("source_url"),
                    "confidence": min((r.get("confidence") or 0.0) for r in row),
                    "duration_s": round(total_dur, 3),
                    "f0_hz":      round(f0_hz, 1) if f0_hz else None,
                    "speed":      speed,
                    "pitch":      pitch,
                    "gap":        gap,
                    "fragments": [
                        {"tier": "syllable" if "phones" in r else "phone",
                         "instance_id": r["id"], "run_id": r.get("run_id")}
                        for r in row
                    ],
                }
            )
        else:
            src       = corpus_dir / row["filename"]
            word_adj  = adj.get(i, {})
            clip      = _cut_clip(
                src, row["start_s"], row["end_s"],
                speed=word_adj.get("speed", 1.0),
                pitch=word_adj.get("pitch", 0.0),
            )
            parts.append(clip)
            manifest_words.append(
                {
                    "token":            token,
                    "found":            True,
                    "word_instance_id": row["id"],
                    "word":             row["word"],
                    "source_file":      row["filename"],
                    **index.attribution(row),
                    "date":             row["date"],
                    "source_url":       row["source_url"],
                    "start_s":          round(row["start_s"], 4),
                    "end_s":            round(row["end_s"], 4),
                    "confidence":       row["confidence"],
                    "speed":            word_adj.get("speed", 1.0),
                    "pitch":            word_adj.get("pitch", 0.0),
                    "gap":              gap,
                    "fragments":        [{"tier": "word", "instance_id": row["id"]}],
                }
            )

    if not parts:
        return np.zeros(SAMPLE_RATE // 10, dtype=np.float32), []

    audio = parts[0]
    for part in parts[1:]:
        audio = _crossfade_join(audio, part)

    return audio, manifest_words


def compute_output_times(words: list[dict]) -> list[dict]:
    """
    Estimate per-word positions (seconds) in the assembled output.
    Mirrors the client-side computeOutputTimes() — approximation that ignores crossfade overlap.
    Words is the render['words'] list (needs 'found', 'duration_s', 'speed').
    A phrase run (consecutive words sharing 'phrase_id') is ONE continuous cut in the
    real audio (no per-member pad/gap — see assemble()'s phrase-unit branch), so its
    members share one padded span, subdivided proportionally by each member's own
    duration rather than each getting an independent pad+gap.
    Returns list of {'start': float, 'end': float} parallel to words.
    """
    GAP_S     = WORD_GAP    / SAMPLE_RATE
    MISSING_S = MISSING_GAP / SAMPLE_RATE
    t = 0.0
    times: list[dict] = []
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        if not w.get("found"):
            times.append({"start": round(t, 4), "end": round(t + MISSING_S, 4)})
            t += MISSING_S + GAP_S
            i += 1
            continue

        if w.get("phrase_id") is not None and w.get("phrase_pos") == 0:
            plen      = w.get("phrase_len", 1)
            group     = words[i:i + plen]
            durs      = [g.get("duration_s", 0.0) for g in group]
            total_nat = sum(durs)
            pad       = min(CLIP_PAD, max(total_nat * 0.25, 0.010))
            speed     = max(group[0].get("speed", 1.0), 0.1)
            inner_span = total_nat / speed
            sub_t = t + pad / speed
            for d in durs:
                seg = (d / total_nat) * inner_span if total_nat > 0 else 0.0
                times.append({"start": round(sub_t, 4), "end": round(sub_t + seg, 4)})
                sub_t += seg
            t = sub_t + pad / speed + GAP_S
            i += plen
            continue

        dur   = w.get("duration_s", 0.0)
        pad   = min(CLIP_PAD, max(dur * 0.25, 0.010))
        speed = max(w.get("speed", 1.0), 0.1)
        effective = (dur + 2 * pad) / speed
        times.append({"start": round(t, 4), "end": round(t + effective, 4)})
        t += effective + GAP_S
        i += 1
    return times


def write_wav(audio: np.ndarray, output_path: Path) -> None:
    import librosa
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_RATE != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=OUTPUT_RATE)
    sf.write(str(output_path), audio, OUTPUT_RATE, subtype="PCM_16")
