"""Cut clips from corpus, crossfade joins, concatenate to output WAV."""

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000      # internal / corpus rate (WhisperX requires 16 kHz)
OUTPUT_RATE = 44_100      # output WAV rate (standard for DAWs)
CLIP_PAD    = 0.04        # 40 ms handle each side of word timestamp
CROSSFADE   = int(0.025 * SAMPLE_RATE)   # 25 ms crossfade, capped per clip below
FADE_LEN    = int(0.005 * SAMPLE_RATE)   # 5 ms anti-click fade
WORD_GAP    = int(0.10 * SAMPLE_RATE)    # 100 ms silence between words
MISSING_GAP = int(0.40 * SAMPLE_RATE)   # 400 ms silence for missing words

TARGET_RMS  = 0.08   # normalise every clip to ~-22 dBFS
PEAK_LIMIT  = 0.95   # hard ceiling after normalisation


def _apply_fades(clip: np.ndarray) -> np.ndarray:
    if len(clip) < 2 * FADE_LEN:
        return clip
    out = clip.copy()
    ramp = np.linspace(0.0, 1.0, FADE_LEN, dtype=np.float32)
    out[:FADE_LEN]  *= ramp
    out[-FADE_LEN:] *= ramp[::-1]
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


def _crossfade_join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xf = min(CROSSFADE, len(a) // 4, len(b) // 4)
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


def assemble(
    matches: list[tuple[str, dict | None]],
    corpus_dir: Path,
    adjustments: dict[int, dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build audio from match results.
    adjustments: {word_index: {speed, pitch}} for per-word rubberband processing.
    Returns (audio_array_float32, manifest_words).
    """
    adj             = adjustments or {}
    gap_silence     = np.zeros(WORD_GAP,    dtype=np.float32)
    missing_silence = np.zeros(MISSING_GAP, dtype=np.float32)

    parts: list[np.ndarray] = []
    manifest_words: list[dict] = []

    for i, (token, row) in enumerate(matches):
        if i > 0:
            parts.append(gap_silence.copy())

        if row is None:
            print(f"  [MISSING] '{token}' — gap inserted")
            parts.append(missing_silence.copy())
            manifest_words.append(
                {"token": token, "found": False, "gap_s": round(MISSING_GAP / SAMPLE_RATE, 3)}
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
                    "speaker":          row["speaker"],
                    "date":             row["date"],
                    "source_url":       row["source_url"],
                    "start_s":          round(row["start_s"], 4),
                    "end_s":            round(row["end_s"], 4),
                    "confidence":       row["confidence"],
                    "speed":            word_adj.get("speed", 1.0),
                    "pitch":            word_adj.get("pitch", 0.0),
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
    Returns list of {'start': float, 'end': float} parallel to words.
    """
    GAP_S     = WORD_GAP    / SAMPLE_RATE
    MISSING_S = MISSING_GAP / SAMPLE_RATE
    t = 0.0
    times: list[dict] = []
    for w in words:
        if not w.get("found"):
            times.append({"start": round(t, 4), "end": round(t + MISSING_S, 4)})
            t += MISSING_S + GAP_S
        else:
            dur   = w.get("duration_s", 0.0)
            pad   = min(CLIP_PAD, max(dur * 0.25, 0.010))
            speed = max(w.get("speed", 1.0), 0.1)
            effective = (dur + 2 * pad) / speed
            times.append({"start": round(t, 4), "end": round(t + effective, 4)})
            t += effective + GAP_S
    return times


def write_wav(audio: np.ndarray, output_path: Path) -> None:
    import librosa
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_RATE != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=OUTPUT_RATE)
    sf.write(str(output_path), audio, OUTPUT_RATE, subtype="PCM_16")
