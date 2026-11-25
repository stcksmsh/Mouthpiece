"""Acoustic features per word instance: pitch, loudness and spectral tilt.

A source is analysed once into frame features (20 ms hop at 16 kHz), processed in
blocks so a 9-hour file never needs a whole-file frame matrix in memory. Each word
then aggregates the frames whose centre lies inside [start_s, end_s]; a window too
short to contain a frame centre uses the single frame nearest its midpoint.

Stored values:
  f0_mean        mean f0 in Hz over voiced frames (pyin voicing); NULL if none are voiced
  f0_var         variance of f0 in semitones^2 over the same frames; NULL if none
  rms            20*log10 of the mean frame RMS, in dBFS (a full-scale sine is -3 dB),
                 floored at -100 dB, so it is never NULL once a word has been analysed
  spectral_tilt  least-squares slope, in dB/octave, of the mean power in octave bands
                 centred 250 Hz..4 kHz; speech is typically negative

f0 uses librosa.pyin at a 20 ms hop: about 19x real time measured on a real
16 kHz source, against 8x at a 10 ms hop. librosa.yin was ~360x but has no voicing
decision, so unvoiced and silent frames would leak into the pitch statistics.
"""

import math
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
HOP = 320              # 20 ms
F0_FRAME = 1024        # 64 ms: two periods of the 65 Hz floor
RMS_FRAME = 512        # 32 ms
FMIN, FMAX = 65.0, 500.0
BLOCK_FRAMES = 6000    # 2 minutes of frames per analysis block
BAND_CENTRES = np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0])
RMS_FLOOR_DB = -100.0


def _segment(y: np.ndarray, first: int, n: int, frame: int) -> np.ndarray:
    """Samples for `n` frames of length `frame` centred on hops first..first+n-1, zero-padded at the edges."""
    a = first * HOP - frame // 2
    b = a + (n - 1) * HOP + frame
    seg = y[max(a, 0):min(b, len(y))]
    return np.pad(seg, (max(-a, 0), max(b - len(y), 0)))


def analyse(y: np.ndarray) -> dict[str, np.ndarray]:
    """Frame features for a mono 16 kHz signal. Frame t is centred on sample t*HOP."""
    import librosa

    n = len(y) // HOP + 1
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=RMS_FRAME)
    lo, hi = BAND_CENTRES / math.sqrt(2), BAND_CENTRES * math.sqrt(2)
    band_masks = [(freqs >= l) & (freqs < h) for l, h in zip(lo, hi, strict=True)]

    f0 = np.full(n, np.nan)
    rms = np.zeros(n)
    bands = np.zeros((n, len(BAND_CENTRES)))
    for t0 in range(0, n, BLOCK_FRAMES):
        k = min(BLOCK_FRAMES, n - t0)
        seg = _segment(y, t0, k, F0_FRAME)
        pitch, voiced, _ = librosa.pyin(
            seg, fmin=FMIN, fmax=FMAX, sr=SAMPLE_RATE,
            frame_length=F0_FRAME, hop_length=HOP, center=False,
        )
        f0[t0:t0 + k] = np.where(voiced, pitch, np.nan)[:k]

        seg = _segment(y, t0, k, RMS_FRAME)
        rms[t0:t0 + k] = librosa.feature.rms(
            y=seg, frame_length=RMS_FRAME, hop_length=HOP, center=False)[0][:k]
        power = np.abs(librosa.stft(seg, n_fft=RMS_FRAME, hop_length=HOP, center=False)) ** 2
        bands[t0:t0 + k] = np.stack([power[m].sum(axis=0) for m in band_masks], axis=1)[:k]
    return {"f0": f0, "rms": rms, "bands": bands}


def aggregate(frames: dict[str, np.ndarray], windows: list[tuple[float, float]]) -> list[dict]:
    """Per-window features from `analyse` output; see the module docstring for units."""
    n = len(frames["rms"])
    octaves = np.log2(BAND_CENTRES)
    out = []
    for start, end in windows:
        i0 = max(math.ceil(start * SAMPLE_RATE / HOP), 0)
        i1 = min(math.floor(end * SAMPLE_RATE / HOP) + 1, n)
        if i1 <= i0:
            i0 = min(max(round((start + end) / 2 * SAMPLE_RATE / HOP), 0), n - 1)
            i1 = i0 + 1

        f0 = frames["f0"][i0:i1]
        f0 = f0[np.isfinite(f0)]
        if len(f0):
            f0_mean = float(np.mean(f0))
            f0_var = float(np.var(12 * np.log2(f0)))
        else:
            f0_mean = f0_var = None

        level = float(np.mean(frames["rms"][i0:i1]))
        rms_db = max(20 * math.log10(level), RMS_FLOOR_DB) if level > 0 else RMS_FLOOR_DB

        band_db = 10 * np.log10(np.mean(frames["bands"][i0:i1], axis=0) + 1e-12)
        tilt = float(np.polyfit(octaves, band_db, 1)[0])

        out.append({"f0_mean": f0_mean, "f0_var": f0_var, "rms": rms_db, "spectral_tilt": tilt})
    return out


def for_file(path: Path, windows: list[tuple[float, float]]) -> list[dict]:
    """Load `path` once at 16 kHz mono and return features for each window."""
    import librosa

    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return aggregate(analyse(y), windows)


def backfill(conn, corpus_dir: Path, *, source_id: int | None = None, force: bool = False):
    """Compute features for word instances that lack them, one source at a time.

    `rms` is never NULL after analysis, so it marks which rows are done; each source
    is committed before the next starts, so a rerun resumes where an interrupted one
    stopped. Yields one dict per source that had work: id, filename, updated (rows
    written), seconds, and missing (audio file not found; rows left NULL).
    """
    from . import index

    sql = "SELECT id, filename FROM sources" + (" WHERE id=?" if source_id is not None else "") + " ORDER BY id"
    sources = conn.execute(sql, () if source_id is None else (source_id,)).fetchall()
    for src in sources:
        rows = conn.execute(
            "SELECT id, start_s, end_s FROM word_instances WHERE source_id=?"
            + ("" if force else " AND rms IS NULL") + " ORDER BY id",
            (src["id"],),
        ).fetchall()
        if not rows:
            continue
        report = {"id": src["id"], "filename": src["filename"], "updated": 0, "seconds": 0.0, "missing": False}
        path = corpus_dir / src["filename"]
        if not path.is_file():
            yield {**report, "missing": True}
            continue
        t = time.monotonic()
        feats = for_file(path, [(r["start_s"], r["end_s"]) for r in rows])
        index.update_word_features(conn, [(r["id"], f) for r, f in zip(rows, feats, strict=True)])
        yield {**report, "updated": len(rows), "seconds": time.monotonic() - t}
