"""Reference-audio (voiceover) analysis: extract per-word pitch/duration from a
short recording and compute the pitch/speed adjustments needed to make a render
follow it."""

import difflib
from pathlib import Path

SAMPLE_RATE = 16_000

PITCH_CLAMP_ST = 12.0
SPEED_MIN = 0.25
SPEED_MAX = 4.0


def save_reference_temp(audio_bytes: bytes, dest_dir: Path, name: str = "reference") -> Path:
    """Write raw uploaded audio bytes to disk, normalize to 16kHz mono WAV via ffmpeg."""
    import subprocess

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = dest_dir / f"{name}_raw"
    dest = dest_dir / f"{name}.wav"
    raw.write_bytes(audio_bytes)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(dest), "-loglevel", "error",
        ],
        check=True,
    )
    raw.unlink(missing_ok=True)
    return dest


def analyze_reference_audio(audio_path: Path, model_size: str = "tiny") -> list[dict]:
    """Transcribe + compute per-word f0/rms for a reference recording.
    Uses librosa.pyin (more robust than yin on clean, close-mic'd speech)."""
    import librosa
    import numpy as np
    import soundfile as sf

    from . import transcribe

    words = transcribe.transcribe_audio(audio_path, model_size=model_size)
    # The file is ffmpeg's own WAV output (save_reference_temp), so a read failure is
    # a real error and surfaces to the caller instead of silently dropping pitch.
    audio, _ = sf.read(str(audio_path), dtype="float32", always_2d=False)

    enriched = []
    for w in words:
        s = int(w["start"] * SAMPLE_RATE)
        e = int(w["end"] * SAMPLE_RATE)
        clip = audio[max(0, s): min(len(audio), e)]
        if len(clip) < 256:
            enriched.append({**w, "f0_mean": None, "rms": None})
            continue
        rms = float(np.sqrt(np.mean(clip ** 2)))
        try:
            f0, voiced_flag, _ = librosa.pyin(
                clip, fmin=65, fmax=500, sr=SAMPLE_RATE,
                frame_length=max(256, min(2048, len(clip) // 2 * 2)),
            )
            voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            voiced = voiced[~np.isnan(voiced)]
            f0_mean = float(np.median(voiced)) if len(voiced) else None
        except (ValueError, librosa.util.exceptions.ParameterError):
            # pyin rejects some clip/frame length combinations; that word just gets no pitch.
            f0_mean = None
        enriched.append({**w, "f0_mean": f0_mean, "rms": rms})

    return enriched


def _align_indices(render_tokens: list[str], ref_tokens: list[str]) -> list[tuple[int, int]]:
    """Align two word sequences by longest-common-subsequence of tokens;
    returns list of (render_index, reference_index) pairs for matched positions."""
    sm = difflib.SequenceMatcher(a=render_tokens, b=ref_tokens, autojunk=False)
    pairs = []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            pairs.append((block.a + k, block.b + k))
    return pairs


def compute_pitch_adjustments(
    render_words: list[dict],
    reference_words: list[dict],
) -> list[dict]:
    """Given the current render's words (with token/f0_hz/duration_s) and a reference
    recording's word list (with word/f0_mean/start/end), return per-word
    {word_index, pitch_st, speed} adjustments that push the render toward the
    reference's prosody. Falls back to positional alignment when token sets differ."""
    render_tokens = [w["token"] for w in render_words]
    ref_tokens = [w["word"] for w in reference_words]
    pairs = _align_indices(render_tokens, ref_tokens)

    import math

    out = []
    for ri, fi in pairs:
        rw = render_words[ri]
        fw = reference_words[fi]
        if not rw.get("found"):
            continue

        pitch_st = 0.0
        clip_f0 = rw.get("f0_hz")
        ref_f0 = fw.get("f0_mean")
        if clip_f0 and ref_f0:
            pitch_st = 12.0 * math.log2(ref_f0 / clip_f0)
            pitch_st = max(-PITCH_CLAMP_ST, min(PITCH_CLAMP_ST, pitch_st))

        speed = rw.get("speed", 1.0)
        clip_dur = rw.get("duration_s")
        ref_dur = (fw.get("end") or 0) - (fw.get("start") or 0)
        if clip_dur and ref_dur and ref_dur > 0:
            speed = max(SPEED_MIN, min(SPEED_MAX, clip_dur / ref_dur))

        out.append({"word_index": ri, "pitch_st": pitch_st, "speed": speed})

    return out
