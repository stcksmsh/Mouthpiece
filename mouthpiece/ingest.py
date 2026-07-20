"""Ingest a source audio file: normalize, transcribe, and index."""

import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

_lemmatizer = None


def _lemmatize(word: str) -> str | None:
    """WordNet lemma for a word (verb-form first, falls back to noun-form)."""
    global _lemmatizer
    try:
        if _lemmatizer is None:
            from nltk.stem import WordNetLemmatizer
            _lemmatizer = WordNetLemmatizer()
        w = word.lower().strip()
        if not w.isalpha():
            return None
        try:
            lemma = _lemmatizer.lemmatize(w, pos="v")
        except LookupError:
            import nltk
            nltk.download("wordnet", quiet=True)
            lemma = _lemmatizer.lemmatize(w, pos="v")
        if lemma == w:
            lemma = _lemmatizer.lemmatize(w, pos="n")
        return lemma
    except Exception:
        return None


def _spectral_tilt(clip: np.ndarray, sr: int) -> float | None:
    """Log ratio of sub-1kHz energy to 1-4kHz energy — rough voice-quality proxy."""
    try:
        spec = np.abs(np.fft.rfft(clip))
        freqs = np.fft.rfftfreq(len(clip), d=1.0 / sr)
        low = spec[(freqs > 0) & (freqs < 1000)]
        high = spec[(freqs >= 1000) & (freqs < 4000)]
        low_e = float(np.sum(low ** 2))
        high_e = float(np.sum(high ** 2))
        if low_e <= 0 or high_e <= 0:
            return None
        return float(np.log(low_e / high_e))
    except Exception:
        return None


def _compute_prosodics(audio_path: Path, words: list[dict]) -> list[dict]:
    """Enrich word dicts with f0_mean/f0_var (Hz), rms, spectral_tilt, and lemma."""
    try:
        import soundfile as sf
        import librosa
        audio, _ = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        print(f"  Warning: could not load audio for prosodic analysis: {exc}")
        return [
            {**w, "f0_mean": None, "f0_var": None, "rms": None,
             "spectral_tilt": None, "lemma": _lemmatize(w["word"])}
            for w in words
        ]

    enriched = []
    for w in words:
        s = int(w["start"] * SAMPLE_RATE)
        e = int(w["end"]   * SAMPLE_RATE)
        clip = audio[max(0, s) : min(len(audio), e)]
        lemma = _lemmatize(w["word"])

        if len(clip) < 256:
            enriched.append({
                **w, "f0_mean": None, "f0_var": None, "rms": None,
                "spectral_tilt": None, "lemma": lemma,
            })
            continue

        rms = float(np.sqrt(np.mean(clip ** 2)))
        tilt = _spectral_tilt(clip, SAMPLE_RATE)

        try:
            # frame_length must be < clip length; use a power-of-2 ≤ clip//4
            fl = 1
            while fl * 2 <= len(clip) // 4:
                fl *= 2
            fl = max(fl, 128)
            f0 = librosa.yin(clip, fmin=65, fmax=500, sr=SAMPLE_RATE, frame_length=fl)
            voiced = f0[(f0 > 65) & (f0 < 490)]
            f0_mean = float(np.median(voiced)) if len(voiced) else None
            f0_var = float(np.var(voiced)) if len(voiced) else None
        except Exception:
            f0_mean = None
            f0_var = None

        enriched.append({
            **w, "f0_mean": f0_mean, "f0_var": f0_var, "rms": rms,
            "spectral_tilt": tilt, "lemma": lemma,
        })

    return enriched


def _normalize_audio(src: Path, dst: Path) -> None:
    import subprocess
    # Stream through ffmpeg — never loads the full file into RAM.
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(dst), "-loglevel", "error",
        ],
        check=True,
    )


def _load_metadata(meta_path: Path | None) -> dict:
    if meta_path is None:
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _ingest(
    audio_path: Path,
    meta: dict,
    *,
    conn,
    corpus_dir: Path,
    model_size: str = "base",
    reindex: bool = False,
) -> int:
    from . import index, transcribe

    corpus_dir.mkdir(parents=True, exist_ok=True)
    dest = corpus_dir / (audio_path.stem + ".wav")

    if dest.exists() and not reindex:
        print(f"  Normalized WAV exists at {dest} (skipping conversion)")
    else:
        print(f"  Normalizing → {dest}")
        _normalize_audio(audio_path, dest)

    source_id = index.insert_source(
        conn,
        filename=dest.name,
        speaker=meta.get("speaker"),
        date=meta.get("date"),
        event=meta.get("event"),
        source_url=meta.get("source_url"),
        license=meta.get("license"),
    )
    print(f"  Source id={source_id}  ({dest.name})")

    existing = index.word_count(conn, source_id)
    if existing > 0 and not reindex:
        print(f"  Already indexed ({existing} words). Use --reindex to redo.")
        return source_id

    if existing > 0 and reindex:
        conn.execute("DELETE FROM word_instances WHERE source_id=?", (source_id,))
        conn.commit()
        print(f"  Cleared {existing} existing word instances.")

    words = transcribe.transcribe_audio(dest, model_size=model_size)
    print(f"  Computing prosodics ({len(words)} clips) ...")
    words = _compute_prosodics(dest, words)
    print(f"  Inserting {len(words)} word instances ...")
    index.insert_word_instances(conn, source_id, words)

    print(f"  Done. {len(words)} words indexed from '{dest.name}'.")
    return source_id


def ingest_file(
    audio_path: Path,
    metadata_path: Path | None,
    *,
    conn,
    corpus_dir: Path,
    model_size: str = "base",
    reindex: bool = False,
) -> int:
    return _ingest(
        audio_path, _load_metadata(metadata_path),
        conn=conn, corpus_dir=corpus_dir,
        model_size=model_size, reindex=reindex,
    )


def ingest_file_with_meta(
    audio_path: Path,
    meta: dict,
    *,
    conn,
    corpus_dir: Path,
    model_size: str = "base",
    reindex: bool = False,
) -> int:
    return _ingest(
        audio_path, meta,
        conn=conn, corpus_dir=corpus_dir,
        model_size=model_size, reindex=reindex,
    )
