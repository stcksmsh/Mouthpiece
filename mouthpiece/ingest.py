"""Ingest a source audio file: normalize, transcribe, and index."""

import json
from pathlib import Path

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


def _compute_prosodics(audio_path: Path, words: list[dict]) -> list[dict]:
    """Enrich word dicts with lemma and the acoustic features from features.py."""
    from . import features

    try:
        feats = features.for_file(audio_path, [(w["start"], w["end"]) for w in words])
    except Exception as exc:
        # Rows keep NULL features; `mouthpiece features` fills them later.
        print(f"  Warning: could not load audio for prosodic analysis: {exc}")
        feats = [{"f0_mean": None, "f0_var": None, "rms": None, "spectral_tilt": None}] * len(words)
    return [{**w, **f, "lemma": _lemmatize(w["word"])} for w, f in zip(words, feats, strict=True)]


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
    from . import index, phonemes, transcribe

    # Checked before any audio is written, so bad metadata leaves nothing behind.
    index.check_tier(meta.get("tier"))
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
        **{k: meta.get(k) for k in index.SOURCE_FIELDS},
    )
    print(f"  Source id={source_id}  ({dest.name})")

    existing = index.word_count(conn, source_id)
    if existing > 0 and not reindex:
        print(f"  Already indexed ({existing} words). Use --reindex to redo.")
        return source_id

    if existing > 0 and reindex:
        conn.execute(
            """DELETE FROM phone_instances WHERE word_instance_id IN
                   (SELECT id FROM word_instances WHERE source_id=?)""",
            (source_id,),
        )
        conn.execute(
            """DELETE FROM syllable_instances WHERE word_instance_id IN
                   (SELECT id FROM word_instances WHERE source_id=?)""",
            (source_id,),
        )
        conn.execute("DELETE FROM word_instances WHERE source_id=?", (source_id,))
        conn.commit()
        print(f"  Cleared {existing} existing word instances.")

    words = transcribe.transcribe_audio(dest, model_size=model_size)
    print(f"  Computing prosodics ({len(words)} clips) ...")
    words = _compute_prosodics(dest, words)
    print(f"  Inserting {len(words)} word instances ...")
    index.insert_word_instances(conn, source_id, words)

    print(f"  Computing syllables and phones ...")
    n_syll = n_phone = 0
    for row in conn.execute(
        "SELECT id, word, start_s, end_s FROM word_instances WHERE source_id=?", (source_id,)
    ).fetchall():
        rows = phonemes.compute_syllable_rows(row["word"], row["start_s"], row["end_s"])
        if rows:
            index.insert_syllable_instances(conn, row["id"], rows)
            n_syll += len(rows)
        prows = phonemes.compute_phone_rows(row["word"], row["start_s"], row["end_s"])
        if prows:
            index.insert_phone_instances(conn, row["id"], prows)
            n_phone += len(prows)
    print(f"  {n_syll} syllables, {n_phone} phones indexed.")

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
