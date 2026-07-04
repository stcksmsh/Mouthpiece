"""Ingest a source audio file: normalize, transcribe, and index."""

import json
from pathlib import Path

SAMPLE_RATE = 16_000


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
