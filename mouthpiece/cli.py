"""CLI entry point. Supports both subcommand and bare-text render forms:

    mouthpiece "some text" -o out.wav          (bare render)
    mouthpiece render "some text" -o out.wav   (explicit)
    mouthpiece ingest audio.wav -m meta.json
    mouthpiece batch-ingest sources.csv
    mouthpiece coverage "some text"
    mouthpiece sources
    mouthpiece resyllabify
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


_SUBCOMMANDS = {"ingest", "batch-ingest", "render", "coverage", "sources", "remove", "wipe", "candidates", "serve", "resyllabify"}


def _paths() -> dict[str, Path]:
    return {
        "db":     Path(os.environ.get("MOUTHPIECE_DB",      "db/mouthpiece.db")),
        "corpus": Path(os.environ.get("MOUTHPIECE_CORPUS",  "corpus")),
        "renders":Path(os.environ.get("MOUTHPIECE_RENDERS", "renders")),
    }


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    from . import index, ingest

    p = _paths()
    conn = index.get_connection(p["db"])

    audio_path = Path(args.audio)
    meta_path  = Path(args.metadata) if args.metadata else None

    if not audio_path.exists():
        sys.exit(f"Error: audio file not found: {audio_path}")
    if meta_path and not meta_path.exists():
        sys.exit(f"Error: metadata file not found: {meta_path}")

    print(f"Ingesting {audio_path} ...")
    ingest.ingest_file(
        audio_path, meta_path,
        conn=conn, corpus_dir=p["corpus"],
        model_size=args.model,
        reindex=args.reindex,
    )
    conn.close()


def _parse_nth(nth_args: list[str]) -> dict[str, int]:
    """Parse ['me:2', 'the:3'] → {'me': 2, 'the': 3} (1-indexed ranks)."""
    overrides = {}
    for item in (nth_args or []):
        if ":" not in item:
            sys.exit(f"--nth must be WORD:N (e.g. --nth me:2), got: {item!r}")
        word, _, n = item.partition(":")
        if not n.isdigit() or int(n) < 1:
            sys.exit(f"--nth rank must be a positive integer, got: {item!r}")
        overrides[word.lower().strip()] = int(n)
    return overrides


def cmd_render(args: argparse.Namespace) -> None:
    import numpy as np
    from . import assemble, index, manifest, match

    p = _paths()
    conn = index.get_connection(p["db"])
    p["renders"].mkdir(parents=True, exist_ok=True)

    text = args.text
    output = (
        Path(args.output)
        if args.output
        else p["renders"] / f"render_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )
    nth = _parse_nth(getattr(args, "nth", None) or [])

    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2**31))
    rng  = np.random.default_rng(seed)
    print(f"Seed: {seed}  (re-render with --seed {seed} to reproduce)")

    print(f"Matching: {text!r}")
    matches = match.match_text(
        text, conn, rng=rng, nth_overrides=nth,
        granularity=args.granularity, diction=args.diction, emotion=args.emotion,
    )
    report  = match.coverage_report(matches)

    print(f"Coverage: {report['found']}/{report['total']} words found "
          f"({report['found_word']} whole-word, {report['found_phrase']} phrase, "
          f"{report['found_syllables']} syllable-stitched)")
    if report["missing"]:
        print(f"Missing:  {', '.join(report['missing_tokens'])}")

    if report["found"] == 0:
        conn.close()
        sys.exit("Error: no words found in index. Run 'mouthpiece ingest' first.")

    print("Assembling ...")
    audio, words = assemble.assemble(matches, p["corpus"])
    assemble.write_wav(audio, output)

    manifest_path = manifest.write_manifest(text, words, output, seed=seed)

    conn.close()
    print(f"\nOutput:   {output}")
    print(f"Manifest: {manifest_path}")
    duration = len(audio) / assemble.SAMPLE_RATE
    print(f"Duration: {duration:.2f}s")


def cmd_candidates(args: argparse.Namespace) -> None:
    from . import index

    p = _paths()
    conn = index.get_connection(p["db"])
    word = args.word.lower().strip()
    rows = index.search_word(conn, word, limit=20)
    conn.close()

    if not rows:
        print(f"No candidates found for '{word}'.")
        return

    print(f"{len(rows)} candidate(s) for '{word}':\n")
    for i, r in enumerate(rows, 1):
        conf = f"{r['confidence']:.3f}" if r['confidence'] else "n/a"
        dur  = round(r['end_s'] - r['start_s'], 3)
        speaker = index.speaker_label(r)
        print(f"  [{i}] id={r['id']}  conf={conf}  dur={dur}s  {speaker}  {r['date'] or ''}  {r['filename']}")


def cmd_coverage(args: argparse.Namespace) -> None:
    import numpy as np
    from . import index, match

    p = _paths()
    conn = index.get_connection(p["db"])
    # Phrase-search and forced-syllable branches only trigger with an rng — supply
    # one whenever a slider is actually in use, else stay deterministic (unchanged).
    rng = np.random.default_rng() if (args.granularity or args.diction or args.emotion) else None
    matches = match.match_text(
        args.text, conn, rng=rng,
        granularity=args.granularity, diction=args.diction, emotion=args.emotion,
    )
    conn.close()

    report = match.coverage_report(matches)
    print(f"Coverage: {report['found']}/{report['total']} words "
          f"({report['found_word']} whole-word, {report['found_phrase']} phrase, "
          f"{report['found_syllables']} syllable-stitched)")
    if report["found_phrase_tokens"]:
        print(f"Phrases:  {', '.join(report['found_phrase_tokens'])}")
    if report["found_syllable_tokens"]:
        print(f"Stitched: {', '.join(report['found_syllable_tokens'])}")
    if report["missing_tokens"]:
        print(f"Missing:  {', '.join(report['missing_tokens'])}")
    else:
        print("All words covered.")


def cmd_batch_ingest(args: argparse.Namespace) -> None:
    import csv
    from . import index, ingest

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: CSV file not found: {csv_path}")

    # Resolve relative audio paths from the CSV's own directory.
    csv_dir = csv_path.parent

    p = _paths()
    conn = index.get_connection(p["db"])

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        sys.exit("CSV is empty.")

    missing_col = "file" not in (reader.fieldnames or [])
    if missing_col:
        sys.exit("CSV must have a 'file' column.")

    ok = skipped = failed = 0

    for i, row in enumerate(rows, 1):
        raw = Path(row["file"].strip())
        if raw.is_absolute():
            audio_path = raw
        elif (csv_dir / raw).exists():
            audio_path = csv_dir / raw
        elif (p["corpus"] / raw.name).exists():
            audio_path = p["corpus"] / raw.name
        else:
            audio_path = raw  # will fail the exists() check below with a clear message

        print(f"\n[{i}/{len(rows)}] {audio_path.name}")

        if not audio_path.exists():
            print(f"  ERROR: file not found (tried {csv_dir / raw} and {p['corpus'] / raw.name}) — skipping")
            failed += 1
            continue

        # Write a temp metadata dict from CSV columns; ingest expects a JSON file
        # or None. We pass None and override via a small shim.
        meta = {k: v.strip() for k, v in row.items() if k != "file" and v.strip()}

        try:
            ingest.ingest_file_with_meta(
                audio_path, meta,
                conn=conn, corpus_dir=p["corpus"],
                model_size=args.model,
                reindex=args.reindex,
            )
            ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    conn.close()
    print(f"\nDone. {ok} ingested, {failed} failed, {skipped} skipped.")


def cmd_remove(args: argparse.Namespace) -> None:
    from . import index

    p = _paths()
    conn = index.get_connection(p["db"])

    # Accept numeric ID or filename/stem
    target = args.source.strip()
    if target.isdigit():
        row = conn.execute(
            "SELECT id, filename, speaker, source_label, date FROM sources WHERE id=?", (int(target),)
        ).fetchone()
    else:
        # match on full filename or stem
        row = conn.execute(
            "SELECT id, filename, speaker, source_label, date FROM sources WHERE filename=? OR filename=?",
            (target, target + ".wav"),
        ).fetchone()

    if row is None:
        conn.close()
        sys.exit(f"No source found matching '{target}'. Run 'mouthpiece sources' to list them.")

    source_id, filename, speaker = row["id"], row["filename"], index.speaker_label(dict(row))
    word_count = index.word_count(conn, source_id)

    print(f"  [{source_id}] {filename}  ({speaker}, {word_count} words)")

    if not args.yes:
        answer = input("Delete this source and all its word instances? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            conn.close()
            return

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
    conn.execute("DELETE FROM sources WHERE id=?", (source_id,))
    conn.commit()
    print(f"  Removed {word_count} word instances and source record.")

    if args.files:
        wav = p["corpus"] / filename
        if wav.exists():
            wav.unlink()
            print(f"  Deleted {wav}")
        else:
            print(f"  Corpus WAV not found at {wav} (already gone?)")

    conn.close()


def cmd_wipe(args: argparse.Namespace) -> None:
    from . import index

    p = _paths()
    conn = index.get_connection(p["db"])

    sources = index.list_sources(conn)
    if not sources:
        print("Nothing to wipe — corpus is already empty.")
        conn.close()
        return

    total_words = sum(s["word_count"] for s in sources)
    print(f"  {len(sources)} sources, {total_words} word instances")
    for s in sources:
        print(f"    [{s['id']}] {s['filename']}  ({index.speaker_label(s)})")

    if not args.yes:
        answer = input("\nWipe entire corpus from DB? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            conn.close()
            return

    conn.execute("DELETE FROM phone_instances")
    conn.execute("DELETE FROM syllable_instances")
    conn.execute("DELETE FROM word_instances")
    conn.execute("DELETE FROM sources")
    conn.execute("DELETE FROM word_fts")
    conn.commit()
    print(f"  DB cleared.")

    if args.files:
        wavs = list(p["corpus"].glob("*.wav"))
        for wav in wavs:
            wav.unlink()
        print(f"  Deleted {len(wavs)} WAV file(s) from {p['corpus']}/")

    conn.close()


def cmd_resyllabify(args: argparse.Namespace) -> None:
    from . import index, phonemes

    p = _paths()
    conn = index.get_connection(p["db"])

    n_norm = index.normalize_stored_words(conn)
    if n_norm:
        print(f"Stripped punctuation from {n_norm} stored word(s) (e.g. 'week,' -> 'week').")

    if args.reindex:
        all_words = index.all_word_instances(conn)
        conn.execute("DELETE FROM syllable_instances")
        conn.execute("DELETE FROM phone_instances")
        conn.commit()
        needs_syll  = {w["id"] for w in all_words}
        needs_phone = {w["id"] for w in all_words}
        target_words = all_words
    else:
        missing_syll  = index.word_instances_missing_units(conn, "syllable")
        missing_phone = index.word_instances_missing_units(conn, "phone")
        needs_syll  = {w["id"] for w in missing_syll}
        needs_phone = {w["id"] for w in missing_phone}
        by_id = {w["id"]: w for w in missing_syll + missing_phone}
        target_words = list(by_id.values())

    print(f"Processing {len(target_words)} word instance(s) ...")
    n_words = n_syll = n_phone = n_oov = 0
    for w in target_words:
        did_something = False
        if w["id"] in needs_syll:
            rows = phonemes.compute_syllable_rows(w["word"], w["start_s"], w["end_s"])
            if rows:
                index.insert_syllable_instances(conn, w["id"], rows)
                n_syll += len(rows)
                did_something = True
        if w["id"] in needs_phone:
            prows = phonemes.compute_phone_rows(w["word"], w["start_s"], w["end_s"])
            if prows:
                index.insert_phone_instances(conn, w["id"], prows)
                n_phone += len(prows)
                did_something = True
        if did_something:
            n_words += 1
        else:
            n_oov += 1

    conn.close()
    print(f"Done. {n_words} words processed ({n_syll} syllables, {n_phone} phones), "
          f"{n_oov} not in CMUdict.")


def cmd_sources(args: argparse.Namespace) -> None:
    from . import index

    p = _paths()
    conn = index.get_connection(p["db"])
    sources = index.list_sources(conn)
    conn.close()

    if not sources:
        print("No sources ingested yet.")
        return

    for s in sources:
        label = index.speaker_label(s)
        # The unidentified label already ends with the date; add it only for named speakers.
        if label == s["speaker"]:
            label += f", {s['date'] or 'no date'}"
        print(f"  [{s['id']}] {s['filename']}  — {label}  ({s['word_count']} words)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # Allow bare-text render: mouthpiece "text" -o out.wav
    if (
        len(sys.argv) > 1
        and sys.argv[1] not in _SUBCOMMANDS
        and not sys.argv[1].startswith("-")
    ):
        sys.argv.insert(1, "render")

    parser = argparse.ArgumentParser(
        prog="mouthpiece",
        description="Concatenative speech-collage engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest a source audio file")
    p_ingest.add_argument("audio", help="Path to source audio (WAV, MP3, FLAC, …)")
    p_ingest.add_argument("--metadata", "-m", help="Path to JSON metadata file")
    p_ingest.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="WhisperX model size (default: base)",
    )
    p_ingest.add_argument(
        "--reindex", action="store_true",
        help="Re-transcribe even if already indexed",
    )

    # --- render ---
    p_render = sub.add_parser("render", help="Render text to speech-collage WAV")
    p_render.add_argument("text", help="Text to render")
    p_render.add_argument("--output", "-o", help="Output WAV path")
    p_render.add_argument(
        "--nth", action="append", metavar="WORD:N",
        help="Use Nth-best clip for a word, e.g. --nth me:2  (repeatable)",
    )
    p_render.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible renders (printed after each render)",
    )
    p_render.add_argument(
        "--granularity", type=float, default=0.0,
        help="-1 (syllable-per-syllable) .. 0 (default) .. +1 (long uniform phrases)",
    )
    p_render.add_argument(
        "--diction", type=float, default=0.0,
        help="-1 (mumbled) .. 0 (default) .. +1 (crisp)",
    )
    p_render.add_argument(
        "--emotion", type=float, default=0.0,
        help="-1 (flat) .. 0 (default) .. +1 (expressive)",
    )

    # --- coverage ---
    p_cov = sub.add_parser("coverage", help="Check word coverage without rendering")
    p_cov.add_argument("text", help="Text to check")
    p_cov.add_argument("--granularity", type=float, default=0.0)
    p_cov.add_argument("--diction", type=float, default=0.0)
    p_cov.add_argument("--emotion", type=float, default=0.0)

    # --- batch-ingest ---
    p_batch = sub.add_parser("batch-ingest", help="Ingest multiple files from a CSV")
    p_batch.add_argument("csv", help="Path to CSV file (see corpus/batch_example.csv)")
    p_batch.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="WhisperX model size (default: base)",
    )
    p_batch.add_argument(
        "--reindex", action="store_true",
        help="Re-transcribe files that are already indexed",
    )

    # --- remove ---
    p_remove = sub.add_parser("remove", help="Remove a source and its word instances")
    p_remove.add_argument("source", help="Source ID (number) or filename/stem")
    p_remove.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p_remove.add_argument(
        "--files", action="store_true",
        help="Also delete the normalized WAV from corpus/",
    )

    # --- wipe ---
    p_wipe = sub.add_parser("wipe", help="Clear the entire corpus from the DB")
    p_wipe.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    p_wipe.add_argument(
        "--files", action="store_true",
        help="Also delete all normalized WAVs from corpus/",
    )

    # --- candidates ---
    p_cand = sub.add_parser("candidates", help="List all clips for a word, ranked by confidence")
    p_cand.add_argument("word", help="Word to look up")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start the local web UI server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8745)

    # --- sources ---
    sub.add_parser("sources", help="List ingested sources")

    # --- resyllabify ---
    p_resyll = sub.add_parser(
        "resyllabify",
        help="Strip punctuation from stored words, then backfill syllable and phone entries",
    )
    p_resyll.add_argument(
        "--reindex", action="store_true",
        help="Recompute syllables for all words, not just ones missing them",
    )

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        print(f"Starting MOUTHPIECE at http://{args.host}:{args.port}")
        uvicorn.run("mouthpiece.server:app", host=args.host, port=args.port, reload=False)
        return

    {
        "ingest":        cmd_ingest,
        "batch-ingest":  cmd_batch_ingest,
        "render":        cmd_render,
        "coverage":      cmd_coverage,
        "candidates":    cmd_candidates,
        "remove":        cmd_remove,
        "wipe":          cmd_wipe,
        "sources":       cmd_sources,
        "resyllabify":   cmd_resyllabify,
    }[args.command](args)
