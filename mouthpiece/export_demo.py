"""Export curated demo clips + provenance manifest for the stcksmsh.github.io
project page (federated via the io-page branch's demo/ folder). Not part of
the regular CLI surface — a one-off/occasional tool, run manually:

    python -m mouthpiece.export_demo [--db PATH] [--corpus DIR] [--out DIR]

For each curated text snippet, runs the real match_text() pipeline, groups
the result into segments the exact same way assemble.py does for real
renders (_group_phrase_units / _group_syllable_runs — reused directly, not
reimplemented), cuts each segment's real audio, and writes:

    demo/manifest.json
    demo/clips/<slug>-<n>.opus

Every clip in the output is real corpus audio; the manifest records real
provenance (parent word, speaker, source url) per segment so the site can
caption what it's playing.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import assemble, index, match

# Curated to span all four tiers this session's work produced: plain
# whole-word matches, natural phrase reuse, syllable-level natural-run
# stitching, and phoneme-level fallback stitching — plus the project's own
# tagline, spoken by the tool itself.
CURATED_TEXTS = [
    "mouthpiece",
    "concatenative",
    "synthesis",
    "collage",
    "a concatenative speech collage engine",
    "good for the mission",
    "for the crew",
    "photography",
    "navigation",
    "gravitational",
    "astronaut",
    "engine",
    "speech",
    "real people's real words",
    "text in a wave out",
]

SEED = 42


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _write_opus(clip: np.ndarray, dest: Path) -> None:
    import soundfile as sf

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, clip, assemble.SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp.name, "-c:a", "libopus", "-b:a", "32k",
             str(dest), "-loglevel", "error"],
            check=True,
        )


def _row_label(rows: list[dict]) -> str:
    """Human-readable label for a segment: joined phones for a stitched
    segment, or the plain word for a whole-word/phrase segment."""
    if "word" in rows[0] and "phones" not in rows[0] and "phone" not in rows[0]:
        return " ".join(r["word"] for r in rows)
    return "-".join(re.sub(r"\d$", "", p) for r in rows for p in assemble._row_phones(r))


def _export_segment(rows: list[dict], slug: str, seg_idx: int, corpus_dir: Path,
                     out_dir: Path, cut_fn) -> dict:
    first, last = rows[0], rows[-1]
    clip = cut_fn(corpus_dir / first["filename"], first["start_s"], last["end_s"])
    seg_id = f"{slug}-{seg_idx}"
    _write_opus(clip, out_dir / "clips" / f"{seg_id}.opus")
    return {
        "id":          seg_id,
        "audioFile":   f"clips/{seg_id}.opus",
        "durationMs":  round(len(clip) / assemble.SAMPLE_RATE * 1000),
        "label":       _row_label(rows),
        "parentWord":  first.get("word") or first.get("parent_word"),
        "speaker":     first.get("speaker"),
        "sourceUrl":   first.get("source_url"),
        "natural":     len(rows) > 1,  # >1 row glued as one real continuous span
    }


def export(db_path: Path, corpus_dir: Path, out_dir: Path) -> list[dict]:
    conn = index.get_connection(db_path)
    rng = np.random.default_rng(SEED)
    manifest = []

    for text in CURATED_TEXTS:
        matches = match.match_text(text, conn, rng=rng, granularity=0.3)
        report = match.coverage_report(matches)
        if report["missing"]:
            print(f"  SKIP {text!r}: missing {report['missing_tokens']}")
            continue

        slug = _slugify(text)
        segments = []
        seg_idx = 0
        for _first_i, tokens_u, rows_u in assemble._group_phrase_units(matches):
            if len(rows_u) > 1:
                # phrase unit — one continuous real span across multiple tokens
                segments.append(_export_segment(rows_u, slug, seg_idx, corpus_dir,
                                                 out_dir, assemble._cut_clip))
                seg_idx += 1
                continue

            row = rows_u[0]
            if isinstance(row, list):
                # stitched word — further group into natural syllable/phone runs
                for sub in assemble._group_syllable_runs(row):
                    segments.append(_export_segment(sub, slug, seg_idx, corpus_dir,
                                                      out_dir, assemble._cut_syllable_clip))
                    seg_idx += 1
            else:
                segments.append(_export_segment([row], slug, seg_idx, corpus_dir,
                                                  out_dir, assemble._cut_clip))
                seg_idx += 1

        manifest.append({"text": text, "slug": slug, "segments": segments})
        print(f"  {text!r}: {len(segments)} segment(s)")

    conn.close()
    return manifest


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="db/mouthpiece.db")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--out", default="demo")
    args = p.parse_args()

    out_dir = Path(args.out)
    print(f"Exporting demo clips to {out_dir}/ ...")
    manifest = export(Path(args.db), Path(args.corpus), out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_clips = sum(len(e["segments"]) for e in manifest)
    print(f"Done. {len(manifest)} entries, {n_clips} clips, manifest at {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
