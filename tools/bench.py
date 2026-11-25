"""Listening bench: render a fixed sentence list with fixed seeds and count what the engine did.

    python tools/bench.py [--label NAME] [--seed-offset N] [--out DIR] [--sentences FILE]
                          [--granularity X] [--diction X] [--emotion X]
                          [--selector greedy|viterbi] [--smooth X] [--continuity X]

Writes DIR/<sha>[-dirty][-label]/NN.wav, NN.manifest.json, summary.csv and index.txt, where NN
is the 1-based line number and the render seed is NN + seed-offset. DB, corpus and render paths
come from the same MOUTHPIECE_* environment variables the CLI reads.

Counts come from the v2 manifest. A unit's source is its `source_file`; a syllable/phone-stitched
word has one source per fragment. `source_switches` counts changes between consecutive fragments
across the sentence, so a word stitched from 3 sources adds 2, and a gap does not break the chain.
`runtime_s` is the wall time of a CLI subprocess, process start included.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENTENCES = ROOT / "bench" / "sentences.txt"
COLUMNS = ["line", "words", "source_switches", "distinct_sources", "subword_words",
           "subword_fragments", "gap_words", "duration_s", "runtime_s"]
SUBWORD_TIERS = {"syllable", "phone"}


def unit_sources(unit: dict) -> list[str]:
    """Source file of each fragment of a found unit, in playback order."""
    if unit.get("stitched"):
        return [s["source_file"] for s in unit["syllables"]]
    return [unit["source_file"]]


def manifest_metrics(manifest: dict) -> dict:
    """Counts for one render. `words` counts input tokens (a phrase unit holds several)."""
    sources: list[str] = []
    words = subword_words = subword_fragments = gap_words = 0
    for unit in manifest["words"]:
        words += len(unit.get("tokens") or [unit["token"]])
        if not unit["found"]:
            gap_words += 1
            continue
        sources += unit_sources(unit)
        if {f["tier"] for f in unit["fragments"]} & SUBWORD_TIERS:
            subword_words += 1
            subword_fragments += len(unit["fragments"])
    return {
        "words": words,
        "source_switches": sum(a != b for a, b in pairwise(sources)),
        "distinct_sources": len(set(sources)),
        "subword_words": subword_words,
        "subword_fragments": subword_fragments,
        "gap_words": gap_words,
    }


def speaker_sequence(manifest: dict) -> str:
    return " | ".join(u["speaker_label"] if u["found"] else "(gap)" for u in manifest["words"])


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def run_name(label: str | None) -> str:
    name = git("rev-parse", "--short", "HEAD")
    if git("status", "--porcelain"):
        name += "-dirty"
    return f"{name}-{label}" if label else name


def render_line(text: str, seed: int, wav: Path, controls: list[str]) -> tuple[dict | None, float]:
    """Render via the CLI. Returns (manifest or None if nothing was found, wall seconds)."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "mouthpiece", "render", text, "--seed", str(seed), "-o", str(wav), *controls],
        env=env, capture_output=True, text=True, check=False,
    )
    runtime = time.perf_counter() - t0
    manifest_path = wav.with_name(wav.stem + ".manifest.json")
    if proc.returncode != 0 or not manifest_path.exists():
        # "no words found in index" exits nonzero; any other failure is a bench error.
        if "no words found" not in proc.stdout + proc.stderr:
            sys.exit(f"render failed for {text!r}:\n{proc.stdout}{proc.stderr}")
        return None, runtime
    return json.loads(manifest_path.read_text()), runtime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("renders/bench"))
    ap.add_argument("--label")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    for flag in ("granularity", "diction", "emotion", "smooth", "continuity"):
        ap.add_argument(f"--{flag}", type=float)
    ap.add_argument("--selector", choices=("greedy", "viterbi"))
    args = ap.parse_args()

    lines = [ln.strip() for ln in args.sentences.read_text().splitlines() if ln.strip()]
    controls = [x for f in ("granularity", "diction", "emotion", "selector", "smooth", "continuity")
                if getattr(args, f) is not None for x in (f"--{f}", str(getattr(args, f)))]

    run_dir = args.out / run_name(args.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, index = [], []
    for n, text in enumerate(lines, 1):
        wav = run_dir / f"{n:02d}.wav"
        wav.unlink(missing_ok=True)
        manifest, runtime = render_line(text, n + args.seed_offset, wav, controls)
        if manifest is None:
            m = {"words": len(text.split()), "source_switches": 0, "distinct_sources": 0,
                 "subword_words": 0, "subword_fragments": 0, "gap_words": len(text.split())}
            duration, seq = 0.0, "(nothing found)"
        else:
            m = manifest_metrics(manifest)
            duration, seq = sf.info(str(wav)).duration, speaker_sequence(manifest)
        rows.append({"line": n, **m, "duration_s": f"{duration:.3f}", "runtime_s": f"{runtime:.2f}"})
        index.append(f"{n:02d}  {text}\n    {seq}\n")
        print(f"{n:02d}/{len(lines)}  switches={m['source_switches']}  gaps={m['gap_words']}  {runtime:.1f}s", flush=True)

    total = {"line": "TOTAL"}
    for col in COLUMNS[1:-2]:
        total[col] = sum(r[col] for r in rows)
    total["duration_s"] = f"{sum(float(r['duration_s']) for r in rows):.3f}"
    total["runtime_s"] = f"{sum(float(r['runtime_s']) for r in rows) / max(len(rows), 1):.2f}"
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows([*rows, total])
    (run_dir / "index.txt").write_text("\n".join(index))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
