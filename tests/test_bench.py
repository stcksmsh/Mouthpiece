"""Bench sentence list, metric counts, and an end-to-end run on the fixture corpus."""

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tools" / "bench.py"

_spec = importlib.util.spec_from_file_location("bench", BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_sentence_file_shape():
    lines = (ROOT / "bench" / "sentences.txt").read_text().splitlines()
    assert len(lines) == 20
    assert all(ln.strip() for ln in lines)
    assert sum(len(ln.split()) >= 28 for ln in lines) == 1


def _word(token, sources, tier="word"):
    if tier == "word":
        return {"token": token, "found": True, "source_file": sources[0],
                "fragments": [{"tier": "word", "instance_id": 1}]}
    return {"token": token, "found": True, "stitched": True,
            "syllables": [{"source_file": s} for s in sources],
            "fragments": [{"tier": tier, "instance_id": i} for i, _ in enumerate(sources)]}


def test_manifest_metrics_exact_counts():
    manifest = {"words": [
        _word("a", ["s1"]),
        _word("b", ["s1"]),                                # same source: no switch
        _word("c", ["s2"]),                                # s1->s2: switch 1
        {"token": "zz", "found": False, "fragments": []},  # gap: does not break the chain
        _word("d", ["s2", "s3", "s1"], tier="syllable"),   # s2->s2 none, s2->s3, s3->s1: 2 and 3
        {"token": "e", "tokens": ["e", "f"], "found": True, "source_file": "s1",
         "fragments": [{"tier": "phrase", "instance_id": 1}, {"tier": "phrase", "instance_id": 2}]},
        _word("g", ["s4", "s4"], tier="phone"),            # s1->s4: switch 4
    ]}
    assert bench.manifest_metrics(manifest) == {
        "words": 8,
        "source_switches": 4,
        "distinct_sources": 4,
        "subword_words": 2,
        "subword_fragments": 5,
        "gap_words": 1,
    }


def _run_bench(corpus_env, out, sentences):
    proc = subprocess.run(
        [sys.executable, str(BENCH), "--out", str(out), "--label", "t", "--sentences", str(sentences)],
        env={**os.environ, **corpus_env}, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (run_dir,) = out.iterdir()
    return run_dir


def _rows_without_runtime(run_dir):
    with (run_dir / "summary.csv").open() as f:
        return [{k: v for k, v in r.items() if k != "runtime_s"} for r in csv.DictReader(f)]


def test_bench_end_to_end_and_deterministic(corpus_env, tmp_path):
    sentences = tmp_path / "s.txt"
    sentences.write_text("the tradition of presidents\n"
                         "deliver zxqvbn address\n"
                         "i traveled to chicago to deliver my final farewell address\n")
    run_dir = _run_bench(corpus_env, tmp_path / "a", sentences)
    for n in (1, 2, 3):
        assert (run_dir / f"{n:02d}.wav").exists()
        assert (run_dir / f"{n:02d}.manifest.json").exists()
    assert "Barack Obama" in (run_dir / "index.txt").read_text()

    rows = _rows_without_runtime(run_dir)
    assert [r["line"] for r in rows] == ["1", "2", "3", "TOTAL"]
    assert rows[1]["gap_words"] == "1"
    assert rows[-1]["words"] == str(sum(int(r["words"]) for r in rows[:-1]))

    again = _rows_without_runtime(_run_bench(corpus_env, tmp_path / "b", sentences))
    assert rows == again
