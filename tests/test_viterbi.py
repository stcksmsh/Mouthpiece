"""Viterbi selector: DP against brute force, each transition term, continuity, seeding, edges."""

import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mouthpiece import index, match

ROOT = Path(__file__).resolve().parents[1]


def _row(id=0, source_id=1, start=0.0, end=0.3, word="abc", f0=None, tilt=None, rms=None):
    return {"id": id, "source_id": source_id, "start_s": start, "end_s": end, "word": word,
            "f0_mean": f0, "spectral_tilt": tilt, "rms": rms}


def _node(cost, row):
    return (cost, row, row)


def _path_cost(lattice, path, smooth, continuity):
    nodes = [pos[k] for pos, k in zip(lattice, path, strict=True)]
    return (sum(c for c, _, _ in nodes)
            + sum(match.transition_cost(a[2], b[1], smooth, continuity)
                  for a, b in itertools.pairwise(nodes)))


def _random_row(rng, id):
    def maybe(x):
        return None if rng.random() < 0.2 else x
    start = float(rng.uniform(0, 5))
    return _row(id=id, source_id=int(rng.integers(1, 3)), start=start,
                end=start + float(rng.uniform(0.05, 0.6)), word="x" * int(rng.integers(1, 9)),
                f0=maybe(float(rng.uniform(60, 400))), tilt=maybe(float(rng.uniform(-9, 0))),
                rms=maybe(float(rng.uniform(-50, -15))))


@pytest.mark.parametrize("seed", range(8))
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    lattice = []
    for p in range(n):
        k = int(rng.integers(1, 5))
        nodes = []
        for j in range(k):
            first = _random_row(rng, 10 * p + j)
            last = first if rng.random() < 0.7 else _random_row(rng, 10 * p + j + 500)
            nodes.append((float(rng.uniform(0, 2)), first, last))
        lattice.append(nodes)
    smooth, continuity = float(rng.uniform(0.5, 2)), float(rng.choice([0.0, 1.0]))

    path = match.viterbi(lattice, smooth, continuity)
    brute = min(_path_cost(lattice, p, smooth, continuity)
                for p in itertools.product(*(range(len(pos)) for pos in lattice)))
    assert len(path) == n
    assert _path_cost(lattice, path, smooth, continuity) == pytest.approx(brute, abs=1e-12)


# The rough candidate (index 0) has the cheaper unit cost; the smooth one (index 1) only wins
# through the transition term that tells them apart.
TERMS = {
    "pitch":  ({"f0": 120.0}, {"f0": 240.0}, {"f0": 125.0}),
    "colour": ({"tilt": -2.0}, {"tilt": -9.0}, {"tilt": -2.5}),
    "tempo":  ({"end": 0.3}, {"end": 0.9}, {"end": 0.31}),
    "effort": ({"rms": -20.0}, {"rms": -45.0}, {"rms": -21.0}),
}


@pytest.mark.parametrize("term", TERMS)
def test_each_transition_term_moves_the_choice(term):
    prev, rough, smooth = TERMS[term]
    lattice = [[_node(0.0, _row(id=1, **prev))],
               [_node(0.0, _row(id=2, start=0.0, source_id=2, **rough)),
                _node(0.2, _row(id=3, start=0.0, source_id=3, **smooth))]]
    assert match.viterbi(lattice, smooth=0.0) == [0, 0]
    assert match.viterbi(lattice, smooth=1.0) == [0, 1]


@pytest.mark.parametrize("key", ["f0_mean", "spectral_tilt", "rms"])
def test_null_feature_gives_term_zero(key):
    a = _row(word="", f0=100.0, tilt=-3.0, rms=-20.0)
    b = _row(word="", f0=300.0, tilt=-9.0, rms=-50.0)
    a_only = {**_row(word=""), key: a[key]}
    b_only = {**_row(word=""), key: b[key]}
    assert match.transition_cost(a_only, b_only) > 0
    assert match.transition_cost(a_only, {**b_only, key: None}) == 0.0
    assert match.transition_cost({**a_only, key: None}, b_only) == 0.0
    assert match.transition_cost(a, None) == match.transition_cost(None, b) == 0.0


def test_tempo_null_when_duration_or_length_unknown():
    a, b = _row(end=0.3), _row(end=0.9)
    assert match.transition_cost(a, b) > 0
    assert match.transition_cost(a, {**b, "word": ""}) == 0.0
    assert match.transition_cost(a, {**b, "end_s": 0.0}) == 0.0


def test_tempo_compares_words_and_subword_fragments_in_one_unit():
    # "the" is DH AH: 2 phones in 0.1 s, like the 2-phone fragment that ends a stitched word.
    word = _row(word="the", start=0.0, end=0.1)
    fragment = {**_row(word=None, start=5.0, end=5.1), "phone_key": "DH-AH"}
    assert match.transition_cost(fragment, word) == pytest.approx(0.0)
    assert match.transition_cost(word, fragment) == pytest.approx(0.0)
    slower = {**fragment, "end_s": 5.2}
    assert match.transition_cost(word, slower) == pytest.approx(match.W_TEMPO * math.log(2))


def test_source_change_alone_costs_nothing():
    a = _row(id=1, source_id=1, f0=150.0, tilt=-4.0, rms=-25.0)
    same = _row(id=2, source_id=1, start=5.0, end=5.4, f0=170.0, tilt=-5.0, rms=-28.0)
    other = {**same, "id": 3, "source_id": 9}
    assert match.transition_cost(a, same) == match.transition_cost(a, other)
    assert match.transition_cost(a, same, continuity=1.0) == match.transition_cost(a, other, continuity=1.0)


def test_continuity_bonus_makes_the_adjacent_clip_win():
    prev = _row(id=1, source_id=1, start=0.5, end=1.0)
    elsewhere = _row(id=2, source_id=2, start=3.0, end=3.3)
    follows = _row(id=3, source_id=1, start=1.1, end=1.4)
    lattice = [[_node(0.0, prev)], [_node(0.0, elsewhere), _node(0.5, follows)]]
    assert match.viterbi(lattice, continuity=0.0) == [0, 0]
    assert match.viterbi(lattice, continuity=1.0) == [0, 1]

    too_late = {**follows, "start_s": 1.0 + match.CONTINUITY_GAP_S + 0.05, "end_s": 1.7}
    before = {**follows, "start_s": 0.2, "end_s": 0.4}
    for row in (too_late, before):
        assert match.viterbi([[_node(0.0, prev)], [_node(0.0, elsewhere), _node(0.5, row)]],
                             continuity=1.0) == [0, 0]


def test_fixed_nodes_use_their_edge_fragments():
    stitched_first = _row(id=10, f0=100.0)
    stitched_last = _row(id=11, f0=400.0)
    lattice = [[_node(0.0, _row(id=1, f0=400.0)), _node(0.1, _row(id=2, f0=100.0))],
               [(0.0, stitched_first, stitched_last)],
               [_node(0.0, _row(id=3, f0=100.0)), _node(0.1, _row(id=4, f0=400.0))]]
    assert match.viterbi(lattice) == [1, 0, 1]


def test_all_features_null_reduces_to_unit_cost_plus_continuity():
    rows = [_row(id=i, source_id=i, word="") for i in range(3)]
    lattice = [[_node(0.3, rows[0]), _node(0.1, rows[1])], [_node(0.2, rows[2]), _node(0.0, rows[0])]]
    assert match.viterbi(lattice, smooth=5.0) == [1, 1]
    adj = _row(id=7, source_id=1, start=rows[1]["end_s"] + 0.1, end=1.0, word="")
    assert match.viterbi([[_node(0.3, rows[0]), _node(0.1, rows[1])],
                          [_node(0.0, rows[2]), _node(0.5, adj)]], continuity=1.0) == [1, 1]


def test_ties_go_to_the_lower_index():
    rows = [_row(id=i, word="") for i in range(3)]
    assert match.viterbi([[_node(0.0, r) for r in rows]] * 3) == [0, 0, 0]
    assert match.viterbi([]) == []


@pytest.fixture
def conn(corpus_env):
    c = index.get_connection(Path(corpus_env["MOUTHPIECE_DB"]))
    yield c
    c.close()


def _ids(results):
    return [[r["id"] for r in clip] if isinstance(clip, list) else (clip and clip["id"])
            for _, clip in results]


SENTENCE = "to the to the to the the to"  # every word has 2 or 3 candidates in the fixture


def test_seeding(conn):
    def run(seed):
        return _ids(match.match_text(SENTENCE, conn, rng=np.random.default_rng(seed),
                                     selector="viterbi"))
    assert run(3) == run(3)
    assert len({tuple(run(s)) for s in range(1, 6)}) > 1


def test_temperature_zero_is_deterministic(conn, monkeypatch):
    monkeypatch.setattr(match, "TEMPERATURE", 0.0)
    paths = {tuple(_ids(match.match_text(SENTENCE, conn, rng=np.random.default_rng(s),
                                         selector="viterbi"))) for s in range(10)}
    assert len(paths) == 1
    assert paths == {tuple(_ids(match.match_text(SENTENCE, conn, selector="viterbi")))}


def test_same_seed_same_manifest(run_cli, tmp_path):
    words = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.wav"
        code, stdout = run_cli("render", SENTENCE, "-o", str(out), "--seed", "4",
                               "--selector", "viterbi", "--continuity", "1")
        assert code == 0, stdout
        m = json.loads(out.with_name(f"{name}.manifest.json").read_text())
        assert m["controls"] == {"granularity": 0.0, "diction": 0.0, "emotion": 0.0, "nth": {},
                                 "selector": "viterbi", "smooth": 1.0, "continuity": 1.0,
                                 "sources": {"tiers": ["clean", "unverified"], "only": [], "exclude": []}}
        words.append([w["fragments"] for w in m["words"]])
    assert words[0] == words[1]


def test_edges(conn):
    rng = np.random.default_rng
    one = match.match_text("to", conn, rng=rng(1), selector="viterbi")
    assert len(one) == 1 and one[0][1]["word"] == "to"

    assert match.match_text("zxqvbn qqqqx", conn, rng=rng(1), selector="viterbi") == [
        ("zxqvbn", None), ("qqqqx", None)]

    mid = match.match_text("to presidential the", conn, rng=rng(2), selector="viterbi")
    assert isinstance(mid[0][1], dict) and isinstance(mid[2][1], dict)
    assert isinstance(mid[1][1], list) and len(mid[1][1]) > 1

    third_to = index.search_word(conn, "to")[2]["id"]
    for seed in range(5):
        fixed = match.match_text("to the", conn, rng=rng(seed), nth_overrides={"to": 3},
                                 selector="viterbi")
        assert fixed[0][1]["id"] == third_to


def test_unknown_selector_is_rejected(conn):
    with pytest.raises(ValueError, match="selector"):
        match.match_text("to", conn, selector="beam")


def test_thirty_words_select_under_a_second(conn):
    text = ("to the nation to the tradition of presidents before me this week i traveled "
            "to chicago to deliver my final farewell address to the nation following in "
            "the tradition of")
    assert len(match.tokenize(text)) == 30
    t0 = time.perf_counter()
    results = match.match_text(text, conn, rng=np.random.default_rng(1), selector="viterbi")
    elapsed = time.perf_counter() - t0
    assert all(clip is not None for _, clip in results)
    assert elapsed < 1.0, elapsed


def test_bench_passes_selector_controls(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("bench", ROOT / "tools" / "bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    seen = []

    def fake_render(*args):
        seen.append(args[3])
        return None, 0.0

    monkeypatch.setattr(bench, "render_line", fake_render)
    sentences = tmp_path / "s.txt"
    sentences.write_text("to the\n")
    monkeypatch.setattr(sys, "argv", ["bench", "--out", str(tmp_path / "out"), "--sentences", str(sentences),
                                      "--selector", "viterbi", "--smooth", "2", "--continuity", "1"])
    bench.main()
    assert seen == [["--selector", "viterbi", "--smooth", "2.0", "--continuity", "1.0"]]
