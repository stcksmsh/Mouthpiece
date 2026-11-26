"""Candidate scoring in match.py, and golden selections that pin its behaviour."""

import json
from pathlib import Path

import numpy as np
import pytest

from mouthpiece import match

GOLDEN = sorted((Path(__file__).parent / "golden").glob("*.json"))


def _sequence(m: dict) -> list:
    return [{"token": w["token"], "fragments": [[f["tier"], f["instance_id"]] for f in w["fragments"]]}
            for w in m["words"]]


def test_golden_cases_exist():
    assert len(GOLDEN) >= 6


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_golden_manifests_unchanged(run_cli, tmp_path, path):
    golden = json.loads(path.read_text())
    out = tmp_path / "r.wav"
    code, stdout = run_cli("render", golden["text"], "-o", str(out), *golden["args"])
    assert code == 0, stdout
    m = json.loads(out.with_name("r.manifest.json").read_text())
    assert _sequence(m) == golden["words"]


def _row(conf, f0_var=None, rms=None, id=0, source_id=None):
    return {"id": id, "confidence": conf, "f0_var": f0_var, "rms": rms, "source_id": source_id}


@pytest.mark.parametrize("conf", [0.9, 0.4, None])
def test_zero_sliders_give_the_confidence_weight(conf):
    assert match.unit_weight(_row(conf)) == (conf or 0.5)
    assert match.unit_weight(_row(conf), 0.0, 0.0, intensity_score=0.9) == (conf or 0.5)


def test_diction_moves_weight_towards_clear_or_mumbled_clips():
    clear, mumbled = _row(0.95), _row(0.4)

    def ratio(diction):
        return match.unit_weight(clear, diction) / match.unit_weight(mumbled, diction)

    neutral = ratio(0.0)
    assert ratio(1.0) > ratio(0.5) > neutral
    assert ratio(-1.0) < ratio(-0.5) < neutral


def test_emotion_moves_weight_towards_expressive_or_flat_clips():
    # The bias multiplier never exceeds 1, so direction shows as the ratio of a
    # lively clip's weight to a flat one's.
    row = _row(0.8)

    def ratio(emotion):
        lively = match.unit_weight(row, intensity=emotion, intensity_score=0.9)
        flat = match.unit_weight(row, intensity=emotion, intensity_score=0.1)
        return lively / flat

    assert ratio(0.0) == 1.0
    assert ratio(1.0) > ratio(0.5) > 1.0
    assert ratio(-1.0) < ratio(-0.5) < 1.0
    assert match.unit_weight(row, intensity=1.0, intensity_score=1.0) == match.unit_weight(row)
    assert match.unit_weight(row, intensity=-1.0, intensity_score=0.0) == match.unit_weight(row)
    assert match.unit_weight(row, intensity=-1.0, intensity_score=1.0) < match.unit_weight(row)


def test_pool_weights_rank_emotion_within_the_pool():
    flat = _row(0.8, f0_var=1.0, rms=0.01, id=1, source_id=1)
    lively = _row(0.8, f0_var=50.0, rms=0.20, id=2, source_id=1)
    norms = {1: {"f0_var": 10.0, "rms": 0.1}}
    up = match.pool_weights([flat, lively], intensity=1.0, norms=norms)
    down = match.pool_weights([flat, lively], intensity=-1.0, norms=norms)
    assert up[1] > up[0]
    assert down[0] > down[1]
    assert match.pool_weights([flat, lively]) == [0.8, 0.8]


def test_pool_weights_single_and_empty():
    assert match.pool_weights([_row(0.7)], diction=1.0, intensity=1.0) == [
        match.unit_weight(_row(0.7), 1.0, 1.0, 0.5)]
    assert match.pool_weights([]) == []


def test_candidate_pool_drops_low_confidence_and_falls_back_when_all_low():
    good, low, unknown = _row(0.9, id=1), _row(0.1, id=2), _row(None, id=3)
    assert match.candidate_pool([good, low, unknown]) == [good]
    everything_low = [low, unknown]
    assert match.candidate_pool(everything_low) == everything_low
    assert match.candidate_pool([]) == []


def test_candidate_pool_keeps_first_top_k():
    rows = [_row(0.9, id=i) for i in range(match.TOP_K + 3)]
    assert match.candidate_pool(rows) == rows[:match.TOP_K]


def test_pick_single_candidate_and_all_below_floor():
    only = _row(0.1, id=7)
    assert match._pick([only], np.random.default_rng(0), None) is only
    low = [_row(0.1, id=1), _row(0.2, id=2)]
    picks = {match._pick(low, np.random.default_rng(s), None)["id"] for s in range(20)}
    assert picks == {1, 2}
