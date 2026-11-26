"""Search-bias controls: diction, intensity, pitch and tempo, normalised per source."""

import numpy as np
import pytest

from mouthpiece import index, match

SEEDS = range(200)
FILLERS = 6  # "world" words per source that set its medians; above MIN_SOURCE_VALUES


def _db(tmp_path, sources: dict[str, tuple[dict, list[dict]]]):
    """One source per entry: FILLERS "world" words with the `base` features,
    then the "hello" clips. Returns (conn, {(source, n): hello row id})."""
    conn = index.get_connection(tmp_path / "db" / "mouthpiece.db")
    ids = {}
    for name, (base, hellos) in sources.items():
        sid = index.insert_source(conn, f"{name}.wav", speaker=name)
        words, t = [], 0.0
        for w in [{"word": "world", **base}] * FILLERS + [{"word": "hello", **h} for h in hellos]:
            dur = w.get("dur", 0.4)
            words.append({"word": w["word"], "start": t, "end": t + dur, "confidence": 0.9,
                          "f0_mean": w.get("f0"), "f0_var": w.get("f0_var"), "rms": w.get("rms")})
            t += dur + 1.0
        index.insert_word_instances(conn, sid, words)
        rows = conn.execute("SELECT id FROM word_instances WHERE source_id=? AND word='hello' ORDER BY id",
                            (sid,)).fetchall()
        ids |= {(name, n): r["id"] for n, r in enumerate(rows)}
    return conn, ids


def _picks(conn, **bias) -> list[int]:
    return [match.find_clip(conn, "hello", rng=np.random.default_rng(s), **bias)["id"] for s in SEEDS]


def _master_pool_weights(pool, diction):
    """pool_weights as it was before the per-source biases, with emotion at 0."""
    out = []
    for r in pool:
        conf = r["confidence"]
        dscore = conf if conf is not None else 0.5
        out.append((conf or 0.5) * match._bias_weight(dscore, diction) * match._bias_weight(0.5, 0.0))
    return out


@pytest.mark.parametrize("diction", [-1.0, -0.4, 0.0, 0.3, 1.0])
def test_new_biases_at_zero_give_master_weights(tmp_path, diction):
    conn, _ = _db(tmp_path, {
        "low":  ({"f0": 100, "f0_var": 2, "rms": -30}, [{"f0": 130, "f0_var": 9, "rms": -20, "dur": 0.2},
                                                         {"f0": 90, "f0_var": 1, "rms": -35, "dur": 0.6}]),
        "high": ({"f0": 200, "f0_var": 8, "rms": -10}, [{"f0": 190, "f0_var": 3, "rms": -12},
                                                         {}]),
    })
    pool = match.candidate_pool(index.search_word(conn, "hello"))
    pool[0]["confidence"] = 0.5
    pool[1]["confidence"] = None
    norms = match.source_norms(conn)
    assert norms  # the zero path must ignore real per-source data, not just its absence
    master = _master_pool_weights(pool, diction)
    assert match.pool_weights(pool, diction, 0.0, 0.0, 0.0, norms) == master
    assert match.pool_weights(pool, diction) == master
    conn.close()


def test_pitch_prefers_high_for_its_own_source_not_high_in_hz(tmp_path):
    conn, ids = _db(tmp_path, {
        "low":  ({"f0": 100}, [{"f0": 130}]),   # +4.5 semitones above its speaker
        "high": ({"f0": 200}, [{"f0": 190}]),   # higher in Hz, but below its speaker
    })
    assert set(_picks(conn, pitch=1.0)) == {ids["low", 0]}
    assert set(_picks(conn, pitch=-1.0)) == {ids["high", 0]}
    assert len(set(_picks(conn))) == 2
    conn.close()


def _mean_measure(picks: list[int], measure: dict[int, float]) -> float:
    return sum(measure[i] for i in picks) / len(picks)


def test_intensity_picks_louder_livelier_clips_for_their_source(tmp_path):
    # The quiet recording's "hello"s are both above its own level, the loud
    # recording's both below; in absolute dB the loud ones are still louder, so
    # ranking raw rms would reverse every assertion below.
    conn, ids = _db(tmp_path, {
        "quiet": ({"f0_var": 2, "rms": -30}, [{"f0_var": 3, "rms": -28}, {"f0_var": 8, "rms": -22}]),
        "loud":  ({"f0_var": 20, "rms": -10}, [{"f0_var": 6, "rms": -18}, {"f0_var": 16, "rms": -12}]),
    })
    # dB above each source's median rms (-30 and -10).
    rel_rms = {ids["quiet", 0]: 2, ids["quiet", 1]: 8, ids["loud", 0]: -8, ids["loud", 1]: -2}
    calm = _mean_measure(_picks(conn, intensity=-1.0), rel_rms)
    neutral = _mean_measure(_picks(conn), rel_rms)
    intense = _mean_measure(_picks(conn, intensity=1.0), rel_rms)
    assert calm < neutral < intense
    conn.close()


def test_tempo_picks_faster_clips_for_their_source(tmp_path):
    # "world" and "hello" are 4 phones each: the slow source speaks 8 phones/s,
    # the fast one 16. Its "hello"s are fast for the slow source (9.6, 10.8/s)
    # and slow for the fast one (11.2, 12.8/s), so ranking raw phone rate would
    # reverse every assertion below.
    durs = {"slow": (4 / 9.6, 4 / 10.8), "fast": (4 / 11.2, 4 / 12.8)}
    conn, ids = _db(tmp_path, {
        "slow": ({"dur": 0.5}, [{"dur": d} for d in durs["slow"]]),
        "fast": ({"dur": 0.25}, [{"dur": d} for d in durs["fast"]]),
    })
    base = {"slow": 8, "fast": 16}
    rate_ratio = {ids[s, n]: (4 / d) / base[s] for s in durs for n, d in enumerate(durs[s])}
    slow = _mean_measure(_picks(conn, tempo=-1.0), rate_ratio)
    neutral = _mean_measure(_picks(conn), rate_ratio)
    fast = _mean_measure(_picks(conn, tempo=1.0), rate_ratio)
    assert slow < neutral < fast
    conn.close()


@pytest.mark.parametrize("control", ["intensity", "pitch", "tempo"])
def test_rows_without_features_are_neutral(control):
    norms = {1: {"f0_mean": 100.0, "f0_var": 4.0, "rms": -20.0, "rate": 10.0}}

    def row(id, f0=None, f0_var=None, rms=None, dur=None):
        return {"id": id, "source_id": 1, "word": "hello", "confidence": 0.8,
                "f0_mean": f0, "f0_var": f0_var, "rms": rms, "start_s": 0.0, "end_s": dur or 0.0}

    low = row(1, f0=80, f0_var=1, rms=-30, dur=0.8)
    high = row(2, f0=130, f0_var=20, rms=-10, dur=0.2)
    blank = row(3)
    pool = [low, blank, high]
    up = match.pool_weights(pool, norms=norms, **{control: 1.0})
    down = match.pool_weights(pool, norms=norms, **{control: -1.0})
    # The featureless row sits in the middle of the pool whichever way the bias points.
    assert up[1] == down[1] == match.unit_weight(blank, **{control: 1.0, f"{control}_score": 0.5})
    assert up[0] < up[1] < up[2]
    assert down[0] > down[1] > down[2]
    # A source without per-source medians (too few analysed words) is neutral too.
    assert match.pool_weights(pool, norms={}, **{control: 1.0}) == [up[1]] * 3


def test_sparse_source_has_no_medians_and_single_pool_is_unchanged(tmp_path):
    conn = index.get_connection(tmp_path / "db" / "mouthpiece.db")
    sid = index.insert_source(conn, "sparse.wav")
    index.insert_word_instances(conn, sid, [
        {"word": "hello", "start": float(i), "end": i + 0.4, "confidence": 0.9,
         "f0_mean": 100.0 + i, "f0_var": 2.0, "rms": -20.0} for i in range(match.MIN_SOURCE_VALUES - 1)])
    assert match.source_norms(conn) == {sid: {}}
    only = index.search_word(conn, "hello", limit=1)
    for bias in ({"pitch": 1.0}, {"tempo": -1.0}, {"intensity": 1.0}):
        assert match._pick(only, np.random.default_rng(0), None, **bias) is only[0]
    conn.close()


def test_bias_outside_range_is_clamped():
    rows = [{"id": i, "confidence": 0.8, "source_id": 1, "f0_mean": f0} for i, f0 in enumerate((90, 110))]
    norms = {1: {"f0_mean": 100.0}}
    assert match.pool_weights(rows, pitch=5.0, norms=norms) == match.pool_weights(rows, pitch=1.0, norms=norms)
    assert match.pool_weights(rows, pitch=-3.0, norms=norms) == match.pool_weights(rows, pitch=-1.0, norms=norms)


def test_source_norms_cache_follows_feature_backfill(tmp_path):
    conn, _ = _db(tmp_path, {"a": ({"f0": 100}, [])})
    (sid,) = match.source_norms(conn)
    assert "rms" not in match.source_norms(conn)[sid]
    conn.execute("UPDATE word_instances SET rms = -40")
    conn.commit()
    assert match.source_norms(conn)[sid]["rms"] == -40
    conn.close()
