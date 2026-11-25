"""Manifest v2 fields and `mouthpiece replay`, run against the fixture corpus."""

import json
from pathlib import Path

import pytest
import soundfile as sf

import mouthpiece
from mouthpiece import index, manifest

# Seeds picked so that together the cases use every fragment tier and a gap.
CASES = [
    pytest.param("deliver my final farewell zxqvbn address", ("--seed", "3", "--granularity", "0.9"),
                 id="phrase-gap-word"),
    pytest.param("the tradition of presidents before zxqvbn me farewell",
                 ("--seed", "1", "--granularity", "-0.4", "--nth", "of:2"), id="syllable"),
    pytest.param("presidential traveler", ("--seed", "3", "--diction", "0.5", "--emotion", "-0.5"),
                 id="phone"),
]

V2_WORD_KEYS = {"nth", "locked", "swaps", "rerolls", "speed", "pitch", "gap", "fragments"}


def _render(run, out: Path, text: str, *extra: str) -> dict:
    code, stdout = run("render", text, "-o", str(out), *extra)
    assert code == 0, stdout
    return json.loads(out.with_name(f"{out.stem}.manifest.json").read_text())


def _replay(run, manifest_path: Path, out: Path) -> tuple[int, str]:
    return run("replay", str(manifest_path), "-o", str(out))


def _fragments(m: dict) -> list:
    return [w["fragments"] for w in m["words"]]


def _option(extra: tuple, flag: str, default: float) -> float:
    return float(extra[extra.index(flag) + 1]) if flag in extra else default


@pytest.mark.parametrize(("text", "extra"), CASES)
def test_cli_manifest_has_every_v2_field(run_cli, tmp_path, text, extra):
    m = _render(run_cli, tmp_path / "r.wav", text, *extra)

    assert m["schema_version"] == 2
    assert m["engine_version"] == mouthpiece.__version__
    assert len(m["corpus_hash"]) == 64
    int(m["corpus_hash"], 16)
    assert m["seed"] == int(extra[1])
    assert m["text"] == text
    nth = {"of": 2} if "--nth" in extra else {}
    assert m["controls"] == {
        "granularity": _option(extra, "--granularity", 0.0),
        "diction":     _option(extra, "--diction", 0.0),
        "emotion":     _option(extra, "--emotion", 0.0),
        "nth":         nth,
    }

    if nth:
        # "of" must start its own unit, so its rank is really recorded, not None == None.
        assert [w["nth"] for w in m["words"] if (w.get("tokens") or [w["token"]])[0] == "of"] == [2]

    for u, w in enumerate(m["words"]):
        assert w.keys() >= V2_WORD_KEYS, w["token"]
        first_token = (w.get("tokens") or [w["token"]])[0]
        assert w["nth"] == m["controls"]["nth"].get(first_token)
        assert w["locked"] is False
        assert (w["swaps"], w["rerolls"]) == (0, 0)
        assert (w["speed"], w["pitch"]) == (1.0, 0.0)
        assert w["gap"] == (0.0 if u == 0 else 0.1)
        if not w["found"]:
            assert w["fragments"] == []
            continue
        assert w["fragments"]
        for f in w["fragments"]:
            assert f["tier"] in {"phrase", "word", "syllable", "phone"}
            assert isinstance(f["instance_id"], int)


def test_cases_cover_every_tier_and_a_gap(run_cli, tmp_path):
    tiers, gaps = set(), 0
    for i, case in enumerate(CASES):
        text, extra = case.values
        m = _render(run_cli, tmp_path / f"c{i}.wav", text, *extra)
        tiers |= {f["tier"] for frags in _fragments(m) for f in frags}
        gaps += sum(not w["found"] for w in m["words"])
    assert tiers == {"phrase", "word", "syllable", "phone"}
    assert gaps


@pytest.mark.parametrize(("text", "extra"), CASES)
def test_replay_reproduces_instances_and_samples(run_cli, tmp_path, text, extra):
    original = _render(run_cli, tmp_path / "r.wav", text, *extra)

    code, out = _replay(run_cli, tmp_path / "r.manifest.json", tmp_path / "p.wav")
    assert code == 0, out
    assert "mismatch" not in out
    replayed = json.loads((tmp_path / "p.manifest.json").read_text())

    assert _fragments(replayed) == _fragments(original)
    for key in ("schema_version", "corpus_hash", "seed", "text", "controls"):
        assert replayed[key] == original[key]
    for a, b in zip(original["words"], replayed["words"], strict=True):
        assert {k: a[k] for k in V2_WORD_KEYS} == {k: b[k] for k in V2_WORD_KEYS}

    rendered, sr_a = sf.read(tmp_path / "r.wav", dtype="int16")
    again, sr_b = sf.read(tmp_path / "p.wav", dtype="int16")
    assert sr_a == sr_b == 44100
    assert rendered.shape == again.shape
    assert (rendered == again).all()


def test_replaying_twice_is_byte_identical(run_cli, tmp_path):
    text, extra = CASES[1].values
    _render(run_cli, tmp_path / "r.wav", text, *extra)
    for name in ("a", "b"):
        code, out = _replay(run_cli, tmp_path / "r.manifest.json", tmp_path / f"{name}.wav")
        assert code == 0, out
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_replay_warns_on_corpus_change_and_still_renders(own_db_cli, tmp_path):
    run, db = own_db_cli
    m = _render(run, tmp_path / "r.wav", "deliver my final address", "--seed", "1")

    conn = index.get_connection(db)
    index.insert_source(conn, "added_later.wav", speaker="Someone Else")
    current = manifest.corpus_hash(conn)
    conn.close()
    assert current != m["corpus_hash"]

    code, out = _replay(run, tmp_path / "r.manifest.json", tmp_path / "p.wav")
    assert code == 0, out
    assert "corpus hash mismatch" in out
    assert m["corpus_hash"] in out and current in out
    assert (tmp_path / "p.wav").exists()
    replayed = json.loads((tmp_path / "p.manifest.json").read_text())
    assert _fragments(replayed) == _fragments(m)


def test_replay_refuses_missing_instance(own_db_cli, tmp_path):
    run, db = own_db_cli
    m = _render(run, tmp_path / "r.wav", "deliver my final address", "--seed", "1")
    gone = m["words"][-1]["fragments"][0]
    assert gone["tier"] == "word"

    conn = index.get_connection(db)
    for table in ("phone_instances", "syllable_instances"):
        conn.execute(f"DELETE FROM {table} WHERE word_instance_id=?", (gone["instance_id"],))
    conn.execute("DELETE FROM word_instances WHERE id=?", (gone["instance_id"],))
    conn.commit()
    conn.close()

    code, out = _replay(run, tmp_path / "r.manifest.json", tmp_path / "p.wav")
    assert code != 0
    assert f"word_instance {gone['instance_id']}" in out
    assert not (tmp_path / "p.wav").exists()
    assert not (tmp_path / "p.manifest.json").exists()


def test_replay_refuses_v1_manifest(run_cli, tmp_path):
    v1 = {"rendered_at": "2026-01-01T00:00:00+00:00", "seed": 1, "text": "address",
          "output_wav": "x.wav", "words": [{"token": "address", "found": True, "word_instance_id": 12}]}
    path = tmp_path / "old.manifest.json"
    path.write_text(json.dumps(v1))

    code, out = _replay(run_cli, path, tmp_path / "p.wav")
    assert code != 0
    assert "v1 manifest cannot be replayed" in out
    assert not (tmp_path / "p.wav").exists()


def test_corpus_hash_ignores_ingest_time(own_db_cli):
    _, db = own_db_cli
    conn = index.get_connection(db)
    before = manifest.corpus_hash(conn)
    conn.execute("UPDATE sources SET ingested_at = '1999-01-01 00:00:00'")
    conn.commit()
    assert manifest.corpus_hash(conn) == before
    conn.execute("UPDATE word_instances SET end_s = end_s + 0.001 WHERE id = 1")
    conn.commit()
    assert manifest.corpus_hash(conn) != before
    conn.close()


def _edit_manifest(path: Path, edit) -> None:
    m = json.loads(path.read_text())
    edit(m)
    path.write_text(json.dumps(m))


def _no_fragments(m: dict) -> None:
    m["words"][-1]["fragments"] = []


def _short_phrase(m: dict) -> None:
    m["words"][0]["fragments"].pop()


@pytest.mark.parametrize(("edit", "expected"), [
    pytest.param(_no_fragments, "is found but has no fragments", id="found-without-fragments"),
    pytest.param(_short_phrase, "has 4 tokens but 3 fragments", id="phrase-count-mismatch"),
])
def test_replay_refuses_malformed_manifest(run_cli, tmp_path, edit, expected):
    text, extra = CASES[0].values
    original = _render(run_cli, tmp_path / "r.wav", text, *extra)
    assert original["words"][0]["fragments"][0]["tier"] == "phrase"
    assert len(original["words"][0]["tokens"]) == 4
    _edit_manifest(tmp_path / "r.manifest.json", edit)

    code, out = _replay(run_cli, tmp_path / "r.manifest.json", tmp_path / "p.wav")
    assert code != 0
    assert "malformed manifest:" in out and expected in out
    assert not (tmp_path / "p.wav").exists()


def test_non_default_gap_round_trips_exactly(run_cli, tmp_path):
    text, extra = CASES[1].values
    _render(run_cli, tmp_path / "r.wav", text, *extra)
    odd_gap = 1601 / 16000  # one sample past the default; rounding to 4 decimals gives 1602

    def set_gap(m):
        m["words"][1]["gap"] = odd_gap

    _edit_manifest(tmp_path / "r.manifest.json", set_gap)

    code, out = _replay(run_cli, tmp_path / "r.manifest.json", tmp_path / "p.wav")
    assert code == 0, out
    first = json.loads((tmp_path / "p.manifest.json").read_text())
    assert first["words"][1]["gap"] == odd_gap
    assert round(first["words"][1]["gap"] * 16000) == 1601

    code, out = _replay(run_cli, tmp_path / "p.manifest.json", tmp_path / "q.wav")
    assert code == 0, out
    assert (tmp_path / "q.wav").read_bytes() == (tmp_path / "p.wav").read_bytes()
