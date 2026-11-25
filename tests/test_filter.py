"""Source filter: licence tiers and --only/--exclude, from the index queries up to the CLI and server."""

import json
import shutil
import sqlite3

import pytest
from conftest import _cli_runner, fixture_data, index_units

from mouthpiece import index, ingest, match, phonemes, server

OBAMA = "obama_20170114_0-8s.wav"
CHURCHILL = "churchill_copy.wav"
TEXT = "i traveled to chicago to deliver my final farewell address to the nation"
DEFAULT_CONTROLS = {"tiers": ["clean", "unverified"], "only": [], "exclude": []}


@pytest.fixture
def mixed(corpus_env, tmp_path, monkeypatch, capsys):
    """The fixture corpus plus a restricted copy of it under another speaker. The copy's
    clips are a little more confident, so an unfiltered render prefers them, and it holds
    one extra word, "zxqvbn", that no other source has. Returns (run_cli, env)."""
    corpus = tmp_path / "corpus"
    shutil.copytree(corpus_env["MOUTHPIECE_CORPUS"], corpus)
    shutil.copy(corpus / OBAMA, corpus / CHURCHILL)
    db = tmp_path / "db" / "mouthpiece.db"
    db.parent.mkdir()
    src, dst = sqlite3.connect(corpus_env["MOUTHPIECE_DB"]), sqlite3.connect(db)
    src.backup(dst)
    src.close()
    dst.close()

    words = [{**w, "confidence": min(w["confidence"] + 0.01, 1.0)} for w in fixture_data()["words"]]
    words.append({"word": "zxqvbn", "start": 7.0, "end": 7.4, "confidence": 0.9})
    conn = index.get_connection(db)
    sid = index.insert_source(conn, CHURCHILL, speaker="Winston Churchill", tier="restricted")
    index.insert_word_instances(conn, sid, words)
    index_units(conn, sid)
    conn.close()

    env = {**corpus_env, "MOUTHPIECE_DB": str(db), "MOUTHPIECE_CORPUS": str(corpus)}
    return _cli_runner(env, monkeypatch, capsys), env


_SOURCE_OF = {
    "word": "SELECT s.filename FROM word_instances wi JOIN sources s ON s.id = wi.source_id WHERE wi.id = ?",
    "syllable": "SELECT s.filename FROM syllable_instances u JOIN word_instances wi ON wi.id = u.word_instance_id"
                " JOIN sources s ON s.id = wi.source_id WHERE u.id = ?",
    "phone": "SELECT s.filename FROM phone_instances u JOIN word_instances wi ON wi.id = u.word_instance_id"
             " JOIN sources s ON s.id = wi.source_id WHERE u.id = ?",
}
_SOURCE_OF["phrase"] = _SOURCE_OF["word"]


def _render(run, tmp_path, *args):
    out = tmp_path / "out.wav"
    code, stdout = run("render", *args, "-o", str(out))
    assert code == 0, stdout
    return json.loads(out.with_name("out.manifest.json").read_text()), stdout


def _fragments(env, manifest) -> list[tuple[str, str]]:
    """(tier, source filename) of every fragment in the manifest."""
    conn = sqlite3.connect(env["MOUTHPIECE_DB"])
    out = [(f["tier"], conn.execute(_SOURCE_OF[f["tier"]], (f["instance_id"],)).fetchone()[0])
           for w in manifest["words"] for f in w["fragments"]]
    conn.close()
    return out


def _files(rows) -> set[str]:
    return {r["filename"] for r in rows}


MODES = [(), ("--granularity", "0.8"), ("--granularity", "-0.8"), ("--selector", "viterbi")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", ["1", "2"])
def test_default_render_never_uses_restricted(mixed, tmp_path, mode, seed):
    run, env = mixed
    m, _ = _render(run, tmp_path, TEXT, "--seed", seed, *mode)
    frags = _fragments(env, m)
    assert frags
    assert {f for _, f in frags} == {OBAMA}


@pytest.mark.parametrize("mode", MODES)
def test_tier_all_can_use_restricted(mixed, tmp_path, mode):
    run, env = mixed
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", "--tier", "all", *mode)
    assert CHURCHILL in {f for _, f in _fragments(env, m)}


def test_word_only_in_restricted_source_becomes_gap(mixed, tmp_path):
    run, _ = mixed
    m, stdout = _render(run, tmp_path, "deliver zxqvbn address", "--seed", "1")
    found = {w["token"]: w["found"] for w in m["words"]}
    assert found == {"deliver": True, "zxqvbn": False, "address": True}
    missing = [line for line in stdout.splitlines() if line.startswith("Missing:")]
    assert len(missing) == 1 and "zxqvbn" in missing[0] and "selected sources" in missing[0]

    m, _ = _render(run, tmp_path, "deliver zxqvbn address", "--seed", "1", "--tier", "all")
    assert all(w["found"] for w in m["words"])


def test_only_matching_nothing_renders_gaps(mixed, tmp_path):
    run, _ = mixed
    m, stdout = _render(run, tmp_path, "deliver my address", "--seed", "1", "--only", "nobody")
    assert [w["found"] for w in m["words"]] == [False] * 3
    assert "Missing:  deliver, my, address" in stdout


@pytest.mark.parametrize(("args", "allowed"), [
    (("--tier", "all", "--only", "CHURCHILL"), {CHURCHILL}),         # speaker label
    (("--tier", "all", "--only", "obama_2017"), {OBAMA}),             # filename
    (("--tier", "all", "--only", "nobody", "--only", "winston"), {CHURCHILL}),  # any one may match
    (("--tier", "all", "--exclude", "barack"), {CHURCHILL}),
    (("--tier", "all", "--exclude", "copy.wav"), {OBAMA}),
    (("--tier", "restricted,clean", "--exclude", "obama"), {CHURCHILL}),
])
@pytest.mark.parametrize("mode", MODES)
def test_only_and_exclude(mixed, tmp_path, args, allowed, mode):
    run, env = mixed
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", *args, *mode)
    frags = _fragments(env, m)
    assert all(w["found"] for w in m["words"])
    assert {f for _, f in frags} == allowed
    tiers = {t for t, _ in frags}
    # The modes must really exercise phrase runs and stitched sub-word units.
    if mode == ("--granularity", "0.8"):
        assert "phrase" in tiers
    if mode == ("--granularity", "-0.8"):
        assert tiers & {"syllable", "phone"}


def test_manifest_records_filter(mixed, tmp_path):
    run, _ = mixed
    m, _ = _render(run, tmp_path, "deliver", "--seed", "1")
    assert m["controls"]["sources"] == DEFAULT_CONTROLS

    m, _ = _render(run, tmp_path, "deliver", "--seed", "1", "--tier", "all",
                   "--only", "Obama", "--only", "churchill", "--exclude", "copy")
    assert m["controls"]["sources"] == {
        "tiers": ["clean", "unverified", "restricted"],
        "only": ["Obama", "churchill"],
        "exclude": ["copy"],
    }


@pytest.mark.parametrize("args", [("--tier", "public"), ("--tier", ""), ("--only", " ")])
def test_bad_filter_args_exit_with_message(mixed, tmp_path, args):
    run, _ = mixed
    code, out = run("render", "deliver", "-o", str(tmp_path / "x.wav"), *args)
    assert code != 0
    assert "Error:" in out
    assert not (tmp_path / "x.wav").exists()


def test_server_render_excludes_restricted(mixed, monkeypatch):
    _, env = mixed
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    r = server.api_render(server.RenderRequest(text=TEXT, seed=1, granularity=0.8))
    r = server.api_reroll(r["id"], 2)
    m = server.api_manifest(r["id"])
    assert m["controls"]["sources"] == DEFAULT_CONTROLS
    assert {f for _, f in _fragments(env, m)} == {OBAMA}


@pytest.fixture
def crowded(tmp_path):
    """25 restricted "hello" clips at high confidence and one clean one at low confidence."""
    conn = index.get_connection(tmp_path / "crowded.db")
    clean = index.insert_source(conn, "clean.wav", tier="clean")
    restricted = index.insert_source(conn, "restricted.wav", tier="restricted")
    index.insert_word_instances(conn, restricted, [
        {"word": "hello", "start": float(i), "end": i + 0.5, "confidence": 0.99, "f0_mean": 120.0}
        for i in range(25)
    ])
    index.insert_word_instances(conn, clean, [
        {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.1, "f0_mean": 300.0},
    ])
    index_units(conn, clean)
    index_units(conn, restricted)
    yield conn
    conn.close()


def test_filter_is_applied_inside_candidate_queries(crowded):
    conn = crowded
    assert _files(index.search_word(conn, "hello", limit=20)) == {"clean.wav"}
    assert _files(index.search_word_by_pitch(conn, "hello", 120.0, limit=20)) == {"clean.wav"}
    key = phonemes.phone_key(phonemes.syllabify("hello")[0])
    assert _files(index.search_syllable(conn, key, limit=20)) == {"clean.wav"}
    assert match.find_clip(conn, "hello")["filename"] == "clean.wav"
    assert match.find_clip_near_pitch(conn, "hello", 120.0)["filename"] == "clean.wav"
    # Without the filter the restricted clips fill the whole limit.
    everything = index.SourceFilter(tiers=index.TIERS)
    assert _files(index.search_word(conn, "hello", limit=20, source_filter=everything)) == {"restricted.wav"}


def test_old_db_gains_tier_column_and_null_is_unverified(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sources (id INTEGER PRIMARY KEY, filename TEXT NOT NULL UNIQUE,"
                 " speaker TEXT, date TEXT, event TEXT, source_url TEXT, license TEXT,"
                 " ingested_at TEXT NOT NULL DEFAULT (datetime('now')))")
    conn.execute("INSERT INTO sources (filename) VALUES ('old.wav')")
    conn.commit()
    conn.close()

    conn = index.get_connection(path)
    assert conn.execute("SELECT tier FROM sources").fetchone()["tier"] is None
    index.insert_word_instances(conn, 1, [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9}])
    assert len(index.search_word(conn, "hello")) == 1
    assert index.search_word(conn, "hello", source_filter=index.SourceFilter(tiers=("unverified",)))
    assert not index.search_word(conn, "hello", source_filter=index.SourceFilter(tiers=("clean",)))
    conn.close()


def test_ingest_stores_and_validates_tier(tmp_path, monkeypatch):
    from mouthpiece import transcribe

    normalized = []
    monkeypatch.setattr(ingest, "_normalize_audio", lambda _src, dst: normalized.append(dst))
    monkeypatch.setattr(transcribe, "transcribe_audio", lambda _path, **_kw: [])
    monkeypatch.setattr(ingest, "_compute_prosodics", lambda _path, words: words)
    conn = index.get_connection(tmp_path / "db.sqlite")
    audio = tmp_path / "talk.mp3"

    with pytest.raises(ValueError, match="tier must be one of"):
        ingest.ingest_file_with_meta(audio, {"tier": "pd"}, conn=conn, corpus_dir=tmp_path / "corpus")
    assert normalized == []
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0

    ingest.ingest_file_with_meta(audio, {"tier": "restricted", "speaker": "X"}, conn=conn,
                                 corpus_dir=tmp_path / "corpus")
    assert conn.execute("SELECT tier FROM sources WHERE filename='talk.wav'").fetchone()["tier"] == "restricted"
    conn.close()
