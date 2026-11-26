"""Source filter: licence tiers and --only/--exclude, from the index queries up to the CLI and server."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from conftest import _cli_runner, fixture_data, index_units, scratch_dir

from mouthpiece import index, ingest, match, phonemes, server

OBAMA = "obama_20170114_0-8s.wav"
CHURCHILL = "churchill_copy.wav"
TEXT = "i traveled to chicago to deliver my final farewell address to the nation"
DEFAULT_CONTROLS = {"tiers": ["clean", "unverified"], "only": [], "exclude": [],
                    "langs": ["en"]}


@pytest.fixture(scope="module")
def mixed_env(corpus_env):
    """The fixture corpus plus a restricted copy of it under another speaker. The copy's
    clips are a little more confident, so an unfiltered render prefers them, and it holds
    one extra word, "zxqvbn", that no other source has. Built once: renders and the
    server only read the DB."""
    with scratch_dir("pytest-mixed-") as root:
        corpus = root / "corpus"
        shutil.copytree(corpus_env["MOUTHPIECE_CORPUS"], corpus)
        shutil.copy(corpus / OBAMA, corpus / CHURCHILL)
        db = root / "db" / "mouthpiece.db"
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

        yield {**corpus_env, "MOUTHPIECE_DB": str(db), "MOUTHPIECE_CORPUS": str(corpus)}


@pytest.fixture
def mixed(mixed_env, monkeypatch, capsys):
    """Returns (run_cli, env) on the mixed corpus."""
    return _cli_runner(mixed_env, monkeypatch, capsys), mixed_env


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
def test_default_render_never_uses_restricted(mixed, tmp_path, mode):
    run, env = mixed
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", *mode)
    frags = _fragments(env, m)
    assert frags
    assert {f for _, f in frags} == {OBAMA}


def test_tier_all_can_use_restricted(mixed, tmp_path):
    run, env = mixed
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", "--tier", "all")
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


@pytest.mark.parametrize(("kwargs", "allowed"), [
    ({"tiers": index.TIERS, "only": ("CHURCHILL",)}, {CHURCHILL}),         # speaker label
    ({"tiers": index.TIERS, "only": ("obama_2017",)}, {OBAMA}),             # filename
    ({"tiers": index.TIERS, "only": ("nobody", "winston")}, {CHURCHILL}),  # any one may match
    ({"tiers": index.TIERS, "exclude": ("barack",)}, {CHURCHILL}),
    ({"tiers": index.TIERS, "exclude": ("copy.wav",)}, {OBAMA}),
    ({"tiers": ("restricted", "clean"), "exclude": ("obama",)}, {CHURCHILL}),
])
def test_only_and_exclude_queries(mixed_env, kwargs, allowed):
    source_filter = index.SourceFilter(**kwargs)
    conn = index.get_connection(Path(mixed_env["MOUTHPIECE_DB"]))
    key = phonemes.phone_key(phonemes.syllabify("deliver")[0])
    words = index.search_word(conn, "deliver", source_filter=source_filter)
    sylls = index.search_syllable(conn, key, source_filter=source_filter)
    conn.close()
    assert _files(words) == allowed
    assert _files(sylls) == allowed


@pytest.mark.parametrize("mode", MODES)
def test_only_and_exclude_render(mixed, tmp_path, mode):
    run, env = mixed
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", "--tier", "all", "--exclude", "barack", *mode)
    frags = _fragments(env, m)
    assert all(w["found"] for w in m["words"])
    assert {f for _, f in frags} == {CHURCHILL}
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
        "langs": ["en"],
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


CHURCHILL_ONLY = index.SourceFilter(tiers=index.TIERS, only=("churchill",))


def test_server_swap_refuses_excluded_source(mixed, monkeypatch):
    _, env = mixed
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    conn = index.get_connection(Path(env["MOUTHPIECE_DB"]))

    r = server.api_render(server.RenderRequest(text="deliver address", seed=1))
    restricted = index.search_word(conn, "deliver", source_filter=CHURCHILL_ONLY)[0]
    with pytest.raises(server.HTTPException) as exc:
        server.api_swap(r["id"], 0, server.SwapRequest(word_instance_id=restricted["id"]))
    assert exc.value.status_code == 400 and "excludes" in exc.value.detail
    allowed = index.search_word(conn, "deliver")[0]
    server.api_swap(r["id"], 0, server.SwapRequest(word_instance_id=allowed["id"]))

    r = server.api_render(server.RenderRequest(text=TEXT, seed=1, granularity=-0.8))
    idx, sylls = next((i, s) for i, (_, s) in enumerate(server._renders[r["id"]]["matches"])
                      if isinstance(s, list))
    restricted = index.search_syllable(conn, sylls[0]["phone_key"], source_filter=CHURCHILL_ONLY)[0]
    conn.close()
    with pytest.raises(server.HTTPException) as exc:
        server.api_swap_syllable(r["id"], idx, 0,
                                 server.SwapSyllableRequest(syllable_instance_id=restricted["id"]))
    assert exc.value.status_code == 400 and "excludes" in exc.value.detail
    m = server.api_manifest(r["id"])
    assert {f for _, f in _fragments(env, m)} == {OBAMA}


def test_cli_candidates_use_filter(mixed):
    run, _ = mixed
    code, out = run("candidates", "deliver")
    assert code == 0 and OBAMA in out and CHURCHILL not in out
    code, out = run("candidates", "deliver", "--tier", "all", "--only", "churchill")
    assert code == 0 and CHURCHILL in out and OBAMA not in out


def test_cli_coverage_uses_filter(mixed):
    run, _ = mixed
    code, out = run("coverage", "deliver zxqvbn")
    assert code == 0 and "Missing:  zxqvbn" in out
    code, out = run("coverage", "deliver zxqvbn", "--tier", "all")
    assert code == 0 and "All words covered." in out
    code, out = run("coverage", "deliver", "--only", " ")
    assert code != 0 and "Error:" in out


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


FRENCH = "french_copy.wav"
ALL_LANGS = index.SourceFilter(langs=None)


@pytest.fixture(scope="module")
def french_env(corpus_env):
    """The fixture corpus (its source has no lang, so it counts as en) plus a more
    confident copy of it marked lang='fr', so an unfiltered render prefers the French clips."""
    with scratch_dir("pytest-french-") as root:
        corpus = root / "corpus"
        shutil.copytree(corpus_env["MOUTHPIECE_CORPUS"], corpus)
        shutil.copy(corpus / OBAMA, corpus / FRENCH)
        db = root / "db" / "mouthpiece.db"
        db.parent.mkdir()
        src, dst = sqlite3.connect(corpus_env["MOUTHPIECE_DB"]), sqlite3.connect(db)
        src.backup(dst)
        src.close()
        dst.close()

        words = [{**w, "confidence": min(w["confidence"] + 0.01, 1.0)} for w in fixture_data()["words"]]
        conn = index.get_connection(db)
        sid = index.insert_source(conn, FRENCH, speaker="Livre", lang="fr")
        index.insert_word_instances(conn, sid, words)
        index_units(conn, sid)
        conn.close()

        yield {**corpus_env, "MOUTHPIECE_DB": str(db), "MOUTHPIECE_CORPUS": str(corpus)}


@pytest.fixture
def french(french_env, monkeypatch, capsys):
    return _cli_runner(french_env, monkeypatch, capsys), french_env


def test_default_never_picks_french(french):
    _, env = french
    conn = index.get_connection(Path(env["MOUTHPIECE_DB"]))
    assert _files(index.search_word(conn, "deliver", source_filter=index.DEFAULT_FILTER)) == {OBAMA}
    assert _files(index.search_word(conn, "deliver", source_filter=ALL_LANGS)) == {OBAMA, FRENCH}
    conn.close()


def test_french_render_default_and_all(french, tmp_path):
    run, env = french
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1")
    assert {f for _, f in _fragments(env, m)} == {OBAMA}
    assert m["controls"]["sources"]["langs"] == ["en"]

    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", "--lang", "all")
    assert FRENCH in {f for _, f in _fragments(env, m)}
    assert m["controls"]["sources"]["langs"] is None

    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", "--lang", "fr")
    assert {f for _, f in _fragments(env, m)} == {FRENCH}
    assert m["controls"]["sources"]["langs"] == ["fr"]


def test_null_lang_counts_as_english(french):
    _, env = french
    conn = index.get_connection(Path(env["MOUTHPIECE_DB"]))
    assert conn.execute("SELECT lang FROM sources WHERE filename=?", (OBAMA,)).fetchone()[0] is None
    assert OBAMA in _files(index.search_word(conn, "deliver", source_filter=index.DEFAULT_FILTER))
    conn.close()


def test_replay_without_langs_applies_no_language_filter(french, tmp_path):
    run, env = french
    m, _ = _render(run, tmp_path, TEXT, "--seed", "1", "--lang", "all")
    assert FRENCH in {f for _, f in _fragments(env, m)}
    del m["controls"]["sources"]["langs"]
    old = tmp_path / "old.manifest.json"
    old.write_text(json.dumps(m))
    code, out = run("replay", str(old), "-o", str(tmp_path / "p.wav"))
    assert code == 0, out
    replayed = json.loads((tmp_path / "p.manifest.json").read_text())
    assert _fragments(env, replayed) == _fragments(env, m)
    assert "langs" not in replayed["controls"]["sources"]


def test_server_swap_refuses_french_source(french, monkeypatch):
    _, env = french
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    conn = index.get_connection(Path(env["MOUTHPIECE_DB"]))
    r = server.api_render(server.RenderRequest(text="deliver address", seed=1))
    fr = next(x for x in index.search_word(conn, "deliver", source_filter=ALL_LANGS) if x["filename"] == FRENCH)
    with pytest.raises(server.HTTPException) as exc:
        server.api_swap(r["id"], 0, server.SwapRequest(word_instance_id=fr["id"]))
    assert exc.value.status_code == 400 and "excludes" in exc.value.detail
    conn.close()


def test_bad_lang_exits_with_message(french, tmp_path):
    run, _ = french
    code, out = run("render", "deliver", "-o", str(tmp_path / "x.wav"), "--lang", "")
    assert code != 0
    assert "Error:" in out
