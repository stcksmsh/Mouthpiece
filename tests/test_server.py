"""Server handlers, called directly (no HTTP client dependency) against the fixture corpus."""

import pytest

from mouthpiece import server


@pytest.fixture
def env(corpus_env, monkeypatch):
    for key, value in corpus_env.items():
        monkeypatch.setenv(key, value)


def test_every_route_is_bound_to_a_public_handler():
    # Regression: the reroll decorator once sat on the _replace_match helper.
    for route in server.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None and getattr(route, "path", "").startswith("/api/"):
            assert endpoint.__name__.startswith("api_"), (route.path, endpoint.__name__)


@pytest.mark.usefixtures("env")
def test_render_then_reroll_and_adjust():
    r = server.api_render(server.RenderRequest(text="deliver my final farewell address", seed=5))
    assert [w["found"] for w in r["words"]] == [True] * 5

    r = server.api_reroll(r["id"], 0)
    assert r["words"][0]["token"] == "deliver"

    r = server.api_adjust(r["id"], 1, server.AdjustRequest(speed=1.5, pitch=2.0))
    assert (r["words"][1]["speed"], r["words"][1]["pitch"]) == (1.5, 2.0)
    assert r["duration_s"] > 0


@pytest.mark.usefixtures("env")
def test_reroll_gap_token_is_400():
    r = server.api_render(server.RenderRequest(text="deliver zzyzxqq address", seed=5))
    assert r["words"][1]["found"] is False
    with pytest.raises(server.HTTPException) as exc:
        server.api_reroll(r["id"], 1)
    assert exc.value.status_code == 400


# "to" has three instances in the fixture, the only word with real choice to reroll between.
TEXT = "to to to"


def _ids(r):
    return [w.get("word_instance_id") for w in r["words"]]


def _render():
    return server.api_render(server.RenderRequest(text=TEXT, seed=5))


@pytest.mark.usefixtures("env")
def test_reroll_is_reproducible_for_same_seed():
    # Compare every intermediate state so an unseeded RNG is very unlikely to match by chance.
    runs = []
    for _ in range(2):
        r = _render()
        runs.append([_ids(server.api_reroll(r["id"], i % 3)) for i in range(30)])
    assert runs[0] == runs[1]


@pytest.mark.usefixtures("env")
def test_reroll_order_across_words_does_not_matter():
    def picks(order):
        r = _render()
        seq = {0: [], 2: []}
        for _ in range(12):
            for w in order:
                seq[w].append(_ids(server.api_reroll(r["id"], w))[w])
        return seq

    # Each word's full pick sequence must match whichever word goes first.
    assert picks((0, 2)) == picks((2, 0))


@pytest.mark.usefixtures("env")
def test_failed_reroll_does_not_advance_sequence():
    r = server.api_render(server.RenderRequest(text="to zzyzxqq to", seed=5))
    ref = server.api_render(server.RenderRequest(text="to zzyzxqq to", seed=5))
    for _ in range(5):
        with pytest.raises(server.HTTPException):
            server.api_reroll(r["id"], 1)
    got = [_ids(server.api_reroll(r["id"], 0)) for _ in range(10)]
    assert got == [_ids(server.api_reroll(ref["id"], 0)) for _ in range(10)]


@pytest.mark.usefixtures("env")
def test_render_response_and_reaper_export_use_display_label(tmp_path):
    r = server.api_render(server.RenderRequest(text="deliver my final address", seed=5))
    assert [w["speaker_label"] for w in r["words"]] == ["Barack Obama"] * 4
    script = server._make_reaper_script(server._renders[r["id"]], tmp_path / "x.wav")
    assert script.count('speaker="Barack Obama"') == 4
    cands = server.api_candidates("address")
    assert cands
    assert all(c["speaker_label"] == "Barack Obama" for c in cands)


def test_reaper_export_labels_unattributed_sources(unattributed_db, tmp_path):
    from mouthpiece import index

    conn = index.get_connection(unattributed_db)
    words = [
        server._row_to_word(i, tok, index.search_word(conn, tok)[0])
        for i, tok in enumerate(["hello", "world"])
    ]
    conn.close()
    render = {"id": "r1", "text": "hello world", "words": words}
    script = server._make_reaper_script(render, tmp_path / "x.wav")
    assert 'speaker="unidentified speaker, SCOTUS, 2019-04-23"' in script
    assert 'speaker="unidentified speaker, NASA"' in script
    assert "@" not in script
    assert "unknown" not in script
