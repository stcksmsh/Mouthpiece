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
    ab, ba = _render(), _render()
    for _ in range(10):
        server.api_reroll(ab["id"], 0)
        ab_r = server.api_reroll(ab["id"], 2)
    for _ in range(10):
        server.api_reroll(ba["id"], 2)
        ba_r = server.api_reroll(ba["id"], 0)
    assert _ids(ab_r) == _ids(ba_r)


@pytest.mark.usefixtures("env")
def test_failed_reroll_does_not_advance_sequence():
    r = server.api_render(server.RenderRequest(text="to zzyzxqq to", seed=5))
    ref = server.api_render(server.RenderRequest(text="to zzyzxqq to", seed=5))
    for _ in range(5):
        with pytest.raises(server.HTTPException):
            server.api_reroll(r["id"], 1)
    got = [_ids(server.api_reroll(r["id"], 0)) for _ in range(10)]
    assert got == [_ids(server.api_reroll(ref["id"], 0)) for _ in range(10)]
