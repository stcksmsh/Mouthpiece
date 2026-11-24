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
