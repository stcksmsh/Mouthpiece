"""End-to-end and unit checks for the render path, run against the fixture corpus."""

import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from mouthpiece import assemble


def _render(run_cli, tmp_path, name: str, text: str, *extra: str) -> tuple[dict, str]:
    out = tmp_path / f"{name}.wav"
    code, stdout = run_cli("render", text, "-o", str(out), *extra)
    assert code == 0, stdout
    manifest = json.loads(out.with_name(f"{name}.manifest.json").read_text())
    return manifest, stdout


@pytest.mark.parametrize("extra", [(), ("--granularity", "0.8")])
def test_seeded_render_is_reproducible(run_cli, tmp_path, extra):
    text = "i traveled to chicago to deliver my final farewell address"
    a, _ = _render(run_cli, tmp_path, "a", text, "--seed", "7", *extra)
    b, _ = _render(run_cli, tmp_path, "b", text, "--seed", "7", *extra)
    assert a["seed"] == b["seed"] == 7
    assert a["words"] == b["words"]


def test_missing_word_becomes_gap_not_crash(run_cli, tmp_path):
    manifest, stdout = _render(run_cli, tmp_path, "gap", "deliver zxqvbn address", "--seed", "1")
    tokens = {w["token"]: w for w in manifest["words"]}
    assert tokens["zxqvbn"]["found"] is False
    assert tokens["zxqvbn"]["gap_s"] == pytest.approx(0.4)
    assert "MISSING" in stdout and "zxqvbn" in stdout  # the user is warned
    assert tokens["deliver"]["found"] and tokens["address"]["found"]


def test_output_is_44_1_khz(run_cli, tmp_path):
    _render(run_cli, tmp_path, "rate", "final address", "--seed", "1")
    info = sf.info(str(tmp_path / "rate.wav"))
    assert info.samplerate == 44100
    assert info.channels == 1


def test_manifest_attributes_every_found_word(run_cli, tmp_path):
    manifest, _ = _render(run_cli, tmp_path, "attr", "the tradition of presidents", "--seed", "1")
    for w in manifest["words"]:
        assert w["found"]
        assert w["speaker"] == "Barack Obama"
        assert w["source_url"].startswith("https://obamawhitehouse.archives.gov/")


ATTRIBUTION_KEYS = {
    "speaker", "author", "reader", "uploader", "source_label",
    "license", "licence_url", "lang", "speaker_label",
}


@pytest.mark.parametrize("extra", [(), ("--granularity", "0.8")])
def test_manifest_words_carry_attribution(run_cli, tmp_path, extra):
    manifest, _ = _render(run_cli, tmp_path, "keys", "the tradition of presidents", "--seed", "1", *extra)
    for w in manifest["words"]:
        assert w.keys() >= ATTRIBUTION_KEYS, w["token"]
        assert w["speaker_label"] == "Barack Obama"
        for s in w.get("syllables", []):
            assert s.keys() >= ATTRIBUTION_KEYS


@pytest.mark.parametrize(("len_a", "len_b"), [(16000, 16000), (400, 16000), (16000, 300), (40, 40)])
def test_crossfade_never_exceeds_quarter_of_shorter_clip(len_a, len_b):
    a = np.ones(len_a, dtype=np.float32)
    b = np.ones(len_b, dtype=np.float32)
    out = assemble._crossfade_join(a, b)
    overlap = len_a + len_b - len(out)
    assert 0 <= overlap <= min(len_a, len_b) // 4


def test_speed_two_halves_duration():
    # Needs the rubberband CLI (apt install rubberband-cli), a runtime dependency.
    sr = assemble.SAMPLE_RATE
    t = np.arange(sr, dtype=np.float32) / sr
    clip = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = assemble._apply_rubberband(clip, speed=2.0)
    assert len(out) / sr == pytest.approx(0.5, abs=0.02)


def test_render_path_does_not_import_torch():
    code = (
        "import sys, mouthpiece.cli, mouthpiece.match, mouthpiece.assemble, mouthpiece.server;"
        "sys.exit('torch' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False, timeout=120).returncode == 0


def test_punctuated_transcript_words_are_matchable(run_cli, tmp_path):
    # The fixture transcript stores 'week,' 'nation,' and 'me.' exactly as WhisperX emitted them.
    manifest, _ = _render(run_cli, tmp_path, "punct", "this week the nation before me", "--seed", "1")
    assert all(w["found"] and not w.get("stitched") for w in manifest["words"])
