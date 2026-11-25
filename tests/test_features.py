"""Acoustic features per word instance (features.py) and the `mouthpiece features` backfill."""

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from mouthpiece import features, index

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = "obama_20170114_0-8s"
SR = features.SAMPLE_RATE
COLS = ("f0_mean", "f0_var", "rms", "spectral_tilt")


def _tone(freq: float, amp: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_synthetic_tone_silence_and_level():
    # 1 s of 220 Hz at amplitude 0.5, 1 s of silence, 1 s of 220 Hz at amplitude 0.05.
    y = np.concatenate([_tone(220, 0.5, 1), np.zeros(SR, np.float32), _tone(220, 0.05, 1)])
    loud, silent, quiet, tiny = features.aggregate(
        features.analyse(y), [(0.1, 0.9), (1.1, 1.9), (2.1, 2.9), (0.5, 0.505)])

    assert loud["f0_mean"] == pytest.approx(220, rel=0.02)
    assert quiet["f0_mean"] == pytest.approx(220, rel=0.02)
    assert loud["f0_var"] < 0.01  # semitones^2: a steady tone barely varies
    # A sine of amplitude A has RMS A/sqrt(2): -9.03 dBFS at 0.5, and 20 dB less at 0.05.
    assert loud["rms"] == pytest.approx(-9.03, abs=0.2)
    assert loud["rms"] - quiet["rms"] == pytest.approx(20, abs=0.5)

    assert silent["f0_mean"] is None and silent["f0_var"] is None
    assert silent["rms"] == features.RMS_FLOOR_DB

    # A 5 ms window holds no frame centre and falls back to the nearest frame.
    assert tiny["f0_mean"] == pytest.approx(220, rel=0.02)
    assert tiny["rms"] == pytest.approx(loud["rms"], abs=0.5)


def test_spectral_tilt_is_steeper_for_a_darker_sound():
    bright = _tone(250, 0.3, 1) + _tone(4000, 0.3, 1)
    dark = _tone(250, 0.3, 1) + _tone(4000, 0.003, 1)
    b, d = features.aggregate(features.analyse(bright), [(0.2, 0.8)])[0], \
        features.aggregate(features.analyse(dark), [(0.2, 0.8)])[0]
    # Only the top of the five bands changes, by -40 dB. A least-squares line over
    # octaves 0..4 moves by 40 * (4 - 2) / 10 = 8 dB/octave.
    assert b["spectral_tilt"] - d["spectral_tilt"] == pytest.approx(8, abs=0.5)


def _fixture_words() -> list[dict]:
    return json.loads((FIXTURES / f"{FIXTURE}.json").read_text())["words"]


def _features(db) -> list[tuple]:
    conn = sqlite3.connect(db)
    rows = conn.execute(f"SELECT {', '.join(COLS)} FROM word_instances ORDER BY id").fetchall()
    conn.close()
    return rows


def test_cli_fills_features_then_is_idempotent_and_force_recomputes(own_db_cli):
    run, db = own_db_cli
    conn = sqlite3.connect(db)
    conn.execute("UPDATE word_instances SET f0_mean=NULL, f0_var=NULL, rms=NULL, spectral_tilt=NULL")
    conn.commit()
    conn.close()
    words = _fixture_words()

    code, out = run("features")
    assert code == 0, out
    assert f"{len(words)} words updated" in out
    stored = _features(db)
    for w, row in zip(words, stored, strict=True):
        for col, value in zip(COLS, row, strict=True):
            if w[col] is None:  # documented NULL: no voiced frame in the window
                assert value is None, (w["word"], col)
            else:
                assert value == pytest.approx(w[col], abs=1e-3), (w["word"], col)
    assert sum(r[0] is not None for r in stored) >= len(words) - 4

    code, out = run("features")
    assert code == 0, out
    assert "Done. 0 word instance(s) updated." in out

    conn = sqlite3.connect(db)
    conn.execute("UPDATE word_instances SET rms=1.0")
    conn.commit()
    conn.close()
    code, out = run("features", "--force", "--source", "1")
    assert code == 0, out
    assert f"Done. {len(words)} word instance(s) updated." in out
    assert _features(db) == stored


def _two_source_db(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    conn = index.get_connection(tmp_path / "db" / "mouthpiece.db")
    words = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in _fixture_words()]
    for name in ("a.wav", "b.wav"):
        shutil.copy(FIXTURES / f"{FIXTURE}.wav", corpus / name)
        index.insert_word_instances(conn, index.insert_source(conn, name), words)
    return conn, corpus


def test_interrupted_backfill_resumes_without_recomputing(tmp_path, monkeypatch):
    conn, corpus = _two_source_db(tmp_path)
    real, loaded = features.for_file, []

    def interrupt_on_second(path, windows):
        if loaded:
            raise KeyboardInterrupt
        loaded.append(path.name)
        return real(path, windows)

    monkeypatch.setattr(features, "for_file", interrupt_on_second)
    with pytest.raises(KeyboardInterrupt):
        list(features.backfill(conn, corpus))
    assert loaded == ["a.wav"]

    def record(path, windows):
        loaded.append(path.name)
        return real(path, windows)

    monkeypatch.setattr(features, "for_file", record)
    reports = list(features.backfill(conn, corpus))
    assert loaded == ["a.wav", "b.wav"]
    assert [(r["filename"], r["updated"]) for r in reports] == [("b.wav", len(_fixture_words()))]
    assert conn.execute("SELECT COUNT(*) FROM word_instances WHERE rms IS NULL").fetchone()[0] == 0
    assert list(features.backfill(conn, corpus)) == []


def test_missing_audio_is_skipped_and_left_null(tmp_path):
    conn, corpus = _two_source_db(tmp_path)
    (corpus / "a.wav").unlink()
    reports = list(features.backfill(conn, corpus))
    assert [(r["filename"], r["missing"], r["updated"]) for r in reports] == [
        ("a.wav", True, 0), ("b.wav", False, len(_fixture_words()))]
    null_by_source = dict(conn.execute(
        "SELECT source_id, SUM(rms IS NULL) FROM word_instances GROUP BY source_id").fetchall())
    assert null_by_source == {1: len(_fixture_words()), 2: 0}


def test_ingest_stores_features_for_voiced_words(tmp_path, monkeypatch):
    from mouthpiece import ingest, transcribe

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    audio = corpus / f"{FIXTURE}.wav"
    shutil.copy(FIXTURES / f"{FIXTURE}.wav", audio)  # already normalised, so ingest skips ffmpeg
    words = [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in _fixture_words()]
    monkeypatch.setattr(transcribe, "transcribe_audio", lambda *_args, **_kwargs: words)

    conn = index.get_connection(tmp_path / "db" / "mouthpiece.db")
    sid = ingest.ingest_file_with_meta(audio, {}, conn=conn, corpus_dir=corpus)
    rows = conn.execute(
        "SELECT word, f0_mean, f0_var, rms, spectral_tilt FROM word_instances WHERE source_id=? ORDER BY id",
        (sid,)).fetchall()
    assert len(rows) == len(words)
    assert all(r["rms"] is not None and r["spectral_tilt"] is not None for r in rows)
    farewell = next(r for r in rows if r["word"] == "farewell")
    assert 80 < farewell["f0_mean"] < 200 and farewell["f0_var"] is not None
    assert sum(r["f0_mean"] is not None for r in rows) >= len(rows) - 4
