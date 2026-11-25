"""Shared fixture: a tiny real corpus (8 s of a public-domain address) indexed without WhisperX.

The fixture JSON holds word timestamps exported from a real WhisperX ingest of the same
audio, so tests exercise the real index and assembly code with no model download. Its
acoustic features come from mouthpiece.features run on the fixture WAV.
"""

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = "obama_20170114_0-8s"


def build_fixture_db(root: Path) -> dict[str, str]:
    """Index the fixture under `root` and return the env vars that point the CLI at it."""
    from mouthpiece import index, phonemes

    data = json.loads((FIXTURES / f"{FIXTURE}.json").read_text())
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    shutil.copy(FIXTURES / f"{FIXTURE}.wav", corpus / data["source"]["filename"])

    conn = index.get_connection(root / "db" / "mouthpiece.db")
    source_id = index.insert_source(conn, **data["source"])
    index.insert_word_instances(conn, source_id, data["words"])
    for row in conn.execute("SELECT id, word, start_s, end_s FROM word_instances").fetchall():
        sylls = phonemes.compute_syllable_rows(row["word"], row["start_s"], row["end_s"])
        if sylls:
            index.insert_syllable_instances(conn, row["id"], sylls)
        phones = phonemes.compute_phone_rows(row["word"], row["start_s"], row["end_s"])
        if phones:
            index.insert_phone_instances(conn, row["id"], phones)
    conn.close()

    return {
        "MOUTHPIECE_DB": str(root / "db" / "mouthpiece.db"),
        "MOUTHPIECE_CORPUS": str(corpus),
        "MOUTHPIECE_RENDERS": str(root / "renders"),
    }


@pytest.fixture(scope="session")
def corpus_env():
    # Built under the repo's git-ignored scratch dir, not the system temp dir: SQLite
    # commits once per insert, and on a slow /tmp that made setup take ~25 s.
    scratch = Path(__file__).resolve().parents[1] / ".claude" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="pytest-corpus-") as root:
        yield build_fixture_db(Path(root))


@pytest.fixture
def unattributed_db(tmp_path):
    """A DB with one source lacking a speaker and one whose speaker is an uploader's email,
    as legacy rows look before the S0.8 backfill. Each source has one word: "hello", "world"."""
    from mouthpiece import index

    db = tmp_path / "db" / "mouthpiece.db"
    conn = index.get_connection(db)
    a = index.insert_source(conn, "a.wav", source_label="SCOTUS", date="2019-04-23")
    b = index.insert_source(conn, "b.wav", speaker="uploader@nasa.gov", source_label="NASA")
    index.insert_word_instances(conn, a, [{"word": "hello", "start": 0.0, "end": 0.5}])
    index.insert_word_instances(conn, b, [{"word": "world", "start": 0.0, "end": 0.5}])
    conn.close()
    return db


def _cli_runner(env: dict[str, str], monkeypatch, capsys):
    from mouthpiece import cli

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    def run(*args: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["mouthpiece", *args])
        capsys.readouterr()
        try:
            cli.main()
            code, message = 0, ""
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            # sys.exit("text") prints only when the interpreter exits; keep it visible here.
            message = exc.code if isinstance(exc.code, str) else ""
        return code, capsys.readouterr().out + message

    return run


@pytest.fixture
def run_cli(corpus_env, monkeypatch, capsys):
    """Run the `mouthpiece` CLI in-process against the fixture corpus.

    Returns (exit_code, stdout plus any sys.exit message). In-process keeps the
    suite fast: a subprocess pays librosa's import cost on every call.
    """
    return _cli_runner(corpus_env, monkeypatch, capsys)


@pytest.fixture
def own_db_cli(corpus_env, tmp_path, monkeypatch, capsys):
    """Like run_cli, but on a private copy of the fixture DB that the test may edit.
    Returns (run, db_path)."""
    db = tmp_path / "db" / "mouthpiece.db"
    db.parent.mkdir()
    src, dst = sqlite3.connect(corpus_env["MOUTHPIECE_DB"]), sqlite3.connect(db)
    src.backup(dst)
    src.close()
    dst.close()
    return _cli_runner({**corpus_env, "MOUTHPIECE_DB": str(db)}, monkeypatch, capsys), db
