"""Shared fixture: a tiny real corpus (8 s of a public-domain address) indexed without WhisperX.

The fixture JSON holds word timestamps exported from a real WhisperX ingest of the same
audio, so tests exercise the real index and assembly code with no model download.
"""

import json
import shutil
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
def run_cli(corpus_env, monkeypatch, capsys):
    """Run the `mouthpiece` CLI in-process against the fixture corpus.

    Returns (exit_code, stdout). In-process keeps the suite fast: a subprocess
    pays librosa's import cost on every call.
    """
    from mouthpiece import cli

    for key, value in corpus_env.items():
        monkeypatch.setenv(key, value)

    def run(*args: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["mouthpiece", *args])
        capsys.readouterr()
        try:
            cli.main()
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        return code, capsys.readouterr().out

    return run
