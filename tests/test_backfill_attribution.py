"""tools/backfill_attribution.py against small fixture DBs."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from mouthpiece import index

TOOL = Path(__file__).resolve().parents[1] / "tools" / "backfill_attribution.py"
HEADER = "filename,speaker,author,reader,uploader,source_label,licence_url,lang\n"
TABLE = HEADER + (
    "nasa.wav,,,,,NASA Audio,,en\n"
    "book.wav,Ezwa,Jules Lermina,Ezwa,,LibriVox,,fr\n"
)

# The pre-S0.7 `sources` table, as the real DB still has it.
OLD_SCHEMA = """
CREATE TABLE sources (
    id          INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL UNIQUE,
    speaker     TEXT,
    date        TEXT,
    event       TEXT,
    source_url  TEXT,
    license     TEXT,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def run(db: Path, csv: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--db", str(db), "--csv", str(csv), *extra],
        capture_output=True, text=True, check=False,
    )


def sources(db: Path) -> dict[str, dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = {r["filename"]: dict(r) for r in conn.execute("SELECT * FROM sources")}
    conn.close()
    return rows


@pytest.fixture
def db(tmp_path):
    """Legacy rows: an uploader's email as NASA speaker, the book author as LibriVox speaker,
    and one source the table does not list."""
    path = tmp_path / "mouthpiece.db"
    conn = index.get_connection(path)
    index.insert_source(conn, "nasa.wav", speaker="someone@example.org", event="NASA Audio", license="public domain")
    index.insert_source(conn, "book.wav", speaker="Jules Lermina", event="LibriVox reading")
    index.insert_source(conn, "other.wav", speaker="Winston Churchill")
    conn.close()
    return path


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "attribution.csv"
    path.write_text(TABLE)
    return path


def test_applies_table(db, table):
    result = run(db, table)
    assert result.returncode == 0, result.stderr
    rows = sources(db)
    assert rows["nasa.wav"]["speaker"] is None
    assert rows["nasa.wav"]["uploader"] is None
    assert rows["nasa.wav"]["source_label"] == "NASA Audio"
    assert rows["nasa.wav"]["license"] == "public domain"
    assert rows["book.wav"]["speaker"] == "Ezwa"
    assert rows["book.wav"]["author"] == "Jules Lermina"
    assert rows["book.wav"]["reader"] == "Ezwa"
    assert rows["book.wav"]["lang"] == "fr"
    assert rows["other.wav"]["speaker"] == "Winston Churchill"
    assert "other.wav: not in csv" in result.stdout
    # The email is masked in the change log, never printed.
    assert "someone@example.org" not in result.stdout
    assert "speaker: '<email>' → NULL" in result.stdout


def test_second_run_changes_nothing(db, table):
    assert run(db, table).returncode == 0
    before = sources(db)
    result = run(db, table)
    assert result.returncode == 0
    assert sources(db) == before
    assert "nasa.wav:" not in result.stdout
    assert "book.wav:" not in result.stdout
    assert "0 row(s) changed" in result.stdout


def test_email_in_csv_refuses_and_writes_nothing(db, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(TABLE + "other.wav,,,,someone@example.org,,,en\n")
    before = sources(db)
    result = run(db, bad)
    assert result.returncode != 0
    assert "'@'" in result.stderr
    assert sources(db) == before


def test_unknown_filename_skipped_with_warning(db, tmp_path):
    extra = tmp_path / "extra.csv"
    extra.write_text(TABLE + "missing.wav,Someone,,,,,,en\n")
    result = run(db, extra)
    assert result.returncode == 0
    assert "warning: missing.wav: not in db, skipped" in result.stderr
    assert "missing.wav" not in sources(db)
    assert sources(db)["book.wav"]["speaker"] == "Ezwa"


def test_dry_run_writes_nothing(db, table):
    before = db.read_bytes()
    rows_before = sources(db)
    result = run(db, table, "--dry-run")
    assert result.returncode == 0
    assert "book.wav: speaker: 'Jules Lermina' → 'Ezwa'" in result.stdout
    assert "2 row(s) would change" in result.stdout
    assert sources(db) == rows_before
    assert db.read_bytes() == before


def test_duplicate_filename_refuses_and_writes_nothing(db, tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text(TABLE + "book.wav,Someone Else,,,,,,en\n")
    before = db.read_bytes()
    result = run(db, dup)
    assert result.returncode != 0
    assert "duplicate filename book.wav" in result.stderr
    assert db.read_bytes() == before


def old_schema_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO sources (filename, speaker) VALUES ('book.wav', 'Jules Lermina')")
    conn.execute("INSERT INTO sources (filename, speaker) VALUES ('nasa.wav', 'someone@example.org')")
    conn.commit()
    conn.close()
    return path


def test_dry_run_on_old_schema_writes_nothing_and_matches_real_run(tmp_path, table):
    path = old_schema_db(tmp_path / "old.db")
    before = path.read_bytes()
    dry = run(path, table, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert path.read_bytes() == before

    copy = tmp_path / "copy.db"
    copy.write_bytes(before)
    real = run(copy, table)
    assert real.returncode == 0, real.stderr

    dry_lines, real_lines = dry.stdout.splitlines(), real.stdout.splitlines()
    assert dry_lines[-1] == "2 row(s) would change"
    assert real_lines[-1] == "2 row(s) changed"
    assert dry_lines[:-1] == real_lines[:-1]
    assert "book.wav: speaker: 'Jules Lermina' → 'Ezwa'; author: NULL → 'Jules Lermina'" in dry.stdout


def test_old_schema_migrated_and_backfilled(tmp_path, table):
    path = old_schema_db(tmp_path / "old.db")

    result = run(path, table)
    assert result.returncode == 0, result.stderr
    row = sources(path)["book.wav"]
    assert row["author"] == "Jules Lermina"
    assert row["speaker"] == "Ezwa"
    assert row["lang"] == "fr"
