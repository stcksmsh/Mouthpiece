"""SQLite schema, connection management, and low-level DB operations."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY,
    filename    TEXT NOT NULL UNIQUE,
    speaker     TEXT,
    date        TEXT,
    event       TEXT,
    source_url  TEXT,
    license     TEXT,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS word_instances (
    id                  INTEGER PRIMARY KEY,
    source_id           INTEGER NOT NULL REFERENCES sources(id),
    word                TEXT NOT NULL,
    lemma               TEXT,
    phonemes            TEXT,
    start_s             REAL NOT NULL,
    end_s               REAL NOT NULL,
    confidence          REAL,
    diarization_speaker TEXT,
    f0_mean             REAL,
    f0_var              REAL,
    rms                 REAL,
    spectral_tilt       REAL
);

CREATE INDEX IF NOT EXISTS idx_wi_word     ON word_instances(word);
CREATE INDEX IF NOT EXISTS idx_wi_source   ON word_instances(source_id);

CREATE VIRTUAL TABLE IF NOT EXISTS word_fts USING fts5(
    word,
    lemma,
    content='word_instances',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS wi_ai AFTER INSERT ON word_instances BEGIN
    INSERT INTO word_fts(rowid, word, lemma) VALUES (new.id, new.word, new.lemma);
END;

CREATE TRIGGER IF NOT EXISTS wi_ad AFTER DELETE ON word_instances BEGIN
    INSERT INTO word_fts(word_fts, rowid, word, lemma) VALUES('delete', old.id, old.word, old.lemma);
END;

CREATE TRIGGER IF NOT EXISTS wi_au AFTER UPDATE ON word_instances BEGIN
    INSERT INTO word_fts(word_fts, rowid, word, lemma) VALUES('delete', old.id, old.word, old.lemma);
    INSERT INTO word_fts(rowid, word, lemma) VALUES (new.id, new.word, new.lemma);
END;
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def insert_source(
    conn: sqlite3.Connection,
    filename: str,
    speaker: str | None = None,
    date: str | None = None,
    event: str | None = None,
    source_url: str | None = None,
    license: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO sources (filename, speaker, date, event, source_url, license)"
        " VALUES (?,?,?,?,?,?)",
        (filename, speaker, date, event, source_url, license),
    )
    conn.commit()
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute("SELECT id FROM sources WHERE filename=?", (filename,)).fetchone()[0]


def insert_word_instances(conn: sqlite3.Connection, source_id: int, words: list[dict]) -> None:
    conn.executemany(
        """INSERT INTO word_instances
               (source_id, word, lemma, start_s, end_s, confidence, diarization_speaker,
                f0_mean, f0_var, rms, spectral_tilt)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                source_id,
                w["word"].lower().strip(),
                w.get("lemma"),
                w["start"],
                w["end"],
                w.get("confidence"),
                w.get("speaker"),
                w.get("f0_mean"),
                w.get("f0_var"),
                w.get("rms"),
                w.get("spectral_tilt"),
            )
            for w in words
        ],
    )
    conn.commit()


def word_count(conn: sqlite3.Connection, source_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM word_instances WHERE source_id=?", (source_id,)
    ).fetchone()[0]


def search_word(conn: sqlite3.Connection, word: str, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """SELECT wi.id, wi.source_id, wi.word, wi.lemma, wi.start_s, wi.end_s,
                  COALESCE(wi.confidence, 0) AS confidence,
                  wi.diarization_speaker, wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM word_instances wi
           JOIN sources s ON wi.source_id = s.id
           WHERE wi.word = ?
           ORDER BY wi.confidence DESC
           LIMIT ?""",
        (word.lower().strip(), limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_word_by_pitch(
    conn: sqlite3.Connection, word: str, target_f0: float, limit: int = 20
) -> list[dict]:
    """Like search_word, but ordered by proximity to a target F0 (Hz) instead of confidence."""
    rows = conn.execute(
        """SELECT wi.id, wi.source_id, wi.word, wi.lemma, wi.start_s, wi.end_s,
                  COALESCE(wi.confidence, 0) AS confidence,
                  wi.diarization_speaker, wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM word_instances wi
           JOIN sources s ON wi.source_id = s.id
           WHERE wi.word = ? AND wi.f0_mean IS NOT NULL
           ORDER BY ABS(wi.f0_mean - ?) ASC
           LIMIT ?""",
        (word.lower().strip(), target_f0, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_word_instance(conn: sqlite3.Connection, instance_id: int) -> dict | None:
    row = conn.execute(
        """SELECT wi.id, wi.source_id, wi.word, wi.lemma, wi.start_s, wi.end_s,
                  COALESCE(wi.confidence, 0) AS confidence,
                  wi.diarization_speaker, wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM word_instances wi
           JOIN sources s ON wi.source_id = s.id
           WHERE wi.id = ?""",
        (instance_id,),
    ).fetchone()
    return dict(row) if row else None


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT s.id, s.filename, s.speaker, s.date, s.event,
                  COUNT(wi.id) AS word_count
           FROM sources s
           LEFT JOIN word_instances wi ON wi.source_id = s.id
           GROUP BY s.id
           ORDER BY s.ingested_at""",
    ).fetchall()
    return [dict(r) for r in rows]
