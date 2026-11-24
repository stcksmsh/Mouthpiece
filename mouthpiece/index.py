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

CREATE TABLE IF NOT EXISTS syllable_instances (
    id               INTEGER PRIMARY KEY,
    word_instance_id INTEGER NOT NULL REFERENCES word_instances(id),
    syll_index       INTEGER NOT NULL,
    syll_count       INTEGER NOT NULL,
    phones           TEXT NOT NULL,
    phone_key        TEXT NOT NULL,
    start_s          REAL NOT NULL,
    end_s            REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_syll_key  ON syllable_instances(phone_key);
CREATE INDEX IF NOT EXISTS idx_syll_word ON syllable_instances(word_instance_id);

CREATE TABLE IF NOT EXISTS phone_instances (
    id               INTEGER PRIMARY KEY,
    word_instance_id INTEGER NOT NULL REFERENCES word_instances(id),
    syll_index       INTEGER NOT NULL,
    phone_index      INTEGER NOT NULL,
    phone_count      INTEGER NOT NULL,
    phone            TEXT NOT NULL,
    phone_key        TEXT NOT NULL,
    start_s          REAL NOT NULL,
    end_s            REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_phone_key  ON phone_instances(phone_key);
CREATE INDEX IF NOT EXISTS idx_phone_word ON phone_instances(word_instance_id);
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


def next_word_instance(
    conn: sqlite3.Connection, word_instance_id: int, max_gap: float = 0.5
) -> dict | None:
    """The word instance immediately following `word_instance_id` in the same
    source recording, within `max_gap` seconds — a proxy for 'spoken as part
    of the same continuous utterance' (small negative slack allows for slight
    timestamp overlap/rounding)."""
    row = conn.execute(
        """SELECT wi2.id, wi2.source_id, wi2.word, wi2.lemma, wi2.start_s, wi2.end_s,
                  COALESCE(wi2.confidence, 0) AS confidence,
                  wi2.diarization_speaker, wi2.f0_mean, wi2.f0_var, wi2.rms, wi2.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM word_instances wi1
           JOIN word_instances wi2 ON wi2.source_id = wi1.source_id
                  AND wi2.id != wi1.id
                  AND wi2.start_s >= wi1.end_s - 0.05
                  AND wi2.start_s - wi1.end_s <= ?
           JOIN sources s ON wi2.source_id = s.id
           WHERE wi1.id = ?
           ORDER BY wi2.start_s ASC
           LIMIT 1""",
        (max_gap, word_instance_id),
    ).fetchone()
    return dict(row) if row else None


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


def insert_syllable_instances(conn: sqlite3.Connection, word_instance_id: int, syllables: list[dict]) -> None:
    """syllables: list of {syll_index, syll_count, phones, phone_key, start_s, end_s}."""
    conn.executemany(
        """INSERT INTO syllable_instances
               (word_instance_id, syll_index, syll_count, phones, phone_key, start_s, end_s)
           VALUES (?,?,?,?,?,?,?)""",
        [
            (
                word_instance_id,
                s["syll_index"],
                s["syll_count"],
                s["phones"],
                s["phone_key"],
                s["start_s"],
                s["end_s"],
            )
            for s in syllables
        ],
    )
    conn.commit()


def delete_syllables_for_word(conn: sqlite3.Connection, word_instance_id: int) -> None:
    conn.execute("DELETE FROM syllable_instances WHERE word_instance_id=?", (word_instance_id,))
    conn.commit()


def next_syllable_instance(
    conn: sqlite3.Connection, syllable_instance_id: int, max_gap: float = 0.5
) -> dict | None:
    """The syllable instance immediately following `syllable_instance_id` in the
    same source recording — either the next syllable within the same parent
    word, or (if it's that word's last syllable) the first syllable of
    whatever word comes next via next_word_instance(). Returns None if the
    word doesn't continue, or the next word has no syllable rows (OOV)."""
    cur = conn.execute(
        "SELECT word_instance_id, syll_index, syll_count FROM syllable_instances WHERE id=?",
        (syllable_instance_id,),
    ).fetchone()
    if cur is None:
        return None

    if cur["syll_index"] < cur["syll_count"] - 1:
        row = conn.execute(
            """SELECT si.id, si.word_instance_id, si.syll_index, si.syll_count,
                      si.phones, si.phone_key, si.start_s, si.end_s,
                      wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                      wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                      s.filename, s.speaker, s.date, s.source_url
               FROM syllable_instances si
               JOIN word_instances wi ON si.word_instance_id = wi.id
               JOIN sources s ON wi.source_id = s.id
               WHERE si.word_instance_id = ? AND si.syll_index = ?""",
            (cur["word_instance_id"], cur["syll_index"] + 1),
        ).fetchone()
        return dict(row) if row else None

    nxt_word = next_word_instance(conn, cur["word_instance_id"], max_gap=max_gap)
    if nxt_word is None:
        return None
    row = conn.execute(
        """SELECT si.id, si.word_instance_id, si.syll_index, si.syll_count,
                  si.phones, si.phone_key, si.start_s, si.end_s,
                  wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                  wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM syllable_instances si
           JOIN word_instances wi ON si.word_instance_id = wi.id
           JOIN sources s ON wi.source_id = s.id
           WHERE si.word_instance_id = ? AND si.syll_index = 0""",
        (nxt_word["id"],),
    ).fetchone()
    return dict(row) if row else None


def search_syllable(conn: sqlite3.Connection, phone_key: str, limit: int = 20) -> list[dict]:
    """Candidate syllable instances for a phone_key, ordered by parent word's confidence."""
    rows = conn.execute(
        """SELECT si.id, si.word_instance_id, si.syll_index, si.syll_count,
                  si.phones, si.phone_key, si.start_s, si.end_s,
                  wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                  wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM syllable_instances si
           JOIN word_instances wi ON si.word_instance_id = wi.id
           JOIN sources s ON wi.source_id = s.id
           WHERE si.phone_key = ?
           ORDER BY wi.confidence DESC
           LIMIT ?""",
        (phone_key, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_syllable_instance(conn: sqlite3.Connection, instance_id: int) -> dict | None:
    row = conn.execute(
        """SELECT si.id, si.word_instance_id, si.syll_index, si.syll_count,
                  si.phones, si.phone_key, si.start_s, si.end_s,
                  wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                  wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM syllable_instances si
           JOIN word_instances wi ON si.word_instance_id = wi.id
           JOIN sources s ON wi.source_id = s.id
           WHERE si.id = ?""",
        (instance_id,),
    ).fetchone()
    return dict(row) if row else None


def insert_phone_instances(conn: sqlite3.Connection, word_instance_id: int, phones: list[dict]) -> None:
    """phones: list of {syll_index, phone_index, phone_count, phone, phone_key, start_s, end_s}."""
    conn.executemany(
        """INSERT INTO phone_instances
               (word_instance_id, syll_index, phone_index, phone_count, phone, phone_key, start_s, end_s)
           VALUES (?,?,?,?,?,?,?,?)""",
        [
            (
                word_instance_id,
                p["syll_index"],
                p["phone_index"],
                p["phone_count"],
                p["phone"],
                p["phone_key"],
                p["start_s"],
                p["end_s"],
            )
            for p in phones
        ],
    )
    conn.commit()


def delete_phones_for_word(conn: sqlite3.Connection, word_instance_id: int) -> None:
    conn.execute("DELETE FROM phone_instances WHERE word_instance_id=?", (word_instance_id,))
    conn.commit()


def next_phone_instance(
    conn: sqlite3.Connection, phone_instance_id: int, max_gap: float = 0.5
) -> dict | None:
    """The phone instance immediately following `phone_instance_id` in the
    same source recording — either the next phone within the same parent
    word, or (if it's that word's last phone) the first phone of whatever
    word comes next via next_word_instance(). Returns None if the word
    doesn't continue, or the next word has no phone rows (OOV)."""
    cur = conn.execute(
        "SELECT word_instance_id, phone_index, phone_count FROM phone_instances WHERE id=?",
        (phone_instance_id,),
    ).fetchone()
    if cur is None:
        return None

    if cur["phone_index"] < cur["phone_count"] - 1:
        row = conn.execute(
            """SELECT pi.id, pi.word_instance_id, pi.syll_index, pi.phone_index, pi.phone_count,
                      pi.phone, pi.phone_key, pi.start_s, pi.end_s,
                      wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                      wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                      s.filename, s.speaker, s.date, s.source_url
               FROM phone_instances pi
               JOIN word_instances wi ON pi.word_instance_id = wi.id
               JOIN sources s ON wi.source_id = s.id
               WHERE pi.word_instance_id = ? AND pi.phone_index = ?""",
            (cur["word_instance_id"], cur["phone_index"] + 1),
        ).fetchone()
        return dict(row) if row else None

    nxt_word = next_word_instance(conn, cur["word_instance_id"], max_gap=max_gap)
    if nxt_word is None:
        return None
    row = conn.execute(
        """SELECT pi.id, pi.word_instance_id, pi.syll_index, pi.phone_index, pi.phone_count,
                  pi.phone, pi.phone_key, pi.start_s, pi.end_s,
                  wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                  wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM phone_instances pi
           JOIN word_instances wi ON pi.word_instance_id = wi.id
           JOIN sources s ON wi.source_id = s.id
           WHERE pi.word_instance_id = ? AND pi.phone_index = 0""",
        (nxt_word["id"],),
    ).fetchone()
    return dict(row) if row else None


def search_phone(conn: sqlite3.Connection, phone_key: str, limit: int = 20) -> list[dict]:
    """Candidate phone instances for a phone_key, ordered by parent word's confidence."""
    rows = conn.execute(
        """SELECT pi.id, pi.word_instance_id, pi.syll_index, pi.phone_index, pi.phone_count,
                  pi.phone, pi.phone_key, pi.start_s, pi.end_s,
                  wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                  wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM phone_instances pi
           JOIN word_instances wi ON pi.word_instance_id = wi.id
           JOIN sources s ON wi.source_id = s.id
           WHERE pi.phone_key = ?
           ORDER BY wi.confidence DESC
           LIMIT ?""",
        (phone_key, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_phone_instance(conn: sqlite3.Connection, instance_id: int) -> dict | None:
    row = conn.execute(
        """SELECT pi.id, pi.word_instance_id, pi.syll_index, pi.phone_index, pi.phone_count,
                  pi.phone, pi.phone_key, pi.start_s, pi.end_s,
                  wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                  wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.speaker, s.date, s.source_url
           FROM phone_instances pi
           JOIN word_instances wi ON pi.word_instance_id = wi.id
           JOIN sources s ON wi.source_id = s.id
           WHERE pi.id = ?""",
        (instance_id,),
    ).fetchone()
    return dict(row) if row else None


def all_word_instances(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT id, word, start_s, end_s FROM word_instances").fetchall()
    return [dict(r) for r in rows]


def word_instances_missing_syllables(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT id, word, start_s, end_s FROM word_instances
           WHERE id NOT IN (SELECT DISTINCT word_instance_id FROM syllable_instances)"""
    ).fetchall()
    return [dict(r) for r in rows]


def word_instances_missing_phones(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT id, word, start_s, end_s FROM word_instances
           WHERE id NOT IN (SELECT DISTINCT word_instance_id FROM phone_instances)"""
    ).fetchall()
    return [dict(r) for r in rows]


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
