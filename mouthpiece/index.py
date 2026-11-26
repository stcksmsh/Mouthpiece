"""SQLite schema, connection management, and low-level DB operations."""

import re
import sqlite3
from dataclasses import dataclass
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
    _migrate_sources(conn)
    conn.create_function("mp_source_text", 4, _source_text, deterministic=True)
    return conn


# Attribution columns added after the first schema. They are added in place on
# open so existing DBs keep their rows; old rows read NULL.
_ADDED_SOURCE_COLS = ("author", "reader", "uploader", "source_label", "licence_url", "lang", "tier")

# Source metadata keys accepted from ingest (JSON / CSV); other keys are ignored.
SOURCE_FIELDS = ("speaker", "date", "event", "source_url", "license", *_ADDED_SOURCE_COLS)

# Attribution carried on every clip row and into render manifests.
ATTRIBUTION_FIELDS = (
    "speaker", "author", "reader", "uploader", "source_label", "license", "licence_url", "lang",
)


# Licence tiers a source can carry; NULL is read as "unverified".
TIERS = ("clean", "unverified", "restricted")


def check_tier(tier: str | None) -> None:
    if tier is not None and tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}, got {tier!r}")


@dataclass(frozen=True)
class SourceFilter:
    """Which sources clips may come from.

    A source passes when its tier (NULL counts as unverified) is in `tiers`, it
    matches at least one `only` needle (when any are given), and it matches no
    `exclude` needle. A needle matches when it is a case-insensitive substring of
    the source's speaker_label() or its filename.
    """
    tiers: tuple[str, ...] = ("clean", "unverified")
    only: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.tiers:
            raise ValueError("at least one tier is needed")
        for t in self.tiers:
            check_tier(t)
        if not all(n.strip() for n in self.only + self.exclude):
            raise ValueError("--only/--exclude need a non-empty name")

    def sql(self) -> tuple[str, tuple]:
        """A WHERE condition on the sources table aliased `s`, and its parameters."""
        conds = [f"COALESCE(s.tier, 'unverified') IN ({', '.join('?' * len(self.tiers))})"]
        params = list(self.tiers)
        hit = "instr(mp_source_text(s.speaker, s.source_label, s.date, s.filename), ?) > 0"
        if self.only:
            conds.append("(" + " OR ".join([hit] * len(self.only)) + ")")
            params += [n.casefold() for n in self.only]
        for n in self.exclude:
            conds.append(f"NOT {hit}")
            params.append(n.casefold())
        return " AND ".join(conds), tuple(params)

    def controls(self) -> dict:
        """The filter as recorded in a render manifest."""
        return {"tiers": list(self.tiers), "only": list(self.only), "exclude": list(self.exclude)}


DEFAULT_FILTER = SourceFilter()


def _source_text(speaker, source_label, date, filename) -> str:
    """What --only/--exclude needles are matched against (see SourceFilter)."""
    label = speaker_label({"speaker": speaker, "source_label": source_label, "date": date})
    return f"{label}\n{filename}".casefold()


def _migrate_sources(conn: sqlite3.Connection) -> None:
    have = {r["name"] for r in conn.execute("PRAGMA table_info(sources)")}
    for col in _ADDED_SOURCE_COLS:
        if col not in have:
            conn.execute(f"ALTER TABLE sources ADD COLUMN {col} TEXT")
    conn.commit()


def speaker_label(row: dict) -> str:
    """The one speaker string shown to users.

    Legacy rows sometimes hold an uploader's email in `speaker`; that names no
    speaker, so it is treated as unset and never shown.
    """
    speaker = row.get("speaker")
    if speaker and "@" not in speaker:
        return speaker
    parts = ["unidentified speaker"]
    parts += [v for k in ("source_label", "date") if (v := row.get(k)) and "@" not in v]
    return ", ".join(parts)


def attribution(row: dict) -> dict:
    """A clip row's attribution fields plus its display label."""
    out = {k: row.get(k) for k in ATTRIBUTION_FIELDS}
    out["speaker_label"] = speaker_label(row)
    return out


def stitched_attribution(rows: list[dict]) -> dict:
    """Attribution for a word stitched from several clips.

    Speakers and labels are joined over all clips. Any other field the clips
    disagree on is None, since no single value is true for the whole word;
    each clip keeps its own attribution.
    """
    out = {k: v if all(r.get(k) == v for r in rows) else None for k, v in attribution(rows[0]).items()}
    speakers = list(dict.fromkeys(r["speaker"] for r in rows if r.get("speaker")))
    out["speaker"] = ", ".join(speakers) or None
    out["speaker_label"] = ", ".join(dict.fromkeys(speaker_label(r) for r in rows))
    return out


def insert_source(
    conn: sqlite3.Connection,
    filename: str,
    speaker: str | None = None,
    date: str | None = None,
    event: str | None = None,
    source_url: str | None = None,
    license: str | None = None,
    author: str | None = None,
    reader: str | None = None,
    uploader: str | None = None,
    source_label: str | None = None,
    licence_url: str | None = None,
    lang: str | None = None,
    tier: str | None = None,
) -> int:
    check_tier(tier)
    cur = conn.execute(
        "INSERT OR IGNORE INTO sources (filename, speaker, date, event, source_url, license,"
        " author, reader, uploader, source_label, licence_url, lang, tier)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (filename, speaker, date, event, source_url, license,
         author, reader, uploader, source_label, licence_url, lang, tier),
    )
    conn.commit()
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute("SELECT id FROM sources WHERE filename=?", (filename,)).fetchone()[0]


_EDGE_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")
_QUOTED = re.compile(r"^'(.+)'$")


def normalize_word(word: str) -> str:
    """Lowercase and strip leading/trailing punctuation ('week,' -> 'week', 'mr.' -> 'mr').

    WhisperX attaches punctuation to words, but lookups use clean tokens, so an
    unstripped 'week,' can never match. Inner punctuation ('eye-to-eye') and
    apostrophes stay, except a pair wrapping the whole token ("'hello'" -> 'hello');
    one-sided ones ("'em", "goin'") are contractions. A word that is only punctuation
    is kept as-is.
    """
    w = word.lower().strip()
    w = _EDGE_PUNCT.sub("", w) or w
    return _QUOTED.sub(r"\1", w)


def normalize_stored_words(conn: sqlite3.Connection) -> int:
    """Apply normalize_word() to every stored word instance. Returns rows changed."""
    rows = conn.execute("SELECT id, word FROM word_instances").fetchall()
    changes = [(normalize_word(r["word"]), r["id"]) for r in rows if normalize_word(r["word"]) != r["word"]]
    conn.executemany("UPDATE word_instances SET word=? WHERE id=?", changes)
    conn.commit()
    return len(changes)


def insert_word_instances(conn: sqlite3.Connection, source_id: int, words: list[dict]) -> None:
    conn.executemany(
        """INSERT INTO word_instances
               (source_id, word, lemma, start_s, end_s, confidence, diarization_speaker,
                f0_mean, f0_var, rms, spectral_tilt)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                source_id,
                normalize_word(w["word"]),
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


def update_word_features(conn: sqlite3.Connection, rows: list[tuple[int, dict]]) -> None:
    """Store acoustic features (see features.py) for (word_instance_id, features) pairs."""
    conn.executemany(
        "UPDATE word_instances SET f0_mean=?, f0_var=?, rms=?, spectral_tilt=? WHERE id=?",
        [(f["f0_mean"], f["f0_var"], f["rms"], f["spectral_tilt"], wid) for wid, f in rows],
    )
    conn.commit()


def word_count(conn: sqlite3.Connection, source_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM word_instances WHERE source_id=?", (source_id,)
    ).fetchone()[0]


def search_word(
    conn: sqlite3.Connection, word: str, limit: int = 20, source_filter: SourceFilter = DEFAULT_FILTER,
) -> list[dict]:
    cond, params = source_filter.sql()
    rows = conn.execute(
        """SELECT wi.id, wi.source_id, wi.word, wi.lemma, wi.start_s, wi.end_s,
                  COALESCE(wi.confidence, 0) AS confidence,
                  wi.diarization_speaker, wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.date, s.source_url,
                  s.speaker, s.author, s.reader, s.uploader, s.source_label,
                  s.license, s.licence_url, s.lang
           FROM word_instances wi
           JOIN sources s ON wi.source_id = s.id
           WHERE wi.word = ? AND {cond}
           ORDER BY wi.confidence DESC
           LIMIT ?""".format(cond=cond),
        (word.lower().strip(), *params, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def next_word_instance(
    conn: sqlite3.Connection, word_instance_id: int, max_gap: float = 0.5,
    source_filter: SourceFilter = DEFAULT_FILTER,
) -> dict | None:
    """The word instance immediately following `word_instance_id` in the same
    source recording, within `max_gap` seconds — a proxy for 'spoken as part
    of the same continuous utterance' (small negative slack allows for slight
    timestamp overlap/rounding)."""
    cond, params = source_filter.sql()
    row = conn.execute(
        """SELECT wi2.id, wi2.source_id, wi2.word, wi2.lemma, wi2.start_s, wi2.end_s,
                  COALESCE(wi2.confidence, 0) AS confidence,
                  wi2.diarization_speaker, wi2.f0_mean, wi2.f0_var, wi2.rms, wi2.spectral_tilt,
                  s.filename, s.date, s.source_url,
                  s.speaker, s.author, s.reader, s.uploader, s.source_label,
                  s.license, s.licence_url, s.lang
           FROM word_instances wi1
           JOIN word_instances wi2 ON wi2.source_id = wi1.source_id
                  AND wi2.id != wi1.id
                  AND wi2.start_s >= wi1.end_s - 0.05
                  AND wi2.start_s - wi1.end_s <= ?
           JOIN sources s ON wi2.source_id = s.id
           WHERE wi1.id = ? AND {cond}
           ORDER BY wi2.start_s ASC
           LIMIT 1""".format(cond=cond),
        (max_gap, word_instance_id, *params),
    ).fetchone()
    return dict(row) if row else None


def search_word_by_pitch(
    conn: sqlite3.Connection, word: str, target_f0: float, limit: int = 20,
    source_filter: SourceFilter = DEFAULT_FILTER,
) -> list[dict]:
    """Like search_word, but ordered by proximity to a target F0 (Hz) instead of confidence."""
    cond, params = source_filter.sql()
    rows = conn.execute(
        """SELECT wi.id, wi.source_id, wi.word, wi.lemma, wi.start_s, wi.end_s,
                  COALESCE(wi.confidence, 0) AS confidence,
                  wi.diarization_speaker, wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.date, s.source_url,
                  s.speaker, s.author, s.reader, s.uploader, s.source_label,
                  s.license, s.licence_url, s.lang
           FROM word_instances wi
           JOIN sources s ON wi.source_id = s.id
           WHERE wi.word = ? AND wi.f0_mean IS NOT NULL AND {cond}
           ORDER BY ABS(wi.f0_mean - ?) ASC
           LIMIT ?""".format(cond=cond),
        (word.lower().strip(), *params, target_f0, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_word_instance(conn: sqlite3.Connection, instance_id: int) -> dict | None:
    row = conn.execute(
        """SELECT wi.id, wi.source_id, wi.word, wi.lemma, wi.start_s, wi.end_s,
                  COALESCE(wi.confidence, 0) AS confidence,
                  wi.diarization_speaker, wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                  s.filename, s.date, s.source_url,
                  s.speaker, s.author, s.reader, s.uploader, s.source_label,
                  s.license, s.licence_url, s.lang
           FROM word_instances wi
           JOIN sources s ON wi.source_id = s.id
           WHERE wi.id = ?""",
        (instance_id,),
    ).fetchone()
    return dict(row) if row else None


def source_allowed(conn: sqlite3.Connection, word_instance_id: int, source_filter: SourceFilter) -> bool:
    """Whether the word instance's source passes `source_filter`."""
    cond, params = source_filter.sql()
    return conn.execute(
        f"SELECT 1 FROM word_instances wi JOIN sources s ON wi.source_id = s.id WHERE wi.id = ? AND {cond}",
        (word_instance_id, *params),
    ).fetchone() is not None


# Sub-word unit tables. Syllables and phones share one query shape; only these
# column lists differ. Names come from this fixed dict, never from callers.
_UNITS = {
    "syllable": {
        "table": "syllable_instances",
        "cols":  ["syll_index", "syll_count", "phones", "phone_key", "start_s", "end_s"],
        "pos":   "syll_index",
        "count": "syll_count",
    },
    "phone": {
        "table": "phone_instances",
        "cols":  ["syll_index", "phone_index", "phone_count", "phone", "phone_key", "start_s", "end_s"],
        "pos":   "phone_index",
        "count": "phone_count",
    },
}


def _unit_select(kind: str) -> str:
    u = _UNITS[kind]
    cols = ", ".join(f"u.{c}" for c in u["cols"])
    return f"""SELECT u.id, u.word_instance_id, {cols},
                      wi.word AS parent_word, COALESCE(wi.confidence, 0) AS confidence,
                      wi.f0_mean, wi.f0_var, wi.rms, wi.spectral_tilt,
                      s.filename, s.date, s.source_url,
                      s.speaker, s.author, s.reader, s.uploader, s.source_label,
                      s.license, s.licence_url, s.lang
               FROM {u["table"]} u
               JOIN word_instances wi ON u.word_instance_id = wi.id
               JOIN sources s ON wi.source_id = s.id"""


def _unit_one(conn: sqlite3.Connection, kind: str, where: str, params: tuple) -> dict | None:
    row = conn.execute(f"{_unit_select(kind)} WHERE {where}", params).fetchone()
    return dict(row) if row else None


def _insert_units(conn: sqlite3.Connection, kind: str, word_instance_id: int, rows: list[dict]) -> None:
    u = _UNITS[kind]
    cols = ["word_instance_id", *u["cols"]]
    conn.executemany(
        f"INSERT INTO {u['table']} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [(word_instance_id, *(r[c] for c in u["cols"])) for r in rows],
    )
    conn.commit()


def _next_unit(conn: sqlite3.Connection, kind: str, unit_id: int, max_gap: float,
               source_filter: SourceFilter) -> dict | None:
    """The unit right after `unit_id` in the same recording: the next unit of the same
    word, or the first unit of the following word (via next_word_instance). None if
    speech does not continue or the next word has no units (not in CMUdict).
    The next unit of the same word shares its source, so only the step to the
    next word needs `source_filter`."""
    u = _UNITS[kind]
    cur = conn.execute(
        f"SELECT word_instance_id, {u['pos']} AS pos, {u['count']} AS n FROM {u['table']} WHERE id=?",
        (unit_id,),
    ).fetchone()
    if cur is None:
        return None
    if cur["pos"] < cur["n"] - 1:
        return _unit_one(conn, kind, f"u.word_instance_id = ? AND u.{u['pos']} = ?",
                         (cur["word_instance_id"], cur["pos"] + 1))
    nxt_word = next_word_instance(conn, cur["word_instance_id"], max_gap=max_gap,
                                  source_filter=source_filter)
    if nxt_word is None:
        return None
    return _unit_one(conn, kind, f"u.word_instance_id = ? AND u.{u['pos']} = 0", (nxt_word["id"],))


def _search_units(conn: sqlite3.Connection, kind: str, phone_key: str, limit: int,
                  source_filter: SourceFilter) -> list[dict]:
    cond, params = source_filter.sql()
    rows = conn.execute(
        f"{_unit_select(kind)} WHERE u.phone_key = ? AND {cond} ORDER BY wi.confidence DESC LIMIT ?",
        (phone_key, *params, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_syllable_instances(conn: sqlite3.Connection, word_instance_id: int, syllables: list[dict]) -> None:
    """syllables: list of {syll_index, syll_count, phones, phone_key, start_s, end_s}."""
    _insert_units(conn, "syllable", word_instance_id, syllables)


def insert_phone_instances(conn: sqlite3.Connection, word_instance_id: int, phones: list[dict]) -> None:
    """phones: list of {syll_index, phone_index, phone_count, phone, phone_key, start_s, end_s}."""
    _insert_units(conn, "phone", word_instance_id, phones)


def next_syllable_instance(conn: sqlite3.Connection, syllable_instance_id: int, max_gap: float = 0.5,
                          source_filter: SourceFilter = DEFAULT_FILTER) -> dict | None:
    return _next_unit(conn, "syllable", syllable_instance_id, max_gap, source_filter)


def next_phone_instance(conn: sqlite3.Connection, phone_instance_id: int, max_gap: float = 0.5,
                          source_filter: SourceFilter = DEFAULT_FILTER) -> dict | None:
    return _next_unit(conn, "phone", phone_instance_id, max_gap, source_filter)


def search_syllable(conn: sqlite3.Connection, phone_key: str, limit: int = 20,
                  source_filter: SourceFilter = DEFAULT_FILTER) -> list[dict]:
    """Candidate syllable instances for a phone_key, ordered by parent word's confidence."""
    return _search_units(conn, "syllable", phone_key, limit, source_filter)


def search_phone(conn: sqlite3.Connection, phone_key: str, limit: int = 20,
                  source_filter: SourceFilter = DEFAULT_FILTER) -> list[dict]:
    """Candidate phone instances for a phone_key, ordered by parent word's confidence."""
    return _search_units(conn, "phone", phone_key, limit, source_filter)


def get_syllable_instance(conn: sqlite3.Connection, instance_id: int) -> dict | None:
    return _unit_one(conn, "syllable", "u.id = ?", (instance_id,))


def get_phone_instance(conn: sqlite3.Connection, instance_id: int) -> dict | None:
    return _unit_one(conn, "phone", "u.id = ?", (instance_id,))


def all_word_instances(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT id, word, start_s, end_s FROM word_instances").fetchall()
    return [dict(r) for r in rows]


def word_instances_missing_units(conn: sqlite3.Connection, kind: str) -> list[dict]:
    """Word instances with no rows in the `kind` ("syllable" or "phone") table."""
    table = _UNITS[kind]["table"]
    rows = conn.execute(
        f"""SELECT id, word, start_s, end_s FROM word_instances
            WHERE id NOT IN (SELECT DISTINCT word_instance_id FROM {table})"""
    ).fetchall()
    return [dict(r) for r in rows]


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT s.id, s.filename, s.speaker, s.source_label, s.date, s.event,
                  COUNT(wi.id) AS word_count
           FROM sources s
           LEFT JOIN word_instances wi ON wi.source_id = s.id
           GROUP BY s.id
           ORDER BY s.ingested_at""",
    ).fetchall()
    return [dict(r) for r in rows]
