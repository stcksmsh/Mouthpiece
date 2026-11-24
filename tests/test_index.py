"""Index-level behaviour that does not need audio."""

import pytest

from mouthpiece import index


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("week,", "week"),
        ("Me.", "me"),
        ("mr.", "mr"),
        ('"hello"', "hello"),
        ("don't", "don't"),
        ("'hello'", "hello"),
        ("'em", "'em"),
        ("goin'", "goin'"),
        ("eye-to-eye", "eye-to-eye"),
        ("что", "что"),
        ("...", "..."),
    ],
)
def test_normalize_word(raw, expected):
    assert index.normalize_word(raw) == expected


def test_normalize_stored_words_fixes_old_rows(tmp_path):
    conn = index.get_connection(tmp_path / "t.db")
    sid = index.insert_source(conn, "f.wav")
    # Simulate a row written before normalisation existed.
    conn.execute("INSERT INTO word_instances (source_id, word, start_s, end_s) VALUES (?, 'nation,', 0, 1)", (sid,))
    conn.commit()
    assert index.normalize_stored_words(conn) == 1
    assert [r["word"] for r in index.search_word(conn, "nation")] == ["nation"]
    assert index.normalize_stored_words(conn) == 0


@pytest.mark.parametrize("kind", ["syllable", "phone"])
def test_word_instances_missing_units(tmp_path, kind):
    from mouthpiece import phonemes

    conn = index.get_connection(tmp_path / "t.db")
    sid = index.insert_source(conn, "a.wav")
    index.insert_word_instances(conn, sid, [{"word": "hello", "start": 0.0, "end": 0.5}])
    assert [w["word"] for w in index.word_instances_missing_units(conn, kind)] == ["hello"]
    wid = conn.execute("SELECT id FROM word_instances").fetchone()[0]
    if kind == "syllable":
        index.insert_syllable_instances(conn, wid, phonemes.compute_syllable_rows("hello", 0.0, 0.5))
    else:
        index.insert_phone_instances(conn, wid, phonemes.compute_phone_rows("hello", 0.0, 0.5))
    assert index.word_instances_missing_units(conn, kind) == []


def test_old_schema_db_is_migrated_in_place(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)
    old.execute(
        """CREATE TABLE sources (
            id INTEGER PRIMARY KEY, filename TEXT NOT NULL UNIQUE, speaker TEXT, date TEXT,
            event TEXT, source_url TEXT, license TEXT,
            ingested_at TEXT NOT NULL DEFAULT (datetime('now')))"""
    )
    old.executemany(
        "INSERT INTO sources (filename, speaker, license) VALUES (?, ?, ?)",
        [("a.wav", "Ann", "CC0"), ("b.wav", None, None)],
    )
    old.commit()
    old.close()

    for _ in range(2):  # the second open must be a no-op
        conn = index.get_connection(db)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(sources)")}
        rows = [dict(r) for r in conn.execute("SELECT * FROM sources ORDER BY id")]
        conn.close()
        assert {"author", "reader", "uploader", "source_label", "licence_url", "lang"} <= cols
        assert {"speaker", "license"} <= cols
        assert len(rows) == 2
        assert (rows[0]["speaker"], rows[0]["license"], rows[0]["author"]) == ("Ann", "CC0", None)


def test_insert_source_stores_attribution(tmp_path):
    conn = index.get_connection(tmp_path / "t.db")
    sid = index.insert_source(conn, "f.wav", reader="Rae", uploader="u@x.org", source_label="LibriVox", lang="en")
    index.insert_word_instances(conn, sid, [{"word": "hello", "start": 0.0, "end": 0.5}])
    (row,) = index.search_word(conn, "hello")
    assert (row["reader"], row["uploader"], row["source_label"], row["lang"]) == ("Rae", "u@x.org", "LibriVox", "en")


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"speaker": "Barack Obama", "source_label": "White House", "date": "2017-01-10"}, "Barack Obama"),
        ({"speaker": None, "source_label": "SCOTUS", "date": "2019-04-23"}, "unidentified speaker, SCOTUS, 2019-04-23"),
        ({"speaker": None, "source_label": None, "date": "2019-04-23"}, "unidentified speaker, 2019-04-23"),
        ({"speaker": None, "source_label": None, "date": None}, "unidentified speaker"),
        ({}, "unidentified speaker"),
        ({"speaker": "uploader@nasa.gov", "source_label": "NASA", "date": None}, "unidentified speaker, NASA"),
        ({"speaker": "uploader@nasa.gov"}, "unidentified speaker"),
    ],
)
def test_speaker_label(row, expected):
    label = index.speaker_label(row)
    assert label == expected
    assert "@" not in label


def test_stitched_attribution_joins_speakers_and_labels():
    rows = [
        {"speaker": "Ann", "license": "CC0"},
        {"speaker": "a@b.org", "source_label": "NASA"},
        {"speaker": "Ann"},
    ]
    out = index.stitched_attribution(rows)
    assert out["speaker_label"] == "Ann, unidentified speaker, NASA"
    assert out["license"] == "CC0"
