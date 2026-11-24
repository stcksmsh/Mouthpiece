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
