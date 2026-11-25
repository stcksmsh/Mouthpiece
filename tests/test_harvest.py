"""harvest.py licence and attribution handling, with archive.org, HTTP and ffmpeg stubbed out."""

import csv
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# harvest.py is a script at the repo root, not part of the package.
_spec = importlib.util.spec_from_file_location(
    "harvest", Path(__file__).resolve().parents[1] / "harvest.py")
harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harvest)

# Must match the adapter table in the S0.10 pull request.
EXPECTED_TIERS = {
    "obama": "clean",
    "trump": "restricted",
    "scotus": "unverified",
    "nasa": "unverified",
    "nasa_reels": "clean",
    "apollo11": "clean",
    "librivox": "clean",
    "speeches": "unverified",
    "churchill": "restricted",
    "voa": "clean",
}

EXPECTED_LICENCE_URLS = {
    "librivox": "https://librivox.org/pages/public-domain/",
    "voa": "https://www.voanews.com/p/5338.html",
}


class _FakeIAItem:
    def __init__(self, metadata, files):
        self.metadata = metadata
        self.files = files


def _fake_ia(monkeypatch, items):
    """Install a stand-in `internetarchive` module serving `items` {identifier: _FakeIAItem}."""
    mod = types.ModuleType("internetarchive")
    mod.search_items = lambda _query: [{"identifier": i} for i in items]
    mod.get_item = lambda ident: items[ident]
    monkeypatch.setitem(sys.modules, "internetarchive", mod)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, _size):
        yield b"audio"


@pytest.fixture
def harvest_dir(tmp_path, monkeypatch):
    """Point the harvester at tmp_path and stub the download and ffmpeg steps."""
    monkeypatch.setattr(harvest, "OUT", str(tmp_path))
    monkeypatch.setattr(harvest, "MANIFEST", str(tmp_path / "batch.csv"))
    monkeypatch.setattr(harvest.requests, "get", lambda *_a, **_k: _FakeResponse())

    def fake_ffmpeg(cmd, **_kwargs):
        Path(cmd[-1]).write_bytes(b"wav")

    monkeypatch.setattr(harvest.subprocess, "run", fake_ffmpeg)
    return tmp_path


def _harvest_rows(adapter):
    for item in adapter:
        assert harvest.normalize_and_write(item, set())
    with open(harvest.MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == harvest.FIELDS
        return list(reader)


def test_item_has_no_licence_default():
    with pytest.raises(TypeError):
        harvest.Item(media_url="u", speaker="s", date="", event="e", source_url="u")


@pytest.mark.usefixtures("harvest_dir")
def test_undeclared_adapter_writes_empty_licence_and_unverified(monkeypatch):
    _fake_ia(monkeypatch, {"bundle": _FakeIAItem({}, [{"name": "talk.mp3"}])})
    adapter = harvest.IAItemFilesAdapter(identifier="bundle", speaker="Someone", event="Talk")

    [row] = _harvest_rows(adapter)

    assert row["license"] == ""
    assert row["tier"] == "unverified"
    assert row["licence_url"] == ""


def test_invalid_tier_is_rejected():
    with pytest.raises(ValueError):
        harvest.IAItemFilesAdapter(identifier="x", speaker="s", event="e", tier="public domain")


def test_every_source_declares_the_expected_tier():
    assert set(harvest.SOURCES) == set(EXPECTED_TIERS)
    for name, make in harvest.SOURCES.items():
        licence = make().licence
        assert licence["tier"] == EXPECTED_TIERS[name], name
        if licence["tier"] == "clean":
            assert licence["licence_url"].startswith("https://"), name
            assert licence["license"], name
        if name in EXPECTED_LICENCE_URLS:
            assert licence["licence_url"] == EXPECTED_LICENCE_URLS[name], name


@pytest.mark.usefixtures("harvest_dir")
def test_restricted_source_still_records_its_tier(monkeypatch):
    _fake_ia(monkeypatch, {"cspan_1": _FakeIAItem(
        {"creator": "CSPAN", "date": "2019-01-01"}, [{"name": "clip.mp4", "size": "10"}])})

    [row] = _harvest_rows(harvest.SOURCES["trump"]())

    assert row["tier"] == "restricted"


@pytest.mark.usefixtures("harvest_dir")
def test_nasa_drops_uploader_email(monkeypatch):
    _fake_ia(monkeypatch, {"nasa_item": _FakeIAItem(
        {"creator": "someone@example.com", "uploader": "someone@example.com",
         "date": "1969-07-20", "title": "Liftoff"},
        [{"name": "liftoff.mp3"}])})

    [row] = _harvest_rows(harvest.SOURCES["nasa"]())

    assert not any("@" in v for v in row.values())
    assert row["speaker"] == ""
    assert row["source_label"] == "NASA Audio"
    assert row["author"] == ""


@pytest.mark.usefixtures("harvest_dir")
def test_librivox_creator_is_author_not_speaker(monkeypatch):
    _fake_ia(monkeypatch, {"pride_lv": _FakeIAItem(
        {"creator": ["Jane Austen"], "date": "2007-01-01", "title": "Pride and Prejudice"},
        [{"name": "pride_01.mp3"}])})

    [row] = _harvest_rows(harvest.SOURCES["librivox"]())

    assert not any("@" in v for v in row.values())
    assert row["author"] == "Jane Austen"
    assert row["speaker"] == ""


_VOA_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>VOA Learning English Podcast</title>
<item><title>Learning English Podcast - March 31, 2025</title>
<link>https://learningenglish.voanews.com/a/8011955.html</link>
<pubDate>Mon, 31 Mar 2025 00:30:03 +0000</pubDate>
<author>someone@example.com</author>
<enclosure url="https://voa-audio.voanews.eu/vle/2025/03/31/20250331-003003-vle122-program_hq.mp3" type="audio/mpeg" length="1"/>
</item>
<item><title>No audio here</title><link>https://learningenglish.voanews.com/a/1.html</link></item>
</channel></rss>"""


class _FakeFeed:
    content = _VOA_FEED

    def raise_for_status(self):
        return None


@pytest.mark.usefixtures("harvest_dir")
def test_voa_feed_yields_labelled_items_without_speaker(monkeypatch):
    monkeypatch.setattr(harvest.requests, "get", lambda *_a, **_k: _FakeFeed())
    items = list(harvest.SOURCES["voa"]())
    monkeypatch.setattr(harvest.requests, "get", lambda *_a, **_k: _FakeResponse())

    rows = _harvest_rows(items)

    [row] = rows
    assert not any("@" in v for v in row.values())
    assert row["speaker"] == ""
    assert row["source_label"] == "VOA Learning English"
    assert row["date"] == "2025-03-31"
    assert row["event"] == "Learning English Podcast - March 31, 2025"
    assert row["source_url"] == "https://learningenglish.voanews.com/a/8011955.html"
    assert row["tier"] == "clean"
    assert row["license"] == "US federal government work (Voice of America)"
    assert row["licence_url"] == "https://www.voanews.com/p/5338.html"


class _BodyFeed:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


@pytest.mark.parametrize("body, reason", [
    (_VOA_FEED + b" " * (10 * 1024 * 1024), "over"),
    (_VOA_FEED.replace(b"<rss", b'<!DOCTYPE rss [<!ENTITY a "aaaa">]><rss', 1), "entities"),
])
def test_rss_feed_refuses_oversized_or_entity_bodies(monkeypatch, capsys, body, reason):
    monkeypatch.setattr(harvest.requests, "get", lambda *_a, **_k: _BodyFeed(body))

    assert list(harvest.SOURCES["voa"]()) == []
    out = capsys.readouterr().out
    assert "feed refused" in out
    assert reason in out


def test_old_header_batch_file_is_refused_untouched(tmp_path, monkeypatch, capsys):
    batch = tmp_path / "batch.csv"
    old = "file,speaker,date,event,source_url,license\ncorpus/a.wav,X,,E,u,public domain\n"
    batch.write_text(old)
    monkeypatch.setattr(harvest, "MANIFEST", str(batch))
    monkeypatch.setattr(sys, "argv", ["harvest.py", "--source", "obama", "--max", "1"])
    monkeypatch.setitem(harvest.SOURCES, "obama", lambda: pytest.fail("adapter must not run"))

    with pytest.raises(SystemExit) as exc:
        harvest.main()

    assert exc.value.code not in (0, None)
    assert batch.read_text() == old
    assert "new batch file" in capsys.readouterr().out
