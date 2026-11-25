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
    "librivox": "unverified",
    "speeches": "unverified",
    "churchill": "restricted",
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
