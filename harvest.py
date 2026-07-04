#!/usr/bin/env python3
"""
MOUTHPIECE corpus harvester — multi-source, pluggable, automated.

Each SOURCE is an adapter that yields Item(media_url, meta) records.
The core downloads, normalizes to mono 16k WAV via ffmpeg, and writes a
sidecar JSON matching the mouthpiece ingest schema. Source-agnostic:
add an adapter, get corpus files, no core changes.

Two adapter styles are provided:
  - GovArchiveAdapter: scrape a frozen .gov index for a media-URL pattern
                       (Obama / Trump White House archives). PD, direct files.
  - IAAdapter:         enumerate archive.org items via the internetarchive
                       API by collection/query, pull the audio derivative
                       (Free Law Project oral arguments, NASA/agency audio).

Only .gov-frozen and IA-community/FedFlix/FreeLaw items are used — NOT the
TV News Archive (C-SPAN) captures, which are access-restricted (403).

Requires: python3, ffmpeg on PATH, `pip install requests internetarchive`
Usage:
  ./harvest.py --source obama         --max 5
  ./harvest.py --source trump         --max 5
  ./harvest.py --source scotus        --max 5
  ./harvest.py --source nasa          --max 5
  ./harvest.py --source all           --max 20
"""

import argparse, csv, hashlib, os, re, subprocess, sys, tempfile
from dataclasses import dataclass, field
from typing import Iterator, Optional

import requests

OUT = "corpus"
MANIFEST = "corpus/batch.csv"          # ingest manifest: file,speaker,date,event,source_url,license
FIELDS = ["file", "speaker", "date", "event", "source_url", "license"]
UA = {"User-Agent": "Mozilla/5.0 (mouthpiece-harvest; research)"}
os.makedirs(OUT, exist_ok=True)


@dataclass
class Item:
    media_url: str
    speaker: str
    date: str            # ISO YYYY-MM-DD or "" if unknown
    event: str
    source_url: str
    license: str = "public domain"
    stem: str = ""       # output filename stem; auto-derived if empty


# ------------- manifest (append-only, dedup by file path) -------------

def _load_manifest_paths() -> set:
    if not os.path.exists(MANIFEST):
        return set()
    with open(MANIFEST, newline="") as f:
        return {row["file"] for row in csv.DictReader(f)}

def _append_manifest(row: dict):
    exists = os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------------- core -------------------------

def normalize_and_write(item: Item, done_paths: set) -> bool:
    stem = item.stem or _derive_stem(item)
    wav = os.path.join(OUT, f"{stem}.wav")
    if wav in done_paths or os.path.exists(wav):
        print(f"  = {stem} exists, skip")
        done_paths.add(wav)
        return True

    is_youtube = "youtube.com/watch" in item.media_url or "youtu.be/" in item.media_url

    if is_youtube:
        # yt-dlp extracts + transcodes to wav directly; skip requests/ffmpeg branch.
        try:
            subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "wav",
                 "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
                 "-o", os.path.join(OUT, f"{stem}.%(ext)s"),
                 item.media_url],
                check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  ! yt-dlp download failed: {e.stderr[:200] if e.stderr else e}")
            return False
        if not os.path.exists(wav):
            print(f"  ! yt-dlp produced no {wav}")
            return False
    else:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=_ext(item.media_url)) as tmp:
                raw = tmp.name
            with requests.get(item.media_url, headers=UA, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(raw, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
        except Exception as e:
            print(f"  ! download failed: {e}")
            return False
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                 "-i", raw, "-ac", "1", "-ar", "16000", wav],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ! ffmpeg failed: {e}")
            return False
        finally:
            try: os.remove(raw)
            except OSError: pass
    # append a manifest row in the mouthpiece batch schema
    _append_manifest({
        "file": wav,
        "speaker": item.speaker,
        "date": item.date,
        "event": item.event,
        "source_url": item.source_url,
        "license": item.license,
    })
    done_paths.add(wav)
    print(f"  + {stem}")
    return True


def _ext(url: str) -> str:
    m = re.search(r"\.(mp3|mp4|m4a|m4v|mkv|webm|ogv|wav|ogg)(\?|$)", url, re.I)
    return "." + m.group(1).lower() if m else ".bin"


def _make_stem(identifier: str, base: str, maxlen: int = 80) -> str:
    """Readable, collision-proof stem: slugified identifier+title, capped,
    with a short hash of the full base appended so distinct titles that share
    a truncated prefix never collapse to the same filename."""
    slug = re.sub(r"[^a-z0-9]+", "_", f"{identifier}_{base}".lower()).strip("_")
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:6]
    return f"{slug[:maxlen]}_{h}"


def _derive_stem(item: Item) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", item.speaker.lower()).strip("_") or "clip"
    d = item.date.replace("-", "") if item.date else "nodate"
    tail = re.sub(r"[^a-z0-9]+", "", os.path.basename(item.media_url).lower())[:8]
    return f"{base}_{d}_{tail}"


def run(adapter, max_items: int, done_paths: set):
    n = 0
    for item in adapter:
        if n >= max_items:
            break
        n += 1
        print(f"[{n}] {item.date} {item.speaker}")
        normalize_and_write(item, done_paths)
    print(f"Done: attempted {n} items from this source.")


# ------------------------- .gov archive adapter -------------------------

class GovArchiveAdapter:
    """Scrape a frozen .gov archive for media files.

    Two modes:
      * Direct: index pages link straight to media (media_re matches on index).
      * Two-level: index pages link to ARTICLE pages (article_re), each of
        which is fetched and scanned for the media link (media_re).

    The Obama weekly-address archive is two-level: the index lists article
    pages, and the .mp3 only appears on the article page itself.
    """
    def __init__(self, base, index_urls, media_re, speaker, event,
                 article_re=None, date_from_url=None, date_from_name=None):
        self.base = base
        self.index_urls = index_urls
        self.media_re = re.compile(media_re)
        self.article_re = re.compile(article_re) if article_re else None
        self.speaker = speaker
        self.event = event
        self.date_from_url = date_from_url    # fn(article_url)->iso or None
        self.date_from_name = date_from_name  # fn(media_filename)->iso or None

    def _get(self, url):
        try:
            return requests.get(url, headers=UA, timeout=30).text
        except Exception as e:
            print(f"  ! fetch failed {url}: {e}")
            return ""

    def _emit(self, media_url, article_url=None):
        url = media_url if media_url.startswith("http") else self.base + media_url
        date = ""
        if self.date_from_url and article_url:
            date = self.date_from_url(article_url) or ""
        if not date and self.date_from_name:
            date = self.date_from_name(os.path.basename(url)) or ""
        return Item(media_url=url, speaker=self.speaker, date=date,
                    event=self.event, source_url=article_url or url)

    def __iter__(self) -> Iterator[Item]:
        seen_media = set()
        for idx in self.index_urls:
            html = self._get(idx)
            if not html:
                continue
            if self.article_re is None:
                # direct: media links live on the index page
                for m in self.media_re.finditer(html):
                    mu = m.group(0)
                    if mu in seen_media:
                        continue
                    seen_media.add(mu)
                    yield self._emit(mu)
            else:
                # two-level: collect article links, fetch each, find media
                articles = []
                seen_art = set()
                for m in self.article_re.finditer(html):
                    au = m.group(0)
                    au = au if au.startswith("http") else self.base + au
                    if au not in seen_art:
                        seen_art.add(au)
                        articles.append(au)
                for au in articles:
                    ahtml = self._get(au)
                    if not ahtml:
                        continue
                    mm = self.media_re.search(ahtml)
                    if not mm:
                        continue
                    mu = mm.group(0)
                    if mu in seen_media:
                        continue
                    seen_media.add(mu)
                    yield self._emit(mu, article_url=au)


def _date_from_press_url(url):
    # .../the-press-office/2016/12/31/weekly-address-... -> 2016-12-31
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _date8(name):  # 20170114_... -> 2017-01-14
    m = re.match(r"(\d{4})(\d{2})(\d{2})", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


# ------------------------- archive.org adapters -------------------------

class IAItemFilesAdapter:
    """Iterate the audio FILES inside a single archive.org item.

    Use when one IA item bundles many recordings as files, e.g.
    'SCOTUSOralArugments' (note: identifier is misspelled on IA itself).
    Each audio file becomes one Item. Per-file date is usually not in
    metadata, so date is parsed from the filename if possible, else "".
    """
    def __init__(self, identifier, speaker, event, date_from_name=None,
                 speaker_from_name=None):
        self.identifier = identifier
        self.speaker = speaker
        self.event = event
        self.date_from_name = date_from_name
        self.speaker_from_name = speaker_from_name  # fn(filename)->str or None

    def __iter__(self) -> Iterator[Item]:
        try:
            import internetarchive as ia
        except ImportError:
            print("  ! pip install internetarchive"); return
        try:
            item = ia.get_item(self.identifier)
        except Exception as e:
            print(f"  ! get_item {self.identifier}: {e}"); return
        seen_base = set()
        for f in item.files:
            name = f.get("name", "")
            if not name.lower().endswith((".mp3", ".m4a", ".ogg", ".wav")):
                continue
            # IA often lists the same recording in several formats/bitrates
            # (foo.ogg, foo.mp3, foo_64kb.mp3). Collapse to one per recording:
            # strip extension and common derivative suffixes, then dedup.
            base = os.path.splitext(name)[0]
            base = re.sub(r"[_.-]?(64kb|128kb|vbr|64_?kbps|spoken)$", "", base, flags=re.I)
            key = base.lower()
            if key in seen_base:
                continue
            seen_base.add(key)
            url = f"https://archive.org/download/{self.identifier}/{name}"
            date = ""
            if self.date_from_name:
                date = self.date_from_name(name) or ""
            speaker = self.speaker
            if self.speaker_from_name:
                speaker = self.speaker_from_name(name) or self.speaker
            stem = _make_stem(self.identifier, base)
            yield Item(media_url=url, speaker=speaker, date=date,
                       event=self.event,
                       source_url=f"https://archive.org/details/{self.identifier}",
                       stem=stem)


class IACollectionAdapter:
    """Enumerate SEPARATE archive.org items via advanced-search, one audio
    file each. Use for true collections where each recording is its own item.

    `query` is an IA search query, e.g. 'collection:<slug>'. Confirm the slug
    first:  ia search 'collection:<slug>' --itemlist | head
    """
    def __init__(self, query, speaker_field="creator", event="",
                 default_speaker="unknown"):
        self.query = query
        self.speaker_field = speaker_field
        self.event = event
        self.default_speaker = default_speaker

    def __iter__(self) -> Iterator[Item]:
        try:
            import internetarchive as ia
        except ImportError:
            print("  ! pip install internetarchive"); return
        for res in ia.search_items(self.query):
            ident = res["identifier"]
            try:
                item = ia.get_item(ident)
            except Exception as e:
                print(f"  ! get_item {ident}: {e}"); continue
            md = item.metadata or {}
            # Prefer a real audio derivative; fall back to video (ffmpeg extracts
            # audio downstream). C-SPAN items are typically video-only.
            AUDIO = (".mp3", ".m4a", ".ogg", ".wav")
            VIDEO = (".mp4", ".m4v", ".mkv", ".webm", ".ogv")
            pick = None
            for f in item.files:
                if f.get("name", "").lower().endswith(AUDIO):
                    pick = f["name"]; break
            if not pick:
                # choose the smallest video to avoid pulling a 1GB broadcast
                vids = [f for f in item.files
                        if f.get("name", "").lower().endswith(VIDEO)]
                if vids:
                    def _sz(f):
                        try: return int(f.get("size", 0))
                        except (TypeError, ValueError): return 1 << 62
                    pick = min(vids, key=_sz)["name"]
            if not pick:
                continue
            url = f"https://archive.org/download/{ident}/{pick}"
            date = (md.get("date") or "")[:10]
            speaker = md.get(self.speaker_field) or self.default_speaker
            if isinstance(speaker, list):
                speaker = speaker[0] if speaker else self.default_speaker
            yield Item(media_url=url, speaker=str(speaker), date=date,
                       event=self.event or md.get("title", ""),
                       source_url=f"https://archive.org/details/{ident}",
                       stem=re.sub(r"[^a-z0-9]+", "_", ident.lower())[:40])


def _scotus_date(name):
    # SCOTUS files are named by docket (e.g. "09-1159.mp3"); no reliable date
    # in the filename. Return None -> date left blank, which ingest tolerates.
    return None


# ------------------------- yt-dlp adapter -------------------------

class YtDlpAdapter:
    """Enumerate a YouTube channel/playlist and yield each video as an Item
    whose media_url is the YouTube watch URL. The core's downloader special-
    cases these (see normalize_and_write): yt-dlp extracts audio directly.

    Use for sources that only exist as video, e.g. the NARA-run archived
    Trump White House channel (youtube.com/whitehouse45). Requires yt-dlp.

    channel_url: a channel, /videos page, or playlist URL.
    """
    def __init__(self, channel_url, speaker, event, default_speaker=None):
        self.channel_url = channel_url
        self.speaker = speaker
        self.event = event

    def __iter__(self) -> Iterator[Item]:
        import shutil, json as _json
        if not shutil.which("yt-dlp"):
            print("  ! yt-dlp not on PATH (pip install yt-dlp)"); return
        # Flat playlist enumeration: one JSON line per video, no download.
        try:
            proc = subprocess.run(
                ["yt-dlp", "--flat-playlist", "-J", self.channel_url],
                capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ! yt-dlp enumerate failed: {e.stderr[:200]}"); return
        try:
            data = _json.loads(proc.stdout)
        except Exception as e:
            print(f"  ! yt-dlp JSON parse failed: {e}"); return
        entries = data.get("entries", []) or []
        if not entries:
            print(f"  ! 0 videos from {self.channel_url}")
            print(f"    Verify the URL resolves:")
            print(f"    yt-dlp --flat-playlist -J '{self.channel_url}' | head")
            print(f"    If it 404s, the channel handle is wrong — find the real")
            print(f"    channel/playlist URL and pass it to trump() / --url.")
            return
        for entry in entries:
            vid = entry.get("id")
            if not vid:
                continue
            title = entry.get("title", vid)
            # upload_date not always present in flat mode; leave blank if absent
            up = entry.get("upload_date") or ""
            date = f"{up[0:4]}-{up[4:6]}-{up[6:8]}" if len(up) == 8 else ""
            yield Item(
                media_url=f"https://www.youtube.com/watch?v={vid}",
                speaker=self.speaker, date=date, event=self.event,
                source_url=f"https://www.youtube.com/watch?v={vid}",
                stem=re.sub(r"[^a-z0-9]+", "_", f"{self.speaker}_{vid}".lower())[:60])


# ------------------------- source registry -------------------------

def obama():
    # Two-level crawl: index pages (?page=N) list article pages; each article
    # page carries the .mp3 link. Pagination confirmed as ?page=0..43.
    base = "https://obamawhitehouse.archives.gov"
    idx = [f"{base}/briefing-room/weekly-address"] + \
          [f"{base}/briefing-room/weekly-address?page={n}" for n in range(1, 44)]
    return GovArchiveAdapter(
        base, idx,
        media_re=r"https://obamawhitehouse\.archives\.gov/videos/\d{4}/[A-Za-z]+/\d{8}_Weekly_Address\.mp3",
        article_re=r"https://obamawhitehouse\.archives\.gov/the-press-office/\d{4}/\d{2}/\d{2}/weekly-address[^\"'\s)]*",
        speaker="Barack Obama", event="Weekly Address",
        date_from_url=_date_from_press_url)

def trump():
    # The NARA archived YouTube channel handle does not resolve. Reliable source:
    # C-SPAN captures of Trump remarks on archive.org — individual items, each a
    # downloadable file, via the SAME ia mechanism as scotus/nasa (no yt-dlp).
    # Combative/improvisational register (the "rage" end of the corpus).
    # CAVEAT: C-SPAN items carry a Title-17 copyright notice, unlike the pure-PD
    # Obama/SCOTUS material — fine for private use, not a clean-PD story.
    return IACollectionAdapter(
        query='collection:TV-CSPAN AND title:(Trump AND Remarks)',
        event="Trump Remarks (C-SPAN)", default_speaker="Donald Trump")

def scotus():
    # Free Law Project SCOTUS oral arguments live as FILES inside ONE IA item.
    # The identifier is misspelled on IA itself: "SCOTUSOralArugments".
    # Many voices (justices + advocates), formal/argumentative register.
    return IAItemFilesAdapter(identifier="SCOTUSOralArugments",
                              speaker="SCOTUS oral argument",
                              event="Supreme Court Oral Argument",
                              date_from_name=_scotus_date)

def nasa():
    # NASA audio as a true collection (separate items). CONFIRM the slug first:
    #   ia search 'collection:nasaaudiocollection' --itemlist | head
    # Control-room / procedural / institutional texture.
    return IACollectionAdapter(query='collection:nasaaudiocollection',
                               event="NASA Audio", default_speaker="NASA")

def nasa_reels():
    # Single item bundling many NASA highlight/soundbite mp3s — clean PD.
    return IAItemFilesAdapter(identifier="NasaAudioHighlightReels",
                              speaker="NASA", event="NASA Audio Highlight")

def apollo11():
    # Apollo 11 mission audio (air-to-ground). Single item, many files. PD.
    return IAItemFilesAdapter(identifier="Apollo11Audio",
                              speaker="Apollo 11", event="Apollo 11 Mission Audio")

def librivox():
    # LibriVox: public-domain audiobook readings. Thousands of items, huge
    # vocabulary, explicitly PD/CC0. This is the big word-count multiplier.
    # Each item is a separate audiobook; grab one audio file per item.
    # Narrated-prose register (calm to dramatic), many different readers.
    return IACollectionAdapter(query='collection:librivoxaudio',
                               speaker_field="creator",
                               event="LibriVox reading",
                               default_speaker="LibriVox reader")

def _speech_speaker_from_name(name):
    # "Inaugural Address - Franklin D. Roosevelt (1933).ogg" -> "Franklin D. Roosevelt"
    base = os.path.splitext(name)[0]
    m = re.search(r"-\s*([^-(]+?)\s*\(", base)
    return m.group(1).strip() if m else None

def _speech_date_from_name(name):
    # ".... (1933).ogg" -> "1933" (year only; ingest tolerates a bare year)
    m = re.search(r"\((\d{4})", name)
    return m.group(1) if m else None

def speeches():
    # "Greatest Speeches of the 20th Century" — single item, many mp3s,
    # labeled Public Domain. FDR, Churchill, etc. Dense in war/political/
    # institutional-gravitas vocabulary. Speaker+year parsed from filename.
    return IAItemFilesAdapter(identifier="Greatest_Speeches_of_the_20th_Century",
                              speaker="20th-century orator",
                              event="Greatest Speeches of the 20th Century",
                              speaker_from_name=_speech_speaker_from_name,
                              date_from_name=_speech_date_from_name)

def churchill():
    # Winston Churchill speeches/broadcasts — single item, 105 mp3s, PD.
    # Wartime register; the densest single source for the word "war".
    return IAItemFilesAdapter(identifier="Winston_Churchill",
                              speaker="Winston Churchill",
                              event="Churchill Speech/Broadcast")

SOURCES = {"obama": obama, "trump": trump, "scotus": scotus,
           "nasa": nasa, "nasa_reels": nasa_reels, "apollo11": apollo11,
           "librivox": librivox, "speeches": speeches, "churchill": churchill}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=list(SOURCES) + ["all"])
    ap.add_argument("--max", type=int, default=10,
                    help="max items per source")
    ap.add_argument("--url", default=None,
                    help="override channel/playlist URL for yt-dlp sources "
                         "(e.g. --source trump --url 'https://youtube.com/...')")
    args = ap.parse_args()
    done_paths = _load_manifest_paths()
    targets = list(SOURCES) if args.source == "all" else [args.source]
    for name in targets:
        print(f"\n=== {name} ===")
        adapter = SOURCES[name]()
        if args.url and isinstance(adapter, YtDlpAdapter):
            adapter.channel_url = args.url
        run(adapter, args.max, done_paths)
    print(f"\nManifest: {MANIFEST}  ->  mouthpiece ingest --batch {MANIFEST}")


if __name__ == "__main__":
    main()