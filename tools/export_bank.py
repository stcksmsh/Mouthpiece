"""Export a browser-sized sound bank from a local Mouthpiece index.

Run from the website branch with Mouthpiece's own venv (needs numpy, soundfile,
cmudict, and ffmpeg with libopus on PATH):

    ../Mouthpiece/.venv/bin/python tools/export_bank.py \
        --db ../Mouthpiece/db/mouthpiece.db --corpus ../Mouthpiece/corpus --out public/bank

Writes public/bank/index.json (what is where), chunk-NNN.ogg (Opus audio
sprites, ~45 s each, loaded on demand) and cmu.txt (CMUdict, first
pronunciation per word, stress digits kept). Clips are cut and loudness-
normalized the way mouthpiece/assemble.py does it; selection mirrors
mouthpiece/match.py (confidence floor, then the most typical-length instance).
"""

import argparse
import json
import re
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
MIN_CONF = 0.35          # match.py MIN_CONF
TARGET_RMS = 0.08        # assemble.py
PEAK_LIMIT = 0.95
WORD_PAD = 0.04          # assemble.py CLIP_PAD
SUB_PAD = 0.01           # assemble.py _cut_syllable_clip pad
GAP = 0.05               # silence between clips inside a sprite
CHUNK_S = 45.0
MAX_GAP = 0.5            # index.next_word_instance max_gap
N_WORDS = 3000
N_BIGRAMS = 500
N_TRIGRAMS = 150
PHONES_PER_KEY = 4
EXCLUDE_SOURCES = {"10000lieuessansvouloir_2102_librivox.wav"}  # French reading, transcribed as English
# Churchill's broadcasts are not cleared for publication; keep them out of the bank and the name out of cmu.txt.
EXCLUDE_PATTERN = re.compile("churchill", re.I)


def excluded(src):
    return src["filename"] in EXCLUDE_SOURCES or any(
        EXCLUDE_PATTERN.search(src[c] or "") for c in ("filename", "speaker", "event", "source_url"))


def label(src):
    """Short provenance label from the sources row (archive.org metadata)."""
    speaker, event = src["speaker"] or "", src["event"] or ""
    if event.startswith("LibriVox"):
        return f"LibriVox reading: {speaker}" if speaker else "LibriVox reading"
    if "@" in speaker or not speaker:
        return event or "unknown"
    if speaker.lower().startswith("scotus"):
        return event or speaker
    return f"{speaker}, {event}" if event else speaker


class Sprites:
    def __init__(self, out: Path):
        self.out = out
        self.chunks: list[list[np.ndarray]] = []
        self.cur: list[np.ndarray] = []
        self.cur_len = 0
        self.files: dict[str, sf.SoundFile] = {}

    def _open(self, path):
        if path not in self.files:
            self.files[path] = sf.SoundFile(path)
        return self.files[path]

    def cut(self, path, start_s, end_s, pad):
        f = self._open(path)
        a = max(0, int((start_s - pad) * SR))
        b = min(f.frames, int((end_s + pad) * SR))
        if b - a < int(0.02 * SR):
            return None, 0.0
        f.seek(a)
        clip = f.read(b - a, dtype="float32")
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms < 1e-4:
            return None, 0.0
        clip = clip * (TARGET_RMS / rms)
        peak = float(np.max(np.abs(clip)))
        if peak > PEAK_LIMIT:
            clip = clip * (PEAK_LIMIT / peak)
        return clip.astype(np.float32), start_s - a / SR  # actual pad before the core

    def add(self, clip):
        if self.cur_len / SR > CHUNK_S:
            self.flush()
        gap = np.zeros(int(GAP * SR), np.float32)
        self.cur.append(gap)
        self.cur_len += len(gap)
        off = self.cur_len
        self.cur.append(clip)
        self.cur_len += len(clip)
        return [len(self.chunks), round(off / SR, 4), round(len(clip) / SR, 4)]

    def flush(self):
        if self.cur:
            self.cur.append(np.zeros(int(GAP * SR), np.float32))
            self.chunks.append(self.cur)
        self.cur, self.cur_len = [], 0

    def write(self):
        self.flush()
        names = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, parts in enumerate(self.chunks):
                wav = Path(tmp) / f"{i}.wav"
                sf.write(wav, np.concatenate(parts), SR, subtype="PCM_16")
                name = f"chunk-{i:03d}.ogg"
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "libopus",
                     "-b:a", "20k", "-application", "voip",
                     # bitexact: fixed Ogg serials and no encoder tag, so a rebuild is byte-identical
                     "-fflags", "+bitexact", "-flags:a", "+bitexact", str(self.out / name)],
                    check=True,
                )
                names.append(name)
        return names


def typical(rows, dur_of):
    """Highest confidence among instances near the key's median duration."""
    durs = np.array([dur_of(r) for r in rows])
    med = float(np.median(durs))

    def score(r):
        d = dur_of(r)
        return (r["confidence"] or 0) - 0.6 * abs(d - med) / max(med, 1e-3)

    return sorted(rows, key=score, reverse=True), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("chunk-*.ogg"):
        old.unlink()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    sources = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM sources")}
    dropped = sorted(k for k, v in sources.items() if excluded(v))
    sources = {k: v for k, v in sources.items() if k not in dropped}
    print(f"excluded {len(dropped)} sources: {dropped}")
    path_of = {k: str(Path(args.corpus) / v["filename"]) for k, v in sources.items()}
    used_sources = set()
    sp = Sprites(out)

    words = [dict(r) for r in conn.execute(
        "SELECT id, source_id, word, start_s, end_s, COALESCE(confidence,0) AS confidence "
        "FROM word_instances ORDER BY source_id, start_s")]
    words = [w for w in words if w["source_id"] in sources]
    for w in words:
        w["word"] = w["word"].lower()
    good = [w for w in words if w["confidence"] >= MIN_CONF and re.fullmatch(r"[a-z']+", w["word"])
            and 0.05 <= w["end_s"] - w["start_s"] <= 1.6]

    # ---- phrases: consecutive words in one recording (find_phrase's rule) ----
    by_src = defaultdict(list)
    for w in words:
        by_src[w["source_id"]].append(w)
    grams = defaultdict(list)
    for seq in by_src.values():
        for i in range(len(seq)):
            run = [seq[i]]
            for j in range(i + 1, min(i + 3, len(seq))):
                a, b = run[-1], seq[j]
                if not (b["start_s"] >= a["end_s"] - 0.05 and b["start_s"] - a["end_s"] <= MAX_GAP):
                    break
                run.append(b)
                if all(x["confidence"] >= MIN_CONF and re.fullmatch(r"[a-z']+", x["word"]) for x in run):
                    grams[" ".join(x["word"] for x in run)].append(list(run))

    freq = Counter(w["word"] for w in good)
    top_words = [w for w, _ in freq.most_common(N_WORDS)]
    cand = defaultdict(list)
    for w in good:
        cand[w["word"]].append(w)

    bank = {"sampleRate": SR, "sources": {}, "words": {}, "phrases": {}, "sylls": {}, "phones": {}}

    def use(src_id):
        used_sources.add(src_id)
        return src_id

    # ---- whole words, most frequent first (so common words share early chunks) ----
    for word in top_words:
        ranked, _ = typical(cand[word], lambda r: r["end_s"] - r["start_s"])
        for r in ranked[:3]:
            clip, pad = sp.cut(path_of[r["source_id"]], r["start_s"], r["end_s"], WORD_PAD)
            if clip is None:
                continue
            core = r["end_s"] - r["start_s"]
            bank["words"][word] = sp.add(clip) + [use(r["source_id"]), round(pad, 4), round(core, 4)]
            break

    # ---- phrases ----
    def add_grams(n, limit):
        items = [(k, v) for k, v in grams.items() if len(k.split()) == n and len(v) >= 3]
        items.sort(key=lambda kv: len(kv[1]), reverse=True)
        for key, runs in items[:limit]:
            runs.sort(key=lambda run: min(x["confidence"] for x in run), reverse=True)
            run = runs[0]
            clip, pad = sp.cut(path_of[run[0]["source_id"]], run[0]["start_s"], run[-1]["end_s"], WORD_PAD)
            if clip is None:
                continue
            bounds = [round(x["start_s"] - run[0]["start_s"] + pad, 4) for x in run[1:]]
            bank["phrases"][key] = sp.add(clip) + [use(run[0]["source_id"]), bounds]

    add_grams(2, N_BIGRAMS)
    add_grams(3, N_TRIGRAMS)

    # ---- syllables and phones: every key, most frequent first ----
    def sub_units(table, key_col, per_key, label_col):
        rows = [dict(r) for r in conn.execute(
            f"SELECT u.{key_col} AS k, u.start_s, u.end_s, u.{label_col} AS phones, w.word, w.source_id, "
            f"COALESCE(w.confidence,0) AS confidence FROM {table} u JOIN word_instances w ON w.id = u.word_instance_id")]
        rows = [r for r in rows if r["source_id"] in sources and r["end_s"] > r["start_s"]]
        by_key = defaultdict(list)
        for r in rows:
            by_key[r["k"]].append(r)
        keys = sorted(by_key, key=lambda k: len(by_key[k]), reverse=True)
        res = {}
        for k in keys:
            ranked, _ = typical(by_key[k], lambda r: r["end_s"] - r["start_s"])
            picked, seen = [], set()
            for r in ranked:
                if len(picked) >= per_key:
                    break
                if r["source_id"] in seen and len(ranked) > per_key:
                    continue
                clip, pad = sp.cut(path_of[r["source_id"]], r["start_s"], r["end_s"], SUB_PAD)
                if clip is None:
                    continue
                seen.add(r["source_id"])
                picked.append(sp.add(clip) + [use(r["source_id"]), r["word"].lower(), r["phones"]])
            if picked:
                res[k] = picked
        return res

    bank["sylls"] = sub_units("syllable_instances", "phone_key", 1, "phones")
    bank["phones"] = sub_units("phone_instances", "phone_key", PHONES_PER_KEY, "phone")

    bank["chunks"] = sp.write()
    bank["sources"] = {
        str(k): {"label": label(v), "url": v["source_url"], "date": v["date"]}
        for k, v in sources.items() if k in used_sources
    }
    (out / "index.json").write_text(json.dumps(bank, separators=(",", ":")))

    import cmudict
    lines = []
    for w, prons in cmudict.dict().items():
        if re.fullmatch(r"[a-z][a-z']*", w) and not EXCLUDE_PATTERN.search(w):
            lines.append(w + " " + " ".join(prons[0]))
    lines.sort()
    (out / "cmu.txt").write_text("\n".join(lines) + "\n")

    print(f"words {len(bank['words'])}, phrases {len(bank['phrases'])}, syllables {len(bank['sylls'])}, "
          f"phones {len(bank['phones'])}, chunks {len(bank['chunks'])}, sources {len(bank['sources'])}")


if __name__ == "__main__":
    main()
