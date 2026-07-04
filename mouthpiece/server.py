"""FastAPI server — local web UI for MOUTHPIECE."""

import io
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assemble, index, match

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MOUTHPIECE")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _paths() -> dict[str, Path]:
    return {
        "db":      Path(os.environ.get("MOUTHPIECE_DB",      "db/mouthpiece.db")),
        "corpus":  Path(os.environ.get("MOUTHPIECE_CORPUS",  "corpus")),
        "renders": Path(os.environ.get("MOUTHPIECE_RENDERS", "renders")),
    }

def _conn():
    return index.get_connection(_paths()["db"])

# ---------------------------------------------------------------------------
# In-memory render store (last 20; lost on restart)
# ---------------------------------------------------------------------------

MAX_RENDERS = 20
_renders: OrderedDict[str, dict] = OrderedDict()

def _store(render: dict) -> None:
    _renders[render["id"]] = render
    if len(_renders) > MAX_RENDERS:
        _renders.popitem(last=False)

def _get(render_id: str) -> dict:
    r = _renders.get(render_id)
    if r is None:
        raise HTTPException(404, detail="Render not found (server may have restarted — re-render)")
    return r

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_word(
    i: int,
    token: str,
    row: dict | None,
    locked: bool = False,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> dict:
    if row is None:
        return {"index": i, "token": token, "found": False, "locked": locked,
                "speed": speed, "pitch": pitch}
    return {
        "index":            i,
        "token":            token,
        "found":            True,
        "locked":           locked,
        "speed":            speed,
        "pitch":            pitch,
        "word_instance_id": row["id"],
        "word":             row["word"],
        "source_file":      row["filename"],
        "speaker":          row["speaker"],
        "date":             row["date"],
        "source_url":       row["source_url"],
        "start_s":          round(row["start_s"], 4),
        "end_s":            round(row["end_s"], 4),
        "confidence":       row["confidence"],
        "duration_s":       round(row["end_s"] - row["start_s"], 3),
    }


def _build_words(matches: list, old_words: list | None = None) -> list[dict]:
    locked_map = {w["index"]: w.get("locked", False) for w in (old_words or [])}
    speed_map  = {w["index"]: w.get("speed",  1.0)   for w in (old_words or [])}
    pitch_map  = {w["index"]: w.get("pitch",  0.0)   for w in (old_words or [])}
    return [
        _row_to_word(i, tok, row,
                     locked=locked_map.get(i, False),
                     speed=speed_map.get(i, 1.0),
                     pitch=pitch_map.get(i, 0.0))
        for i, (tok, row) in enumerate(matches)
    ]


def _adj_from_words(words: list[dict]) -> dict[int, dict]:
    adj = {}
    for w in words:
        s = w.get("speed", 1.0)
        p = w.get("pitch", 0.0)
        if abs(s - 1.0) >= 0.02 or abs(p) >= 0.05:
            adj[w["index"]] = {"speed": s, "pitch": p}
    return adj


def _assemble(render: dict) -> None:
    p = _paths()
    p["renders"].mkdir(parents=True, exist_ok=True)
    wav  = p["renders"] / f"{render['id']}.wav"
    adj  = _adj_from_words(render["words"])
    audio, _ = assemble.assemble(render["matches"], p["corpus"], adjustments=adj)
    assemble.write_wav(audio, wav)
    render["wav_path"]   = str(wav)
    render["duration_s"] = round(len(audio) / assemble.SAMPLE_RATE, 2)


def _clip_bytes(row: dict) -> bytes:
    p    = _paths()
    src  = p["corpus"] / row["filename"]
    clip = assemble._cut_clip(src, row["start_s"], row["end_s"])
    buf  = io.BytesIO()
    sf.write(buf, clip, assemble.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _render_response(render: dict) -> dict:
    return {
        "id":         render["id"],
        "seed":       render["seed"],
        "duration_s": render["duration_s"],
        "words":      render["words"],
    }

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    text: str
    seed: int | None = None

class SwapRequest(BaseModel):
    word_instance_id: int

class LockRequest(BaseModel):
    locked: bool

class AdjustRequest(BaseModel):
    speed: float = 1.0   # >1 = faster, <1 = slower (0.25–4.0)
    pitch: float = 0.0   # semitones (-12 to +12)

class BatchAdjustItem(BaseModel):
    word_index: int
    speed: float = 1.0
    pitch: float = 0.0

class BatchAdjustRequest(BaseModel):
    adjustments: list[BatchAdjustItem]

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/render")
def api_render(req: RenderRequest):
    conn = _conn()
    seed = req.seed if req.seed is not None else int(np.random.default_rng().integers(2**31))
    rng  = np.random.default_rng(seed)
    matches = match.match_text(req.text, conn, rng=rng)
    conn.close()

    render = {
        "id":         str(uuid.uuid4()),
        "text":       req.text,
        "seed":       seed,
        "matches":    matches,
        "words":      [],
        "wav_path":   None,
        "duration_s": 0.0,
    }
    render["words"] = _build_words(matches)
    _assemble(render)
    _store(render)
    return _render_response(render)


@app.get("/api/render/{render_id}/audio")
def api_audio(render_id: str):
    render = _get(render_id)
    wav = Path(render["wav_path"])
    if not wav.exists():
        raise HTTPException(404, "WAV file missing")
    return FileResponse(str(wav), media_type="audio/wav",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/render/{render_id}/download")
def api_download(render_id: str):
    render = _get(render_id)
    wav = Path(render["wav_path"])
    if not wav.exists():
        raise HTTPException(404, "WAV file missing")
    safe = render["text"][:40].replace(" ", "_").replace("/", "").replace("\\", "")
    return FileResponse(str(wav), media_type="audio/wav",
                        filename=f"mouthpiece_{safe}.wav")


@app.post("/api/render/{render_id}/swap/{word_index}")
def api_swap(render_id: str, word_index: int, req: SwapRequest):
    render = _get(render_id)
    if not (0 <= word_index < len(render["matches"])):
        raise HTTPException(400, "word_index out of range")
    conn = _conn()
    row = index.get_word_instance(conn, req.word_instance_id)
    conn.close()
    if row is None:
        raise HTTPException(404, f"word_instance {req.word_instance_id} not found")
    token  = render["matches"][word_index][0]
    old_w  = render["words"][word_index]
    render["matches"][word_index] = (token, row)
    render["words"][word_index]   = _row_to_word(
        word_index, token, row,
        locked=old_w.get("locked", False),
        speed=old_w.get("speed", 1.0),
        pitch=old_w.get("pitch", 0.0),
    )
    _assemble(render)
    return _render_response(render)


@app.post("/api/render/{render_id}/reroll/{word_index}")
def api_reroll(render_id: str, word_index: int):
    render = _get(render_id)
    if not (0 <= word_index < len(render["matches"])):
        raise HTTPException(400, "word_index out of range")
    token      = render["matches"][word_index][0]
    current_id = render["words"][word_index].get("word_instance_id")
    conn  = _conn()
    rows  = index.search_word(conn, token, limit=20)
    conn.close()
    alts = [r for r in rows if r["id"] != current_id] or rows
    rng  = np.random.default_rng()
    weights = np.array([(r["confidence"] or 0.5) for r in alts], dtype=np.float64)
    weights /= weights.sum()
    row    = alts[int(rng.choice(len(alts), p=weights))]
    old_w  = render["words"][word_index]
    render["matches"][word_index] = (token, row)
    render["words"][word_index]   = _row_to_word(
        word_index, token, row,
        locked=old_w.get("locked", False),
        speed=old_w.get("speed", 1.0),
        pitch=old_w.get("pitch", 0.0),
    )
    _assemble(render)
    return _render_response(render)


@app.post("/api/render/{render_id}/lock/{word_index}")
def api_lock(render_id: str, word_index: int, req: LockRequest):
    render = _get(render_id)
    if not (0 <= word_index < len(render["words"])):
        raise HTTPException(400, "word_index out of range")
    render["words"][word_index]["locked"] = req.locked
    return {"ok": True, "locked": req.locked}


@app.post("/api/render/{render_id}/adjust/{word_index}")
def api_adjust(render_id: str, word_index: int, req: AdjustRequest):
    render = _get(render_id)
    if not (0 <= word_index < len(render["words"])):
        raise HTTPException(400, "word_index out of range")
    speed = max(0.25, min(4.0, req.speed))
    pitch = max(-24.0, min(24.0, req.pitch))
    render["words"][word_index]["speed"] = speed
    render["words"][word_index]["pitch"] = pitch
    _assemble(render)
    return _render_response(render)


@app.post("/api/render/{render_id}/adjust_all")
def api_adjust_all(render_id: str, req: AdjustRequest):
    """Apply global speed/pitch multiplier to all words (replaces existing per-word values)."""
    render = _get(render_id)
    speed = max(0.25, min(4.0, req.speed))
    pitch = max(-24.0, min(24.0, req.pitch))
    for w in render["words"]:
        w["speed"] = speed
        w["pitch"] = pitch
    _assemble(render)
    return _render_response(render)


@app.post("/api/render/{render_id}/adjust_batch")
def api_adjust_batch(render_id: str, req: BatchAdjustRequest):
    render = _get(render_id)
    for item in req.adjustments:
        i = item.word_index
        if 0 <= i < len(render["words"]):
            render["words"][i]["speed"] = max(0.25, min(4.0, item.speed))
            render["words"][i]["pitch"] = max(-24.0, min(24.0, item.pitch))
    _assemble(render)
    return _render_response(render)


@app.get("/api/candidates/{word}")
def api_candidates(word: str):
    conn = _conn()
    rows = index.search_word(conn, word.lower().strip(), limit=20)
    conn.close()
    return [
        {
            "id":          r["id"],
            "word":        r["word"],
            "speaker":     r["speaker"],
            "date":        r["date"],
            "source_file": r["filename"],
            "confidence":  r["confidence"],
            "duration_s":  round(r["end_s"] - r["start_s"], 3),
        }
        for r in rows
    ]


@app.get("/api/clip/{instance_id}/preview")
def api_preview(instance_id: int):
    conn = _conn()
    row = index.get_word_instance(conn, instance_id)
    conn.close()
    if row is None:
        raise HTTPException(404, "Word instance not found")
    return Response(content=_clip_bytes(row), media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=3600"})


# Static files — must be last
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
