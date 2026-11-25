"""FastAPI server — local web UI for MOUTHPIECE."""

import io
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assemble, index, manifest, match, phonemes, reference

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
    row: dict | list[dict] | None,
    locked: bool = False,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> dict:
    ph = phonemes.phoneme_count(token)
    if row is None:
        return {"index": i, "token": token, "found": False, "locked": locked,
                "speed": speed, "pitch": pitch, "phoneme_count": ph}
    if isinstance(row, list):
        durations = [r["end_s"] - r["start_s"] for r in row]
        total_dur = sum(durations)
        f0_pairs  = [(r["f0_mean"], d) for r, d in zip(row, durations, strict=True) if r.get("f0_mean")]
        f0_hz     = (sum(f * d for f, d in f0_pairs) / sum(d for _, d in f0_pairs)) if f0_pairs else None
        return {
            "index":            i,
            "token":            token,
            "found":            True,
            "locked":           locked,
            "speed":            speed,
            "pitch":            pitch,
            "phoneme_count":    ph,
            "stitched":         True,
            "word_instance_id": None,
            "syllables": [
                {
                    "syllable_instance_id": r["id"],
                    "phones":               r["phones"] if "phones" in r else r["phone"],
                    "phone_key":            r["phone_key"],
                    "parent_word":          r["parent_word"],
                    "source_file":          r["filename"],
                    **index.attribution(r),
                    "date":                 r.get("date"),
                    "start_s":              round(r["start_s"], 4),
                    "end_s":                round(r["end_s"], 4),
                    "confidence":           r.get("confidence"),
                    "run_id":               r.get("run_id"),
                    "run_pos":              r.get("run_pos"),
                    "run_len":              r.get("run_len"),
                }
                for r in row
            ],
            "word":             token,
            "source_file":      None,
            **index.stitched_attribution(row),
            "date":             row[0].get("date"),
            "source_url":       row[0].get("source_url"),
            "confidence":       min((r.get("confidence") or 0.0) for r in row),
            "duration_s":       round(total_dur, 3),
            "f0_hz":            round(f0_hz, 1) if f0_hz else None,
            "f0_var":           None,
            "spectral_tilt":    None,
            "lemma":            None,
        }
    return {
        "index":            i,
        "token":            token,
        "found":            True,
        "locked":           locked,
        "speed":            speed,
        "pitch":            pitch,
        "phoneme_count":    ph,
        "word_instance_id": row["id"],
        "word":             row["word"],
        "source_file":      row["filename"],
        **index.attribution(row),
        "date":             row["date"],
        "source_url":       row["source_url"],
        "start_s":          round(row["start_s"], 4),
        "end_s":            round(row["end_s"], 4),
        "confidence":       row["confidence"],
        "duration_s":       round(row["end_s"] - row["start_s"], 3),
        "f0_hz":            round(row["f0_mean"], 1) if row.get("f0_mean") else None,
        "f0_var":           round(row["f0_var"], 1) if row.get("f0_var") else None,
        "spectral_tilt":    round(row["spectral_tilt"], 3) if row.get("spectral_tilt") is not None else None,
        "lemma":            row.get("lemma"),
        "phrase_id":        row.get("phrase_id"),
        "phrase_pos":       row.get("phrase_pos"),
        "phrase_len":       row.get("phrase_len"),
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
    audio, words = assemble.assemble(render["matches"], p["corpus"], adjustments=adj)
    assemble.write_wav(audio, wav)
    render["manifest_words"] = words
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
        "criteria":   render.get("criteria", {"granularity": 0.0, "diction": 0.0, "emotion": 0.0}),
    }

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    text: str
    seed: int | None = None
    granularity: float = 0.0  # -1 (syllable-per-syllable) .. 0 .. +1 (long uniform phrases)
    diction: float = 0.0      # -1 (mumbled) .. 0 .. +1 (crisp)
    emotion: float = 0.0      # -1 (flat) .. 0 .. +1 (expressive)

class SwapRequest(BaseModel):
    word_instance_id: int

class SwapSyllableRequest(BaseModel):
    syllable_instance_id: int

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

class AutoPitchRequest(BaseModel):
    mean_f0_hz: float          # target center pitch for the whole render, in Hz
    pitch_range_st: float = 6.0  # how much of the render's own intonation contour to preserve, in semitones
    speaking_rate: float = 1.0   # multiplier applied to per-word speed
    apply_pitch: bool = True
    apply_speed: bool = False

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

def _clamp1(x: float) -> float:
    return max(-1.0, min(1.0, x))


@app.post("/api/render")
def api_render(req: RenderRequest):
    conn = _conn()
    seed = req.seed if req.seed is not None else int(np.random.default_rng().integers(2**31))
    rng  = np.random.default_rng(seed)
    criteria = {
        "granularity": _clamp1(req.granularity),
        "diction":     _clamp1(req.diction),
        "emotion":     _clamp1(req.emotion),
    }
    matches = match.match_text(req.text, conn, rng=rng, **criteria)
    conn.close()

    render = {
        "id":         str(uuid.uuid4()),
        "text":       req.text,
        "seed":       seed,
        "criteria":   criteria,
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
    _break_phrase(render, word_index)
    _count_swap(render, word_index)
    render["matches"][word_index] = (token, row)
    render["words"][word_index]   = _row_to_word(
        word_index, token, row,
        locked=old_w.get("locked", False),
        speed=old_w.get("speed", 1.0),
        pitch=old_w.get("pitch", 0.0),
    )
    _assemble(render)
    return _render_response(render)


def _break_phrase(render: dict, idx: int) -> None:
    """Drop phrase grouping around idx so assembly cuts each member by its own clip
    (assemble treats a phrase run as one cut and would ignore a swapped member)."""
    row = render["matches"][idx][1]
    if not (isinstance(row, dict) and row.get("phrase_id") is not None):
        return
    start = idx - row["phrase_pos"]
    for k in range(start, start + row["phrase_len"]):
        tok, member = render["matches"][k]
        render["matches"][k] = (tok, {**member, "phrase_id": None, "phrase_pos": None, "phrase_len": None})
        render["words"][k].update(phrase_id=None, phrase_pos=None, phrase_len=None)


def _count_swap(render: dict, idx: int) -> None:
    swaps = render.setdefault("swaps", {})
    swaps[idx] = swaps.get(idx, 0) + 1


def _replace_match(render: dict, idx: int, token: str, row) -> None:
    old_w = render["words"][idx]
    render["matches"][idx] = (token, row)
    render["words"][idx]   = _row_to_word(
        idx, token, row,
        locked=old_w.get("locked", False),
        speed=old_w.get("speed", 1.0),
        pitch=old_w.get("pitch", 0.0),
    )


@app.post("/api/render/{render_id}/reroll/{word_index}")
def api_reroll(render_id: str, word_index: int):
    render = _get(render_id)
    if not (0 <= word_index < len(render["matches"])):
        raise HTTPException(400, "word_index out of range")
    token       = render["matches"][word_index][0]
    current_row = render["matches"][word_index][1]
    old_w       = render["words"][word_index]
    # Per (render, word) counter so the same seed and reroll sequence replays identically;
    # it only advances on success (below), so a 400 leaves the sequence unchanged.
    count       = render.setdefault("rerolls", {}).get(word_index, 0)
    rng         = np.random.default_rng([render["seed"], word_index, count])
    conn        = _conn()
    criteria    = render.get("criteria", {})
    diction     = criteria.get("diction", 0.0)
    emotion     = criteria.get("emotion", 0.0)
    granularity = criteria.get("granularity", 0.0)

    if isinstance(current_row, list):
        # Currently stitched — re-pick per-syllable rather than flipping to whole-word.
        row = match.stitch_syllables(conn, token, rng=rng, diction=diction, emotion=emotion,
                                      granularity=granularity)
        conn.close()
        if row is None:
            raise HTTPException(400, f"Could not re-stitch syllables for '{token}'")
        _replace_match(render, word_index, token, row)

    elif isinstance(current_row, dict) and current_row.get("phrase_id") is not None:
        # Currently part of a multi-word phrase — try to re-find a run of the
        # same length rather than silently breaking it into loose words.
        phrase_pos = current_row["phrase_pos"]
        phrase_len = current_row["phrase_len"]
        start_idx  = word_index - phrase_pos
        tokens_span = [render["matches"][k][0] for k in range(start_idx, start_idx + phrase_len)]
        run = match.find_phrase(conn, tokens_span, 0, rng=rng)

        if run and len(run) == phrase_len:
            pid = int(rng.integers(1, 2**31))
            for pos, r in enumerate(run):
                r["phrase_id"], r["phrase_pos"], r["phrase_len"] = pid, pos, phrase_len
            for pos in range(phrase_len):
                idx = start_idx + pos
                _replace_match(render, idx, render["matches"][idx][0], run[pos])
        else:
            # No run of the same length found — fall back to independent
            # whole-word/syllable picks for each former phrase member.
            for pos in range(phrase_len):
                idx = start_idx + pos
                tok = render["matches"][idx][0]
                new_row = match.find_clip(conn, tok, rng=rng, diction=diction, emotion=emotion)
                _replace_match(render, idx, tok, new_row)
        conn.close()

    else:
        current_id = old_w.get("word_instance_id")
        rows = index.search_word(conn, token, limit=20)
        conn.close()
        if not rows:
            raise HTTPException(400, f"No candidate clips for '{token}'")
        alts = [r for r in rows if r["id"] != current_id] or rows
        weights = np.array([(r["confidence"] or 0.5) for r in alts], dtype=np.float64)
        if diction:
            weights = weights * np.array(
                [match._bias_weight(r["confidence"] if r.get("confidence") is not None else 0.5, diction)
                 for r in alts]
            )
        if emotion:
            f0_rank  = match._rank_normalize(alts, "f0_var")
            rms_rank = match._rank_normalize(alts, "rms")
            weights = weights * np.array(
                [match._bias_weight((f + r) / 2.0, emotion) for f, r in zip(f0_rank, rms_rank, strict=True)]
            )
        weights /= weights.sum()
        row = alts[int(rng.choice(len(alts), p=weights))]
        _replace_match(render, word_index, token, row)

    render["rerolls"][word_index] = count + 1
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

    # A phrase is one physical audio cut — every member must share the same
    # speed/pitch or assemble.py's single cut can't honor them independently.
    phrase_id = render["words"][word_index].get("phrase_id")
    if phrase_id is not None:
        for w in render["words"]:
            if w.get("phrase_id") == phrase_id:
                w["speed"] = speed
                w["pitch"] = pitch

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


@app.post("/api/render/{render_id}/auto_pitch")
def api_auto_pitch(render_id: str, req: AutoPitchRequest):
    """Shift every word's pitch toward a target center Hz, preserving a scaled
    version of the render's own natural intonation contour."""
    render = _get(render_id)
    words = render["words"]
    f0s = [w["f0_hz"] for w in words if w.get("found") and w.get("f0_hz")]
    if not f0s:
        raise HTTPException(400, "No words with pitch data in this render")
    corpus_median = float(np.median(f0s))
    scale = req.pitch_range_st / 12.0

    for w in words:
        if not w.get("found"):
            continue
        if req.apply_speed:
            w["speed"] = max(0.25, min(4.0, req.speaking_rate))
        if req.apply_pitch and w.get("f0_hz"):
            word_f0 = w["f0_hz"]
            dev_st = 12.0 * np.log2(word_f0 / corpus_median)
            shift_st = 12.0 * np.log2(req.mean_f0_hz / word_f0) + dev_st * scale
            w["pitch"] = float(max(-12.0, min(12.0, shift_st)))

    _assemble(render)
    return _render_response(render)


@app.post("/api/render/{render_id}/reference_match")
async def api_reference_match(
    render_id: str,
    file: Annotated[UploadFile, File()],
    apply_pitch: Annotated[bool, Form()] = True,
    apply_speed: Annotated[bool, Form()] = True,
):
    """Analyze an uploaded reference recording and push the render's per-word
    pitch/speed toward what the reference actually did at aligned words."""
    render = _get(render_id)
    data = await file.read()
    p = _paths()
    ref_dir = p["renders"] / render_id
    ref_path = reference.save_reference_temp(data, ref_dir, name="reference")
    ref_words = reference.analyze_reference_audio(ref_path)
    adjustments = reference.compute_pitch_adjustments(render["words"], ref_words)
    if not adjustments:
        raise HTTPException(400, "Could not align any words between render and reference")

    for adj in adjustments:
        w = render["words"][adj["word_index"]]
        if apply_pitch:
            w["pitch"] = max(-24.0, min(24.0, adj["pitch_st"]))
        if apply_speed:
            w["speed"] = max(0.25, min(4.0, adj["speed"]))

    _assemble(render)
    return _render_response(render)


@app.get("/api/render/{render_id}/manifest")
def api_manifest(render_id: str):
    render = _get(render_id)
    swaps, rerolls = render.get("swaps", {}), render.get("rerolls", {})
    state = [
        {"nth": None, "locked": w.get("locked", False),
         "swaps": swaps.get(i, 0), "rerolls": rerolls.get(i, 0)}
        for i, w in enumerate(render["words"])
    ]
    conn = _conn()
    corpus_hash = manifest.corpus_hash(conn)
    conn.close()
    return manifest.build_manifest(
        render["text"], render["manifest_words"], Path(render["wav_path"]),
        seed=render["seed"], controls={**render["criteria"], "nth": {}},
        corpus_hash=corpus_hash, token_state=state,
    )


@app.get("/api/render/{render_id}/reaper")
def api_reaper(render_id: str):
    render = _get(render_id)
    wav    = Path(render["wav_path"])
    if not wav.exists():
        raise HTTPException(404, "WAV file missing — re-render first")
    script = _make_reaper_script(render, wav)
    safe   = render["text"][:40].replace(" ", "_").replace("/", "").replace("\\", "")
    return Response(
        content=script,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="mouthpiece_{safe}.lua"'},
    )


def _make_reaper_script(render: dict, wav: Path) -> str:
    words  = render["words"]
    times  = assemble.compute_output_times(words)
    text   = render["text"]
    rid    = render["id"]

    # collect unique speakers → stable colour index
    speakers = list(dict.fromkeys(w["speaker_label"] for w in words if w.get("found")))

    # Reaper native colour: 0x01RRGGBB — a small muted palette
    _PALETTE = [
        0x014e9cde, 0x01de9c4e, 0x014ede9c, 0x019c4ede,
        0x01de4e9c, 0x019cde4e, 0x014e4ede, 0x01de4e4e,
    ]
    def _color(speaker: str) -> int:
        idx = speakers.index(speaker) if speaker in speakers else 0
        return _PALETTE[idx % len(_PALETTE)]

    # build Lua word table
    rows = []
    for w, t in zip(words, times):
        token   = w["token"].replace('"', '\\"')
        found   = "true" if w.get("found") else "false"
        label   = w.get("speaker_label", "")
        speaker = label.replace('"', '\\"')
        color   = _color(label) if w.get("found") else 0
        rows.append(
            f'  {{start={t["start"]}, stop={t["end"]}, '
            f'token="{token}", speaker="{speaker}", found={found}, color={color}}}'
        )
    word_table = "{\n" + ",\n".join(rows) + "\n}"

    return f"""\
-- MOUTHPIECE — Reaper import script
-- Text:   {text}
-- ID:     {rid}
--
-- Load in Reaper: Actions → ReaScript: Load / run script…

local WAV   = {str(wav)!r}
local WORDS = {word_table}

-- ── helpers ────────────────────────────────────────────────────────────────

local function insert_wav_item(track, path)
  local item = reaper.AddMediaItemToTrack(track)
  local take = reaper.AddTakeToMediaItem(item)
  local src  = reaper.PCM_Source_CreateFromFile(path)
  if not src then
    reaper.ShowMessageBox("Could not load:\\n" .. path, "MOUTHPIECE", 0)
    return nil
  end
  local len = reaper.GetMediaSourceLength(src)
  reaper.SetMediaItemTake_Source(take, src)
  reaper.SetMediaItemLength(item, len, false)
  reaper.SetMediaItemPosition(item, 0, false)
  reaper.SetMediaItemTakeInfo_Value(take, "D_VOL", 1.0)
  return item
end

-- ── main ───────────────────────────────────────────────────────────────────

reaper.Undo_BeginBlock()
reaper.PreventUIRefresh(1)

local tidx  = reaper.CountTracks(0)
reaper.InsertTrackAtIndex(tidx, true)
local track = reaper.GetTrack(0, tidx)
reaper.GetSetMediaTrackInfo_String(track, "P_NAME", "MOUTHPIECE", true)

insert_wav_item(track, WAV)

-- word regions (shown in the timeline ruler)
for _, w in ipairs(WORDS) do
  local label
  if w.found then
    label = w.token .. "  [" .. w.speaker .. "]"
  else
    label = "— " .. w.token
  end
  reaper.AddProjectMarker2(0, true, w.start, w.stop, label, -1, w.color)
end

reaper.PreventUIRefresh(-1)
reaper.Undo_EndBlock("MOUTHPIECE: import " .. #WORDS .. " words", -1)
reaper.UpdateArrange()

reaper.ShowMessageBox(
  "Imported " .. #WORDS .. " words onto track " .. (tidx + 1) .. ".\\n\\n" ..
  "Word regions added to ruler.\\n" ..
  "WAV: " .. WAV,
  "MOUTHPIECE", 0
)
"""


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
            "speaker_label": index.speaker_label(r),
            "date":        r["date"],
            "source_file": r["filename"],
            "confidence":  r["confidence"],
            "duration_s":  round(r["end_s"] - r["start_s"], 3),
            "f0_hz":       round(r["f0_mean"], 1) if r.get("f0_mean") else None,
            "rms":         round(r["rms"], 4) if r.get("rms") else None,
            "f0_var":      round(r["f0_var"], 1) if r.get("f0_var") else None,
            "spectral_tilt": round(r["spectral_tilt"], 3) if r.get("spectral_tilt") is not None else None,
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


@app.get("/api/syllable/{instance_id}/preview")
def api_syllable_preview(instance_id: int):
    conn = _conn()
    row = index.get_syllable_instance(conn, instance_id)
    conn.close()
    if row is None:
        raise HTTPException(404, "Syllable instance not found")
    return Response(content=_clip_bytes(row), media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/syllable_candidates/{word}")
def api_syllable_candidates(word: str):
    conn = _conn()
    syllables = phonemes.syllabify(word.lower().strip())
    if syllables is None:
        conn.close()
        return {"syllables": []}
    out = []
    for idx, group in enumerate(syllables):
        key  = phonemes.phone_key(group)
        rows = index.search_syllable(conn, key, limit=20)
        out.append({
            "index":     idx,
            "phones":    " ".join(group),
            "phone_key": key,
            "candidates": [
                {
                    "id":          r["id"],
                    "parent_word": r["parent_word"],
                    "speaker":     r["speaker"],
                    "speaker_label": index.speaker_label(r),
                    "date":        r["date"],
                    "source_file": r["filename"],
                    "confidence":  r["confidence"],
                    "duration_s":  round(r["end_s"] - r["start_s"], 3),
                    "f0_hz":       round(r["f0_mean"], 1) if r.get("f0_mean") else None,
                }
                for r in rows
            ],
        })
    conn.close()
    return {"syllables": out}


@app.post("/api/render/{render_id}/swap_syllable/{word_index}/{syll_index}")
def api_swap_syllable(render_id: str, word_index: int, syll_index: int, req: SwapSyllableRequest):
    render = _get(render_id)
    if not (0 <= word_index < len(render["matches"])):
        raise HTTPException(400, "word_index out of range")
    token, current = render["matches"][word_index]
    if not isinstance(current, list):
        raise HTTPException(400, "word is not syllable-stitched")
    if not (0 <= syll_index < len(current)):
        raise HTTPException(400, "syll_index out of range")

    conn = _conn()
    row = index.get_syllable_instance(conn, req.syllable_instance_id)
    conn.close()
    if row is None:
        raise HTTPException(404, f"syllable_instance {req.syllable_instance_id} not found")
    if row["phone_key"] != current[syll_index]["phone_key"]:
        raise HTTPException(400, "syllable_instance does not match this syllable's phone_key")

    new_syllables = list(current)

    # Swapping one member of a natural multi-syllable run breaks its real
    # continuity — dissolve the whole run back into independent picks first,
    # so assemble.py never mistakes a partially-replaced run for still-one
    # continuous span of corpus audio.
    run_id = new_syllables[syll_index].get("run_id")
    if run_id is not None:
        for i, r in enumerate(new_syllables):
            if r.get("run_id") == run_id:
                r = dict(r)
                r["run_id"] = r["run_pos"] = r["run_len"] = None
                new_syllables[i] = r

    new_syllables[syll_index] = row
    old_w = render["words"][word_index]
    _count_swap(render, word_index)
    render["matches"][word_index] = (token, new_syllables)
    render["words"][word_index]   = _row_to_word(
        word_index, token, new_syllables,
        locked=old_w.get("locked", False),
        speed=old_w.get("speed", 1.0),
        pitch=old_w.get("pitch", 0.0),
    )
    _assemble(render)
    return _render_response(render)


# Static files — must be last
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
