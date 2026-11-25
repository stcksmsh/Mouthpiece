"""Per-render provenance manifest (schema v2) and the inputs to replay it.

A manifest is written next to each render as `<stem>.manifest.json`. v2 records
everything needed to rebuild the same audio without running selection again.

Top-level fields:
  schema_version  2. Manifests without this field are v1 and cannot be replayed.
  engine_version  mouthpiece.__version__ of the engine that rendered it.
  corpus_hash     sha256 of the corpus rows selection and audio depend on (see corpus_hash()).
  rendered_at     UTC ISO-8601 time the manifest was written.
  seed            RNG seed the render was selected with.
  text            Input text.
  controls        Render controls: granularity, diction, emotion, nth ({word: rank} overrides).
  output_wav      Path of the WAV this manifest describes.
  words           One entry per assembled unit (a word, a gap, or a multi-word phrase).

Fields added to each `words` entry in v2 (the v1 per-word fields are kept as they were):
  nth        --nth rank used for this word, or null.
  locked     Whether the word is locked in the UI (any member, for a phrase).
  swaps      Times the word (or one of its syllables) was swapped by hand (summed over a phrase).
  rerolls    Times the word was rerolled (summed over a phrase).
  speed      Time-stretch factor, >1 is faster.
  pitch      Pitch shift in semitones.
  gap        Seconds of silence inserted before this unit.
  fragments  Clip instances in playback order, each {tier, instance_id}; tier is
             phrase, word, syllable or phone, and instance_id is the row id in the
             matching *_instances table. Syllable and phone fragments also carry
             run_id (consecutive fragments sharing a run_id are one continuous cut).
             A gap has no fragments; its silence length is its v1 field gap_s.
"""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from . import __version__, index

SCHEMA_VERSION = 2

# Columns hashed per table, ordered by id. Attribution and ingested_at are left out:
# they change neither which clip is picked nor how it sounds.
_HASHED = (
    ("sources", ("id", "filename")),
    ("word_instances", ("id", "source_id", "word", "lemma", "start_s", "end_s",
                        "confidence", "f0_mean", "f0_var", "rms", "spectral_tilt")),
    ("syllable_instances", ("id", "word_instance_id", "syll_index", "syll_count",
                            "phones", "phone_key", "start_s", "end_s")),
    ("phone_instances", ("id", "word_instance_id", "syll_index", "phone_index",
                         "phone_count", "phone", "phone_key", "start_s", "end_s")),
)

_TIER_TABLE = {"phrase": "word", "word": "word", "syllable": "syllable", "phone": "phone"}


def corpus_hash(conn: sqlite3.Connection) -> str:
    h = hashlib.sha256()
    for table, cols in _HASHED:
        h.update(f"{table}\n".encode())
        for row in conn.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY id"):
            h.update(json.dumps(list(row)).encode() + b"\n")
    return h.hexdigest()


def build_manifest(
    text: str,
    words: list[dict],
    output_wav: Path,
    *,
    seed: int | None,
    controls: dict,
    corpus_hash: str,
    token_state: list[dict],
) -> dict:
    """words: assemble()'s per-unit list. token_state: one {nth, locked, swaps, rerolls}
    per input token; phrase units fold their members' state together."""
    out_words = []
    t = 0
    for w in words:
        members = token_state[t:t + len(w.get("tokens") or [w["token"]])]
        t += len(members)
        out_words.append({
            **w,
            "nth":     members[0].get("nth"),
            "locked":  any(m.get("locked", False) for m in members),
            "swaps":   sum(m.get("swaps", 0) for m in members),
            "rerolls": sum(m.get("rerolls", 0) for m in members),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": __version__,
        "corpus_hash":    corpus_hash,
        "rendered_at":    datetime.now(UTC).isoformat(),
        "seed":           seed,
        "text":           text,
        "controls":       controls,
        "output_wav":     str(output_wav),
        "words":          out_words,
    }


def write_manifest(manifest: dict, output_wav: Path) -> Path:
    manifest_path = output_wav.with_name(output_wav.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


class ReplayError(Exception):
    pass


def replay_inputs(
    manifest: dict, conn: sqlite3.Connection,
) -> tuple[list[tuple[str, dict | list[dict] | None]], dict[int, dict], list[dict]]:
    """Rebuild assemble()'s inputs from the recorded instance ids, never from selection.
    Returns (matches, adjustments, token_state). Raises ReplayError for a manifest that
    cannot be replayed or names instances no longer in the DB."""
    if not isinstance(manifest, dict):
        raise ReplayError("not a manifest: expected a JSON object")
    if "schema_version" not in manifest:
        raise ReplayError("v1 manifest cannot be replayed: it does not record clip instances")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ReplayError(f"unsupported manifest schema_version {manifest['schema_version']!r}")

    matches: list = []
    adjustments: dict[int, dict] = {}
    token_state: list[dict] = []
    missing: list[str] = []

    for u, w in enumerate(manifest["words"]):
        i = len(matches)
        adjustments[i] = {"speed": w["speed"], "pitch": w["pitch"], "gap": w["gap"]}
        tokens = w.get("tokens") or [w["token"]]
        token_state.append({k: w[k] for k in ("nth", "locked", "swaps", "rerolls")})
        token_state += [{} for _ in tokens[1:]]
        if not w["found"]:
            adjustments[i]["gap_s"] = w["gap_s"]
        rows = _fetch_fragments(conn, w["fragments"], missing) if w["found"] else None
        matches += _unit_matches(tokens, w["fragments"], rows, phrase_id=u + 1)

    if missing:
        raise ReplayError("manifest names instances that are no longer in the DB: " + ", ".join(missing))
    return matches, adjustments, token_state


_GETTERS = {"word": index.get_word_instance, "syllable": index.get_syllable_instance,
            "phone": index.get_phone_instance}


def _fetch_fragments(conn: sqlite3.Connection, fragments: list[dict], missing: list[str]) -> list[dict] | None:
    """DB rows for each fragment, or None if any is gone (its name is added to `missing`)."""
    rows = []
    for f in fragments:
        table = _TIER_TABLE[f["tier"]]
        row = _GETTERS[table](conn, f["instance_id"])
        if row is None:
            missing.append(f"{table}_instance {f['instance_id']}")
        rows.append(row)
    return None if None in rows else rows


def _unit_matches(tokens: list[str], fragments: list[dict], rows: list[dict] | None, phrase_id: int) -> list:
    """The (token, row) entries assemble() takes for one recorded unit."""
    if rows is None:
        return [(tok, None) for tok in tokens]
    tier = fragments[0]["tier"]
    if tier == "phrase":
        for pos, row in enumerate(rows):
            row["phrase_id"], row["phrase_pos"], row["phrase_len"] = phrase_id, pos, len(rows)
        return list(zip(tokens, rows, strict=True))
    if tier == "word":
        return [(tokens[0], rows[0])]
    for row, f in zip(rows, fragments, strict=True):
        row["run_id"] = f.get("run_id")
    _number_runs(rows)
    return [(tokens[0], rows)]


def _number_runs(rows: list[dict]) -> None:
    """Restore run_pos/run_len from consecutive rows sharing a run_id."""
    i = 0
    while i < len(rows):
        rid = rows[i]["run_id"]
        j = i + 1
        while rid is not None and j < len(rows) and rows[j]["run_id"] == rid:
            j += 1
        for pos, row in enumerate(rows[i:j]):
            row["run_pos"], row["run_len"] = (pos, j - i) if rid is not None else (None, None)
        i = j
