# MOUTHPIECE

A concatenative speech-collage engine. Type text; receive a WAV assembled from real clips of real people actually saying each word — cut from a corpus you supply, with full per-word provenance.

**Not TTS.** Every word in the output is an unmodified clip of someone actually saying it.

---

## Setup

### 1. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch (must come first)

**With CUDA (NVIDIA GPU — strongly recommended for fast transcription):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (slow — a 1-hour audio file can take 30–60 min to index):**
```bash
pip install torch torchaudio
```

### 3. Install MOUTHPIECE

```bash
pip install -e .
```

This installs `whisperx`, `librosa`, `soundfile`, and the `mouthpiece` CLI.

**First-run note:** WhisperX will download model weights from HuggingFace on first use (~145 MB for `base`, ~3 GB for `large-v3`). This happens automatically and is cached in `~/.cache/`.

---

## Quick start with a tiny corpus

### Step 1 — Prepare your source

Drop an audio file (WAV, MP3, FLAC, etc.) into `corpus/`. Create a matching metadata JSON:

```bash
cp corpus/example_metadata.json corpus/myspeech.json
# edit myspeech.json with real speaker/date/event
```

**Metadata fields** (all optional except none are required — fill what you know):

```json
{
  "speaker": "Firstname Lastname",
  "date": "2003-02-05",
  "event": "Senate Armed Services Committee",
  "source_url": "https://example.gov/audio",
  "license": "public domain"
}
```

### Step 2 — Ingest

```bash
mouthpiece ingest corpus/myspeech.wav --metadata corpus/myspeech.json
```

This normalizes the audio to 16 kHz mono WAV, transcribes it with WhisperX, aligns word-level timestamps, and writes everything to `db/mouthpiece.db`.

Use `--model large-v2` for better transcription quality (slower, larger download).

### Step 3 — Check coverage

```bash
mouthpiece coverage "we must act now"
```

### Step 4 — Render

```bash
mouthpiece "we must act now" -o renders/out.wav
```

or equivalently:

```bash
mouthpiece render "we must act now" -o renders/out.wav
```

This writes `renders/out.wav` (16 kHz mono PCM WAV) and `renders/out.manifest.json`.

---

## Commands

```
mouthpiece ingest <audio> [-m metadata.json] [--model base] [--reindex]
mouthpiece render "text" [-o out.wav]          # "text" alone also works
mouthpiece coverage "text"
mouthpiece sources
```

**`ingest`** — Ingest one source file. Safe to re-run; skips already-indexed sources unless `--reindex` is passed.

**`render`** — Assemble a WAV from the indexed corpus. Missing words become 400 ms silence gaps with a warning; the render never crashes on missing words.

**`coverage`** — Show which words in the text are (or aren't) in the index, without rendering.

**`sources`** — List all ingested sources with word counts.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MOUTHPIECE_DB` | `db/mouthpiece.db` | SQLite database path |
| `MOUTHPIECE_CORPUS` | `corpus/` | Normalized WAV storage |
| `MOUTHPIECE_RENDERS` | `renders/` | Output WAV + manifest directory |
| `MOUTHPIECE_HF_TOKEN` | *(unset)* | HuggingFace token for speaker diarization (optional) |

Run from the repo root and defaults just work.

---

## Speaker diarization (optional)

If you set `MOUTHPIECE_HF_TOKEN` to a valid HuggingFace access token, and you have accepted the pyannote model terms at <https://hf.co/pyannote/speaker-diarization-3.1>, WhisperX will label each word with a speaker ID. These are stored in the `diarization_speaker` column and appear in manifests. Without the token, diarization is skipped and `speaker` in the manifest comes from the source metadata instead.

---

## Manifest format

Every render produces a `.manifest.json` alongside the WAV:

```json
{
  "rendered_at": "2026-07-03T12:00:00+00:00",
  "seed": null,
  "text": "we must act now",
  "output_wav": "renders/out.wav",
  "words": [
    {
      "token": "we",
      "found": true,
      "word_instance_id": 42,
      "word": "we",
      "source_file": "myspeech.wav",
      "speaker": "Firstname Lastname",
      "date": "2003-02-05",
      "source_url": "https://example.gov/audio",
      "start_s": 12.34,
      "end_s": 12.56,
      "confidence": 0.97
    },
    {
      "token": "now",
      "found": false,
      "gap_s": 0.4
    }
  ]
}
```

Every word is attributable. Missing words appear with `found: false`.

---

## Importing into Reaper

Reaper handles 16 kHz WAV natively. In your project:

1. Set project sample rate to 16000 Hz (optional — Reaper resamples on playback), **or** resample `out.wav` to 44100 Hz first: `sox out.wav -r 44100 out_44k.wav`
2. Insert → Media File → select `renders/out.wav`

A Reaper ReaScript bridge to import renders automatically is planned for v0.2.

---

## Project layout

```
corpus/          normalized 16 kHz WAVs + metadata JSONs (git-ignored data)
db/              SQLite database (git-ignored)
renders/         output WAVs + manifests (git-ignored)
mouthpiece/
  cli.py         entry point, subcommands
  index.py       SQLite schema + queries
  ingest.py      normalize audio + kick off transcription
  transcribe.py  WhisperX wrapper
  match.py       tokenize + find best clips
  assemble.py    cut clips, fade, concatenate
  manifest.py    write provenance JSON
```

---

## Roadmap

- **v0.1** (current): ingest → index → render with naive per-word-best matching
- **v0.2**: crossfades, `--seed` for reproducibility, word-gap control, Reaper ReaScript
- **v0.3**: prosodic features, affect sliders, Viterbi sequence selection, FastAPI + web UI
- **v0.4**: rhythmic quantization, MIDI sample-map export, degrade chain, phoneme fallback
