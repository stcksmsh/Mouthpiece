"""WhisperX transcription and word-level alignment."""

import os
from pathlib import Path

import numpy as np

CHUNK_S = 600  # 10-minute chunks for large files


def _load_chunk(audio_path: Path, start_s: float, end_s: float) -> np.ndarray:
    import soundfile as sf

    with sf.SoundFile(str(audio_path)) as f:
        start_frame = int(start_s * f.samplerate)
        n_frames = int((end_s - start_s) * f.samplerate)
        f.seek(start_frame)
        chunk = f.read(n_frames, dtype="float32", always_2d=False)
    return chunk


def _extract_words(result: dict, time_offset: float = 0.0) -> list[dict]:
    words = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            text = w.get("word", "").strip().lower()
            if not text:
                continue
            words.append(
                {
                    "word": text,
                    "start": round(float(w["start"]) + time_offset, 4),
                    "end": round(float(w["end"]) + time_offset, 4),
                    "confidence": w.get("score"),
                    "speaker": w.get("speaker"),
                }
            )
    return words


def transcribe_audio(audio_path: Path, model_size: str = "base") -> list[dict]:
    import soundfile as sf
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    hf_token = os.environ.get("MOUTHPIECE_HF_TOKEN")

    info = sf.info(str(audio_path))
    total_s = info.duration
    n_chunks = max(1, int(np.ceil(total_s / CHUNK_S)))

    print(f"  Duration: {total_s/60:.1f} min — {n_chunks} chunk(s) of {CHUNK_S//60} min")
    print(f"  Loading WhisperX '{model_size}' on {device} ...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # Detect language from the first chunk before loading alignment model.
    first_chunk = _load_chunk(audio_path, 0.0, min(60.0, total_s))
    lang_result = model.transcribe(first_chunk, batch_size=4 if device == "cpu" else 8)
    lang = lang_result.get("language", "en")
    print(f"  Detected language: {lang}")
    print(f"  Loading alignment model ...")
    align_model, align_meta = whisperx.load_align_model(language_code=lang, device=device)

    if hf_token:
        print("  Diarization enabled (speaker labels will be assigned per chunk).")
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        except Exception as exc:
            print(f"  Warning: diarization pipeline failed ({exc}); skipping.")
            diarize_model = None
    else:
        diarize_model = None

    all_words: list[dict] = []

    for i in range(n_chunks):
        start_s = i * CHUNK_S
        end_s = min(start_s + CHUNK_S, total_s)
        print(f"  [{i+1}/{n_chunks}] {start_s/60:.1f}–{end_s/60:.1f} min ...", flush=True)

        chunk = _load_chunk(audio_path, start_s, end_s)

        result = model.transcribe(chunk, batch_size=4 if device == "cpu" else 8)
        result = whisperx.align(
            result["segments"], align_model, align_meta, chunk, device,
            return_char_alignments=False,
        )

        if diarize_model is not None:
            try:
                diarize_segs = diarize_model(chunk)
                result = whisperx.assign_word_speakers(diarize_segs, result)
            except Exception as exc:
                print(f"    Warning: diarization failed on chunk {i+1} ({exc})")

        words = _extract_words(result, time_offset=start_s)
        all_words.extend(words)
        print(f"    {len(words)} words", flush=True)

    return all_words
