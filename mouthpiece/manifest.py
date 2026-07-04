"""Write per-render provenance manifest as JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path


def write_manifest(
    text: str,
    words: list[dict],
    output_wav: Path,
    seed: int | None = None,
) -> Path:
    manifest = {
        "rendered_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "text": text,
        "output_wav": str(output_wav),
        "words": words,
    }
    manifest_path = output_wav.with_name(output_wav.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path
