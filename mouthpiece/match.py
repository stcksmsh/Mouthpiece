"""Word matching: tokenize input text, find best clip candidates per token."""

import re

import numpy as np

TOP_K = 8           # candidate pool size for weighted random selection
MIN_CONF = 0.35     # clips below this confidence are excluded from the pool


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]


def _suffix_variants(word: str) -> list[str]:
    """Naive suffix-stripping for fail-soft lemma fallback."""
    variants = []
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        variants += [stem, stem + "e"]
    if word.endswith("tion") and len(word) > 5:
        variants.append(word[:-3] + "e")
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        variants += [stem, stem + "e"]
    if word.endswith("er") and len(word) > 4:
        variants.append(word[:-2])
    if word.endswith("est") and len(word) > 5:
        variants.append(word[:-3])
    if word.endswith("ly") and len(word) > 4:
        variants.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        variants.append(word[:-1])
    seen: set[str] = set()
    out = []
    for v in variants:
        if v not in seen and v != word:
            seen.add(v)
            out.append(v)
    return out


def _pick(rows: list[dict], rng: np.random.Generator | None, nth: int | None) -> dict:
    """
    Select one row from a candidate list.
    - nth override: deterministic by rank (1-indexed).
    - rng provided: weighted random from top-K, biased by confidence.
    - neither: top-1 deterministic.
    """
    if nth is not None:
        idx = min(nth - 1, len(rows) - 1)
        if nth > len(rows):
            print(f"    [nth] only {len(rows)} candidate(s), using #{len(rows)}")
        return rows[idx]

    if rng is None:
        return rows[0]

    # Filter below confidence floor; fall back to full list if everything is below it.
    pool = [r for r in rows if (r["confidence"] or 0.0) >= MIN_CONF] or rows
    pool = pool[:TOP_K]

    weights = np.array([(r["confidence"] or 0.5) for r in pool], dtype=np.float64)
    weights /= weights.sum()
    idx = rng.choice(len(pool), p=weights)
    return pool[idx]


def _candidates(conn, word: str) -> list[dict]:
    from . import index
    rows = index.search_word(conn, word, limit=TOP_K + 5)
    if rows:
        return rows
    for variant in _suffix_variants(word):
        rows = index.search_word(conn, variant, limit=TOP_K + 5)
        if rows:
            print(f"    [lemma fallback] '{word}' → '{variant}'")
            return rows
    return []


def find_clip(
    conn,
    word: str,
    rng: np.random.Generator | None = None,
    nth: int | None = None,
) -> dict | None:
    rows = _candidates(conn, word)
    if not rows:
        return None
    return _pick(rows, rng, nth)


def find_clip_near_pitch(
    conn,
    word: str,
    target_f0_hz: float,
    exclude_id: int | None = None,
) -> dict | None:
    """Return the clip for `word` whose f0_mean is closest to target_f0_hz."""
    from . import index
    rows = index.search_word_by_pitch(conn, word, target_f0_hz, limit=TOP_K + 5)
    if exclude_id is not None:
        rows = [r for r in rows if r["id"] != exclude_id] or rows
    if rows:
        return rows[0]
    for variant in _suffix_variants(word):
        rows = index.search_word_by_pitch(conn, variant, target_f0_hz, limit=TOP_K + 5)
        if exclude_id is not None:
            rows = [r for r in rows if r["id"] != exclude_id] or rows
        if rows:
            return rows[0]
    return None


def match_text(
    text: str,
    conn,
    rng: np.random.Generator | None = None,
    nth_overrides: dict[str, int] | None = None,
) -> list[tuple[str, dict | None]]:
    tokens = tokenize(text)
    overrides = nth_overrides or {}
    results = []
    for token in tokens:
        nth = overrides.get(token)
        clip = find_clip(conn, token, rng=rng, nth=nth)
        if nth and clip:
            print(f"    [nth={nth}] '{token}' → id={clip['id']}")
        results.append((token, clip))
    return results


def coverage_report(matches: list[tuple[str, dict | None]]) -> dict:
    found   = [t for t, r in matches if r is not None]
    missing = [t for t, r in matches if r is None]
    return {
        "total":          len(matches),
        "found":          len(found),
        "missing":        len(missing),
        "found_tokens":   found,
        "missing_tokens": missing,
    }
