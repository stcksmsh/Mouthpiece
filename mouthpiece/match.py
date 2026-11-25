"""Word matching: tokenize input text, find best clip candidates per token."""

import re

import numpy as np

TOP_K = 8           # candidate pool size for weighted random selection
MIN_CONF = 0.35     # clips below this confidence are excluded from the pool
BIAS_K = 3.0        # exponent sharpness for diction/emotion reweighting
MAX_PHRASE_LEN = 6  # cap on how many words a single phrase run can span
PHONEME_FORCE_THRESHOLD = -0.5  # granularity below this skips straight to phoneme-only


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]


def _rank_normalize(rows: list[dict], key: str) -> list[float]:
    """Percentile rank (0..1) of each row's `key` value within this pool.
    Rows with a missing value (or pools with <2 present values) get 0.5 (neutral)."""
    present = [(i, r.get(key)) for i, r in enumerate(rows) if r.get(key) is not None]
    if len(present) < 2:
        return [0.5] * len(rows)
    order = sorted(present, key=lambda p: p[1])
    ranks = {i: k / (len(order) - 1) for k, (i, _) in enumerate(order)}
    return [ranks.get(i, 0.5) for i in range(len(rows))]


def _bias_weight(score: float, slider: float) -> float:
    """score in [0,1] (higher = more of the trait); slider in [-1,1].
    slider=0 -> multiplier 1.0 always (today's unbiased behavior)."""
    if not slider:
        return 1.0
    s = score if slider > 0 else (1.0 - score)
    s = max(1e-6, min(1.0, s))
    return s ** (abs(slider) * BIAS_K)


def candidate_pool(rows: list[dict]) -> list[dict]:
    """The rows weighted selection draws from: those at or above MIN_CONF (all
    of them if none are), cut to the first TOP_K."""
    pool = [r for r in rows if (r["confidence"] or 0.0) >= MIN_CONF] or rows
    return pool[:TOP_K]


def unit_weight(
    row: dict,
    diction: float = 0.0,
    emotion: float = 0.0,
    emotion_score: float = 0.5,
) -> float:
    """Unnormalised selection weight of one candidate; higher is better.
    Confidence, scaled by the diction bias (on confidence) and the emotion bias
    (on `emotion_score`, the row's 0..1 expressiveness within its pool, see
    pool_weights). Sliders at 0 leave the confidence weight unchanged."""
    conf = row["confidence"]
    dscore = conf if conf is not None else 0.5
    # Multiplication order matches the pre-split code, so rng.choice sees
    # bit-identical probabilities.
    return (conf or 0.5) * _bias_weight(dscore, diction) * _bias_weight(emotion_score, emotion)


def pool_weights(pool: list[dict], diction: float = 0.0, emotion: float = 0.0) -> list[float]:
    """unit_weight for each row of `pool`. The emotion score is relative: the
    mean of the row's f0_var and rms percentile ranks within this pool."""
    f0_rank  = _rank_normalize(pool, "f0_var")
    rms_rank = _rank_normalize(pool, "rms")
    return [
        unit_weight(r, diction, emotion, (f + m) / 2.0)
        for r, f, m in zip(pool, f0_rank, rms_rank, strict=True)
    ]


def _pick(
    rows: list[dict],
    rng: np.random.Generator | None,
    nth: int | None,
    diction: float = 0.0,
    emotion: float = 0.0,
) -> dict:
    """
    Select one row from a candidate list.
    - nth override: deterministic by rank (1-indexed).
    - rng provided: weighted random from candidate_pool(), by pool_weights().
    - neither: top-1 deterministic.
    """
    if nth is not None:
        idx = min(nth - 1, len(rows) - 1)
        if nth > len(rows):
            print(f"    [nth] only {len(rows)} candidate(s), using #{len(rows)}")
        return rows[idx]

    if rng is None:
        return rows[0]

    pool = candidate_pool(rows)
    weights = np.array(pool_weights(pool, diction, emotion), dtype=np.float64)
    weights /= weights.sum()
    idx = rng.choice(len(pool), p=weights)
    return pool[idx]


def _candidates(conn, word: str) -> list[dict]:
    from . import index, phonemes
    rows = index.search_word(conn, word, limit=TOP_K + 5)
    if rows:
        return rows
    for variant in phonemes.suffix_variants(word):
        rows = index.search_word(conn, variant, limit=TOP_K + 5)
        if rows:
            print(f"    [lemma fallback] '{word}' → '{variant}'")
            return rows
    return []


def _resolve_groups(conn, word: str, search_fn) -> list[list[str]] | None:
    """Find any way to tile `word`'s CMUdict pronunciation into contiguous
    phone spans that each have at least one corpus candidate under `search_fn`
    (index.search_syllable or index.search_phone) — a word-break-style search
    over every possible span, not limited to the syllabifier's own vowel-
    anchored boundaries. This is strictly more powerful than only trying the
    syllabifier's boundaries (or merging adjacent ones): some words are only
    tileable using a span that cuts across where the syllabifier would have
    placed a boundary. Prefers fewer, larger pieces when multiple tilings
    exist (more natural — fewer joins). Returns None if no full tiling exists
    at all under this search_fn: some phone combination in the word occurs
    nowhere in the corpus at that granularity, at any chunk size. At
    phone-level granularity this degenerates to a per-phone reachability
    check (search_fn only ever matches single-phone spans), which is exactly
    right — phonemize() below produces the same singleton-group result more
    directly, but routing phone-level through this same DP keeps one
    implementation for both granularities instead of two."""
    from . import phonemes

    entries = phonemes._cmu_entries(word)
    if not entries:
        return None
    phones = list(entries[0])
    n = len(phones)

    span_ok: dict[tuple[int, int], bool] = {}

    def ok(i: int, j: int) -> bool:
        if (i, j) not in span_ok:
            span_ok[(i, j)] = bool(
                search_fn(conn, phonemes.phone_key(phones[i:j]), 1)
            )
        return span_ok[(i, j)]

    reachable: list[bool] = [False] * (n + 1)
    back: list[int | None] = [None] * (n + 1)
    reachable[0] = True
    for j in range(1, n + 1):
        for i in range(j):  # ascending i => prefer the largest span ending at j
            if reachable[i] and ok(i, j):
                reachable[j] = True
                back[j] = i
                break
    if not reachable[n]:
        return None

    spans: list[list[str]] = []
    j = n
    while j > 0:
        i = back[j]
        spans.append(phones[i:j])
        j = i
    spans.reverse()
    return spans


def _find_natural_run(
    conn,
    groups: list[list[str]],
    start_idx: int,
    search_fn,
    next_fn,
    max_len: int = MAX_PHRASE_LEN,
) -> list[dict] | None:
    """Find the longest run of consecutive target groups starting at
    `start_idx` that also occur consecutively (within a small gap) in one
    source recording — the sub-word analogue of find_phrase(), generalized
    over search_fn/next_fn so it works at syllable or phone granularity.
    Returns the raw instance rows for the run (>=2 units), or None."""
    from . import phonemes

    # Wide pool on purpose: we're hunting for a rare coincidental continuation,
    # not picking the single best individual candidate (that's _pick()'s job) —
    # a seed with a useful next-unit match can rank well outside TOP_K.
    seed_rows = search_fn(conn, phonemes.phone_key(groups[start_idx]), 50)
    if not seed_rows:
        return None

    best_run: list[dict] | None = None
    for seed in seed_rows:
        run = [seed]
        current = seed
        j = start_idx + 1
        while j < len(groups) and (j - start_idx) < max_len:
            nxt = next_fn(conn, current["id"])
            if nxt is None or nxt["phone_key"] != phonemes.phone_key(groups[j]):
                break
            run.append(nxt)
            current = nxt
            j += 1
        if len(run) >= 2 and (best_run is None or len(run) > len(best_run)):
            best_run = run

    return best_run


def _stitch_at_unit(
    conn,
    word: str,
    search_fn,
    next_fn,
    rng: np.random.Generator | None,
    nth: int | None,
    diction: float,
    emotion: float,
) -> tuple[list[dict], int] | None:
    """Resolve+stitch `word` at one granularity (syllable or phone, per
    search_fn/next_fn). Returns (picks, segment_count) — segment_count is the
    number of independent splice points after natural-run merging, used to
    compare granularities — or None if infeasible at this granularity."""
    from . import phonemes

    groups = _resolve_groups(conn, word, search_fn)
    if groups is None:
        return None

    picks: list[dict] = []
    run_counter = 0
    segment_count = 0
    i, n = 0, len(groups)
    while i < n:
        run = _find_natural_run(conn, groups, i, search_fn, next_fn)
        if run:
            run_counter += 1
            rlen = len(run)
            for pos, row in enumerate(run):
                row["run_id"], row["run_pos"], row["run_len"] = run_counter, pos, rlen
                picks.append(row)
            i += rlen
            segment_count += 1
            continue

        key  = phonemes.phone_key(groups[i])
        rows = search_fn(conn, key, TOP_K + 5)
        if not rows:
            return None
        picks.append(_pick(rows, rng, nth, diction=diction, emotion=emotion))
        i += 1
        segment_count += 1
    return picks, segment_count


def stitch_syllables(
    conn,
    word: str,
    rng: np.random.Generator | None = None,
    nth: int | None = None,
    diction: float = 0.0,
    emotion: float = 0.0,
    granularity: float = 0.0,
) -> list[dict] | None:
    """Stitch `word` from sub-word corpus fragments, preferring natural
    multi-unit runs (_find_natural_run) — real, continuously-spoken corpus
    audio spanning 2+ target units at once — over independently-sourced
    single-unit picks wherever one exists. Rows belonging to a run carry
    run_id/run_pos/run_len (mirrors find_phrase()'s phrase_id/phrase_pos/
    phrase_len on whole-word rows).

    Tries syllable-level and phoneme-level (finer — individual ARPAbet
    phones, via phonemes.phonemize()) granularity and keeps whichever
    produces fewer splice points (more natural), UNLESS `granularity` is
    pushed past PHONEME_FORCE_THRESHOLD, in which case phoneme-level is used
    unconditionally (the explicit 'hear more fragmentation' slider extreme).
    Returns None if the word isn't in CMUdict, or neither granularity can
    fully tile it from what's in the corpus."""
    from . import index

    if granularity < PHONEME_FORCE_THRESHOLD:
        forced = _stitch_at_unit(
            conn, word, index.search_phone, index.next_phone_instance,
            rng, nth, diction, emotion,
        )
        if forced is None:
            return None
        print(f"    [phoneme-forced, granularity={granularity:.2f}] '{word}'")
        return forced[0]

    syll  = _stitch_at_unit(
        conn, word, index.search_syllable, index.next_syllable_instance,
        rng, nth, diction, emotion,
    )
    phone = _stitch_at_unit(
        conn, word, index.search_phone, index.next_phone_instance,
        rng, nth, diction, emotion,
    )

    if syll is None and phone is None:
        return None
    if syll is None:
        print(f"    [phoneme fallback — no syllable tiling] '{word}'")
        return phone[0]
    if phone is None:
        return syll[0]

    syll_picks, syll_segs = syll
    phone_picks, phone_segs = phone
    if phone_segs < syll_segs:
        print(f"    [phoneme wins: {phone_segs} vs {syll_segs} segments] '{word}'")
        return phone_picks
    return syll_picks


def find_phrase(
    conn,
    tokens: list[str],
    start_idx: int,
    rng: np.random.Generator | None = None,
    max_len: int = MAX_PHRASE_LEN,
) -> list[dict] | None:
    """Find the longest run of consecutive tokens starting at `start_idx` that
    also occur consecutively (within a small gap) in one source recording.
    Returns the raw word_instance rows for the run (>=2 words), or None."""
    from . import index

    seed_rows = _candidates(conn, tokens[start_idx])
    if not seed_rows:
        return None

    best_run: list[dict] | None = None
    for seed in seed_rows[:TOP_K]:
        run = [seed]
        current = seed
        j = start_idx + 1
        while j < len(tokens) and (j - start_idx) < max_len:
            nxt = index.next_word_instance(conn, current["id"])
            if nxt is None or nxt["word"] != tokens[j]:
                break
            run.append(nxt)
            current = nxt
            j += 1
        if len(run) >= 2 and (best_run is None or len(run) > len(best_run)):
            best_run = run

    return best_run


def find_clip(
    conn,
    word: str,
    rng: np.random.Generator | None = None,
    nth: int | None = None,
    diction: float = 0.0,
    emotion: float = 0.0,
    granularity: float = 0.0,
) -> dict | list[dict] | None:
    if granularity < 0 and rng is not None and rng.random() < abs(granularity):
        stitched = stitch_syllables(conn, word, rng, nth, diction=diction, emotion=emotion,
                                     granularity=granularity)
        if stitched:
            return stitched

    rows = _candidates(conn, word)
    if rows:
        return _pick(rows, rng, nth, diction=diction, emotion=emotion)
    stitched = stitch_syllables(conn, word, rng, nth, diction=diction, emotion=emotion,
                                 granularity=granularity)
    if stitched:
        print(f"    [stitch] '{word}' → {len(stitched)} unit(s)")
    return stitched


def find_clip_near_pitch(
    conn,
    word: str,
    target_f0_hz: float,
    exclude_id: int | None = None,
) -> dict | None:
    """Return the clip for `word` whose f0_mean is closest to target_f0_hz."""
    from . import index, phonemes
    rows = index.search_word_by_pitch(conn, word, target_f0_hz, limit=TOP_K + 5)
    if exclude_id is not None:
        rows = [r for r in rows if r["id"] != exclude_id] or rows
    if rows:
        return rows[0]
    for variant in phonemes.suffix_variants(word):
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
    granularity: float = 0.0,
    diction: float = 0.0,
    emotion: float = 0.0,
) -> list[tuple[str, dict | list[dict] | None]]:
    tokens = tokenize(text)
    overrides = nth_overrides or {}
    results: list[tuple[str, dict | list[dict] | None]] = []
    phrase_counter = 0
    i = 0
    n = len(tokens)

    while i < n:
        token = tokens[i]
        nth = overrides.get(token)

        if granularity > 0 and rng is not None and nth is None and rng.random() < granularity:
            run = find_phrase(conn, tokens, i, rng=rng)
            if run:
                phrase_counter += 1
                plen = len(run)
                for pos, row in enumerate(run):
                    row["phrase_id"]  = phrase_counter
                    row["phrase_pos"] = pos
                    row["phrase_len"] = plen
                    results.append((tokens[i + pos], row))
                print(f"    [phrase] '{' '.join(tokens[i:i + plen])}' → {plen}-word run")
                i += plen
                continue

        clip = find_clip(conn, token, rng=rng, nth=nth, diction=diction, emotion=emotion,
                          granularity=granularity)
        if nth and clip and isinstance(clip, dict):
            print(f"    [nth={nth}] '{token}' → id={clip['id']}")
        results.append((token, clip))
        i += 1

    return results


def coverage_report(matches: list[tuple[str, dict | list[dict] | None]]) -> dict:
    found_word      = []
    found_syllables = []
    found_phrase    = []
    missing         = []
    for t, r in matches:
        if r is None:
            missing.append(t)
        elif isinstance(r, list):
            found_syllables.append(t)
        elif r.get("phrase_id") is not None:
            found_phrase.append(t)
        else:
            found_word.append(t)
    found = found_word + found_syllables + found_phrase
    return {
        "total":                 len(matches),
        "found":                 len(found),
        "missing":               len(missing),
        "found_tokens":          found,
        "found_word":            len(found_word),
        "found_word_tokens":     found_word,
        "found_syllables":       len(found_syllables),
        "found_syllable_tokens": found_syllables,
        "found_phrase":          len(found_phrase),
        "found_phrase_tokens":   found_phrase,
        "missing_tokens":        missing,
    }
