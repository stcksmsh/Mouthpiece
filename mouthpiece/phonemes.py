"""Phoneme count from CMUdict, with a syllable-heuristic fallback."""

import re

_cmu: dict | None = None


def _load_cmu() -> dict:
    global _cmu
    if _cmu is not None:
        return _cmu
    # 1. standalone 'cmudict' package (preferred — bundled, no download)
    try:
        import cmudict as _c  # pip install cmudict
        _cmu = _c.dict()
        return _cmu
    except ImportError:
        pass
    # 2. NLTK cmudict corpus (whisperx already requires nltk)
    try:
        from nltk.corpus import cmudict as _cd
        try:
            _cmu = _cd.dict()
        except LookupError:
            import nltk
            nltk.download("cmudict", quiet=True)
            _cmu = _cd.dict()
        return _cmu
    except Exception:
        pass
    _cmu = {}
    return _cmu


def suffix_variants(word: str) -> list[str]:
    """Naive suffix-stripping for fail-soft word-form fallback — shared by
    match.py's whole-word search and this module's CMUdict lookups, so a
    derived form CMUdict doesn't have its own entry for (e.g. 'concatenative')
    can still borrow its base form's pronunciation ('concatenate')."""
    variants = []
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        variants += [stem, stem + "e"]
    if word.endswith("tion") and len(word) > 5:
        variants.append(word[:-3] + "e")
    if word.endswith("ive") and len(word) > 5:
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


def _split_compound(word: str, cmu: dict, max_pieces: int, memo: dict) -> list[str] | None:
    """Try to cover `word` end-to-end with 2+ known CMUdict headwords (e.g.
    'ironman' -> 'iron' + 'man'), returning the concatenated phones of the
    first pronunciation of each piece. Both pieces of any split must be at
    least 2 letters, to avoid single-letter junk splits. Memoized on suffix
    so recursive re-checks of the same tail are cheap."""
    if word in memo:
        return memo[word]
    result = None
    if max_pieces >= 2:
        for i in range(2, len(word) - 1):
            left, right = word[:i], word[i:]
            left_entries = cmu.get(left)
            if not left_entries:
                continue
            right_entries = cmu.get(right)
            if right_entries:
                result = left_entries[0] + right_entries[0]
                break
            right_split = _split_compound(right, cmu, max_pieces - 1, memo)
            if right_split is not None:
                result = left_entries[0] + right_split
                break
    memo[word] = result
    return result


def _cmu_entries(word: str) -> list[list[str]] | None:
    """CMUdict pronunciation for `word`, trying suffix-stripped variants
    (e.g. 'concatenative' -> 'concatenate') and then compound-word splitting
    (e.g. 'ironman' -> 'iron' + 'man') before giving up."""
    cmu = _load_cmu()
    w = word.lower()
    entries = cmu.get(w)
    if entries:
        return entries
    for variant in suffix_variants(w):
        entries = cmu.get(variant)
        if entries:
            return entries
    if len(w) >= 4:
        compound = _split_compound(w, cmu, max_pieces=3, memo={})
        if compound:
            return [compound]
    return None


def phoneme_count(word: str) -> int:
    """Return phoneme count from CMUdict first pronunciation.
    Falls back to vowel-group syllable estimate when the word isn't in the dict."""
    entries = _cmu_entries(word)
    if entries:
        return len(entries[0])
    # strip silent terminal 'e' then count vowel clusters
    w = re.sub(r"e\b", "", word.lower())
    return max(1, len(re.findall(r"[aeiou]+", w)))


def _is_vowel(phone: str) -> bool:
    return phone[-1:].isdigit()


def syllabify(word: str) -> list[list[str]] | None:
    """Split a word's first CMUdict pronunciation into syllables.

    Vowel phones (stress-marked, e.g. 'AE1') are syllable nuclei. Consonants
    between two vowels are split with a simplified maximal-onset rule: the
    last consonant of the run joins the following syllable's onset, the rest
    stay as the preceding syllable's coda. Leading/trailing consonant runs
    attach to the first/last syllable. Returns None for words CMUdict doesn't
    know (out-of-vocabulary words stay unmatched at syllable level too).
    """
    entries = _cmu_entries(word)
    if not entries:
        return None
    phones = entries[0]
    vowel_idx = [i for i, p in enumerate(phones) if _is_vowel(p)]
    if not vowel_idx:
        return [list(phones)]

    syllables: list[list[str]] = [[] for _ in vowel_idx]
    for n, vi in enumerate(vowel_idx):
        syllables[n].append(phones[vi])

    # leading consonants -> onset of syllable 0
    syllables[0][0:0] = phones[:vowel_idx[0]]

    # consonant runs between vowels
    for n in range(len(vowel_idx) - 1):
        run = phones[vowel_idx[n] + 1: vowel_idx[n + 1]]
        if not run:
            continue
        if len(run) == 1:
            syllables[n + 1].insert(0, run[0])
        else:
            syllables[n].extend(run[:-1])
            syllables[n + 1].insert(0, run[-1])

    # trailing consonants -> coda of last syllable
    syllables[-1].extend(phones[vowel_idx[-1] + 1:])

    return syllables


def phonemize(word: str) -> list[list[str]] | None:
    """Like syllabify(), but each phone is its own singleton group — the
    finest possible decomposition, used as a fallback search granularity
    when whole-syllable matching can't cover a word. Returns None for words
    CMUdict doesn't know (same coverage as syllabify())."""
    entries = _cmu_entries(word)
    if not entries:
        return None
    return [[p] for p in entries[0]]


def phone_key(phones: list[str]) -> str:
    """Stress-insensitive matching key for a syllable's phone list."""
    return "-".join(re.sub(r"\d$", "", p) for p in phones)


# Rough relative duration weights per ARPAbet phone class (arbitrary units,
# not measured — stops shortest, diphthongs longest) so a word's audio
# window gets split closer to where syllable boundaries actually fall,
# instead of assuming every phone takes equal time.
_PHONE_WEIGHTS = {
    # stops
    "B": 0.7, "D": 0.7, "G": 0.7, "P": 0.7, "T": 0.7, "K": 0.7,
    # affricates
    "CH": 0.9, "JH": 0.9,
    # fricatives
    "F": 0.9, "V": 0.8, "TH": 0.9, "DH": 0.7, "S": 0.9, "Z": 0.8,
    "SH": 0.9, "ZH": 0.8, "HH": 0.7,
    # nasals / liquids / glides
    "M": 0.8, "N": 0.8, "NG": 0.8, "L": 0.8, "R": 0.8, "W": 0.7, "Y": 0.6,
    # monophthong vowels
    "AA": 1.5, "AE": 1.5, "AH": 1.1, "AO": 1.5, "EH": 1.4, "IH": 1.2,
    "IY": 1.5, "UH": 1.2, "UW": 1.5, "ER": 1.4,
    # diphthongs
    "AY": 1.8, "AW": 1.8, "EY": 1.7, "OW": 1.7, "OY": 1.9,
}
_DEFAULT_PHONE_WEIGHT = 1.0


def _phone_weight(phone: str) -> float:
    return _PHONE_WEIGHTS.get(re.sub(r"\d$", "", phone), _DEFAULT_PHONE_WEIGHT)


def phone_weight_sum(phones: list[str]) -> float:
    """Total estimated relative duration of a phone sequence — used to compare
    how fast/slow a syllable clip was actually spoken against its phonetic content."""
    return sum(_phone_weight(p) for p in phones)


def compute_syllable_rows(word: str, start_s: float, end_s: float) -> list[dict] | None:
    """Syllabify `word` and split [start_s, end_s] proportionally by each
    syllable's estimated relative duration (phone-class weighted, not just
    phone count — a syllable with a long vowel/diphthong gets more of the
    window than one with the same phone count but all short consonants).
    Returns rows ready for index.insert_syllable_instances(), or None if the
    word isn't in CMUdict. Pure text/arithmetic — no audio access."""
    syllables = syllabify(word)
    if syllables is None:
        return None
    total_phones = sum(sum(_phone_weight(p) for p in g) for g in syllables)
    if total_phones == 0:
        return None
    dur = end_s - start_s
    rows = []
    t = start_s
    n = len(syllables)
    for i, group in enumerate(syllables):
        if i == n - 1:
            seg_end = end_s
        else:
            group_weight = sum(_phone_weight(p) for p in group)
            seg_end = t + dur * (group_weight / total_phones)
        rows.append({
            "syll_index": i,
            "syll_count": n,
            "phones":     " ".join(group),
            "phone_key":  phone_key(group),
            "start_s":    round(t, 4),
            "end_s":      round(seg_end, 4),
        })
        t = seg_end
    return rows


def compute_phone_rows(word: str, start_s: float, end_s: float) -> list[dict] | None:
    """Like compute_syllable_rows(), but split into individual phones instead
    of syllables — same phone-weight proportional-split technique, one level
    finer. Each row also records which syllable (by index) it belongs to,
    for provenance. Returns rows ready for index.insert_phone_instances(), or
    None if the word isn't in CMUdict. Pure text/arithmetic — no audio access."""
    syllables = syllabify(word)
    if syllables is None:
        return None
    all_phones = [(si, p) for si, g in enumerate(syllables) for p in g]
    total_weight = sum(_phone_weight(p) for _, p in all_phones)
    if total_weight == 0:
        return None
    dur = end_s - start_s
    rows = []
    t = start_s
    n = len(all_phones)
    for idx, (si, p) in enumerate(all_phones):
        if idx == n - 1:
            seg_end = end_s
        else:
            seg_end = t + dur * (_phone_weight(p) / total_weight)
        rows.append({
            "syll_index":  si,
            "phone_index": idx,
            "phone_count": n,
            "phone":       p,
            "phone_key":   phone_key([p]),
            "start_s":     round(t, 4),
            "end_s":       round(seg_end, 4),
        })
        t = seg_end
    return rows
