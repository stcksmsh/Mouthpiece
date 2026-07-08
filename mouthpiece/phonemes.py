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


def phoneme_count(word: str) -> int:
    """Return phoneme count from CMUdict first pronunciation.
    Falls back to vowel-group syllable estimate when the word isn't in the dict."""
    entries = _load_cmu().get(word.lower(), [])
    if entries:
        return len(entries[0])
    # strip silent terminal 'e' then count vowel clusters
    w = re.sub(r"e\b", "", word.lower())
    return max(1, len(re.findall(r"[aeiou]+", w)))
