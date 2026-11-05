"""Rule-based Spanish phonotactics: does a word violate Spanish syllable
structure?

A hand-crafted, Spanish-specific complement to the statistical surprisal
scores of :mod:`cascabel.charlm`. It flags words whose syllable onsets,
codas, endings or consonant transitions are impossible in Spanish — a strong
signal that the word is a lexical borrowing (e.g. an anglicism).

Requires the optional dependency ``pylabeador``::

    pip install cascabel[spanish]

Use it as a custom dimension (since 0.7.0 it no longer replaces the shape
model, which is now continuous)::

    from cascabel.contrib import spanish
    dims = {"phonotactic_anomaly": lambda ent: any(
        spanish.is_phonotactically_anomalous(t.text)
        for t in ent if t.text.isalpha())}
    casc.dimension_report()  # built-in dimensions
    casc.scores_by_dimension(dims)
"""

from __future__ import annotations

VALID_ONSETS = frozenset(
    ["bl", "br", "pl", "pr", "cl", "cr", "gr", "gl", "fr", "fl",
     "tr", "dr", "ch", "qu", "gu", "rr", "ll"]
)

VALID_CODAS = frozenset(["ns"])

VOWELS = frozenset(["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú"])

VALID_CONSONANT_ENDINGS = frozenset(["r", "s", "l", "n", "d", "z", "y"])

VALID_TRANSITIONS = frozenset(
    ["ct", "cc", "gn", "mp", "mb", "mn", "pt", "pc", "nn"]
)

FORBIDDEN_PATTERNS = (
    "qua", "quo", "quu", "ze", "zi", "pca", "pco", "pcu",
    "ash", "ish", "osh", "ush", "sh",
)


def _require_pylabeador():
    try:
        import pylabeador  # noqa: F401

        return pylabeador
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "cascabel.contrib.spanish requires the optional dependency "
            "'pylabeador'. Install it with: pip install cascabel[spanish]"
        ) from error


def is_invalid_onset(onset: str) -> bool:
    return len(onset) > 1 and onset not in VALID_ONSETS


def is_invalid_coda(coda: str) -> bool:
    return len(coda) > 1 and coda not in VALID_CODAS


def is_invalid_ending(word: str) -> bool:
    if len(word) == 1 or word[-1] in VOWELS:
        return False
    return word[-2] not in VOWELS or word[-1] not in VALID_CONSONANT_ENDINGS


def is_invalid_transition(first: str, second: str) -> bool:
    """Whether the syllable-boundary consonant pair ``first``-``second`` is
    impossible in Spanish."""
    if (first == "b" and second != "b") or (first == "x" and second != "x"):
        return False
    if first in VALID_CONSONANT_ENDINGS and first != second:
        return False
    return first + second not in VALID_TRANSITIONS


def contains_forbidden_pattern(word: str) -> bool:
    return any(pattern in word for pattern in FORBIDDEN_PATTERNS)


def is_phonotactically_anomalous(word: str) -> bool:
    """True if ``word`` violates Spanish phonotactics (syllable structure,
    onsets/codas, endings, consonant transitions, or forbidden patterns).

    Non-alphabetic strings return ``False``. Words that cannot be
    syllabified at all are considered anomalous.
    """
    if not word or not word.isalpha():
        return False
    pylabeador = _require_pylabeador()
    word = word.lower()
    if contains_forbidden_pattern(word):
        return True
    try:
        syllabified = pylabeador.syllabify_with_details(word)
    except pylabeador.errors.HyphenatorError:
        return True
    for syllable in syllabified.syllables:
        if is_invalid_onset(syllable.onset):
            return True
        if is_invalid_coda(syllable.coda):
            return True
    if is_invalid_ending(word):
        return True
    for first, second in zip(syllabified.syllables, syllabified.syllables[1:]):
        if first.coda and second.onset:
            if is_invalid_transition(first.coda[-1], second.onset[0]):
                return True
    return False


#: Backwards-compatible alias for the original research code's naming.
is_perplexing = is_phonotactically_anomalous
