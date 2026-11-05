"""spaCy custom attribute (``._.``) extensions used by cascabel.

Surface features that depend only on the document itself (length, casing,
position, quotation, adjacency) are registered as *getters* so they are always available
and always up to date. Features that depend on external information (the
training set, a lexicon, a gazetteer, the character language models) are registered
with a default of ``None`` and filled in by :class:`cascabel.Cascabel`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from spacy.tokens import Doc, Span, Token

from cascabel.constants import QUOTATION_MARKS


# ---------------------------------------------------------------------------
# Token-level getters
# ---------------------------------------------------------------------------

def token_is_quotated(token: Token) -> bool:
    """A token is quotated when it is directly enclosed in quotation marks."""
    doc = token.doc
    if token.i == 0 or token.i == len(doc) - 1:
        return False
    return (
        doc[token.i - 1].text in QUOTATION_MARKS
        and doc[token.i + 1].text in QUOTATION_MARKS
    )


# ---------------------------------------------------------------------------
# Span-level getters
# ---------------------------------------------------------------------------

def is_multitoken(span: Span) -> bool:
    return len(span) > 1


def _is_punctuation_like(token: Token) -> bool:
    # Docs built on a bare Vocab have no lexeme attributes (is_punct is
    # always False), so punctuation is detected from the text itself.
    return bool(token.text) and not any(char.isalnum() for char in token.text)


def is_sentence_initial(span: Span) -> bool:
    """True if the span starts the sentence, ignoring leading punctuation."""
    if span.doc[span.start].is_sent_start:
        return True
    # A span preceded exclusively by punctuation (e.g. an opening quote)
    # is still considered sentence initial.
    for i in range(span.sent.start, span.start):
        if not _is_punctuation_like(span.doc[i]):
            return False
    return True


def is_quotated(span: Span) -> bool:
    """True if the span is directly enclosed in quotation marks."""
    doc = span.doc
    if span.start == 0 or span.end >= len(doc):
        return False
    if doc[span.start].is_sent_start or doc[span.end - 1].is_sent_end:
        return False
    return (
        doc[span.start - 1].text in QUOTATION_MARKS
        and doc[span.end].text in QUOTATION_MARKS
    )


def _has_case(text: str) -> bool:
    """Whether the text contains any caseable character. Caseless scripts
    (Han, digits, punctuation) make casing dimensions inapplicable."""
    return text.upper() != text.lower()


def _first_cased_char(text: str) -> Optional[str]:
    for char in text:
        if char.upper() != char.lower():
            return char
    return None


def casing(span: Span) -> Optional[str]:
    """Classify a span's casing as written, independently of its position.

    The first cased letter decides:

    - ``"upper"``: every cased letter is uppercase (*NATO*, *EEUU*, *3M*).
    - ``"lower"``: no cased letter is uppercase (*iphone*).
    - ``"title"``: the first cased letter is uppercase, whatever follows
      (*Bank of England*, *McDonald*, *New york*, *Ñúñez*).
    - ``"mixed"``: the first cased letter is lowercase but a later one is
      uppercase (*iPhone*, *eBay Inc.*).

    Sentence-initial spans are classified as written: position is the
    separate ``sentence_initial`` dimension. ``None`` (not applicable) when
    the span has no cased letter (numbers, symbols, caseless scripts).
    """
    text = span.text
    if not _has_case(text):
        return None
    if text == text.upper():
        return "upper"
    if text == text.lower():
        return "lower"
    if _first_cased_char(text).isupper():
        return "title"
    return "mixed"


def span_length(span: Span) -> int:
    """Number of tokens in the span."""
    return len(span)


def is_adjacent(span: Span) -> bool:
    """True if the span is immediately preceded by another entity."""
    if span.doc[span.start].is_sent_start or span.start == 0:
        return False
    return span.doc[span.start - 1].ent_iob_ != "O"


def surprisal_profile(span: Span) -> Optional[List[float]]:
    """Per-token surprisal (bits/char under the ``O``-token model) of the
    span, in order; ``None`` when surprisal has not been computed."""
    values = [tok._.surprisal for tok in span]
    if not values or any(value is None for value in values):
        return None
    return values


def surprisal_mean(span: Span) -> Optional[float]:
    """Mean token surprisal of the span (overall oddness)."""
    profile = span._.surprisal_profile
    return None if profile is None else sum(profile) / len(profile)


def surprisal_spread(span: Span) -> Optional[float]:
    """Maximum minus minimum token surprisal (0 for single tokens): high
    values mark spans that are only *partially* surprising."""
    profile = span._.surprisal_profile
    return None if profile is None else max(profile) - min(profile)


def least_surprising_position(span: Span) -> Optional[str]:
    """Where the least surprising token sits: ``"single"`` (one-token
    span), ``"first"``, ``"last"`` or ``"interior"``. Ties go to the
    earliest token."""
    profile = span._.surprisal_profile
    if profile is None:
        return None
    if len(profile) == 1:
        return "single"
    index = profile.index(min(profile))
    if index == 0:
        return "first"
    if index == len(profile) - 1:
        return "last"
    return "interior"


def _perplexity(measure: str):
    """Getter for the per-character perplexity of a bits/char measure:
    ``2 ** value`` (same ranking as the measure itself)."""

    def getter(span: Span) -> Optional[float]:
        value = getattr(span._, measure)
        return None if value is None else 2 ** value

    getter.__doc__ = (f"Per-character perplexity of ``{measure}`` (``2 ** "
                      f"{measure}``); ``None`` when it is unavailable.")
    return getter


def is_unseen(span: Span) -> Optional[bool]:
    """True if the span's exact word sequence never occurs in training as a
    whole span or as ``O`` (with the default ``span_match="whole"``);
    ``None`` without a training set."""
    count = span._.train_count
    return None if count is None else count == 0


def identity(span: Span) -> Optional[str]:
    """Classify a span by how its exact word sequence occurs in training.

    With the default ``span_match="whole"``, only occurrences that are a
    whole span or entirely ``O`` count: *Madrid* inside *Comunidad de
    Madrid* does not (it is left to the token layer, ``word_exposure``).
    Labels are compared by type, B/I ignored, and ``O`` counts as a label.
    ``"unseen"`` (never occurs), ``"seen_same_label"`` (every occurrence has
    the span's own label), ``"seen_ambiguous"`` (its own label plus others),
    ``"seen_other_label"`` (it occurs, but never with its own label: only as
    ``O`` or with other labels); ``None`` without a training set.
    """
    count, consistency = span._.train_count, span._.label_consistency
    if count is None:
        return None
    if count == 0:
        return "unseen"
    if consistency == 1.0:
        return "seen_same_label"
    if consistency == 0.0:
        return "seen_other_label"
    return "seen_ambiguous"


def token_exposure(token: Token) -> Optional[str]:
    """How the token's text was seen in training, B/I ignored: ``"in_span"``
    (at least once inside a span, any label), ``"only_O"`` (only outside
    spans) or ``"unseen"``; ``None`` without a training set."""
    labels = token._.train_labels
    if labels is None:
        return None
    if not labels:
        return "unseen"
    return "only_O" if set(labels) == {"O"} else "in_span"


def word_exposure(span: Span) -> Optional[Dict[str, int]]:
    """Token layer of identity: how many of the span's tokens were seen in
    training inside a span, only as ``O``, or never (see
    :func:`token_exposure`), e.g. ``{"in_span": 2, "only_O": 0, "unseen":
    1}``. Function words are included. ``None`` without a training set."""
    exposures = [tok._.exposure for tok in span]
    if any(exposure is None for exposure in exposures):
        return None
    return {key: exposures.count(key) for key in ("in_span", "only_O", "unseen")}


def has_word_seen_in_span(span: Span) -> Optional[bool]:
    """True if at least one token of the span was seen inside a span in
    training (any label); ``None`` without a training set."""
    exposure = span._.word_exposure
    return None if exposure is None else exposure["in_span"] > 0


def is_in_lexicon(span: Span) -> Optional[bool]:
    """True if every alphabetic token also exists in the background lexicon;
    ``None`` without a lexicon or when the span has no alphabetic token."""
    flags = [tok._.is_in_lexicon for tok in span if tok.text.isalpha()]
    if not flags or any(flag is None for flag in flags):
        return None
    return all(flags)


def is_partial_in_lexicon(span: Span) -> Optional[bool]:
    """True if some (but not all) alphabetic tokens exist in the lexicon."""
    flags = [tok._.is_in_lexicon for tok in span if tok.text.isalpha()]
    if not flags or any(flag is None for flag in flags):
        return None
    return any(flags) and not all(flags)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_TOKEN_GETTERS = {
    "is_sentence_initial": lambda tok: bool(tok.is_sent_start),
    "is_quotated": token_is_quotated,
    "exposure": token_exposure,
}

_TOKEN_DEFAULTS = (
    "iob_pred",            # full BIO tag predicted for this token
    "surprisal",           # bits/char under the O-token character LM
    "is_unseen_token",     # token text never seen in training
    "is_unseen_tagging",   # token seen in training, but never with this tag
    "is_type_confusable",  # token seen in training with more than one tag
    "train_labels",        # label counts of the token text in training, B/I ignored, "O" too
    "variant_labels",      # same, for training texts differing only in casing/accents
    "is_in_lexicon",       # token present in the user-provided lexicon
    "is_in_gazetteer",     # token present in the user-provided gazetteer
)

_SPAN_GETTERS = {
    "length": span_length,
    "is_multitoken": is_multitoken,
    "is_sentence_initial": is_sentence_initial,
    "is_quotated": is_quotated,
    "casing": casing,
    "is_adjacent": is_adjacent,
    "is_unseen": is_unseen,
    "identity": identity,
    "word_exposure": word_exposure,
    "has_word_seen_in_span": has_word_seen_in_span,
    "surprisal_profile": surprisal_profile,
    "surprisal_mean": surprisal_mean,
    "surprisal_perplexity": _perplexity("surprisal_mean"),
    "span_typicality_perplexity": _perplexity("span_typicality"),
    "prototypicality_perplexity": _perplexity("prototypicality"),
    "surprisal_spread": surprisal_spread,
    "least_surprising_position": least_surprising_position,
    "is_in_lexicon": is_in_lexicon,
    "is_partial_in_lexicon": is_partial_in_lexicon,
}

_SPAN_DEFAULTS = (
    "ents_pred",           # predicted entities aligned to this sentence
    "train_count",         # occurrences of the exact word sequence in training
    "train_labels",        # label counts of those occurrences, B/I ignored ("O", "MIXED" too)
    "variant_labels",      # same, for occurrences differing only in casing/accents
    "label_consistency",   # share of those occurrences with the span's own label
    "majority_label",      # most frequent label of those occurrences, e.g. "LOC" or "O"
    "majority_share",      # its share of the occurrences
    "span_typicality",     # bits/char under the any-span LM (labels ignored)
    "prototypicality",     # form typicality: bits/char under own label's LM
    "label_fit",           # own-label minus best-other-label log2 prob, per char
    "closest_other_label", # best-fitting other label under the label LMs
    "is_in_gazetteer",     # entity present in the user-provided gazetteer
)

_DOC_DEFAULTS = ("ents_pred",)


def register_extensions() -> None:
    """Register every cascabel extension. Safe to call repeatedly."""
    for name, getter in _TOKEN_GETTERS.items():
        Token.set_extension(name, getter=getter, force=True)
    for name in _TOKEN_DEFAULTS:
        if not Token.has_extension(name):
            Token.set_extension(name, default=None)
    for name, getter in _SPAN_GETTERS.items():
        Span.set_extension(name, getter=getter, force=True)
    for name in _SPAN_DEFAULTS:
        if not Span.has_extension(name):
            Span.set_extension(name, default=None)
    for name in _DOC_DEFAULTS:
        if not Doc.has_extension(name):
            Doc.set_extension(name, default=None)
