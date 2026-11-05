"""Controlled perturbations of annotated documents.

These functions build *new* documents with the same annotations but altered
surface forms (casing, quotation marks). Feeding perturbed documents to a
model and comparing scores against the original is a simple, targeted stress
test: does performance rely on casing? on quotation cues?
"""

from __future__ import annotations

from typing import List, Optional

from spacy.tokens import Doc
from spacy.vocab import Vocab

from cascabel.doc import bio_tag

_CASES = {None, "lower", "title", "upper"}


def _apply_case(text: str, case: Optional[str]) -> str:
    if case is None:
        return text
    if case == "lower":
        return text.lower()
    if case == "title":
        return text.capitalize()
    if case == "upper":
        return text.upper()
    raise ValueError(f"Unknown case {case!r}; expected one of {sorted(c for c in _CASES if c)}")


def _build(words: List[str], tags: List[str], sent_starts: List[bool]) -> Doc:
    return Doc(Vocab(strings=[]), words=words, ents=tags, sent_starts=sent_starts)


def recase(doc: Doc, case: str, scope: str = "doc") -> Doc:
    """Return a copy of ``doc`` with tokens re-cased.

    ``scope="doc"`` re-cases every token; ``scope="entities"`` re-cases only
    the tokens inside entities. Annotations are preserved.
    """
    if scope not in ("doc", "entities"):
        raise ValueError(f"Unknown scope {scope!r}; expected 'doc' or 'entities'")
    words = []
    for tok in doc:
        inside_entity = tok.ent_iob_ != "O"
        if scope == "doc" or inside_entity:
            words.append(_apply_case(tok.text, case))
        else:
            words.append(tok.text)
    tags = [bio_tag(tok) for tok in doc]
    sent_starts = [bool(tok.is_sent_start) for tok in doc]
    return _build(words, tags, sent_starts)


def add_quotes(
    doc: Doc,
    quote: str = '"',
    doc_case: Optional[str] = None,
    entity_case: Optional[str] = None,
) -> Doc:
    """Return a copy of ``doc`` with every unquoted entity wrapped in quotes.

    Optionally re-case non-entity tokens (``doc_case``) and/or entity tokens
    (``entity_case``; defaults to ``doc_case`` when only that one is given).
    """
    entity_case = entity_case if entity_case is not None else doc_case
    words: List[str] = []
    tags: List[str] = []
    sent_starts: List[bool] = []

    cursor = 0
    for ent in doc.ents:
        for tok in doc[cursor : ent.start]:
            words.append(_apply_case(tok.text, doc_case))
            tags.append(bio_tag(tok))
            sent_starts.append(bool(tok.is_sent_start))
        quoted = ent._.is_quotated
        if not quoted:
            words.append(quote)
            tags.append("O")
            sent_starts.append(bool(ent[0].is_sent_start))
        for i, tok in enumerate(ent):
            words.append(_apply_case(tok.text, entity_case))
            tags.append(bio_tag(tok))
            if i == 0 and not quoted:
                # the inserted quote took over the sentence-initial position
                sent_starts.append(False)
            else:
                sent_starts.append(bool(tok.is_sent_start))
        if not quoted:
            words.append(quote)
            tags.append("O")
            sent_starts.append(False)
        cursor = ent.end
    for tok in doc[cursor:]:
        words.append(_apply_case(tok.text, doc_case))
        tags.append(bio_tag(tok))
        sent_starts.append(bool(tok.is_sent_start))
    return _build(words, tags, sent_starts)


def strip_quotes(
    doc: Doc,
    doc_case: Optional[str] = None,
    entity_case: Optional[str] = None,
) -> Doc:
    """Return a copy of ``doc`` with the quotes around quoted entities removed.

    The inverse perturbation of :func:`add_quotes`. Optional re-casing works
    the same way.
    """
    entity_case = entity_case if entity_case is not None else doc_case
    words: List[str] = []
    tags: List[str] = []
    sent_starts: List[bool] = []

    cursor = 0
    for ent in doc.ents:
        for tok in doc[cursor : ent.start]:
            words.append(_apply_case(tok.text, doc_case))
            tags.append(bio_tag(tok))
            sent_starts.append(bool(tok.is_sent_start))
        quoted = ent._.is_quotated
        was_sentence_initial = False
        if quoted:
            # drop the opening quote emitted by the loop above; the entity
            # inherits its sentence-initial position if it had one
            words.pop()
            tags.pop()
            was_sentence_initial = sent_starts.pop()
        for i, tok in enumerate(ent):
            words.append(_apply_case(tok.text, entity_case))
            tags.append(bio_tag(tok))
            if i == 0 and quoted:
                sent_starts.append(was_sentence_initial)
            else:
                sent_starts.append(bool(tok.is_sent_start))
        cursor = ent.end + 1 if quoted else ent.end  # skip the closing quote
    for tok in doc[cursor:]:
        words.append(_apply_case(tok.text, doc_case))
        tags.append(bio_tag(tok))
        sent_starts.append(bool(tok.is_sent_start))
    return _build(words, tags, sent_starts)
