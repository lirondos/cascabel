"""Building spaCy ``Doc`` objects from tokens, sentences, CoNLL files and binaries."""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from spacy.tokens import Doc, DocBin, Span, Token
from spacy.vocab import Vocab

#: A sentence is a pair of (tokens, BIO tags), e.g. (["Un", "look"], ["O", "B-ENG"]).
Sentence = Tuple[Sequence[str], Sequence[str]]

PathLike = Union[str, Path]


def bio_tag(token: Token) -> str:
    """Return the full BIO tag of a token, e.g. ``"B-ENG"`` or ``"O"``."""
    if token.ent_iob_ == "O":
        return "O"
    return f"{token.ent_iob_}-{token.ent_type_}"


def fold_text(text: str) -> str:
    """Fold casing and accents: *MADRID*, *madrid* and *Madrid* all give
    ``"madrid"``; *El Pais* and *EL PAÍS* give ``"el pais"``. *ñ* is kept
    (a letter of its own in Spanish, not an accented *n*): *PEÑA* gives
    ``"peña"``, distinct from *Pena*.

    Used to find training occurrences of a span written differently
    (``variant_labels``), never for the exact matching behind ``identity``.
    """
    decomposed = unicodedata.normalize("NFD", text)
    kept = [
        ch for i, ch in enumerate(decomposed)
        if unicodedata.category(ch) != "Mn"
        or (ch == "̃" and i > 0 and decomposed[i - 1] in "nN")
    ]
    return unicodedata.normalize("NFC", "".join(kept)).casefold()


def occurrence_label(tags: Sequence[str]) -> str:
    """Span label of one occurrence of a word sequence, B/I ignored.

    ``"O"`` when every token is outside any span; the shared label when every
    token carries the same label (*Madrid* as ``B-LOC`` alone or ``I-LOC``
    inside *Comunidad de Madrid* are both ``"LOC"``); ``"MIXED"`` otherwise
    (partly ``O``, or several labels).
    """
    if all(tag == "O" for tag in tags):
        return "O"
    if "O" in tags:
        return "MIXED"
    labels = {tag.split("-", 1)[1] if "-" in tag else "" for tag in tags}
    return labels.pop() if len(labels) == 1 else "MIXED"


def ents_by_sentence(doc: Doc) -> Dict[int, List[Span]]:
    """Map each sentence (keyed by its start-token index) to the entities it
    contains, in a single O(entities + sentences) pass.

    spaCy's ``Span.ents`` scans the whole document's entity list on every
    access, so calling it once per sentence is O(sentences x entities) — minutes
    on corpora with tens of thousands of sentences. Both ``doc.ents`` and
    ``doc.sents`` are already sorted by token offset, so a single merge assigns
    every entity to its sentence."""
    sents = list(doc.sents)
    buckets: Dict[int, List[Span]] = {sent.start: [] for sent in sents}
    si = 0
    for ent in doc.ents:
        while si < len(sents) and sents[si].end <= ent.start:
            si += 1
        if si < len(sents) and sents[si].start <= ent.start:
            buckets[sents[si].start].append(ent)
    return buckets


def repair_bio(tags: Sequence[str]) -> List[str]:
    """Repair an invalid BIO sequence.

    Fixes the transitions that make a tag sequence ill-formed:

    - ``I-X`` at the start of a sentence or after ``O`` becomes ``B-X``.
    - ``I-X`` after a tag with a different label ``Y`` becomes ``B-X``.
    - A bare ``I`` (no label) after ``O`` becomes a bare ``B``.
    """
    repaired: List[str] = []
    previous = "O"
    for tag in tags:
        if tag.startswith("I"):
            label = tag.split("-", 1)[1] if "-" in tag else None
            previous_label = previous.split("-", 1)[1] if "-" in previous else None
            if previous == "O" or previous_label != label:
                tag = f"B-{label}" if label is not None else "B"
        repaired.append(tag)
        previous = tag
    return repaired


def repair_bio_sentences(sentences: Iterable[Sentence]) -> List[Sentence]:
    """Apply :func:`repair_bio` to every sentence of a corpus."""
    return [(list(tokens), repair_bio(tags)) for tokens, tags in sentences]


#: Tagging schemes accepted on loading (aliases map to these names).
SCHEMES: Tuple[str, ...] = ("BIO", "IOB1", "BIOES")
_SCHEME_ALIASES = {"BIO": "BIO", "IOB2": "BIO", "IOB1": "IOB1",
                   "BIOES": "BIOES", "IOBES": "BIOES", "BILOU": "BIOES"}


def normalize_scheme(scheme: str) -> str:
    """Canonical name of a tagging scheme (``"IOB2"`` → ``"BIO"``, ``"BILOU"``
    → ``"BIOES"``); raises ``ValueError`` for an unknown scheme."""
    try:
        return _SCHEME_ALIASES[scheme.upper()]
    except KeyError:
        raise ValueError(
            f"Unknown tagging scheme {scheme!r}; choose one of {SCHEMES} "
            "(aliases: IOB2, IOBES, BILOU)"
        ) from None


def to_bio(tags: Sequence[str], scheme: str = "BIO") -> List[str]:
    """Convert one sentence's tags from ``scheme`` to BIO (IOB2).

    - ``"BIO"``: returned unchanged.
    - ``"IOB1"`` (original CoNLL-2003 files): a span starts with ``I-X``;
      ``B-X`` only marks a span right after another span of the same label.
      Every ``I-X`` that starts a span becomes ``B-X``.
    - ``"BIOES"`` / ``"BILOU"``: ``S-X``/``U-X`` → ``B-X``; ``E-X``/``L-X`` →
      ``I-X``.
    """
    scheme = normalize_scheme(scheme)
    if scheme == "IOB1":
        return repair_bio(tags)
    if scheme == "BIOES":
        mapping = {"S": "B", "U": "B", "E": "I", "L": "I"}
        return [
            f"{mapping[tag[0]]}{tag[1:]}" if tag[:1] in mapping else tag
            for tag in tags
        ]
    return list(tags)


def doc_from_sentences(
    sentences: Iterable[Sentence], repair: bool = True, scheme: str = "BIO"
) -> Doc:
    """Build a single ``Doc`` from a corpus of (tokens, tags) sentences.

    Tags are first converted from ``scheme`` (``"BIO"``, the default;
    ``"IOB1"``; ``"BIOES"``, see :func:`to_bio`). Ill-formed BIO sequences
    left after that (a stray ``I-X`` after ``O``, at the start of a sentence
    or after another label) are handled as in ``conlleval`` and ``seqeval``:
    the stray ``I-X`` starts a new span (see :func:`repair_bio`). A warning
    reports how many sentences were affected and where, so the user can fix
    the file or declare its scheme. With ``repair=False`` an ill-formed
    sequence raises ``ValueError``.
    """
    scheme = normalize_scheme(scheme)
    words: List[str] = []
    tags: List[str] = []
    sent_starts: List[bool] = []
    ill_formed: List[int] = []
    for number, (tokens, sentence_tags) in enumerate(sentences, start=1):
        tokens = list(tokens)
        sentence_tags = list(sentence_tags)
        if len(tokens) != len(sentence_tags):
            raise ValueError(
                f"Sentence has {len(tokens)} tokens but {len(sentence_tags)} tags: "
                f"{tokens[:5]}..."
            )
        sentence_tags = to_bio(sentence_tags, scheme)
        repaired = repair_bio(sentence_tags)
        if repaired != sentence_tags:
            ill_formed.append(number)
            sentence_tags = repaired
        words.extend(tokens)
        tags.extend(sentence_tags)
        sent_starts.extend([True] + [False] * (len(tokens) - 1))
    if not words:
        raise ValueError("Cannot build a Doc from an empty corpus")
    if ill_formed:
        shown = ", ".join(str(n) for n in ill_formed[:10])
        more = ", ..." if len(ill_formed) > 10 else ""
        message = (
            f"{len(ill_formed)} sentence(s) with ill-formed BIO tags "
            f"(sentence numbers: {shown}{more})"
        )
        if not repair:
            raise ValueError(message)
        warnings.warn(
            message + "; each stray I- tag was read as the start of a new span "
            "(conlleval convention). If the file uses another tagging scheme, "
            "pass scheme='IOB1' or scheme='BIOES'",
            stacklevel=2,
        )
    return Doc(Vocab(strings=[]), words=words, ents=tags, sent_starts=sent_starts)


def doc_from_tokens(
    tokens: Sequence[str], tags: Sequence[str], repair: bool = True,
    scheme: str = "BIO",
) -> Doc:
    """Build a single-sentence ``Doc`` from parallel token and tag lists."""
    return doc_from_sentences([(tokens, tags)], repair=repair, scheme=scheme)


def read_conll(
    path: PathLike,
    token_column: int = 0,
    tag_column: int = -1,
    delimiter: Optional[str] = None,
    comment_prefix: Optional[str] = None,
    encoding: str = "utf-8",
    skip_docstart: bool = True,
) -> List[Sentence]:
    """Read a CoNLL-style column file into a list of (tokens, tags) sentences.

    Sentences are separated by blank lines. By default the token is the first
    column and the tag the last one, split on any whitespace; pass ``delimiter``
    (e.g. ``"\\t"``) for strict splitting and ``comment_prefix`` (e.g. ``"#"``)
    to skip comment lines. Document-boundary lines (token ``-DOCSTART-``,
    as in the CoNLL-2002/2003 files) are skipped and end the current
    sentence, so they never count as sentences or tokens; pass
    ``skip_docstart=False`` to keep them.
    """
    sentences: List[Sentence] = []
    tokens: List[str] = []
    tags: List[str] = []
    with open(path, encoding=encoding) as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if comment_prefix and line.startswith(comment_prefix):
                continue
            if not line.strip():
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            columns = line.split(delimiter)
            if skip_docstart and columns[token_column] == "-DOCSTART-":
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            tokens.append(columns[token_column])
            tags.append(columns[tag_column])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def doc_from_conll(
    path: PathLike, repair: bool = True, scheme: str = "BIO", **kwargs
) -> Doc:
    """Build a ``Doc`` straight from a CoNLL-style file (see :func:`read_conll`
    and, for ``scheme``, :func:`doc_from_sentences`)."""
    return doc_from_sentences(read_conll(path, **kwargs), repair=repair, scheme=scheme)


def doc_from_spacy_binary(path: PathLike) -> Doc:
    """Load a serialized ``DocBin`` (``.spacy`` file) and merge it into one ``Doc``."""
    doc_bin = DocBin().from_disk(str(path))
    docs = list(doc_bin.get_docs(Vocab(strings=[])))
    if not docs:
        raise ValueError(f"No documents found in {path}")
    if len(docs) == 1:
        return docs[0]
    return Doc.from_docs(docs)


def load_wordlist(path: PathLike, lowercase: bool = True, encoding: str = "utf-8") -> Set[str]:
    """Load a lexicon/gazetteer file with one entry per line into a set."""
    entries: Set[str] = set()
    with open(path, encoding=encoding) as handle:
        for line in handle:
            entry = line.strip()
            if entry:
                entries.add(entry.lower() if lowercase else entry)
    return entries
