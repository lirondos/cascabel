"""Alignment of predicted entities against gold entities.

Instead of only counting exact matches, cascabel classifies every
gold/prediction disagreement into a fine-grained error taxonomy:

============================  ====================================================
``missed``                    a gold entity with no overlapping prediction
``spurious``                  a prediction with no overlapping gold entity
``type``                      correct boundaries, wrong label
``fused``                     one prediction covering two gold entities
``split``                     two predictions inside one gold entity
``missegmented``              prediction boundaries straddle gold entity boundaries
``overlap_missed``            prediction strictly inside a gold entity
``overlap_spurious``          prediction strictly containing a gold entity
``overlap_spurious_missed``   staggered overlap (extra tokens on one side,
                              missing tokens on the other)
============================  ====================================================

Each alignment carries two scores:

- ``span_score``: strict span-level counting (a correct alignment is one TP;
  every gold span involved in an error is one FN, every predicted span one FP).
- ``boundary_score``: the same counting with labels ignored (unlabelled
  exact match): a predicted span is a TP when its start and end equal a gold
  span's, whatever the label.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from spacy.tokens import Span, Token

from cascabel.doc import ents_by_sentence
from cascabel.metrics import Score


class ErrorType(str, Enum):
    MISSED = "missed"
    SPURIOUS = "spurious"
    TYPE = "type"
    FUSED = "fused"
    SPLIT = "split"
    MISSEGMENTED = "missegmented"
    OVERLAP_MISSED = "overlap_missed"
    OVERLAP_SPURIOUS = "overlap_spurious"
    OVERLAP_SPURIOUS_MISSED = "overlap_spurious_missed"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


#: Error types that reflect a segmentation problem (boundaries, not labels).
SEGMENTATION_ERRORS = frozenset(
    {ErrorType.MISSEGMENTED, ErrorType.SPLIT, ErrorType.FUSED}
)


# ---------------------------------------------------------------------------
# Span-pair predicates (token offsets; spans within the same Doc)
# ---------------------------------------------------------------------------

def is_exact_match(gold: Span, predicted: Span) -> bool:
    return (
        gold.start == predicted.start
        and gold.end == predicted.end
        and gold.label_ == predicted.label_
    )


def is_type_swap(gold: Span, predicted: Span) -> bool:
    return (
        gold.start == predicted.start
        and gold.end == predicted.end
        and gold.label_ != predicted.label_
    )


def is_fused(gold_first: Span, gold_second: Span, predicted: Span) -> bool:
    """One prediction spanning from the first gold entity through the second."""
    return predicted.start == gold_first.start and predicted.end >= gold_second.end


def is_split(gold: Span, predicted_first: Span, predicted_second: Span) -> bool:
    """Two predictions falling inside a single gold entity."""
    return gold.start <= predicted_first.start and gold.end >= predicted_second.end


def is_contained(gold: Span, predicted: Span) -> bool:
    """Prediction inside the gold entity (some gold tokens missing)."""
    return predicted.start >= gold.start and predicted.end <= gold.end


def is_containing(gold: Span, predicted: Span) -> bool:
    """Prediction containing the gold entity (some spurious tokens added)."""
    return predicted.start <= gold.start and predicted.end >= gold.end


def is_staggered(gold: Span, predicted: Span) -> bool:
    """Partial overlap with extra tokens on one side and missing on the other."""
    return (
        predicted.start < gold.start < predicted.end < gold.end
        or gold.start < predicted.start < gold.end < predicted.end
    )


def is_straddling(gold_first: Span, gold_second: Span, predicted: Span) -> bool:
    """Prediction starting inside the first gold entity and reaching into the
    second (``end`` is exclusive: a prediction ending right before the second
    entity does not reach it)."""
    return (
        gold_first.start <= predicted.start < gold_first.end
        and predicted.end > gold_second.start
    )


# ---------------------------------------------------------------------------
# Alignment record
# ---------------------------------------------------------------------------

@dataclass
class Alignment:
    """One unit of gold/prediction correspondence within a sentence.

    ``gold`` and ``predicted`` hold the spans involved (either may be empty
    for missed/spurious cases). ``error_type`` is ``None`` for a correct
    alignment.
    """

    gold: Tuple[Span, ...]
    predicted: Tuple[Span, ...]
    sentence: Span
    error_type: Optional[ErrorType] = None
    correct_tokens: List[Token] = field(default_factory=list, repr=False)
    wrong_tokens: List[Token] = field(default_factory=list, repr=False)
    span_score: Score = field(init=False, repr=False)
    boundary_score: Score = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.gold = tuple(self.gold)
        self.predicted = tuple(self.predicted)
        self._classify_tokens()
        self.span_score = self._compute_span_score()
        self.boundary_score = self._compute_boundary_score()

    @property
    def is_error(self) -> bool:
        return self.error_type is not None

    @property
    def is_segmentation_error(self) -> bool:
        return self.error_type in SEGMENTATION_ERRORS

    # -- token classification -------------------------------------------------
    #
    # Note: predicted spans may belong to the prediction Doc, while gold tags
    # and the ``iob_pred`` overlay live on the test Doc. The two documents are
    # token-aligned, so predicted span offsets are always resolved against
    # ``self.sentence.doc`` (the test Doc) before reading token attributes.

    def _tokens_at(self, span: Span) -> List[Token]:
        doc = self.sentence.doc
        return [doc[i] for i in range(span.start, span.end)]

    def _classify_tokens(self) -> None:
        error = self.error_type
        if error is None:
            self.correct_tokens = [tok for span in self.gold for tok in span]
        elif error is ErrorType.MISSED:
            self.wrong_tokens = list(self.gold[0])
        elif error is ErrorType.SPURIOUS:
            self.wrong_tokens = self._tokens_at(self.predicted[0])
        elif error is ErrorType.FUSED:
            # The wrong tokens are those that should have started an entity
            # (gold B) but did not: they caused the fusing.
            for span in self.gold:
                for tok in span:
                    if tok.ent_iob_ == "B" and not str(tok._.iob_pred or "").startswith("B"):
                        self.wrong_tokens.append(tok)
                    else:
                        self.correct_tokens.append(tok)
        elif error is ErrorType.SPLIT:
            # The wrong tokens are those that started an entity (predicted B)
            # but should not have: they caused the splitting.
            doc = self.sentence.doc
            start = min(self.gold[0].start, self.predicted[0].start)
            end = max(self.gold[-1].end, self.predicted[-1].end)
            for i in range(start, end):
                tok = doc[i]
                if tok.ent_iob_ != "B" and str(tok._.iob_pred or "").startswith("B"):
                    self.wrong_tokens.append(tok)
                else:
                    self.correct_tokens.append(tok)
        elif error is ErrorType.MISSEGMENTED:
            gold_texts = {tok.text for span in self.gold for tok in span}
            for span in self.predicted:
                for tok in self._tokens_at(span):
                    if tok.text in gold_texts:
                        self.correct_tokens.append(tok)
                    else:
                        self.wrong_tokens.append(tok)
        else:
            # type swaps and the overlap_* family: a token is correct as long
            # as both gold and prediction agree it belongs to an entity.
            doc = self.sentence.doc
            start = min(self.gold[0].start, self.predicted[0].start)
            end = max(self.gold[-1].end, self.predicted[-1].end)
            for i in range(start, end):
                tok = doc[i]
                if tok._.iob_pred not in (None, "O") and tok.ent_iob_ != "O":
                    self.correct_tokens.append(tok)
                else:
                    self.wrong_tokens.append(tok)

    # -- scores ---------------------------------------------------------------

    def _compute_span_score(self) -> Score:
        if self.error_type is None:
            return Score(tp=1)
        if self.error_type is ErrorType.MISSED:
            return Score(fn=1)
        if self.error_type is ErrorType.SPURIOUS:
            return Score(fp=1)
        return Score(fn=len(self.gold), fp=len(self.predicted))

    def _compute_boundary_score(self) -> Score:
        """Unlabelled exact match: a gold and a predicted span of this
        alignment with the same start and end are one TP, whatever their
        labels; every other gold span is an FN and every other predicted
        span an FP."""
        predicted = {(p.start, p.end) for p in self.predicted}
        tp = sum((g.start, g.end) in predicted for g in self.gold)
        return Score(tp=tp, fn=len(self.gold) - tp, fp=len(self.predicted) - tp)


# ---------------------------------------------------------------------------
# The alignment walk
# ---------------------------------------------------------------------------

def align_sentence(
    sentence: Span,
    gold_entities: Sequence[Span],
    predicted_entities: Sequence[Span],
) -> List[Alignment]:
    """Walk gold and predicted entities of one sentence in parallel and
    classify every correspondence. Both sequences must be sorted by start
    offset (spaCy's ``.ents`` already are)."""
    alignments: List[Alignment] = []
    gold = list(gold_entities)
    pred = list(predicted_entities)
    gi = pi = 0

    def add(golds, preds, error_type):
        alignments.append(Alignment(tuple(golds), tuple(preds), sentence, error_type))

    while gi < len(gold) and pi < len(pred):
        g, p = gold[gi], pred[pi]
        next_gold = gold[gi + 1] if gi + 1 < len(gold) else None
        next_pred = pred[pi + 1] if pi + 1 < len(pred) else None

        if is_exact_match(g, p):
            add([g], [p], None)
            gi += 1
            pi += 1
        elif is_type_swap(g, p):
            add([g], [p], ErrorType.TYPE)
            gi += 1
            pi += 1
        elif next_gold is not None and is_fused(g, next_gold, p):
            add([g, next_gold], [p], ErrorType.FUSED)
            gi += 2
            pi += 1
        elif next_pred is not None and is_split(g, p, next_pred):
            add([g], [p, next_pred], ErrorType.SPLIT)
            gi += 1
            pi += 2
        elif is_contained(g, p):
            # e.g. gold "carrot cake" + "vegan friendly", predicted "carrot"
            # + "cake vegan friendly": the boundary drifted right.
            if (
                p.start == g.start
                and p.end < g.end
                and next_pred is not None
                and next_gold is not None
                and next_pred.start < g.end
            ):
                add([g, next_gold], [p, next_pred], ErrorType.MISSEGMENTED)
                gi += 2
                pi += 2
            else:
                add([g], [p], ErrorType.OVERLAP_MISSED)
                gi += 1
                pi += 1
        elif is_staggered(g, p):
            # e.g. gold "carrot cake" + "vegan friendly", predicted a single
            # "cake vegan friendly": the first entity was missed and the
            # second missegmented.
            if (
                next_gold is not None
                and g.start < p.start < g.end
                and p.end == next_gold.end
            ):
                add([g], [], ErrorType.MISSED)
                add([next_gold], [p], ErrorType.MISSEGMENTED)
                gi += 2
                pi += 1
            else:
                add([g], [p], ErrorType.OVERLAP_SPURIOUS_MISSED)
                gi += 1
                pi += 1
        elif next_gold is not None and is_straddling(g, next_gold, p):
            # A "perfect" missegmentation ends exactly where the second gold
            # entity ends and is recorded as a single segmentation error.
            if next_pred is not None and next_pred.end == next_gold.end:
                add([g, next_gold], [p, next_pred], ErrorType.MISSEGMENTED)
                gi += 2
                pi += 2
            else:
                add([g], [p], ErrorType.MISSEGMENTED)
                gi += 1
                pi += 1
        elif is_containing(g, p):
            add([g], [p], ErrorType.OVERLAP_SPURIOUS)
            gi += 1
            pi += 1
        elif g.start_char < p.start_char:
            add([g], [], ErrorType.MISSED)
            gi += 1
        elif g.start_char > p.start_char:
            add([], [p], ErrorType.SPURIOUS)
            pi += 1
        else:  # pragma: no cover - defensive: the cases above are exhaustive
            warnings.warn(
                f"Unclassifiable alignment between gold {g!r} and prediction "
                f"{p!r}; recording as overlap_spurious_missed",
                stacklevel=2,
            )
            add([g], [p], ErrorType.OVERLAP_SPURIOUS_MISSED)
            gi += 1
            pi += 1

    for i in range(pi, len(pred)):  # remaining predictions are spurious
        add([], [pred[i]], ErrorType.SPURIOUS)
    for i in range(gi, len(gold)):  # remaining gold entities are missed
        add([gold[i]], [], ErrorType.MISSED)
    return alignments


def align_doc(test_doc) -> List[Alignment]:
    """Align every sentence of a test ``Doc`` that has the prediction layer
    applied (``sent._.ents_pred`` set by :class:`cascabel.Cascabel`)."""
    # Bucket gold entities per sentence once (O(n)); per-sentence ``Span.ents``
    # would rescan the whole doc each time (O(sentences x entities)).
    gold_by_sent = ents_by_sentence(test_doc)
    alignments: List[Alignment] = []
    for sentence in test_doc.sents:
        predicted = sentence._.ents_pred or []
        gold = gold_by_sent.get(sentence.start, [])
        alignments.extend(align_sentence(sentence, gold, predicted))
    return alignments
