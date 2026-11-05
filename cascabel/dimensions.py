"""Bucketing entities by linguistic dimension.

A *dimension* is a boolean property of an entity span (a length bin,
sentence-initial, title case, quoted, unseen in training, ...). Reports
over dimensions show *where* a dataset is hard and *where* a model fails,
instead of a single aggregate score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from spacy.tokens import Doc, Span

SpanPredicate = Callable[[Span], Optional[bool]]


def _identity_is(ent: Span, value: str) -> Optional[bool]:
    identity = ent._.identity
    return None if identity is None else identity == value


def _casing_is(ent: Span, value: str) -> Optional[bool]:
    casing = ent._.casing
    return None if casing is None else casing == value


#: Fixed span-length bins (in tokens), shared by every dataset so that
#: bucket percentages are comparable across datasets. ``(label, low, high)``
#: with ``high=None`` meaning "or longer". Provisional (2026-10-16).
LENGTH_BINS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("length_1", 1, 1),
    ("length_2", 2, 2),
    ("length_3", 3, 3),
    ("length_4_5", 4, 5),
    ("length_6plus", 6, None),
)


def _length_in(low: int, high: Optional[int]) -> SpanPredicate:
    return lambda ent: low <= len(ent) and (high is None or len(ent) <= high)


#: Registry of entity-level dimensions (binary views). Predicates may return
#: ``None`` when the required information (training set, lexicon) is
#: missing; a dimension whose predicate returns ``None`` for every entity is
#: omitted from reports.
#:
#: Continuous properties (length, training count, label consistency,
#: surprisal, prototypicality, label fit) live on each span; see
#: :data:`NUMERIC_ATTRIBUTES` and :meth:`cascabel.Cascabel.span_table`.
#: Length is also binned here with fixed :data:`LENGTH_BINS`; surprisal and
#: prototypicality have no binary view until fixed cross-dataset bins are
#: chosen.
ENTITY_DIMENSIONS: Dict[str, SpanPredicate] = {
    # length (fixed bins)
    **{name: _length_in(low, high) for name, low, high in LENGTH_BINS},
    # surface dimensions (always available)
    "sentence_initial": lambda ent: ent._.is_sentence_initial,
    "casing_upper": lambda ent: _casing_is(ent, "upper"),
    "casing_title": lambda ent: _casing_is(ent, "title"),
    "casing_lower": lambda ent: _casing_is(ent, "lower"),
    "casing_mixed": lambda ent: _casing_is(ent, "mixed"),
    "quoted": lambda ent: ent._.is_quotated,
    "adjacent": lambda ent: ent._.is_adjacent,
    # training-dependent dimensions
    "unseen": lambda ent: ent._.is_unseen,
    "seen_same_label": lambda ent: _identity_is(ent, "seen_same_label"),
    "seen_ambiguous": lambda ent: _identity_is(ent, "seen_ambiguous"),
    "seen_other_label": lambda ent: _identity_is(ent, "seen_other_label"),
    # optional user resources (lexicon/gazetteer); only present when a word
    # list is passed to Cascabel
    "in_lexicon": lambda ent: ent._.is_in_lexicon,
    "partial_in_lexicon": lambda ent: ent._.is_partial_in_lexicon,
    "in_gazetteer": lambda ent: ent._.is_in_gazetteer,
}

#: One-line meaning of each dimension, shown next to it in every report.
DIMENSION_GLOSSES: Dict[str, str] = {
    "length_1": "the span is 1 token long",
    "length_2": "the span is 2 tokens long",
    "length_3": "the span is 3 tokens long",
    "length_4_5": "the span is 4 or 5 tokens long",
    "length_6plus": "the span is 6 or more tokens long",
    "sentence_initial": "the span starts at the beginning of the sentence",
    "casing_upper": "every cased letter is uppercase (NATO, JAPAN)",
    "casing_title": "the first cased letter is uppercase, not all are (Bank of England)",
    "casing_lower": "no letter is uppercase (iphone)",
    "casing_mixed": "starts lowercase, with a later uppercase letter (iPhone)",
    "quoted": "the span is directly enclosed in quotation marks",
    "adjacent": "the span directly follows another span, no token between",
    "unseen": "never in training as a whole span or as O",
    "seen_same_label": "seen in training, always with its own label",
    "seen_ambiguous": "seen in training with its own label and others (O included)",
    "seen_other_label": "seen in training, never with its own label",
    "in_lexicon": "every word is in the given lexicon",
    "partial_in_lexicon": "some, not all, words are in the given lexicon",
    "in_gazetteer": "the span is in the given gazetteer",
}

#: Continuous span attributes (``span._.<name>``), reported per span by
#: :meth:`cascabel.Cascabel.span_table`. ``None`` when unavailable (e.g. no
#: training set, or a label too rare to have a prototypicality model).
NUMERIC_ATTRIBUTES: Tuple[str, ...] = (
    "length",
    "train_count",
    "label_consistency",
    "majority_share",
    "surprisal_mean",
    "surprisal_perplexity",
    "surprisal_spread",
    "span_typicality",
    "span_typicality_perplexity",
    "prototypicality",
    "prototypicality_perplexity",
    "label_fit",
)

#: Categorical span attributes reported by :meth:`cascabel.Cascabel.span_table`.
CATEGORICAL_ATTRIBUTES: Tuple[str, ...] = (
    "casing",
    "identity",
    "train_labels",
    "variant_labels",
    "word_exposure",
    "has_word_seen_in_span",
    "majority_label",
    "least_surprising_position",
    "closest_other_label",
    "is_sentence_initial",
    "is_quotated",
    "is_adjacent",
)


@dataclass
class DimensionStats:
    """How many entities of a document express one dimension.

    ``applicable`` counts the entities for which the dimension is defined at
    all (predicates return ``None`` when it is not — e.g. casing dimensions
    on caseless scripts); ``percentage`` is computed over the applicable
    entities, never over the inapplicable ones.
    """

    count: int
    total: int
    applicable: int
    spans: List[Span] = field(repr=False, default_factory=list)

    @property
    def percentage(self) -> float:
        return 100 * self.count / self.applicable if self.applicable else 0.0

    @property
    def applicability(self) -> float:
        """Share of entities for which this dimension is defined (0-100)."""
        return 100 * self.applicable / self.total if self.total else 0.0


def dimension_report(
    doc: Doc,
    dimensions: Optional[Mapping[str, SpanPredicate]] = None,
) -> Dict[str, DimensionStats]:
    """Count, for every dimension, the entities of ``doc`` that express it.

    Dimensions whose predicate returns ``None`` for every entity (missing
    training data, lexicon, caseless script, ...) are omitted entirely;
    partially applicable dimensions report their ``applicability``.
    """
    dimensions = dict(dimensions) if dimensions is not None else ENTITY_DIMENSIONS
    entities = list(doc.ents)
    report: Dict[str, DimensionStats] = {}
    for name, predicate in dimensions.items():
        values = [predicate(ent) for ent in entities]
        if entities and all(value is None for value in values):
            continue  # information for this dimension is unavailable
        matching = [ent for ent, value in zip(entities, values) if value]
        report[name] = DimensionStats(
            count=len(matching),
            total=len(entities),
            applicable=sum(value is not None for value in values),
            spans=matching,
        )
    return report
