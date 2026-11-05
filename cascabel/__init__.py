"""cascabel — fine-grained error analysis for NER and span-labeling models.

Instead of a single F1 score, cascabel aligns predictions against gold
annotations, classifies every disagreement into a fine-grained error taxonomy
(missed, spurious, type swap, fused, split, missegmented, partial overlaps),
profiles datasets along linguistic dimensions (length, casing, position,
quotation, seen/unseen, surprisal, prototypicality) and supports controlled perturbations for stress
testing.
"""

from cascabel.alignment import (
    Alignment,
    ErrorType,
    SEGMENTATION_ERRORS,
    align_doc,
    align_sentence,
)
from cascabel.bootstrap import bootstrap_scores
from cascabel.cascabel import Cascabel
from cascabel.dimensions import (
    CATEGORICAL_ATTRIBUTES,
    ENTITY_DIMENSIONS,
    LENGTH_BINS,
    NUMERIC_ATTRIBUTES,
    DimensionStats,
    dimension_report,
)
from cascabel.doc import (
    bio_tag,
    doc_from_conll,
    doc_from_sentences,
    doc_from_spacy_binary,
    doc_from_tokens,
    fold_text,
    load_wordlist,
    occurrence_label,
    read_conll,
    repair_bio,
    repair_bio_sentences,
    to_bio,
)
from cascabel.extensions import register_extensions
from cascabel.metrics import Score
from cascabel.perturb import add_quotes, recase, strip_quotes
from cascabel.charlm import CharLM, CharLMConfig, ShapeModels
from cascabel.shape import word_shape

__version__ = "0.8.0"

__all__ = [
    "Alignment",
    "CATEGORICAL_ATTRIBUTES",
    "Cascabel",
    "CharLM",
    "CharLMConfig",
    "DimensionStats",
    "ENTITY_DIMENSIONS",
    "ErrorType",
    "LENGTH_BINS",
    "NUMERIC_ATTRIBUTES",
    "SEGMENTATION_ERRORS",
    "Score",
    "ShapeModels",
    "__version__",
    "add_quotes",
    "align_doc",
    "align_sentence",
    "bootstrap_scores",
    "bio_tag",
    "dimension_report",
    "doc_from_conll",
    "doc_from_sentences",
    "doc_from_spacy_binary",
    "doc_from_tokens",
    "fold_text",
    "load_wordlist",
    "occurrence_label",
    "read_conll",
    "recase",
    "register_extensions",
    "repair_bio",
    "repair_bio_sentences",
    "to_bio",
    "strip_quotes",
    "word_shape",
]
