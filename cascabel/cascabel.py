"""The :class:`Cascabel` facade: load data, align predictions, analyze errors."""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from spacy.tokens import Doc, Span

from cascabel.alignment import Alignment, ErrorType, align_doc
from cascabel.charlm import CharLMConfig, ShapeModels, span_text
from cascabel.dimensions import (
    CATEGORICAL_ATTRIBUTES,
    ENTITY_DIMENSIONS,
    NUMERIC_ATTRIBUTES,
    DimensionStats,
    SpanPredicate,
    dimension_report,
)
from cascabel.doc import (
    Sentence,
    bio_tag,
    doc_from_conll,
    doc_from_sentences,
    doc_from_spacy_binary,
    doc_from_tokens,
    ents_by_sentence,
    fold_text,
    load_wordlist,
    occurrence_label,
)
from cascabel.extensions import register_extensions
from cascabel.metrics import Score

Wordlist = Union[str, Path, Iterable[str], None]


def _as_wordlist(value: Wordlist, lowercase: bool = True) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (str, Path)):
        return load_wordlist(value, lowercase=lowercase)
    return {entry.lower() if lowercase else entry for entry in value}


class Cascabel:
    """Error analysis for a span-labeling (NER-style) dataset and model.

    Parameters
    ----------
    test:
        The gold-annotated evaluation ``Doc``.
    training:
        Optional training ``Doc``. Enables seen/unseen analysis and the
        character language models (surprisal, prototypicality).
    prediction:
        Optional prediction ``Doc``. Must have exactly the same tokens as
        ``test``. Enables alignments and every score.
    lexicon:
        Optional background lexicon (set of ordinary words of the language,
        or a path to a one-word-per-line file). Enables the ``in_lexicon``
        dimensions.
    gazetteer:
        Optional gazetteer of known entities (set or path). Enables the
        ``in_gazetteer`` dimension.
    shape:
        Train the character language models (:class:`~cascabel.charlm.ShapeModels`)
        on the training set (default ``True``; ignored when there is no
        training data). Training takes seconds to minutes on a CPU.
    charlm_config:
        Hyperparameters of the character language models
        (:class:`~cascabel.charlm.CharLMConfig`; defaults if omitted).
    type_confusable_exclude:
        Token texts to exclude from the token-level type-confusability check
        (typically very frequent function words).
    span_match:
        Which training occurrences of a span's word sequence count for its
        exposure attributes (``train_count``, ``label_consistency``,
        ``identity``). ``"whole"`` (default): only occurrences that are a
        whole span (any label) or entirely ``O``; occurrences inside a
        longer span (*Madrid* in *Comunidad de Madrid*) count only for the
        token layer (``token._.train_labels``). ``"any"``: every occurrence,
        inside longer spans included (the 0.6.0 behaviour). The same rule
        applies to ``variant_labels``.

    Notes
    -----
    Span and token matching against training is exact (case- and
    accent-sensitive): *MADRID*, *madrid* and *El Pais* are different
    strings from *Madrid* and *El País*, because casing can change meaning
    (*El País* vs *el país*), casing is a dimension of its own, and cased
    models see these as different inputs. Occurrences that match only once
    casing and accents are folded (:func:`cascabel.doc.fold_text`) are
    reported separately in ``span._.variant_labels`` and
    ``token._.variant_labels``, so that, e.g., all-caps headlines do not
    hide that a span was seen in ordinary casing.
    """

    def __init__(
        self,
        test: Doc,
        training: Optional[Doc] = None,
        prediction: Optional[Doc] = None,
        lexicon: Wordlist = None,
        gazetteer: Wordlist = None,
        shape: bool = True,
        charlm_config: Optional[CharLMConfig] = None,
        type_confusable_exclude: Iterable[str] = (),
        span_match: str = "whole",
    ):
        if span_match not in ("whole", "any"):
            raise ValueError(f"span_match must be 'whole' or 'any', not {span_match!r}")
        register_extensions()
        self.span_match = span_match
        self.test = test
        self.training = training
        self.prediction: Optional[Doc] = None
        self.lexicon = _as_wordlist(lexicon)
        self.gazetteer = _as_wordlist(gazetteer)
        self.shape_models: Optional[ShapeModels] = None
        self.alignments: List[Alignment] = []
        self._type_confusable_exclude: FrozenSet[str] = frozenset(type_confusable_exclude)

        self.seen_token_tags: Dict[str, Set[str]] = {}
        self.token_train_labels: Dict[str, Counter] = {}
        self.token_variant_labels: Dict[str, Dict[str, Counter]] = {}
        if training is not None and [t.text for t in test] == [t.text for t in training]:
            warnings.warn(
                "The evaluated file has the same tokens as the training file: "
                "every span will count as seen, and shape scores compare the "
                "data with models trained on it",
                stacklevel=2,
            )
        if training is not None:
            self.seen_token_tags = self._collect_seen_token_tags(training)
            self.token_train_labels = self._collect_token_labels(training)
            self.token_variant_labels = self._collect_token_variant_labels(
                self.token_train_labels
            )
            if shape:
                self.shape_models = self._fit_shape_models(training, charlm_config)

        for split in self._splits():
            self._annotate(split)
        if prediction is not None:
            self.set_prediction(prediction)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_tokens(
        cls,
        tokens: Sequence[str],
        tags: Sequence[str],
        training: Optional[Sequence] = None,
        prediction: Optional[Sequence] = None,
        scheme: str = "BIO",
        **kwargs,
    ) -> "Cascabel":
        """Build from a single sentence given as parallel token/tag lists.

        ``training`` and ``prediction`` are optional ``(tokens, tags)`` pairs.
        ``scheme`` is the tagging scheme of every input (``"BIO"``,
        ``"IOB1"`` or ``"BIOES"``; see :func:`cascabel.doc.to_bio`).
        """
        return cls(
            doc_from_tokens(tokens, tags, scheme=scheme),
            training=doc_from_tokens(*training, scheme=scheme) if training else None,
            prediction=doc_from_tokens(*prediction, scheme=scheme) if prediction else None,
            **kwargs,
        )

    @classmethod
    def from_sentences(
        cls,
        test: Iterable[Sentence],
        training: Optional[Iterable[Sentence]] = None,
        prediction: Optional[Iterable[Sentence]] = None,
        scheme: str = "BIO",
        **kwargs,
    ) -> "Cascabel":
        """Build from corpora of ``(tokens, tags)`` sentences (``scheme`` as in
        :meth:`from_tokens`)."""
        return cls(
            doc_from_sentences(test, scheme=scheme),
            training=doc_from_sentences(training, scheme=scheme) if training else None,
            prediction=doc_from_sentences(prediction, scheme=scheme) if prediction else None,
            **kwargs,
        )

    @classmethod
    def from_conll(
        cls,
        test: Union[str, Path],
        training: Union[str, Path, None] = None,
        prediction: Union[str, Path, None] = None,
        conll_options: Optional[dict] = None,
        scheme: str = "BIO",
        **kwargs,
    ) -> "Cascabel":
        """Build from CoNLL-style column files (see :func:`cascabel.doc.read_conll`;
        ``scheme`` as in :meth:`from_tokens`, e.g. ``"IOB1"`` for the original
        CoNLL-2003 files)."""
        options = {"scheme": scheme, **(conll_options or {})}
        return cls(
            doc_from_conll(test, **options),
            training=doc_from_conll(training, **options) if training else None,
            prediction=doc_from_conll(prediction, **options) if prediction else None,
            **kwargs,
        )

    @classmethod
    def from_spacy_binary(
        cls,
        test: Union[str, Path],
        training: Union[str, Path, None] = None,
        prediction: Union[str, Path, None] = None,
        **kwargs,
    ) -> "Cascabel":
        """Build from serialized spaCy ``DocBin`` (``.spacy``) files."""
        return cls(
            doc_from_spacy_binary(test),
            training=doc_from_spacy_binary(training) if training else None,
            prediction=doc_from_spacy_binary(prediction) if prediction else None,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Prediction handling
    # ------------------------------------------------------------------

    def set_prediction(self, prediction: Doc) -> None:
        """Attach (or replace) the prediction ``Doc`` and recompute alignments."""
        self._check_prediction(prediction)
        self.prediction = prediction
        self._annotate(prediction)
        self._overlay_prediction()
        self.alignments = align_doc(self.test)

    def set_prediction_from_tokens(
        self, tokens: Sequence[str], tags: Sequence[str], repair: bool = True,
        scheme: str = "BIO",
    ) -> None:
        self.set_prediction(doc_from_tokens(tokens, tags, repair=repair, scheme=scheme))

    def set_prediction_from_sentences(
        self, sentences: Iterable[Sentence], repair: bool = True,
        scheme: str = "BIO",
    ) -> None:
        self.set_prediction(doc_from_sentences(sentences, repair=repair, scheme=scheme))

    def _check_prediction(self, prediction: Doc) -> None:
        if len(prediction) != len(self.test):
            raise ValueError(
                f"Prediction has {len(prediction)} tokens but test has "
                f"{len(self.test)}; they must be token-aligned"
            )
        for i, (test_tok, pred_tok) in enumerate(zip(self.test, prediction)):
            if test_tok.text != pred_tok.text:
                raise ValueError(
                    f"Prediction and test disagree at token {i}: "
                    f"{test_tok.text!r} != {pred_tok.text!r}"
                )
        test_starts = [bool(t.is_sent_start) for t in self.test]
        pred_starts = [bool(t.is_sent_start) for t in prediction]
        if test_starts != pred_starts:
            raise ValueError(
                "Prediction and test have different sentence boundaries"
            )

    def _overlay_prediction(self) -> None:
        assert self.prediction is not None
        self.test._.ents_pred = list(self.prediction.ents)
        # Bucket predicted entities per sentence once; gold and prediction are
        # token-aligned, so a sentence shares the same start index in both.
        pred_by_sent = ents_by_sentence(self.prediction)
        for sent_gold in self.test.sents:
            sent_gold._.ents_pred = pred_by_sent.get(sent_gold.start, [])
        for tok_gold, tok_pred in zip(self.test, self.prediction):
            tok_gold._.iob_pred = bio_tag(tok_pred)

    # ------------------------------------------------------------------
    # Annotation of external information
    # ------------------------------------------------------------------

    def _splits(self) -> List[Doc]:
        return [d for d in (self.test, self.training, self.prediction) if d is not None]

    @staticmethod
    def _collect_seen_token_tags(training: Doc) -> Dict[str, Set[str]]:
        seen: Dict[str, Set[str]] = defaultdict(set)
        for tok in training:
            seen[tok.text].add(bio_tag(tok))
        return dict(seen)

    @staticmethod
    def _collect_token_labels(training: Doc) -> Dict[str, Counter]:
        """Label counts of every token text in ``training``, B/I ignored and
        ``O`` included (the token layer of identity)."""
        labels: Dict[str, Counter] = defaultdict(Counter)
        for tok in training:
            labels[tok.text][tok.ent_type_ if tok.ent_iob_ != "O" else "O"] += 1
        return dict(labels)

    @staticmethod
    def _collect_token_variant_labels(
        token_labels: Mapping[str, Counter]
    ) -> Dict[str, Dict[str, Counter]]:
        """Group the token label counts by folded text (casing and accents
        ignored): ``{"madrid": {"Madrid": Counter(...), "MADRID": ...}}``."""
        variants: Dict[str, Dict[str, Counter]] = defaultdict(dict)
        for text, labels in token_labels.items():
            variants[fold_text(text)][text] = labels
        return dict(variants)

    @staticmethod
    def _is_whole_occurrence(tags: Sequence[str], start: int, end: int) -> bool:
        """True if ``tags[start:end]`` is entirely ``O`` or exactly one whole
        span: it starts the span, and the span neither continues inside nor
        past the window."""
        window = tags[start:end]
        if all(tag == "O" for tag in window):
            return True
        if not window[0].startswith("B-"):
            return False
        inside = "I-" + window[0][2:]
        if any(tag != inside for tag in window[1:]):
            return False
        return end == len(tags) or not tags[end].startswith("I-")

    @staticmethod
    def _collect_sequence_tags(
        training: Doc, sequences: Set[Tuple[str, ...]], span_match: str = "whole"
    ) -> Dict[Tuple[str, ...], Counter]:
        """Count, for each word sequence in ``sequences``, the BIO tag
        sequences it carries at its occurrences in ``training`` (any split
        can be passed).

        Matching is case-sensitive and never crosses a sentence boundary.
        With ``span_match="whole"`` an occurrence counts only if it is a
        whole span or entirely ``O``; with ``"any"`` occurrences inside a
        longer span (``I-``) or partly ``O`` count too. Either way a
        sequence seen only as ``O`` has a non-empty counter without its gold
        tags.
        """
        found: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        lengths = {len(seq) for seq in sequences}
        whole_only = span_match == "whole"
        for sent in training.sents:
            words = [tok.text for tok in sent]
            tags = [bio_tag(tok) for tok in sent]
            for length in lengths:
                for i in range(len(words) - length + 1):
                    key = tuple(words[i:i + length])
                    if key not in sequences:
                        continue
                    if whole_only and not Cascabel._is_whole_occurrence(tags, i, i + length):
                        continue
                    found[key][tuple(tags[i:i + length])] += 1
        return dict(found)

    @staticmethod
    def _collect_variant_tags(
        training: Doc, folded: Set[Tuple[str, ...]], span_match: str = "whole"
    ) -> Dict[Tuple[str, ...], Counter]:
        """Like :meth:`_collect_sequence_tags`, but matching on folded words
        (:func:`cascabel.doc.fold_text`). Returns, per folded sequence, a
        counter of ``(exact words, BIO tags)`` pairs, so that callers can
        drop the occurrences written exactly like the span."""
        found: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        lengths = {len(seq) for seq in folded}
        whole_only = span_match == "whole"
        for sent in training.sents:
            words = [tok.text for tok in sent]
            folded_words = [fold_text(word) for word in words]
            tags = [bio_tag(tok) for tok in sent]
            for length in lengths:
                for i in range(len(words) - length + 1):
                    key = tuple(folded_words[i:i + length])
                    if key not in folded:
                        continue
                    if whole_only and not Cascabel._is_whole_occurrence(tags, i, i + length):
                        continue
                    found[key][(tuple(words[i:i + length]), tuple(tags[i:i + length]))] += 1
        return dict(found)

    @staticmethod
    def _fit_shape_models(
        training: Doc, config: Optional[CharLMConfig]
    ) -> Optional[ShapeModels]:
        background = [t.text for t in training if t.ent_iob_ == "O"]
        if not background:
            return None
        label_spans = [
            (ent.label_, span_text([t.text for t in ent])) for ent in training.ents
        ]
        return ShapeModels(background, label_spans, config=config)

    def _annotate(self, split: Doc) -> None:
        """Fill the externally-informed extensions on one split."""
        if self.seen_token_tags:
            self._annotate_training(split)
        # Shape scores are left empty on the training split: its strings are
        # what the character LMs were trained on, so their scores would
        # reflect memorization, not typicality.
        if self.shape_models is not None and split is not self.training:
            self._annotate_shape(split)
        if self.lexicon or self.gazetteer:
            self._annotate_wordlists(split)

    def _annotate_training(self, split: Doc) -> None:
        for tok in split:
            unseen = tok.text not in self.seen_token_tags
            tok._.is_unseen_token = unseen
            tok._.is_unseen_tagging = (
                False if unseen else bio_tag(tok) not in self.seen_token_tags[tok.text]
            )
            tok._.is_type_confusable = (
                not unseen
                and tok.text.isalpha()
                and tok.text not in self._type_confusable_exclude
                and len(tok.text) > 2
                and len(self.seen_token_tags[tok.text]) > 1
            )
            tok._.train_labels = dict(
                self.token_train_labels.get(tok.text, Counter()).most_common()
            )
            variant_labels: Counter = Counter()
            for text, labels in self.token_variant_labels.get(fold_text(tok.text), {}).items():
                if text != tok.text:
                    variant_labels.update(labels)
            tok._.variant_labels = dict(variant_labels.most_common())
        assert self.training is not None
        entities = list(split.ents)
        sequences = {tuple(tok.text for tok in ent) for ent in entities}
        found = self._collect_sequence_tags(self.training, sequences, self.span_match)
        folded = {tuple(fold_text(word) for word in seq) for seq in sequences}
        variants = self._collect_variant_tags(self.training, folded, self.span_match)
        for ent in entities:
            words = tuple(tok.text for tok in ent)
            variant_tags: Counter = Counter()
            for (exact, tags), n in variants.get(tuple(map(fold_text, words)), Counter()).items():
                if exact != words:
                    variant_tags[tags] += n
            ent._.variant_labels = dict(self._occurrence_labels(variant_tags).most_common())
            labels = self._occurrence_labels(
                found.get(tuple(tok.text for tok in ent), Counter())
            )
            count = sum(labels.values())
            ent._.train_count = count
            ent._.train_labels = dict(labels.most_common())
            ent._.label_consistency = labels[ent.label_] / count if count else None
            if count:
                majority, majority_count = labels.most_common(1)[0]
                ent._.majority_share = majority_count / count
                ent._.majority_label = majority
            else:
                ent._.majority_share = ent._.majority_label = None

    @staticmethod
    def _occurrence_labels(tag_sequences: Counter) -> Counter:
        """Collapse BIO tag sequences into span labels (B/I ignored, ``O`` and
        ``MIXED`` included; see :func:`cascabel.doc.occurrence_label`)."""
        labels: Counter = Counter()
        for tags, n in tag_sequences.items():
            labels[occurrence_label(tags)] += n
        return labels

    def _annotate_shape(self, split: Doc) -> None:
        """Token surprisal, span typicality and prototypicality / label fit.

        Spans are scored under *their own* label: gold labels on the test and
        training splits, predicted labels on the prediction split.
        """
        assert self.shape_models is not None
        tokens = list(split)
        unique = sorted({tok.text for tok in tokens})
        surprisal = dict(zip(unique, self.shape_models.surprisal(unique)))
        for tok in tokens:
            tok._.surprisal = surprisal[tok.text]
        entities = list(split.ents)
        labelled = [(ent.label_, span_text([t.text for t in ent])) for ent in entities]
        scores = self.shape_models.prototypicality(labelled)
        span_like = self.shape_models.span_typicality([text for _, text in labelled])
        for ent, (typicality, fit, other), any_span in zip(entities, scores, span_like):
            ent._.span_typicality = any_span
            ent._.prototypicality = typicality
            ent._.label_fit = fit
            ent._.closest_other_label = other

    def _annotate_wordlists(self, split: Doc) -> None:
        gazetteer_tokens = {
            token for entry in self.gazetteer for token in entry.split()
        }
        for tok in split:
            if self.lexicon:
                tok._.is_in_lexicon = tok.text.lower() in self.lexicon
            if self.gazetteer:
                tok._.is_in_gazetteer = tok.text.lower() in gazetteer_tokens
        if self.gazetteer:
            for ent in split.ents:
                ent._.is_in_gazetteer = ent.text.lower() in self.gazetteer

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def _require_prediction(self) -> None:
        if self.prediction is None:
            raise RuntimeError(
                "No prediction attached; pass prediction=... or call set_prediction()"
            )

    def span_scores(self) -> Score:
        """Strict span-level precision/recall/F1 over all alignments."""
        self._require_prediction()
        return Score.aggregate(a.span_score for a in self.alignments)

    def token_scores(self) -> Score:
        """Token-level micro precision/recall/F1 over full BIO tags (non-O)."""
        self._require_prediction()
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tok in self.test:
            gold = bio_tag(tok)
            pred = tok._.iob_pred
            if pred == gold:
                counts[gold]["tp"] += 1
            else:
                counts[gold]["fn"] += 1
                counts[pred]["fp"] += 1
        total = Score()
        for tag, tag_counts in counts.items():
            if tag == "O":
                continue
            total = total + Score(
                tp=tag_counts["tp"], fp=tag_counts["fp"], fn=tag_counts["fn"]
            )
        return total

    def boundary_scores(self) -> Score:
        """Span precision/recall/F1 with labels ignored: a predicted span is
        correct when its start and end equal a gold span's (unlabelled exact
        match)."""
        self._require_prediction()
        return Score.aggregate(a.boundary_score for a in self.alignments)

    def scores_by_label(self) -> Dict[str, Score]:
        """Span-level scores broken down by entity label.

        Each gold span involved in an error contributes one FN to *its own*
        label; each predicted span one FP to *its own* label. Summing over
        labels therefore reproduces :meth:`span_scores` exactly, even for
        alignments that mix labels (e.g. a fusion of two differently-labeled
        entities).
        """
        self._require_prediction()
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
        for alignment in self.alignments:
            if alignment.error_type is None:
                counts[alignment.gold[0].label_][0] += 1
            else:
                for gold_span in alignment.gold:
                    counts[gold_span.label_][2] += 1
                for predicted_span in alignment.predicted:
                    counts[predicted_span.label_][1] += 1
        return {
            label: Score(tp=tp, fp=fp, fn=fn)
            for label, (tp, fp, fn) in sorted(counts.items())
        }

    def errors(self) -> List[Alignment]:
        """All alignments that represent an error."""
        self._require_prediction()
        return [a for a in self.alignments if a.is_error]

    def error_counts(self) -> Counter:
        """Frequency of each error type (keys are ``ErrorType`` values)."""
        return Counter(a.error_type.value for a in self.errors())

    def retrieval_capacity(self) -> float:
        """Fraction of gold entity tokens labeled as entity (any label) by the model."""
        self._require_prediction()
        gold_entity_tokens = [t for t in self.test if t.ent_iob_ != "O"]
        if not gold_entity_tokens:
            return 0.0
        retrieved = [t for t in gold_entity_tokens if t._.iob_pred != "O"]
        return len(retrieved) / len(gold_entity_tokens)

    def noise_level(self) -> float:
        """Fraction of gold O tokens labeled as entity by the model."""
        self._require_prediction()
        background_tokens = [t for t in self.test if t.ent_iob_ == "O"]
        if not background_tokens:
            return 0.0
        noisy = [t for t in background_tokens if t._.iob_pred != "O"]
        return len(noisy) / len(background_tokens)

    def segmentation_error_rate(self) -> float:
        """Segmentation errors (fused/split/missegmented) per gold entity."""
        self._require_prediction()
        if not list(self.test.ents):
            return 0.0
        segmentation_errors = [a for a in self.alignments if a.is_segmentation_error]
        return len(segmentation_errors) / len(list(self.test.ents))

    def segmentation_success_rate(self) -> float:
        """Fraction of gold entities adjacent to another entity that were
        nevertheless matched exactly."""
        self._require_prediction()
        adjacent = [
            a for a in self.alignments if a.gold and any(g._.is_adjacent for g in a.gold)
        ]
        if not adjacent:
            return 0.0
        return len([a for a in adjacent if not a.is_error]) / len(adjacent)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def dimension_report(self, split: str = "test") -> Dict[str, DimensionStats]:
        """Profile the entities of one split (``"test"``, ``"training"`` or
        ``"prediction"``) along every available dimension."""
        doc = getattr(self, split, None)
        if not isinstance(doc, Doc):
            raise ValueError(f"No such split: {split!r}")
        return dimension_report(doc)

    def scores_by_dimension(
        self,
        dimensions: Optional[Dict[str, SpanPredicate]] = None,
        side: str = "gold",
    ) -> Dict[str, Score]:
        """Span-level scores per dimension. Shows where the model fails.

        Parameters
        ----------
        dimensions : dict, optional
            Dimension predicates; defaults to ``ENTITY_DIMENSIONS``.
        side : {"gold", "predicted"}
            ``"gold"`` (recall side): alignments whose *gold* entity
            expresses the dimension. Spurious predictions have no gold
            entity and are not attributed to any bucket, so read recall
            here, not precision. ``"predicted"`` (precision side): every
            *predicted* span expressing the dimension (dimensions evaluated
            with the predicted label), counted as a true positive when its
            alignment is correct and as a false positive otherwise; only
            precision is meaningful (``fn`` is 0).

        Returns
        -------
        dict
            Dimension name to :class:`~cascabel.metrics.Score`; dimensions
            inapplicable for every span of that side are omitted.
        """
        self._require_prediction()
        dimensions = dimensions if dimensions is not None else ENTITY_DIMENSIONS
        if side == "predicted":
            return self._precision_by_dimension(dimensions)
        if side != "gold":
            raise ValueError(f"side must be 'gold' or 'predicted', not {side!r}")
        result: Dict[str, Score] = {}
        for name, predicate in dimensions.items():
            values: List[Score] = []
            applicable = False  # did any gold entity yield a non-None flag?
            for alignment in self.alignments:
                if not alignment.gold:
                    continue
                flags = [predicate(g) for g in alignment.gold]
                if any(flag is not None for flag in flags):
                    applicable = True
                if any(flag for flag in flags):
                    values.append(alignment.span_score)
            # Drop the dimension only when it is inapplicable for *every* gold
            # entity (e.g. casing on a caseless script), never because a single
            # caseless entity (a number, a symbol) appears in a cased corpus.
            if not applicable or not values:
                continue
            result[name] = Score.aggregate(values)
        return result

    def _precision_by_dimension(
        self, dimensions: Mapping[str, SpanPredicate]
    ) -> Dict[str, Score]:
        outcomes = self.span_outcomes("predicted")
        result: Dict[str, Score] = {}
        for name, predicate in dimensions.items():
            flags = [(predicate(span), outcome) for span, outcome in outcomes]
            if all(flag is None for flag, _ in flags):
                continue
            selected = [outcome for flag, outcome in flags if flag]
            if not selected:
                continue
            correct = sum(outcome is None for outcome in selected)
            result[name] = Score(tp=correct, fp=len(selected) - correct, fn=0)
        return result

    # ------------------------------------------------------------------
    # Two-sided error profiling
    # ------------------------------------------------------------------

    def span_outcomes(self, side: str = "gold") -> List[Tuple[Span, Optional[str]]]:
        """Every gold or predicted span with the outcome of its alignment.

        Parameters
        ----------
        side : {"gold", "predicted"}
            ``"gold"``: the gold spans of the test set (recall side: missed,
            boundary and type errors). ``"predicted"``: the spans of the
            prediction (precision side: spurious, boundary and type errors),
            carrying their *predicted* labels. Boundary and type errors
            appear on both sides, each side with its own span.

        Returns
        -------
        list of (Span, str or None)
            The outcome is ``None`` for a correct span and the error type
            value (e.g. ``"missed"``, ``"overlap_missed"``) otherwise.
        """
        self._require_prediction()
        if side not in ("gold", "predicted"):
            raise ValueError(f"side must be 'gold' or 'predicted', not {side!r}")
        result: List[Tuple[Span, Optional[str]]] = []
        for alignment in self.alignments:
            outcome = None if alignment.error_type is None else alignment.error_type.value
            spans = alignment.gold if side == "gold" else alignment.predicted
            result.extend((span, outcome) for span in spans)
        return result

    def error_profile(
        self,
        side: str = "gold",
        dimensions: Optional[Mapping[str, SpanPredicate]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """How the spans behind errors differ from correctly handled spans.

        For each dimension, counts the spans of one side (see
        :meth:`span_outcomes`) that express it, split into correct and
        erroneous spans and by error type, and compares two shares: the
        share of *correct* spans that express the dimension and the share
        of *erroneous* spans that do. A dimension is linked to errors only
        when the two shares differ; a share among errors means little on its
        own.

        Parameters
        ----------
        side : {"gold", "predicted"}
            Recall side (gold spans) or precision side (predicted spans).
        dimensions : dict, optional
            Dimension predicates; defaults to ``ENTITY_DIMENSIONS``.

        Returns
        -------
        dict
            Dimension name to a dict with ``n`` (spans expressing it),
            ``correct``, ``errors``, one count per error type,
            ``share_of_correct`` and ``share_of_errors`` (percentages over
            the spans for which the dimension is applicable) and ``rate``
            (recall on the gold side, precision on the predicted side,
            within the dimension). Dimensions inapplicable for every span
            are omitted.
        """
        dimensions = dimensions if dimensions is not None else ENTITY_DIMENSIONS
        outcomes = self.span_outcomes(side)
        error_types = sorted({o for _, o in outcomes if o is not None})
        profile: Dict[str, Dict[str, float]] = {}
        for name, predicate in dimensions.items():
            flags = [(predicate(span), outcome) for span, outcome in outcomes]
            applicable = [(flag, outcome) for flag, outcome in flags if flag is not None]
            if not applicable:
                continue
            all_correct = sum(outcome is None for _, outcome in applicable)
            all_errors = len(applicable) - all_correct
            selected = Counter(outcome for flag, outcome in applicable if flag)
            n = sum(selected.values())
            correct = selected[None]
            row: Dict[str, float] = {"n": n, "correct": correct, "errors": n - correct}
            for error_type in error_types:
                row[error_type] = selected[error_type]
            row["share_of_correct"] = 100 * correct / all_correct if all_correct else 0.0
            row["share_of_errors"] = (
                100 * (n - correct) / all_errors if all_errors else 0.0
            )
            row["rate"] = correct / n if n else 0.0
            profile[name] = row
        return profile

    def span_table(self, split: str = "test") -> List[Dict[str, object]]:
        """One record per span of a split, with every continuous and
        categorical attribute: the basis for analyses and figures that use
        the numbers themselves rather than binary dimensions.

        Parameters
        ----------
        split : {"test", "training", "prediction"}
            Which spans to describe. With a prediction attached, test and
            prediction records also carry ``outcome`` (``"correct"`` or the
            error type), so both sides of :meth:`error_profile` can be
            analysed continuously.

        Returns
        -------
        list of dict
            Keys: ``sentence_start``, ``start``, ``end``, ``text``,
            ``label``, the names in ``NUMERIC_ATTRIBUTES`` and
            ``CATEGORICAL_ATTRIBUTES``, and ``outcome`` when available.
        """
        doc = getattr(self, split, None)
        if not isinstance(doc, Doc):
            raise ValueError(f"No such split: {split!r}")
        outcomes: Dict[Tuple[int, int, str], str] = {}
        if self.prediction is not None and split in ("test", "prediction"):
            side = "gold" if split == "test" else "predicted"
            for span, outcome in self.span_outcomes(side):
                outcomes[(span.start, span.end, span.label_)] = outcome or "correct"
        rows: List[Dict[str, object]] = []
        for ent in doc.ents:
            row: Dict[str, object] = {
                "sentence_start": ent.sent.start,
                "start": ent.start,
                "end": ent.end,
                "text": ent.text,
                "label": ent.label_,
            }
            for name in NUMERIC_ATTRIBUTES + CATEGORICAL_ATTRIBUTES:
                row[name] = getattr(ent._, name)
            if outcomes:
                row["outcome"] = outcomes.get((ent.start, ent.end, ent.label_))
            rows.append(row)
        return rows

    def extreme_spans(
        self,
        measure: str,
        n: int = 10,
        least: bool = False,
        label: Optional[str] = None,
        unseen_only: bool = False,
        split: str = "test",
    ) -> List[Dict[str, object]]:
        """The distinct spans with the highest (or lowest) value of a measure.

        For the shape scores (bits per character) the highest values are the
        *least* typical: ``extreme_spans("prototypicality")`` gives the
        spans that look least like their label, and
        ``extreme_spans("surprisal_mean", least=True)`` the spans that look
        most like ordinary ``O`` tokens. ``extreme_spans("majority_share",
        least=True)`` gives the most ambiguous spans: no tagging accounts for
        most of their training occurrences. Ties are broken by training
        count (more evidence first).

        Parameters
        ----------
        measure : str
            A name in ``NUMERIC_ATTRIBUTES`` (e.g. ``"prototypicality"``,
            ``"span_typicality"``, ``"surprisal_mean"``, ``"label_fit"``).
        n : int
            How many spans to return.
        least : bool
            Return the lowest values instead of the highest.
        label : str, optional
            Only spans with this label.
        unseen_only : bool
            Only spans never seen in training (``train_count == 0``).
        split : {"test", "training", "prediction"}
            Which spans to search (see :meth:`span_table`).

        Returns
        -------
        list of dict
            Rows of :meth:`span_table`, sorted from the most extreme value,
            one per distinct ``(text, label)`` (the first occurrence) with an
            added ``occurrences`` key: how often that span occurs in the
            split. Spans without a value for ``measure`` are skipped.
        """
        if measure not in NUMERIC_ATTRIBUTES:
            raise ValueError(
                f"Unknown measure {measure!r}; choose one of {NUMERIC_ATTRIBUTES}"
            )
        rows = [
            row for row in self.span_table(split)
            if row[measure] is not None
            and (label is None or row["label"] == label)
            and (not unseen_only or row["train_count"] == 0)
        ]
        distinct: Dict[Tuple[object, object], Dict[str, object]] = {}
        for row in rows:
            key = (row["text"], row["label"])
            if key in distinct:
                distinct[key]["occurrences"] += 1
            else:
                distinct[key] = {**row, "occurrences": 1}
        # Ties (e.g. many spans with majority_share 0.5) put the spans with
        # more training evidence first.
        ranked = sorted(
            distinct.values(),
            key=lambda row: (
                row[measure] if least else -row[measure],
                -(row["train_count"] or 0),
            ),
        )
        return ranked[:n]

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """Every available analysis as one JSON-serializable dict.

        Includes the dataset profile (counts, dimensions)
        and, when a prediction is attached, all scores, per-label and
        per-dimension breakdowns, error counts and diagnostics.
        """
        from cascabel.report import build_report

        return build_report(self)

    def summary(self) -> str:
        """The report rendered as an aligned plain-text summary."""
        from cascabel.report import build_summary

        return build_summary(self)

    def bootstrap(
        self,
        n: int = 1000,
        alpha: float = 0.05,
        seed: int = 0,
        dimensions: Optional[Dict[str, SpanPredicate]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Sentence-level percentile bootstrap confidence intervals for the
        span/token/boundary F1 and every applicable per-dimension F1 and
        recall (see :func:`cascabel.bootstrap.bootstrap_scores`)."""
        from cascabel.bootstrap import bootstrap_scores

        return bootstrap_scores(self, n=n, alpha=alpha, seed=seed,
                                dimensions=dimensions)

    def card(
        self,
        path: Union[str, Path, None] = None,
        dataset: str = "unspecified",
        split: str = "test",
        system: Optional[str] = None,
        notes: Optional[str] = None,
        bootstrap: bool = True,
        bootstrap_n: int = 1000,
    ) -> Union[str, Path]:
        """Render a standardized markdown documentation card.

        Without a prediction this is a **dataset profile card**; with one it
        is a **performance profile card** (scores, error taxonomy, per-dimension
        performance, auto-generated caveats). When a prediction is attached,
        95% sentence-level bootstrap CIs are attached to the F1 columns
        (``bootstrap=False`` to skip; ``bootstrap_n`` controls resamples).
        Returns the markdown string, or writes it to ``path``.
        """
        from cascabel.card import build_card, write_card

        ci = None
        if bootstrap and self.prediction is not None:
            ci = self.bootstrap(n=bootstrap_n)
        if path is None:
            return build_card(self, dataset=dataset, split=split,
                              system=system, notes=notes, ci=ci)
        return write_card(self, path, dataset=dataset, split=split,
                          system=system, notes=notes, ci=ci)

    def to_html(
        self, path: Union[str, Path, None] = None, only_errors: bool = True
    ) -> Union[str, Path]:
        """Render a self-contained HTML error report.

        Returns the HTML string, or writes it to ``path`` (returning the
        path) when one is given. ``only_errors`` restricts the annotated
        sentence view to sentences containing at least one error.
        """
        from cascabel.report import build_html, write_html

        if path is None:
            return build_html(self, only_errors=only_errors)
        return write_html(self, path, only_errors=only_errors)

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def identity_distribution(self, split: str = "test") -> Counter:
        """Span occurrences of a split per identity group (``"unseen"``,
        ``"seen_same_label"``, ``"seen_ambiguous"``, ``"seen_other_label"``;
        see :func:`cascabel.extensions.identity`). Empty without training."""
        return Counter(
            ent._.identity for ent in self._split(split).ents
            if ent._.identity is not None
        )

    def identity_summary(self, split: str = "test") -> Dict[str, object]:
        """Identity proportions over the span occurrences of a split.

        The radar identity axes are ``unseen_share``, ``multi_label_share``
        and ``unseen_label_share``.

        - ``unseen_share``: spans never seen in training (see ``span_match``),
          out of all spans;
        - ``multi_label_share``: seen spans that occur in training with more
          than one label (``O`` counts as a label), out of the seen spans;
        - ``unseen_label_share``: seen spans that never occur in training
          with their own label (``label_consistency == 0``: only other
          labels or ``O``), out of the seen spans. Overlaps with
          ``multi_label_share`` (*Madrid* ``LOC``/``ORG`` in training,
          ``PER`` in test counts in both);
        - ``unseen_familiar_word_share``: unseen spans with at least one word
          seen *inside a span* in training (any label; words seen only as
          ``O`` do not count), out of the unseen spans;
        - ``ambiguous_share``: seen spans whose occurrences are not all of
          their own label (``label_consistency < 1``: several labels, only
          other labels, or ``O``), out of the seen spans;
        - ``unseen_variant_share``: unseen spans that occur in training
          written differently (casing or accents, ``variant_labels``), out
          of the unseen spans.

        Returns the counts too (``n_spans``, ``n_unseen``,
        ``n_unseen_familiar_word``, ``n_seen``, ``n_multi_label``,
        ``n_unseen_label``, ``n_ambiguous``, ``n_unseen_variant``). Shares
        are ``None`` when their denominator is 0; everything is empty
        without a training set.
        """
        entities = [ent for ent in self._split(split).ents if ent._.train_count is not None]
        unseen = [ent for ent in entities if ent._.train_count == 0]
        seen = [ent for ent in entities if ent._.train_count]
        familiar = sum(bool(ent._.has_word_seen_in_span) for ent in unseen)
        multi_label = sum(len(ent._.train_labels) > 1 for ent in seen)
        unseen_label = sum(ent._.label_consistency == 0 for ent in seen)
        ambiguous = sum(ent._.label_consistency < 1 for ent in seen)
        variant = sum(bool(ent._.variant_labels) for ent in unseen)

        def share(part: int, whole: int) -> Optional[float]:
            return part / whole if whole else None

        if not entities:
            return {}
        return {
            "split": split,
            "span_match": self.span_match,
            "n_spans": len(entities),
            "n_unseen": len(unseen),
            "unseen_share": share(len(unseen), len(entities)),
            "n_unseen_familiar_word": familiar,
            "unseen_familiar_word_share": share(familiar, len(unseen)),
            "n_seen": len(seen),
            "n_multi_label": multi_label,
            "multi_label_share": share(multi_label, len(seen)),
            "n_unseen_label": unseen_label,
            "unseen_label_share": share(unseen_label, len(seen)),
            "n_ambiguous": ambiguous,
            "ambiguous_share": share(ambiguous, len(seen)),
            "n_unseen_variant": variant,
            "unseen_variant_share": share(variant, len(unseen)),
        }

    def label_ambiguity(self, split: str = "training") -> Dict[str, object]:
        """How many distinct entities of a split carry more than one label
        *within that split*.

        Every distinct span text of the split (*Madrid*) is looked up at all
        its occurrences in the same split, ``O`` included, and its labels are
        counted with B/I ignored (see :func:`cascabel.doc.occurrence_label`).
        An entity is multi-label as soon as two labels occur; the proportions
        are kept so that lopsided cases (*Germany*: ``LOC`` 141, ``ORG`` 2)
        can be told apart from balanced ones.

        Parameters
        ----------
        split : {"training", "test", "prediction"}
            The split to describe (default: the training set).

        Returns
        -------
        dict
            ``n_entities`` (distinct span texts), ``n_multi_label``,
            ``multi_label_share`` (0--1) and ``entities``: one dict per
            distinct text with ``text``, ``occurrences``, ``labels`` (label →
            count, most frequent first), ``majority_label`` and
            ``majority_share``, sorted from the most ambiguous (lowest
            majority share; more occurrences first on ties).
        """
        doc = self._split(split)
        sequences = {tuple(tok.text for tok in ent) for ent in doc.ents}
        found = self._collect_sequence_tags(doc, sequences, self.span_match)
        entities: List[Dict[str, object]] = []
        for sequence, tag_sequences in found.items():
            labels = self._occurrence_labels(tag_sequences)
            total = sum(labels.values())
            majority, majority_count = labels.most_common(1)[0]
            entities.append({
                "text": " ".join(sequence),
                "occurrences": total,
                "labels": dict(labels.most_common()),
                "majority_label": majority,
                "majority_share": majority_count / total,
            })
        entities.sort(key=lambda e: (e["majority_share"], -e["occurrences"], e["text"]))
        n_multi = sum(len(e["labels"]) > 1 for e in entities)
        return {
            "split": split,
            "n_entities": len(entities),
            "n_multi_label": n_multi,
            "multi_label_share": n_multi / len(entities) if entities else 0.0,
            "entities": entities,
        }

    def _split(self, split: str) -> Doc:
        doc = getattr(self, split, None)
        if not isinstance(doc, Doc):
            raise ValueError(f"No such split: {split!r}")
        return doc

    def label_distribution(self, split: str = "test") -> Counter:
        doc = getattr(self, split)
        return Counter(ent.label_ for ent in doc.ents)

    def span_length_distribution(self, split: str = "test") -> Counter:
        doc = getattr(self, split)
        return Counter(len(ent) for ent in doc.ents)

    def spans_per_sentence_distribution(self, split: str = "test") -> Counter:
        doc = getattr(self, split)
        by_sent = ents_by_sentence(doc)
        return Counter(len(by_sent.get(sent.start, [])) for sent in doc.sents)

    def n_entities(self, split: str = "test") -> int:
        return len(list(getattr(self, split).ents))

    def n_sentences(self, split: str = "test") -> int:
        return len(list(getattr(self, split).sents))

    def n_tokens(self, split: str = "test") -> int:
        return len(getattr(self, split))
