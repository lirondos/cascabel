Dimensions
==========

A *dimension* is a boolean property of an entity span. Profiling a dataset
along dimensions shows what makes it hard; crossing dimensions with scores
shows where a model fails.

Many properties are really numbers (length, training count, surprisal,
prototypicality). They are stored on each span and listed by
:meth:`~cascabel.Cascabel.span_table` (see `Continuous attributes`_); the
dimensions below are binary *views* for counting and scoring. Bins are fixed
and shared across datasets (never calibrated per dataset), so percentages
are comparable between datasets.

Available dimensions
--------------------

**Surface dimensions** (always available):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Dimension
     - The entity...
   * - ``length_1``, ``length_2``, ``length_3``, ``length_4_5``, ``length_6plus``
     - spans that many tokens (fixed bins, ``LENGTH_BINS``)
   * - ``sentence_initial``
     - starts the sentence (leading punctuation is ignored)
   * - ``casing_upper``
     - every cased letter is uppercase (*NATO*, *3M*)
   * - ``casing_title``
     - first cased letter uppercase, whatever follows (*Bank of England*,
       *McDonald*, *New york*, *Ñúñez*)
   * - ``casing_lower``
     - no uppercase letter (*iphone*)
   * - ``casing_mixed``
     - first cased letter lowercase, a later one uppercase (*iPhone*,
       *eBay Inc.*)
   * - ``quoted``
     - is directly enclosed in quotation marks
   * - ``adjacent``
     - is immediately preceded by another entity

**Exposure dimensions** (require ``training=``). The span's exact word
sequence (case-sensitive, within one sentence) is looked up in training,
and an occurrence counts only if it is a **whole span** (any label) or
**entirely** ``O``: *Madrid* alone as ``B-LOC`` counts as ``LOC``, *Madrid*
outside any span counts as ``O``, but *Madrid* inside *Comunidad de Madrid*
or *Real Madrid* does not count for the span *Madrid* (it shows up in the
token layer below). Labels ignore B/I. ``Cascabel(..., span_match="any")``
restores the 0.6.0 behaviour, where occurrences inside longer spans count
too (and partly-``O`` or multi-label occurrences are ``MIXED``). Stored on
each span:

- ``span._.train_count``: how many times the sequence occurs;
- ``span._.train_labels``: label counts, most frequent first, e.g.
  ``{"LOC": 14, "ORG": 5, "O": 1}`` (empty when unseen);
- ``span._.label_consistency``: the share with the span's *own* label
  (0--1; ``None`` when unseen);
- ``span._.majority_label`` / ``span._.majority_share``: the most frequent
  label, whatever it is, and its share. A span is truly ambiguous when no
  label has the majority (``majority_share`` at or below 0.5): *Germany*,
  ``LOC`` 141 and ``ORG`` 2, has ``label_consistency`` < 1 but
  ``majority_share`` 0.99. ``casc.extreme_spans("majority_share",
  least=True)`` lists the most ambiguous spans.

The four dimensions are views on these numbers (``span._.identity``;
``casc.identity_distribution()`` counts them):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Dimension
     - The entity...
   * - ``unseen``
     - never occurs in training as a whole span or as ``O``, with any
       label (``train_count == 0``)
   * - ``seen_same_label``
     - always occurs with its own label (``label_consistency == 1``),
       e.g. *Germany* seen only as ``LOC``
   * - ``seen_ambiguous``
     - occurs with its own label and with others (``0 < label_consistency
       < 1``), e.g. *Washington* seen as ``PER`` and ``LOC``
   * - ``seen_other_label``
     - occurs, but never with its own label (``label_consistency == 0``):
       only as ``O`` (*Apple* the fruit) or with other labels

**Token layer.** Every token also records how its text was seen in
training, B/I ignored and ``O`` included: ``token._.train_labels`` (e.g.
``{"ORG": 38, "O": 12, "LOC": 3}``) and ``token._.exposure`` (``"in_span"``:
at least once inside a span; ``"only_O"``; ``"unseen"``). Per span,
``span._.word_exposure`` counts its tokens in each group (``{"in_span": 2,
"only_O": 0, "unseen": 1}`` for *Bank of Tokyo-Mitsubishi*) and
``span._.has_word_seen_in_span`` tells whether any of them was seen inside a
span. Function words are included, so *of* in *Bank of England* counts. For a
one-word span the token layer covers *all* occurrences of the word, while the
span layer only covers whole-span and ``O`` occurrences: *Yastrzhembsky*,
seen only inside *Sergei Yastrzhembsky*, is ``unseen`` with a word seen in a
span.

**Spelling variants (casing and accents).** All matching above is exact:
*MADRID*, *madrid* or *El Pais* in training are not occurrences of *Madrid*
or *El País*. This is deliberate. Casing can change meaning (*El País*, the
newspaper, vs *el país*, "the country"); casing is already a dimension of its
own (``casing``); and cased models see these strings
as different inputs. But exact matching alone would make, e.g., *Japan*
look unseen when training only has it in all-caps headlines (*SOCCER -
JAPAN GET LUCKY WIN*), or *El País* unseen when training writes *El Pais*.
So occurrences that match only after folding casing and accents
(:func:`cascabel.doc.fold_text`: *EL PAÍS* → *el pais*) are reported
separately, without changing ``identity``: ``span._.variant_labels`` (label
counts of those occurrences, same whole-span rule as ``train_labels``) and
``token._.variant_labels`` (the same per token). Folding removes every
diacritic except the tilde of *ñ*, a letter of its own in Spanish: *Peña*
and *Pena*, or *año* and *ano*, are not variants of each other.

**Identity summary.** ``casc.identity_summary("test")`` gives shares over
span occurrences. The three radar axes are ``unseen_share`` (never seen in
training, out of all spans), ``multi_label_share`` (seen in training with
more than one label, ``O`` included, out of the seen spans) and
``unseen_label_share`` (seen, but never with the span's own label, out of
the seen spans). The last two overlap: *Madrid* seen as ``LOC`` and ``ORG``
and tagged ``PER`` in test counts in both. Denominators are the seen spans
so that these axes do not shrink just because a dataset has many new spans.
Also reported: ``unseen_familiar_word_share`` (unseen spans with a word seen inside a span,
out of the unseen spans) and ``ambiguous_share`` (``label_consistency < 1``,
out of the seen spans), with the underlying counts. ``unseen_variant_share``
adds the unseen spans seen in training as a spelling variant, out of the
unseen spans.

**Label ambiguity within one split.** ``casc.label_ambiguity("training")``
looks up every distinct span text of the training set at all its
occurrences *in the training set* and reports how many distinct entities
carry more than one label (``n_multi_label``, ``multi_label_share``), with
each entity's label counts so that lopsided cases (95/5) can be told apart
from balanced ones (50/50). Any split can be passed.

Casing is one category per span (``span._.casing``) and describes the span
*as written*, independently of its position: a sentence-initial span is
classified like any other, and position is the separate
``sentence_initial`` dimension. The first cased letter decides the
category.

**Shape** (surprisal, prototypicality, label fit; see :doc:`shape`) is
continuous only: no binary view until fixed cross-dataset bins are chosen.

**Word-list dimensions** are *optional user resources*: they appear only
when a ``lexicon=`` / ``gazetteer=`` is passed. ``in_lexicon`` (every
alphabetic token is also an ordinary word of the language),
``partial_in_lexicon`` (some but not all are), ``in_gazetteer``. Spans
without alphabetic tokens are *n/a* for the lexicon dimensions.

Continuous attributes
---------------------

.. code-block:: python

   >>> rows = casc.span_table("test")     # or "training", "prediction"
   >>> rows[0]
   {'text': 'Phil Simmons', 'label': 'PER', 'length': 2, 'train_count': 0,
    'label_consistency': None, 'surprisal_mean': 4.69, 'surprisal_spread': 1.89,
    'prototypicality': 2.17, 'label_fit': 1.76, 'casing': 'title',
    'identity': 'unseen', 'least_surprising_position': 'last',
    'closest_other_label': 'ORG', ..., 'outcome': 'correct'}

To retrieve the spans at either end of a measure:

.. code-block:: python

   >>> casc.extreme_spans("prototypicality", n=10)              # least typical for their label
   >>> casc.extreme_spans("surprisal_mean", n=10, least=True)   # most like ordinary O tokens
   >>> casc.extreme_spans("span_typicality", label="PER", unseen_only=True)

Rows are those of ``span_table``, one per distinct span (text and label),
with an ``occurrences`` count; for the shape scores (bits per character) the
default, highest values, is the *least* typical end.

``outcome`` is present when a prediction is attached: ``"correct"`` or the
error type of the span's alignment. The rows load directly into a data
frame (``pandas.DataFrame(rows)``).

Profiling a dataset
-------------------

.. code-block:: python

   >>> report = casc.dimension_report()          # the test split
   >>> report["length_2"].count, report["length_2"].percentage
   (412, 34.6)
   >>> report["unseen"].spans[:3]                # the actual entities
   [total black, jet lag, hat trick]

   >>> casc.dimension_report(split="training")   # profile any split

Dimensions whose prerequisites are missing (e.g. ``unseen`` without a
training set) are omitted from the report rather than reported as zero.

Dimensions also carry **applicability preconditions**: casing dimensions
return ``None`` for text without caseable characters (Han script, digits),
so a Chinese corpus reports *n/a* rather than a spurious near-zero
title-case rate. Percentages are computed over the applicable entities, and
``DimensionStats.applicability`` reports how many entities the dimension is
defined for.

Crossing dimensions with performance
------------------------------------

:meth:`~cascabel.Cascabel.scores_by_dimension` restricts the span scores to
alignments whose gold entity expresses each dimension:

.. code-block:: python

   >>> for name, score in casc.scores_by_dimension().items():
   ...     print(f"{name:<20} F1={score.f1:.3f}  (n={score.tp + score.fn})")
   length_2             F1=0.418  (n=412)
   length_1             F1=0.762  (n=779)
   unseen               F1=0.401  (n=356)
   ...

Read **recall** here: spurious predictions have no gold entity and are not
attributed to any bucket. For precision, use the predicted side (below).

Two-sided error profiling
-------------------------

Errors are profiled on two sides, mirroring recall and precision:

- **Gold side** (recall): gold spans, correctly found vs. erred (missed,
  boundary and type errors).
- **Predicted side** (precision): predicted spans, correct vs. erred
  (spurious, boundary and type errors). Label-dependent attributes use the
  *predicted* label.

Boundary and type errors involve a gold and a predicted span, so they appear
on both sides, each with its own span.

.. code-block:: python

   >>> casc.span_outcomes("predicted")[:2]        # (span, None | error type)
   [(look, None), (black, 'overlap_missed')]
   >>> casc.scores_by_dimension(side="predicted")  # precision per dimension
   >>> profile = casc.error_profile("gold")
   >>> profile["unseen"]
   {'n': 1707, 'correct': 1089, 'errors': 618, 'missed': 402, 'type': 121, ...,
    'share_of_correct': 27.8, 'share_of_errors': 30.9, 'rate': 0.638}

``share_of_correct`` and ``share_of_errors`` are the percentages of correct
and of erroneous spans that express the dimension: a dimension is linked to
errors only when the two differ (a share among errors means little on its
own). ``rate`` is recall (gold side) or precision (predicted side) within
the dimension. Both views come from the same counts.

Entity-level flags
------------------

Every dimension is also available directly on the spaCy spans via the ``._.``
namespace, for custom slicing:

.. code-block:: python

   unseen_long = [
       ent for ent in casc.test.ents
       if ent._.is_unseen and ent._.length > 1
   ]
   mostly_consistent = [
       ent for ent in casc.test.ents
       if ent._.train_count and ent._.label_consistency >= 0.9
   ]

Custom dimensions
-----------------

Both reporting functions accept a custom dimension registry — any mapping
from a name to a span predicate:

.. code-block:: python

   my_dimensions = {
       "long": lambda ent: len(ent) >= 3,
       "hyphenated": lambda ent: "-" in ent.text,
   }
   casc.scores_by_dimension(my_dimensions)

   from cascabel import dimension_report
   dimension_report(casc.test, my_dimensions)
