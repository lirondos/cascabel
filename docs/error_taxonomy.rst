The error taxonomy
==================

Exact-match evaluation treats a prediction that is one token off exactly like
no prediction at all. cascabel aligns gold and predicted entities sentence by
sentence and classifies every disagreement:

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Error type
     - Meaning
     - Example (gold → predicted)
   * - ``missed``
     - gold entity with no overlapping prediction
     - [total black] → ∅
   * - ``spurious``
     - prediction with no overlapping gold entity
     - ∅ → [gusta]
   * - ``type``
     - correct boundaries, wrong label
     - [Paris]\ :sub:`LOC` → [Paris]\ :sub:`PER`
   * - ``fused``
     - one prediction covering two gold entities
     - [look] [total black] → [look total black]
   * - ``split``
     - two predictions inside one gold entity
     - [total black] → [total] [black]
   * - ``missegmented``
     - boundaries straddle gold entity boundaries
     - [leggings animal] [print] → [leggings] [animal print]
   * - ``overlap_missed``
     - prediction strictly inside a gold entity
     - [new york city] → [york]
   * - ``overlap_spurious``
     - prediction strictly containing a gold entity
     - [york] → [new york city]
   * - ``overlap_spurious_missed``
     - staggered overlap
     - [carrot cake] → [cake vegan]

Working with alignments
-----------------------

Every gold/prediction correspondence is an :class:`~cascabel.Alignment` with
the spans involved, the error type (``None`` when correct), and per-alignment
scores:

.. code-block:: python

   from cascabel import Cascabel, ErrorType

   casc = Cascabel.from_tokens(
       ["un", "look", "total", "black"],
       ["O", "B-ENG", "B-ENG", "I-ENG"],
       prediction=(["un", "look", "total", "black"],
                   ["O", "B-ENG", "I-ENG", "I-ENG"]),
   )

   >>> alignment = casc.alignments[0]
   >>> alignment.error_type
   <ErrorType.FUSED: 'fused'>
   >>> [g.text for g in alignment.gold]
   ['look', 'total black']
   >>> [p.text for p in alignment.predicted]
   ['look total black']
   >>> [t.text for t in alignment.wrong_tokens]   # the token that fused them
   ['total']
   >>> alignment.is_segmentation_error
   True

Summaries:

.. code-block:: python

   casc.error_counts()               # Counter of error types
   casc.errors()                     # all erroneous alignments
   casc.segmentation_error_rate()    # fused/split/missegmented per gold entity
   casc.retrieval_capacity()         # gold entity tokens labeled entity
   casc.noise_level()                # O tokens labeled entity

Segmentation-aware scoring
--------------------------

Each alignment carries two scores that aggregate into the corpus-level views
(see :doc:`scores`):

- ``span_score`` — strict: a correct alignment is one TP; every gold span
  involved in an error is one FN and every predicted span one FP.
- ``boundary_score`` — the same counting with labels ignored: a gold and a
  predicted span with the same start and end are one TP, whatever their
  labels (so a ``type`` error is correct here).

Using the alignment machinery directly
--------------------------------------

The aligner is exposed for custom pipelines:

.. code-block:: python

   from cascabel import align_sentence

   alignments = align_sentence(sentence, gold_entities, predicted_entities)
