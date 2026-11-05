Scores
======

All scores are :class:`~cascabel.Score` objects carrying raw counts and
derived metrics:

.. code-block:: python

   >>> score = casc.span_scores()
   >>> score.tp, score.fp, score.fn
   (2, 1, 1)
   >>> score.precision, score.recall, score.f1
   (0.6667, 0.6667, 0.6667)
   >>> score.as_dict()
   {'tp': 2, 'fp': 1, 'fn': 1, 'precision': 0.667, 'recall': 0.667, 'f1': 0.667}

Scores are immutable and addable, so custom aggregations are one-liners:

.. code-block:: python

   from cascabel import Score
   total = Score.aggregate(a.span_score for a in casc.alignments
                           if a.sentence.text.startswith("El"))

The three corpus-level views
----------------------------

.. code-block:: python

   casc.span_scores()       # strict span matching (the "headline" F1)
   casc.token_scores()      # micro P/R/F1 over full BIO tags (non-O)
   casc.boundary_scores()   # span matching with labels ignored

``span_scores`` is the strictest: boundary or label disagreements void the
whole entity. ``boundary_scores`` counts spans the same way but ignores the
label (unlabelled exact match): a predicted span is correct when its start
and end equal a gold span's. The gap between the two is what label errors
cost; what remains below 1 is segmentation.

Per-label breakdown
-------------------

.. code-block:: python

   >>> casc.scores_by_label()
   {'ENG': Score(tp=2, fp=1, fn=1), 'LOC': Score(tp=0, fp=0, fn=1),
    'PER': Score(tp=0, fp=1, fn=0)}

Each gold span involved in an error contributes one FN to *its own* label and
each predicted span one FP to *its own* label — so a LOC entity mislabeled as
PER produces ``fn`` under LOC and ``fp`` under PER, and the per-label scores
always sum exactly to :meth:`~cascabel.Cascabel.span_scores`.

Per-dimension breakdown
-----------------------

.. code-block:: python

   >>> casc.scores_by_dimension()["unseen"]
   Score(tp=1, fp=1, fn=1)

See :doc:`dimensions` for the available dimensions and their meaning.

Diagnostics
-----------

Four aggregate rates complement the P/R/F1 views:

.. code-block:: python

   casc.retrieval_capacity()         # fraction of gold entity tokens the
                                     # model labeled as entity (any label)
   casc.noise_level()                # fraction of gold O tokens the model
                                     # labeled as entity
   casc.segmentation_error_rate()    # fused/split/missegmented per gold entity
   casc.segmentation_success_rate()  # exact matches among entities that are
                                     # adjacent to another entity
