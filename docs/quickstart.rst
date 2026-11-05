Quickstart
==========

A complete error-analysis session in a few lines. We evaluate a (toy) model
that detects anglicisms in Spanish text.

Setting up
----------

A :class:`~cascabel.Cascabel` object holds up to three token-aligned pieces of
data: the gold **test** annotations (required), the **training** set (enables
seen/unseen analysis and the statistical shape model) and the model's
**prediction** (enables scores and error analysis).

.. code-block:: python

   from cascabel import Cascabel

   test = [
       (["Un", "look", "total", "black"], ["O", "B-ENG", "B-ENG", "I-ENG"]),
       (["Me", "gusta", "el", "shopping"], ["O", "O", "O", "B-ENG"]),
   ]
   training = [
       (["Fracaso", "total", "del", "look"], ["O", "O", "O", "B-ENG"]),
       (["El", "streaming", "es", "popular"], ["O", "B-ENG", "O", "O"]),
   ]
   prediction = [
       (["Un", "look", "total", "black"], ["O", "B-ENG", "O", "B-ENG"]),
       (["Me", "gusta", "el", "shopping"], ["O", "O", "O", "B-ENG"]),
   ]

   casc = Cascabel.from_sentences(test, training=training, prediction=prediction)

Scores
------

Three complementary views of the same predictions:

.. code-block:: python

   >>> casc.span_scores()                # strict span-level matching
   P=0.6667 R=0.6667 F1=0.6667 (tp=2 fp=1 fn=1)

   >>> casc.token_scores()               # per-token BIO tags
   P=0.6667 R=0.5000 F1=0.5714 (tp=2 fp=1 fn=2)

   >>> casc.boundary_scores()            # span matching, labels ignored
   P=1.0000 R=0.6000 F1=0.7500 (tp=3 fp=0 fn=2)

   >>> casc.scores_by_label()
   {'ENG': Score(tp=2, fp=1, fn=1)}

What went wrong
---------------

.. code-block:: python

   >>> casc.error_counts()
   Counter({'overlap_missed': 1})

   >>> for a in casc.errors():
   ...     print(a.error_type,
   ...           [g.text for g in a.gold], "→",
   ...           [p.text for p in a.predicted])
   overlap_missed ['total black'] → ['black']

The model found *part* of "total black" — a very different failure from
missing it entirely, and exact-match F1 cannot tell them apart.

Where it went wrong
-------------------

.. code-block:: python

   >>> casc.scores_by_dimension()["length_2"]
   Score(tp=0, fp=1, fn=1)          # every two-token entity failed
   >>> casc.scores_by_dimension()["length_1"]
   Score(tp=2, fp=0, fn=0)          # every single-token entity was correct

One call for everything
-----------------------

.. code-block:: python

   print(casc.summary())        # aligned text report
   data = casc.report()         # JSON-serializable dict
   casc.to_html("report.html")  # self-contained color-coded HTML

Or, without writing any Python:

.. code-block:: bash

   cascabel test.conll pred.conll --training train.conll
