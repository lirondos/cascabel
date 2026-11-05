Shape: surprisal and prototypicality
====================================

Entities often *look* different from ordinary words (foreign character
sequences, rare letter combinations, unusual word patterns), and some look
more like one label than another. cascabel learns both notions from your
training data with character-level neural language models (PyTorch LSTMs):
no hand-written rules, any language.

Each model reads a string letter by letter, spaces and case included, and
scores it in **bits per character** (lower = more familiar). Every score is
relative to *one* training set, so absolute values are not comparable across
datasets.

Surprisal (vs. ordinary tokens)
-------------------------------

One model is trained on the distinct **``O`` tokens** of the training set:
what an ordinary, non-entity word looks like. Every token gets its surprisal
under that model; a span keeps the whole *profile* of its tokens' surprisal
plus three summaries:

.. code-block:: python

   casc = Cascabel.from_sentences(test, training=training)

   >>> casc.test[3]._.surprisal                 # one token, bits/char
   4.31
   >>> ent._.surprisal_profile                  # one value per token
   [4.69, 2.80]
   >>> ent._.surprisal_mean                     # overall oddness
   3.74
   >>> ent._.surprisal_spread                   # max - min: partially odd spans
   1.89
   >>> ent._.least_surprising_position          # "single"|"first"|"last"|"interior"
   'last'

Prototypicality (vs. the training spans of each label)
------------------------------------------------------

One model is trained **per label** on that label's training spans (tokens
joined by single spaces, one string per occurrence, so frequent spans weigh
more: prototypicality is *familiarity* given the training set). Because the
model reads the whole span, it sees both odd character combinations and odd
word patterns (*El Bigotes*, article + noun, as a ``PER``). Two scores:

- ``span._.prototypicality`` — **form typicality**: bits per character under
  the span's *own* label model. Low = typical (*Juan Sánchez* as ``PER``).
- ``span._.label_fit`` — **label fit**: how much better the span fits its own
  label than the best-fitting *other* label, in bits per character.
  Negative = the span looks more like another label (*Imperio Argentina* as
  ``PER`` looking like a ``LOC``); ``span._.closest_other_label`` names that
  label. Since it compares the same string under two models, span length
  largely cancels out.

A third score ignores labels:

- ``span._.span_typicality`` — **span typicality**: bits per character under
  one model trained on *every* training span, whatever its label (rare
  labels included). Low = looks like a span. Read together with
  prototypicality, it separates "a span, just not a typical ``PER``" (low
  span typicality, high prototypicality) from "does not look like a span at
  all" (both high). It differs from surprisal, which compares with ordinary
  ``O`` tokens rather than with spans.

**Perplexity.** ``span._.surprisal_perplexity``, ``span._.span_typicality_perplexity``
and ``span._.prototypicality_perplexity`` give the same scores as
per-character perplexity, ``2 ** bits``: the number of equally likely
characters the model is choosing between, on average. They rank spans
exactly like the bits/char scores. Label fit has no perplexity version: it
is a difference of log probabilities, which would become a ratio.

**In reports.** The text, HTML, card and JSON reports show no shape
numbers: their *Shape* section lists, per label, the 5 least and 5 most
prototypical distinct spans. The scores are span attributes and columns of
``casc.span_table()``.

**Training spans are never scored.** The models learned from them, so their
scores would reflect memorization, not typicality: on the training split,
surprisal, span typicality, prototypicality and label fit are ``None``. The
scores describe test (or dev) spans *relative to* what the models learned
from training. A warning is raised when the evaluated file has the same
tokens as the training file.

Spans are scored under their own label: the gold label for test
spans, the *predicted* label for spans of a prediction. Labels with fewer
than ``CharLMConfig.min_label_spans`` training spans get no model; their
spans' prototypicality and label fit are ``None``.

.. note::

   Form typicality still correlates with span length; how to separate the
   two is an open design question. Training spans are scored by models that
   saw them, so they look more typical than unseen spans.

Configuration and cost
----------------------

All models share one :class:`~cascabel.charlm.CharLMConfig` (embedding and
hidden size, epochs, batch size, learning rate, seed, ``min_label_spans``,
device). Training is seeded and fixed-length (no early stopping). On a CPU,
the CoNLL-03 training set takes about two minutes for all five models.

.. code-block:: python

   from cascabel import CharLMConfig
   casc = Cascabel.from_conll("test.txt", training="train.txt",
                              charlm_config=CharLMConfig(epochs=15, seed=1))
   casc = Cascabel.from_conll("test.txt", training="train.txt",
                              shape=False)   # skip the models entirely

Standalone use
--------------

.. code-block:: python

   from cascabel import CharLM, CharLMConfig, ShapeModels

   model = CharLM(CharLMConfig(epochs=20)).fit(["casa", "perro", "gato", ...])
   model.bits_per_char(["shopping", "casa"])

   models = ShapeModels(o_tokens, [("PER", "Juan Pérez"), ("LOC", "Madrid"), ...])
   models.prototypicality([("PER", "Imperio Argentina")])
