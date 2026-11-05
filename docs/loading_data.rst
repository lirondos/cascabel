Loading data
============

cascabel accepts BIO-tagged data in four forms. All constructors accept the
same keyword arguments (``training=``, ``prediction=``, ``lexicon=``,
``gazetteer=``, ``shape=``, ``charlm_config=`` ...).

From token/tag lists
--------------------

One sentence as two parallel lists:

.. code-block:: python

   from cascabel import Cascabel

   casc = Cascabel.from_tokens(
       ["Un", "look", "total", "black"],
       ["O", "B-ENG", "B-ENG", "I-ENG"],
       prediction=(["Un", "look", "total", "black"],
                   ["O", "B-ENG", "O", "B-ENG"]),
   )

From sentence corpora
---------------------

A corpus is a list of ``(tokens, tags)`` pairs:

.. code-block:: python

   test = [
       (["Un", "look"], ["O", "B-ENG"]),
       (["Total", "black"], ["B-ENG", "I-ENG"]),
   ]
   casc = Cascabel.from_sentences(test)

From CoNLL column files
-----------------------

Files with one token per line, sentences separated by blank lines. By default
the token is the first column and the tag the last one, split on any
whitespace:

.. code-block:: python

   casc = Cascabel.from_conll(
       "test.conll",
       training="train.conll",
       prediction="pred.conll",
   )

   # non-default column layout:
   casc = Cascabel.from_conll(
       "test.tsv",
       conll_options={"delimiter": "\t", "token_column": 0, "tag_column": 2,
                      "comment_prefix": "#"},
   )

The reader itself is also public:

.. code-block:: python

   from cascabel import read_conll, doc_from_conll

   sentences = read_conll("test.conll")     # [(tokens, tags), ...]
   doc = doc_from_conll("test.conll")       # a spaCy Doc

From spaCy binaries
-------------------

Serialized ``DocBin`` (``.spacy``) files are loaded and merged into a single
``Doc``:

.. code-block:: python

   casc = Cascabel.from_spacy_binary(
       "test.spacy", training="train.spacy", prediction="pred.spacy"
   )

Attaching predictions later
---------------------------

Predictions can be attached (or replaced) after construction; alignments and
scores are recomputed:

.. code-block:: python

   casc = Cascabel.from_sentences(test, training=training)
   casc.set_prediction_from_sentences(prediction)
   casc.set_prediction_from_tokens(tokens, tags)

Predictions must be token-aligned with the test set: same tokens, same
sentence boundaries. cascabel raises a ``ValueError`` otherwise.

Tagging schemes
---------------

Files are read as BIO (IOB2) by default. Declare another scheme with
``scheme=`` (or ``--scheme`` on the command line); tags are converted to BIO
on loading:

- ``"IOB1"``: a span starts with ``I-X``, and ``B-X`` only marks a span right
  after another span of the same label (the original CoNLL-2003 files).
- ``"BIOES"`` (alias ``"BILOU"``): ``S-``/``U-`` single-token spans and
  ``E-``/``L-`` span ends.

.. code-block:: python

   >>> casc = Cascabel.from_conll("eng.testb", training="eng.train",
   ...                            scheme="IOB1")

Document-boundary lines (``-DOCSTART-``) are skipped automatically, so they
never count as sentences or tokens (``conll_options={"skip_docstart":
False}`` keeps them).

Repairing invalid BIO
---------------------

Gold files and model outputs sometimes contain invalid transitions (``I-X``
after ``O``, at the start of a sentence, or after a different label). Every
loader (gold, training and prediction alike) handles them as ``conlleval``
and ``seqeval`` do: the stray ``I-X`` starts a new span. A warning reports
how many sentences were affected and their numbers, so the file can be fixed
instead. Pass ``repair=False`` to the ``doc_from_*`` functions to raise an
error. The utility is also public:

.. code-block:: python

   >>> from cascabel import repair_bio
   >>> repair_bio(["O", "I-ENG", "I-ENG"])
   ['O', 'B-ENG', 'I-ENG']
   >>> repair_bio(["B-LOC", "I-ENG"])
   ['B-LOC', 'B-ENG']

Lexicons and gazetteers
-----------------------

Two optional word lists refine the analysis (see :doc:`dimensions`):

- a **lexicon** of ordinary words of the language (enables the ``in_lexicon``
  dimensions: entities whose tokens are also common words);
- a **gazetteer** of known entities (enables ``in_gazetteer``).

Both accept a Python set or a path to a one-entry-per-line file:

.. code-block:: python

   casc = Cascabel.from_conll(
       "test.conll",
       lexicon="es_lexicon.txt",
       gazetteer={"real madrid", "nueva york"},
   )
