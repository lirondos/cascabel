API reference
=============

The facade
----------

.. autoclass:: cascabel.Cascabel
   :members:
   :undoc-members:

Scores
------

.. autoclass:: cascabel.Score
   :members:

Alignments
----------

.. autoclass:: cascabel.ErrorType
   :members:
   :undoc-members:

.. autoclass:: cascabel.Alignment
   :members:

.. autofunction:: cascabel.align_sentence

.. autofunction:: cascabel.align_doc

Dimensions
----------

.. autofunction:: cascabel.dimension_report

.. autoclass:: cascabel.DimensionStats
   :members:

Shape (character language models)
----------------------------------

.. autoclass:: cascabel.CharLMConfig

.. autoclass:: cascabel.CharLM
   :members:

.. autoclass:: cascabel.ShapeModels
   :members:

.. autofunction:: cascabel.word_shape

Perturbations
-------------

.. autofunction:: cascabel.recase

.. autofunction:: cascabel.add_quotes

.. autofunction:: cascabel.strip_quotes

Data loading
------------

.. autofunction:: cascabel.doc_from_tokens

.. autofunction:: cascabel.doc_from_sentences

.. autofunction:: cascabel.doc_from_conll

.. autofunction:: cascabel.doc_from_spacy_binary

.. autofunction:: cascabel.read_conll

.. autofunction:: cascabel.load_wordlist

.. autofunction:: cascabel.repair_bio

.. autofunction:: cascabel.repair_bio_sentences

.. autofunction:: cascabel.bio_tag
