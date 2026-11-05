cascabel 🔔
===========

**Span profiling and fine-grained error analysis for NER and span-labeling datasets and models.**

cascabel stands for **ChAracterizing Spans for SequenCe lABELing**. It
implements the span dimensions and error analysis introduced in
`Álvarez-Mellado & Gonzalo (2024)
<http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6613>`_.

A single F1 score tells you *how much* your model fails — cascabel tells you
*how* and *where*:

- it aligns predictions against gold annotations and classifies every
  disagreement into a fine-grained :doc:`error taxonomy <error_taxonomy>`
  (missed, spurious, type swap, fused, split, missegmented, partial overlaps);
- it profiles datasets and errors along linguistic :doc:`dimensions
  <dimensions>` (casing, position, quotation, seen/unseen in training, token
  shape);
- it learns what an *unusually shaped* token looks like directly from your
  training data (:doc:`shape`), in any language;
- it generates controlled :doc:`perturbations <perturbations>` for stress
  testing;
- it renders everything as text, JSON or HTML :doc:`reports <reports>`, from
  Python or the :doc:`command line <cli>`.

Installation
------------

.. code-block:: bash

   pip install cascabel-ner

The package is called ``cascabel-ner`` on PyPI and is imported as ``cascabel``.
(A different, unrelated package already uses the name ``cascabel`` on PyPI, so do
not install both in one environment.)

cascabel is language-agnostic and works with any BIO-tagged data. It is built
on top of `spaCy <https://spacy.io>`_ ``Doc`` objects and uses
`PyTorch <https://pytorch.org>`_ for its character language models.

Thirty-second tour
------------------

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

   casc.span_scores()      # P=0.6667 R=0.6667 F1=0.6667 (tp=2 fp=1 fn=1)
   casc.error_counts()     # Counter({'overlap_missed': 1})
   print(casc.summary())   # the full formatted report
   casc.to_html("report.html")  # color-coded HTML error report

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   loading_data
   error_taxonomy
   scores
   dimensions
   shape
   perturbations
   reports
   cli
   api

Citing
------

If you use cascabel in your research, please cite:

   Elena Álvarez-Mellado and Julio Gonzalo. 2024. *Characterizing Spans for
   Sequence Labeling: A Case on Anglicism Detection.* Procesamiento del
   Lenguaje Natural, 73, 235–246.

.. code-block:: bibtex

   @article{alvarez-mellado-gonzalo-2024-characterizing,
       title   = {Characterizing Spans for Sequence Labeling: A Case on Anglicism Detection},
       author  = {{\'A}lvarez-Mellado, Elena and Gonzalo, Julio},
       journal = {Procesamiento del Lenguaje Natural},
       volume  = {73},
       pages   = {235--246},
       year    = {2024},
       url     = {http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6613}
   }
