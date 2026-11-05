Reports
=======

Three renderings of the complete analysis, from one object.

Machine-readable: ``report()``
------------------------------

Everything as one JSON-serializable dict — ideal for experiment tracking:

.. code-block:: python

   >>> data = casc.report()
   >>> data.keys()
   dict_keys(['dataset', 'dimensions', 'scores',
              'scores_by_label', 'scores_by_dimension', 'error_counts',
              'diagnostics'])
   >>> data["scores"]["span"]["f1"]
   0.6667

   import json
   json.dump(casc.report(), open("run_042.json", "w"))

When no prediction is attached the report contains only the dataset profile
(counts, dimensions, shape divergence) — useful for characterizing corpora,
not just models.

Human-readable: ``summary()``
-----------------------------

.. code-block:: python

   >>> print(casc.summary())
   cascabel report
   ===============

   Dataset: 8 tokens, 2 sentences, 3 entities (ENG: 3)

   Scores
                                  P       R      F1
     span                    0.6667  0.6667  0.6667   (tp=2 fp=1 fn=1)
     token                   0.6667  0.5000  0.5714   (tp=2 fp=1 fn=2)
     boundary                0.6667  0.6667  0.6667   (tp=2 fp=1 fn=1)
   ...

HTML: ``to_html()``
-------------------

A single self-contained HTML file: score cards, error tables, per-label and
per-dimension tables, and — most usefully — every error sentence rendered
with each predicted span highlighted as one box (missed spans: the gold
span): green when correct, otherwise in the colour of its error type, the
same colour as the error's chip. Hovering a box shows the gold and
predicted spans with their labels; under each sentence one chip per error
gives the gold and predicted spans with their labels.

.. code-block:: python

   casc.to_html("report.html")              # write file, return the path
   html_string = casc.to_html()             # or get the string
   casc.to_html("report.html", only_errors=False)  # include correct sentences

Open the file in any browser; no external assets are required, so it can be
attached to an experiment log or shared as is.
