Command line
============

Installing cascabel provides a ``cascabel`` command (also available as
``python -m cascabel``).

Usage
-----

.. code-block:: bash

   # text report to stdout
   cascabel test.conll pred.conll --training train.conll

   # HTML error report
   cascabel test.conll pred.conll -f html -o report.html

   # JSON, for scripting
   cascabel test.conll pred.conll -f json | jq '.scores.span.f1'

   # dataset profile only (no prediction)
   cascabel test.conll --training train.conll

   # spaCy DocBin inputs
   cascabel test.spacy pred.spacy --spacy

Inputs are CoNLL column files — token in the first column, BIO tag in the
last, blank lines between sentences — or serialized spaCy ``DocBin`` files
with ``--spacy``.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Meaning
   * - ``--training PATH``
     - training file; enables seen/unseen and shape analysis
   * - ``--lexicon PATH``
     - background lexicon (one word per line); enables ``in_lexicon``
   * - ``--gazetteer PATH``
     - gazetteer (one entity per line); enables ``in_gazetteer``
   * - ``-f, --format {text,json,html}``
     - output format (default ``text``)
   * - ``-o, --output PATH``
     - write to a file instead of stdout
   * - ``--spacy``
     - inputs are ``.spacy`` DocBin files
   * - ``--delimiter STR``
     - CoNLL column delimiter (default: any whitespace)
   * - ``--no-shape``
     - skip fitting the statistical shape model
   * - ``--all-sentences``
     - HTML only: also render sentences without errors
   * - ``--version``
     - print the cascabel version

Exit codes
----------

``0`` on success; ``2`` on input errors (missing files, mis-aligned
prediction), with a message on stderr.
