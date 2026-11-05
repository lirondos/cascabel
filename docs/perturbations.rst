Perturbations
=============

Perturbations create *new* documents with the same annotations but altered
surface forms. Run your model on the perturbed data and compare scores with
the original: if performance collapses when casing is removed, the model was
leaning on casing.

All functions take a spaCy ``Doc`` (e.g. ``casc.test``) and return a new,
annotation-preserving ``Doc``.

Re-casing
---------

.. code-block:: python

   from cascabel import recase

   recase(casc.test, "lower")                     # lowercase everything
   recase(casc.test, "upper")                     # SHOUT EVERYTHING
   recase(casc.test, "title")                     # Capitalize Everything
   recase(casc.test, "lower", scope="entities")   # only entity tokens

Quotation marks
---------------

.. code-block:: python

   from cascabel import add_quotes, strip_quotes

   add_quotes(casc.test)      # wrap every unquoted entity: look → " look "
   strip_quotes(casc.test)    # remove quotes around quoted entities

Both accept the same optional re-casing arguments (``doc_case=``,
``entity_case=``), so combined perturbations are one call:

.. code-block:: python

   # lowercase text, but entities quoted and uppercased:
   add_quotes(casc.test, doc_case="lower", entity_case="upper")

A full stress-test loop
-----------------------

.. code-block:: python

   from cascabel import Cascabel, recase, bio_tag

   casc = Cascabel.from_sentences(test, training=training)

   perturbations = {
       "original": casc.test,
       "lowercase": recase(casc.test, "lower"),
       "uppercase": recase(casc.test, "upper"),
   }

   for name, doc in perturbations.items():
       sentences = [
           ([t.text for t in sent], [bio_tag(t) for t in sent])
           for sent in doc.sents
       ]
       predicted = my_model.predict(sentences)          # your model here
       probe = Cascabel.from_sentences(sentences, prediction=predicted)
       print(f"{name:<12} span F1 = {probe.span_scores().f1:.4f}")

A large gap between ``original`` and ``lowercase`` quantifies exactly how
much of the model's performance rests on casing cues.
