# cascabel 🔔

**Span profiling and fine-grained error analysis for NER and span-labeling datasets and models.**

cascabel stands for **ChAracterizing Spans for SequenCe lABELing**. It
implements the span dimensions and error analysis introduced in
[Álvarez-Mellado & Gonzalo (2024)](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6613)
(see [Citing](#citing)).

A single F1 score tells you *how much* your model fails — cascabel tells you
*how* and *where*. It aligns predictions against gold annotations, classifies
every disagreement into a fine-grained error taxonomy, profiles datasets along
linguistic dimensions, and generates controlled perturbations for stress
testing. It is language-agnostic and works with any BIO-tagged data.

```bash
pip install cascabel-ner
```

The package is called `cascabel-ner` on PyPI and is imported as `cascabel`. (A
different, unrelated package already uses the name `cascabel` on PyPI, so do not
install both in one environment.)

## Quick start

```python
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

casc.span_scores()      # strict span-level P/R/F1
casc.token_scores()     # token-level P/R/F1 over BIO tags
casc.boundary_scores()  # span P/R/F1 with labels ignored (exact boundaries only)
casc.scores_by_label()  # span P/R/F1 per entity type
casc.error_counts()     # Counter({'overlap_missed': 1})

print(casc.summary())   # everything above as one formatted text report
casc.report()           # ... or as one JSON-serializable dict
casc.to_html("report.html")  # ... or as a color-coded HTML error report
```

Or straight from the command line:

```bash
cascabel test.conll pred.conll --training train.conll            # text report
cascabel test.conll pred.conll -f html -o report.html            # HTML report
cascabel test.conll pred.conll -f json                           # JSON report
cascabel test.conll --training train.conll                       # dataset profile only
```

Data can also be loaded from CoNLL column files or serialized spaCy `DocBin`s:

```python
casc = Cascabel.from_conll("test.conll", training="train.conll",
                           prediction="pred.conll")
casc = Cascabel.from_spacy_binary("test.spacy", prediction="pred.spacy")
```

## The error taxonomy

Exact-match evaluation treats a prediction that is one token off exactly like
no prediction at all. cascabel distinguishes:

| Error type                 | Meaning                                              |
| -------------------------- | ---------------------------------------------------- |
| `missed`                   | gold entity with no overlapping prediction           |
| `spurious`                 | prediction with no overlapping gold entity           |
| `type`                     | correct boundaries, wrong label                      |
| `fused`                    | one prediction covering two gold entities            |
| `split`                    | two predictions inside one gold entity               |
| `missegmented`             | prediction boundaries straddle gold entity boundaries|
| `overlap_missed`           | prediction strictly inside a gold entity             |
| `overlap_spurious`         | prediction strictly containing a gold entity         |
| `overlap_spurious_missed`  | staggered overlap (extra on one side, missing on the other) |

Every alignment is available for inspection:

```python
for alignment in casc.errors():
    print(alignment.error_type,
          [g.text for g in alignment.gold],
          [p.text for p in alignment.predicted])
```

The HTML report (`casc.to_html("report.html")`) renders every sentence with
each predicted span highlighted in the colour of its error type (green when
correct), plus an error chip per alignment, score cards, and the
per-label and per-dimension tables. It is a single self-contained file you
can open in a browser or attach to an experiment log.

Segmentation-focused diagnostics build on the same alignments:

```python
casc.retrieval_capacity()         # fraction of gold entity tokens retrieved
casc.noise_level()                # fraction of O tokens labeled as entity
casc.segmentation_error_rate()    # fused/split/missegmented per gold entity
casc.segmentation_success_rate()  # exact matches among adjacent entities
```

## Dimensions: where does the model fail?

cascabel profiles entities along linguistic dimensions — surface properties
(length in fixed bins, sentence-initial, casing as upper / title / lower /
mixed, starts capitalized, quoted, adjacent to another entity), exposure properties (how often
the exact span occurs in training as a whole span or as `O`, and in what
share of those occurrences it carries its own label: unseen, seen with the
same label, ambiguous, seen only with other labels; plus, for every word of
the span, how it was seen in training: inside spans, only as `O`, or never)
and shape properties (see below). Continuous properties
(length, training count, label consistency, surprisal, prototypicality,
label fit) are stored on every span; binary dimensions are views on them.

```python
casc.dimension_report()                   # how the dataset distributes over dimensions
casc.scores_by_dimension()                # P/R/F1 per bucket (read recall: gold side)
casc.scores_by_dimension(side="predicted")  # precision per bucket (predicted spans)
casc.error_profile("gold")                # correct vs. erroneous spans, per dimension
casc.error_profile("predicted")           # the same for predicted spans (spurious too)
casc.span_table("test")                   # one record per span, every attribute + outcome
```

`scores_by_dimension()` answers questions like *"how much does performance
drop on entities never seen in training?"*; `error_profile()` asks the
reverse question, *"what do the spans behind the errors look like, compared
with the spans handled correctly?"*, on both the gold (recall) and the
predicted (precision) side.

Lexicon and gazetteer dimensions are optional: they appear only when you
pass your own `lexicon=` / `gazetteer=` word list.

## Shape: surprisal and prototypicality

Entities often *look* different from ordinary words, and some look more like
one label than another. cascabel learns both from the training data with
character-level LSTM language models (PyTorch), which read whole strings,
spaces and case included, and score them in bits per character:

- **Surprisal** (vs. ordinary tokens) — one model trained on the training
  `O` tokens. Every token gets `token._.surprisal`; every span keeps its
  profile (`span._.surprisal_profile`) and summaries: `surprisal_mean`,
  `surprisal_spread` (max − min: partially odd spans) and
  `least_surprising_position` (`"single"` / `"first"` / `"last"` /
  `"interior"`).
- **Prototypicality** (vs. the training spans of each label) — one model per
  label. `span._.prototypicality` is the span's bits per character under its
  own label's model (form typicality: characters *and* word pattern);
  `span._.label_fit` is own label minus best other label (negative = looks
  like another label, named in `span._.closest_other_label`).

Each score also has a per-character perplexity version (`2 ** bits`):
`surprisal_perplexity`, `span_typicality_perplexity`,
`prototypicality_perplexity`. Reports list, per label, the 5 least and
5 most prototypical spans without numbers; the scores are in `span_table()`.

Scores are relative to one training set. Hyperparameters live in
`CharLMConfig` (`charlm_config=`); `shape=False` skips the models.

## Perturbations: controlled stress tests

Generate modified copies of a dataset that isolate one surface cue, run your
model on them, and compare:

```python
from cascabel import recase, add_quotes, strip_quotes

lowercased = recase(casc.test, "lower")             # does casing matter?
uppercased = recase(casc.test, "upper", scope="entities")
quoted     = add_quotes(casc.test)                  # do quotes give it away?
unquoted   = strip_quotes(casc.test)
```

All functions return new spaCy `Doc`s with gold annotations preserved.

## Scope and lineage

cascabel was originally developed to analyze lexical borrowing detection
(anglicisms in Spanish) and generalized into a language-independent library.
The original rule-based Spanish graphotactics check was removed in 0.7.0:
surprisal is now learned from the training data. The evaluation methodology (error taxonomy, dimensions,
perturbations) follows the author's research on borrowing detection
evaluation.

## Citing

If you use cascabel in your research, please cite the paper that introduced
the span dimensions and error analysis this library implements:

> Elena Álvarez-Mellado and Julio Gonzalo. 2024. *Characterizing Spans for
> Sequence Labeling: A Case on Anglicism Detection.* Procesamiento del
> Lenguaje Natural, 73, 235–246.

```bibtex
@article{alvarez-mellado-gonzalo-2024-characterizing,
    title   = {Characterizing Spans for Sequence Labeling: A Case on Anglicism Detection},
    author  = {{\'A}lvarez-Mellado, Elena and Gonzalo, Julio},
    journal = {Procesamiento del Lenguaje Natural},
    volume  = {73},
    pages   = {235--246},
    year    = {2024},
    url     = {http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6613}
}
```

## License

MIT
