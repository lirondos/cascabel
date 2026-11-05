# Changelog

## 0.8.0 (2026-11-05)

First release on PyPI.

### Changed
- **Package name** (2026-11-05, Elena): published as `cascabel-ner`, because
  `cascabel` is taken on PyPI by an unrelated package. The import name stays
  `cascabel`.

### Removed (breaking)
- **`starts_capitalized`** (2026-10-28, Elena): the dimension, the
  `span._.starts_capitalized` attribute and its `span_table` column are
  removed. It was almost the union of `casing_title` and `casing_upper`
  (CoNLL-03 test: 5,625 vs 4,680 + 951 spans), so it added nothing to the
  reports.

### Added
- **Dimension examples** (2026-10-28, Elena): each dimension in `report()`
  gets an `example` (`{"text", "label", "left", "right"}`): the most
  frequent span form that expresses it, with up to 3 tokens of context on
  each side (`cascabel.report.dimension_example`). The text and HTML
  reports show it in an *example* column.

## 0.7.0 (2026-10-28)

### Fixed
- **Alignment** (2026-10-28): a prediction ending right before the next gold span (gold *[SIALKOT] , [Pakistan]*, predicted *[SIALKOT ,] [Pakistan]*) was classified as `missegmented` (off-by-one in `is_straddling`; now `overlap_spurious` + correct); and a "perfect" straddling missegmentation (*[a b c] [d]* for *[a b] [c d]*) consumed the second prediction without recording it, so it was lost from precision counts. Error counts and span precision can change.

### Changed (breaking)
- **Boundary F1 redefined** (2026-10-28, Elena): `boundary_scores()` /
  `Alignment.boundary_score` are now span-level with labels ignored
  (unlabelled exact match: a predicted span is correct when its start and
  end equal a gold span's). The previous token-and-separator counting, with
  partial credit, is removed. Span, token and boundary F1 are the three
  score views.
- **Readable reports** (2026-10-28): every error type is listed, zeros included (`report()["error_counts"]` now has all nine keys; `Cascabel.error_counts()` is unchanged); each dimension has a one-line meaning (`cascabel.dimensions.DIMENSION_GLOSSES`) and each score view a one-line definition (`cascabel.report.SCORE_GLOSSES`), shown in the text, HTML and card reports; the meaning column sits right after the dimension name; in the HTML sentence view, each *predicted* span is highlighted as one box (gold *Ann Battelle* predicted as *Ann* highlights *Ann* only; missed spans highlight the gold span) in its error type's colour, the same as its chip, correct spans in green; new error palette (nine hues at least 15 apart in OKLab for normal vision, chip text dark or white by contrast; error names stay as labels since nine hues cannot all be told apart under colour blindness); the line under each sentence gives every gold and predicted span with its label (a type error reads gold “Nueva York” LOC → predicted “Nueva York” PER); shape lists show most prototypical first.
- **Reports show identity and shape** (2026-10-28): `report()` (JSON), `summary()` (text), `to_html()` and `card()` gain an *Identity* section (`identity_summary`: the three radar axes plus the familiar-word and spelling-variant shares) and a *Shape* section listing, per label, the 5 least and 5 most prototypical distinct spans, without scores (`cascabel.report.shape_summary`), when a training set is given. Per-character perplexity attributes (`2 ** bits`): `surprisal_perplexity`, `span_typicality_perplexity`, `prototypicality_perplexity` (also `span_table` columns); no perplexity for `label_fit` (a difference of logs). The card adds a caveat for dimensions with fewer than 30 gold spans (`SMALL_DIMENSION_FLOOR`).
- **`-DOCSTART-` lines are skipped** (2026-10-28): `read_conll` (and so
  `doc_from_conll`, `Cascabel.from_conll` and the command line) drops
  document-boundary lines, which previously became one-token `O`
  sentences and inflated sentence and token counts on CoNLL-2002/2003.
  `skip_docstart=False` keeps them.
- **Ill-formed BIO is handled the same way in every file, with a warning**
  (2026-10-21): gold, training and prediction loaders read a stray `I-X`
  (after `O`, at sentence start, or after another label) as the start of a
  new span, as `conlleval`/`seqeval` do, and warn with the number of
  affected sentences and their numbers. Previously gold files were repaired
  silently by spaCy and predictions silently by `repair_bio`. `repair`
  now defaults to `True` in `doc_from_sentences` / `doc_from_tokens` /
  `doc_from_conll`; `repair=False` raises `ValueError` instead.
- **Tagging scheme option** (2026-10-21): `scheme="BIO"` (default),
  `"IOB1"` (original CoNLL-2003 files) or `"BIOES"` (alias `"BILOU"`) on
  every loader, `Cascabel.from_*` and `set_prediction_from_*`, and
  `--scheme` on the command line; tags are converted to BIO on loading
  (`cascabel.to_bio`), so declared schemes do not trigger the ill-formed
  BIO warning.
- **PyTorch is now a required dependency** (`torch>=2.0`).
- **Shape scoring uses character-level LSTM language models**
  (`cascabel.charlm`: `CharLM`, `CharLMConfig`, `ShapeModels`). The character
  n-gram scorer is removed: `CharNgramModel`, `ShapeScorer`,
  `Cascabel.shape_scorer` and the constructor arguments `shape_n`,
  `shape_alpha`, `shape_percentile` and `shape_predicate` are gone; use
  `charlm_config=` instead. `Cascabel.shape_models` holds the trained models.
- **Surprisal** (vs. training `O` tokens, now case-preserving) is continuous:
  `token._.surprisal` and, per span, `surprisal_profile`, `surprisal_mean`,
  `surprisal_spread` and `least_surprising_position`. Removed:
  `token._.shape_surprisal`, `is_high_surprisal`, `span._.shape_class` and
  the dimensions `high_surprisal` / `low_surprisal` (no per-dataset
  thresholds).
- **Prototypicality** is span-level, with one model per label:
  `span._.prototypicality` (form typicality, bits/char under the span's own
  label model), `span._.label_fit` (own label minus best other label) and
  `span._.closest_other_label`. Plus `span._.span_typicality` (2026-10-21):
  bits/char under one any-span model trained on all training spans, labels
  ignored (`ShapeModels.span_model`, `ShapeModels.span_typicality`).
  **Training spans are not scored** (2026-10-21): surprisal, span
  typicality, prototypicality and label fit are `None` on the training split
  (the models learned from those strings); a warning is raised when the
  evaluated file has the same tokens as the training file.
  `Cascabel.extreme_spans(measure, n, least, label, unseen_only)` returns
  the distinct spans (text + label, with an `occurrences` count) at either
  end of any numeric measure (2026-10-21).
- **Identity compares labels, not BIO tags** (2026-10-21): each training
  occurrence of a span's word sequence gets a label with B/I ignored
  (`cascabel.occurrence_label`: `"O"`, the shared label, or `"MIXED"`), so
  *Madrid* inside *Comunidad de Madrid* (`I-LOC`) counts as `LOC`, not as a
  different tagging. `label_consistency` and the identity groups use these
  labels. New: `span._.train_labels` (label counts),
  `span._.majority_label` / `span._.majority_share` (most frequent label
  and its share; low = ambiguous), `Cascabel.identity_distribution()` and
  `Cascabel.label_ambiguity(split)` (distinct entities of a split with more
  than one label within that split, with their proportions).
  **Only whole-span or all-`O` occurrences count** (rule B, 2026-10-21):
  *Madrid* inside *Comunidad de Madrid* or *Real Madrid*, or *Madrid* as the
  start of *Madrid Barajas*, is no longer an occurrence of the span
  *Madrid*; partly-`O` occurrences no longer count either. More spans are
  therefore `unseen`. Applies to `label_ambiguity` too.
  `Cascabel(span_match="any")` restores the previous behaviour.
  **Token layer** (2026-10-21): `token._.train_labels` (label counts of the
  token text in training, B/I ignored, `O` included), `token._.exposure`
  (`"in_span"` / `"only_O"` / `"unseen"`), `span._.word_exposure` (tokens per
  group) and `span._.has_word_seen_in_span`; function words included.
  **Spelling variants** (2026-10-28): matching stays exact (case- and
  accent-sensitive), but occurrences that match only once casing and
  accents are folded (`cascabel.fold_text`: *MADRID*, *El Pais* for
  *Madrid*, *El País*) are reported in `span._.variant_labels` (same
  whole-span rule) and `token._.variant_labels`, plus
  `n_unseen_variant` / `unseen_variant_share` in `identity_summary`.
  Rationale: casing can change meaning (*El País* vs *el país*) and is a
  dimension of its own, but all-caps headlines or missing accents should
  not hide that a span was seen. *ñ* is kept (not folded to *n*).
  `Cascabel.identity_summary(split)`: share unseen (of all spans), share of
  unseen spans with a word seen inside a span, share ambiguous
  (`label_consistency < 1`, of seen spans); plus (2026-10-28) the radar
  axes `multi_label_share` (seen with more than one label, `O` included)
  and `unseen_label_share` (never seen with its own label), both out of the
  seen spans. Removed: `token._.entity_surprisal`,
  `is_non_prototypical`, `span._.prototypicality_class` and the dimensions
  `prototypical` / `non_prototypical`.
- **Casing** is one position-independent category, `span._.casing`
  (`"upper"` / `"title"` / `"lower"` / `"mixed"`, `None` without cased
  letters). The dimensions `titlecase` / `uppercase` are replaced by
  `casing_upper`, `casing_title`, `casing_lower`, `casing_mixed`; title case
  no longer excludes sentence-initial spans, and an all-caps span is no
  longer also title case. The first cased letter decides (2026-10-21):
  title = first cased letter uppercase, whatever follows (*New york*,
  *McDonald*); mixed = first cased letter lowercase with a later uppercase
  (*iPhone*). New `span._.starts_capitalized` and dimension
  `starts_capitalized`: first character is an uppercase letter (digits and
  symbols count as not capitalized; sentence-initial capitals count). Span getters `is_titlecase` / `is_upper` and
  token getters `is_titlecase` / `is_upper` are removed.
- **Length**: `span._.length` (tokens); the dimensions `multitoken` /
  `single_token` are replaced by fixed bins shared across datasets
  (`LENGTH_BINS`: `length_1`, `length_2`, `length_3`, `length_4_5`,
  `length_6plus`).
- **Corpus-level Jensen–Shannon divergence removed**:
  `Cascabel.entity_shape_divergence()`, `js_divergence`,
  `ngram_distribution`, and the `shape_divergence` section of the report,
  summary and card. Train/test shape alignment is now read from the
  per-span surprisal and prototypicality scores (`span_table`).
- **`cascabel.contrib.spanish` is removed** (the rule-based Spanish
  phonotactics check, a symbolic approach to surprisal), together with the
  `spanish` optional extra and its `pylabeador` dependency. The module is
  kept for reference in `_Archive/contrib/` (not installed).

### Added
- **Two-sided error profiling**: `Cascabel.span_outcomes(side)`,
  `Cascabel.error_profile(side)` (per dimension: correct vs. erroneous
  spans, counts per error type, share of correct vs. share of erroneous
  spans, recall or precision within the dimension) and
  `scores_by_dimension(side="predicted")` (precision per dimension over
  predicted spans, evaluated with the predicted label). Both are in
  `Cascabel.report()` (`precision_by_dimension`, `error_profile`).
- `Cascabel.span_table(split)`: one record per span with every continuous
  and categorical attribute (`NUMERIC_ATTRIBUTES`,
  `CATEGORICAL_ATTRIBUTES`) and, with a prediction, its outcome.

### Fixed
- Lexicon dimensions (`in_lexicon`, `partial_in_lexicon`) no longer appear
  as 0% when no lexicon is passed: spans without alphabetic tokens are now
  *n/a* instead of "not in lexicon".

### Defaults (confirmed by Elena 2026-10-28)
- `CharLMConfig` defaults (embedding 32, hidden 128, 10 epochs, batch 32,
  Adam 0.003, seed 0, `min_label_spans=20`); the surprisal model trains on
  distinct `O` token texts, label models on every span occurrence.
- Length bins `1 / 2 / 3 / 4–5 / 6+`.
- Form typicality still correlates with length (by design, not corrected).

## 0.6.0 (2026-10-15)

### Changed (breaking)
- **Exposure dimensions redefined.** A span's exact word sequence is now
  looked up at every occurrence in training together with the BIO tags it
  carries there, `O` and `I-` (inside a longer span) included. Matching is
  case-sensitive (previously lowercased) and never crosses a sentence
  boundary. New span attributes: `span._.train_count` (number of
  occurrences), `span._.label_consistency` (share carrying the span's own
  tags; `None` when unseen) and `span._.identity` (`"unseen"`,
  `"seen_same_label"`, `"seen_ambiguous"`, `"seen_other_label"`).
- Dimensions `unseen_tagging` and `type_confusable` are replaced by
  `seen_same_label`, `seen_ambiguous` and `seen_other_label`. The span
  attributes `is_unseen_tagging` / `is_type_confusable` are removed;
  `span._.is_unseen` is now a getter on `train_count`.
- `unseen` is stricter: a span whose text occurs in training only as `O`
  (or inside a longer span) is now `seen_other_label`, not `unseen`.
- `Cascabel.seen_entity_labels` is removed.

### Unchanged
- Token-level `is_unseen_token`, `is_unseen_tagging` and
  `is_type_confusable` are kept as before.

## 0.5.0 (2026-07-23)

### Changed (breaking)
- Lexicon dimensions renamed for clarity: `ambiguous` → `in_lexicon`
  (every alphabetic token of the span is an ordinary lexicon word) and
  `mixed_ambiguous` → `partial_in_lexicon` (some but not all are). The
  span getters follow: `span._.is_ambiguous` → `span._.is_in_lexicon`,
  `span._.is_mixed_ambiguous` → `span._.is_partial_in_lexicon`. The
  token-level `is_in_lexicon` attribute is unchanged.

### Fixed
- `Cascabel.scores_by_dimension` no longer discards a casing dimension
  (`titlecase`, `uppercase`) for the whole corpus when a single caseless
  entity (a number, a symbol) appears in it. A dimension is now dropped
  only when it is inapplicable for *every* gold entity (e.g. casing on a
  caseless script), matching `dimension_report`. Performance cards and
  HTML reports on cased corpora now score these dimensions.
- **Quadratic `Span.ents`**: `align_doc`, `Cascabel._overlay_prediction`
  and `Cascabel.spans_per_sentence_distribution` called spaCy's
  `Span.ents` once per sentence, which rescans the whole document's entity
  list — O(sentences × entities), ~1 hour on Few-NERD. Replaced with a
  single O(n) bucketing pass (`cascabel.doc.ents_by_sentence`); the same
  analysis now runs in seconds.

### Changed
- The HTML report's Dimensions table applies the applicability floor:
  dimensions applicable to under 25% of entities (e.g. casing on a
  caseless script with a few embedded Latin names) are reported as `n/a`
  rather than scored over an unrepresentative subset, and the lexicon
  dimensions are omitted from the table.

## 0.4.0 (2026-07-20)

### Added
- **Sentence-level bootstrap confidence intervals** (`Cascabel.bootstrap()`,
  `cascabel.bootstrap_scores`): resamples sentences with replacement and
  reports percentile CIs for span/token/boundary F1 and every applicable
  per-dimension F1 and recall. All metrics in one draw share the resample.
- Performance profile cards now attach 95% bootstrap CIs to the F1 columns
  (Metrics and Disaggregated evaluation) and report per-dimension gold
  counts; `card(bootstrap=False)` skips it. `numpy` added as an explicit
  dependency (already present via spaCy).

## 0.3.0 (2026-07-20)

### Changed (breaking)
- Shape dimensions now form two explicit axes with distinct training-set
  reference distributions:
  - **Surprisal** (vs. ordinary, non-entity tokens): the former
    `unusual_shape` is renamed `high_surprisal`, its complement
    `low_surprisal`. Token attribute `is_unusual_shape` → `is_high_surprisal`;
    `span._.shape_class` values `"unusual"`/`"usual"` → `"high"`/`"low"`.
  - **Prototypicality** (vs. training spans / entity tokens): new
    dimensions `prototypical` and `non_prototypical`, backed by a
    calibrated entity-model surprisal. New token attributes
    `entity_surprisal`, `is_non_prototypical`; new span getter
    `prototypicality_class` (`"prototypical"`/`"non_prototypical"`/`"mixed"`).
- `mixed_shape` / `prototypical_shape` (0.2.1) dimensions removed in favor of
  the two-axis scheme above.

### Added
- `ShapeScorer.entity_surprisal()` and `ShapeScorer.is_non_prototypical()`,
  with an entity-vocabulary-calibrated `entity_threshold`.

## 0.2.0 (2026-07-17)

### Added
- **Cards** (`Cascabel.card()`, CLI `-f card`): standardized markdown
  documentation artifacts — a *dataset profile card* (no prediction) or a
  *performance profile card* (with prediction), schema v1, with auto-generated
  caveats (inapplicable dimensions, small evaluation sets, missing training
  split) and a reproducibility footer.

## 0.1.1 (2026-07-17)

### Changed
- Casing dimensions (`is_titlecase`, `is_upper`, span- and token-level) now
  return ``None`` (not applicable) for text without caseable characters
  (Han script, digits, punctuation) instead of a misleading ``False``.
- `DimensionStats` gained an ``applicable`` count and an ``applicability``
  property; ``percentage`` is now computed over applicable entities only.
  Reports include both fields. Fully inapplicable dimensions are omitted,
  as before.

## 0.1.0 (2026-07-16)

First public release. Complete refactor of the original research codebase
into a general-purpose library.

### Added
- Fine-grained error taxonomy with 9 error types (`missed`, `spurious`,
  `type`, `fused`, `split`, `missegmented`, `overlap_missed`,
  `overlap_spurious`, `overlap_spurious_missed`) via `ErrorType` and
  `Alignment`.
- Three evaluation views: strict span scores, token scores, and
  token-and-separator boundary scores, all as `Score` objects.
- Statistical token-shape scoring (`cascabel.shape`): character n-gram
  surprisal with auto-calibrated threshold, entity-vs-background log-odds,
  Jensen–Shannon divergence diagnostics, word-shape templates. Replaces the
  hand-written Spanish phonotactics as the default shape signal.
- Dimension registry and reports (`dimension_report`,
  `Cascabel.scores_by_dimension`).
- CoNLL column-file reader and BIO repair utilities.
- Perturbation functions `recase`, `add_quotes`, `strip_quotes`.
- Optional extra `cascabel[spanish]` with the original Spanish phonotactics
  rules (`cascabel.contrib.spanish`).
- Reports: `Cascabel.report()` (JSON-serializable dict), `Cascabel.summary()`
  (aligned plain text) and `Cascabel.to_html()` (self-contained color-coded
  HTML error report).
- `Cascabel.scores_by_label()`: per-entity-type decomposition of the span
  scores (sums exactly to the aggregate).
- Command-line interface: `cascabel` console script / `python -m cascabel`
  with text, JSON and HTML output.
- Test suite (144 tests) and PyPI packaging.

### Changed
- Package identity: published as `cascabel` (previously mispackaged as
  `pylazaro`).
- Error type `splitted` renamed to `split`.
- `attrs` dependency replaced with standard-library dataclasses; dependency
  footprint reduced to spaCy only.

### Fixed
- Syntax error that prevented the package from importing.
- Inverted titlecase span getter.
- Token-level quotation detection comparing `Token` objects to strings
  (was always false).
- Zero-division guards in precision/recall/F1.
- Potential infinite loop in the alignment walk on unclassifiable overlaps.
- Mutable default arguments shared across instances.
- Broken/undefined attributes (`doc_extensions`, `sent_len`, `has_adjacent`)
  in the bucket and segmentation-success reports.
