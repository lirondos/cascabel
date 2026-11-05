"""Report builders: machine-readable dict, plain-text summary, and HTML.

These functions take a :class:`cascabel.Cascabel` instance and turn its
analyses into consumable artifacts. They are exposed on the facade as
``Cascabel.report()``, ``Cascabel.summary()`` and ``Cascabel.to_html()``.
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from cascabel.alignment import ErrorType
from cascabel.dimensions import DIMENSION_GLOSSES
from cascabel.doc import bio_tag

#: Dimensions omitted from the HTML report's Dimensions table (rarely
#: applicable and generally noise for a reader).
_HIDDEN_DIMENSIONS = {"in_lexicon", "partial_in_lexicon"}

#: Below this share of applicable entities a dimension is reported as n/a
#: (matches ``cascabel.card.APPLICABILITY_FLOOR``): e.g. casing on a script
#: that is caseless apart from a handful of embedded Latin-script names.
_APPLICABILITY_FLOOR = 25.0

#: Tokens of context shown on each side of a dimension example.
EXAMPLE_WINDOW = 3


def dimension_example(spans, window: int = EXAMPLE_WINDOW) -> Optional[dict]:
    """One example span for a dimension, with a little context.

    The example is the most frequent surface form among ``spans`` (ties:
    the one that occurs first), taken at its first occurrence. Returns
    ``{"text", "label", "left", "right"}``, where ``left`` and ``right``
    are up to ``window`` tokens of the same sentence (``"…"`` marks that
    the sentence goes on), or ``None`` when there are no spans.
    """
    if not spans:
        return None
    counts = Counter(span.text for span in spans)
    text = max(counts, key=counts.__getitem__)
    span = next(span for span in spans if span.text == text)
    sent, doc = span.sent, span.doc
    start = max(sent.start, span.start - window)
    end = min(sent.end, span.end + window)
    left = ("… " if start > sent.start else "") + doc[start:span.start].text
    right = doc[span.end:end].text + (" …" if end < sent.end else "")
    return {"text": span.text, "label": span.label_,
            "left": left.strip(), "right": right.strip()}


def _example_text(example: Optional[dict]) -> str:
    """``… in [Nueva York] LOC .``: the span in brackets, then its label."""
    if not example:
        return ""
    return " ".join(part for part in (
        example["left"], f"[{example['text']}] {example['label']}", example["right"]
    ) if part)


def _example_html(example: Optional[dict]) -> str:
    if not example:
        return ""
    span = (f"<b>{html.escape(example['text'])}</b>"
            f"<sub>{html.escape(example['label'])}</sub>")
    return " ".join(part for part in (
        html.escape(example["left"]), span, html.escape(example["right"])
    ) if part)

def shape_summary(casc, split: str = "test", n_examples: int = 5) -> dict:
    """Per label, the least and the most prototypical distinct spans of a
    split: ``{label: {"least_prototypical": [texts], "most_prototypical":
    [texts]}}``, most extreme first, labels in alphabetical order.

    Reports show these lists only, without scores; the scores themselves
    (bits/char and perplexity) are span attributes and columns of
    :meth:`cascabel.Cascabel.span_table`. Labels without a prototypicality
    model (fewer than ``min_label_spans`` training spans) are left out; the
    result is empty without any (no training set, ``shape=False``, or the
    training split).
    """
    labels = sorted({ent.label_ for ent in casc._split(split).ents
                     if ent._.prototypicality is not None})

    def texts(label: str, least: bool) -> List[str]:
        rows = casc.extreme_spans("prototypicality", n=n_examples, least=least,
                                  label=label, split=split)
        return [row["text"] for row in rows]

    return {label: {"least_prototypical": texts(label, least=False),
                    "most_prototypical": texts(label, least=True)}
            for label in labels}

# ---------------------------------------------------------------------------
# Machine-readable report
# ---------------------------------------------------------------------------


def build_report(casc) -> dict:
    """Assemble every available analysis into a JSON-serializable dict.

    Sections that need a prediction (scores, errors, diagnostics) are only
    present when one is attached; the dataset profile is always included.
    """
    report: dict = {
        "dataset": {
            "tokens": casc.n_tokens(),
            "sentences": casc.n_sentences(),
            "entities": casc.n_entities(),
            "labels": dict(casc.label_distribution()),
            "span_lengths": {
                str(k): v for k, v in sorted(casc.span_length_distribution().items())
            },
        },
        "dimensions": {
            name: {
                "count": stats.count,
                "total": stats.total,
                "applicable": stats.applicable,
                "percentage": round(stats.percentage, 2),
                "applicability": round(stats.applicability, 2),
                "example": dimension_example(stats.spans),
            }
            for name, stats in casc.dimension_report().items()
        },
    }
    identity = casc.identity_summary("test")
    if identity:
        report["identity"] = identity
    shape = shape_summary(casc, "test")
    if shape:
        report["shape"] = shape
    if casc.prediction is None:
        return report

    report["scores"] = {
        "span": casc.span_scores().as_dict(),
        "token": casc.token_scores().as_dict(),
        "boundary": casc.boundary_scores().as_dict(),
    }
    report["scores_by_label"] = {
        label: score.as_dict() for label, score in casc.scores_by_label().items()
    }
    report["scores_by_dimension"] = {
        name: score.as_dict() for name, score in casc.scores_by_dimension().items()
    }
    report["precision_by_dimension"] = {
        name: score.as_dict()
        for name, score in casc.scores_by_dimension(side="predicted").items()
    }
    report["error_profile"] = {
        side: casc.error_profile(side) for side in ("gold", "predicted")
    }
    # Every error type, zeros included, so that absent errors are visible.
    counts = casc.error_counts()
    report["error_counts"] = {e.value: counts.get(e.value, 0) for e in ErrorType}
    report["diagnostics"] = {
        "retrieval_capacity": casc.retrieval_capacity(),
        "noise_level": casc.noise_level(),
        "segmentation_error_rate": casc.segmentation_error_rate(),
        "segmentation_success_rate": casc.segmentation_success_rate(),
    }
    return report


# ---------------------------------------------------------------------------
# Plain-text summary
# ---------------------------------------------------------------------------


def _score_row(name: str, data: dict, width: int = 22) -> str:
    return (
        f"  {name:<{width}}"
        f"{data['precision']:>8.4f}{data['recall']:>8.4f}{data['f1']:>8.4f}"
        f"   (tp={data['tp']} fp={data['fp']} fn={data['fn']})"
    )


def _share(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


#: One-line meaning of each score view, shown under every Scores section.
SCORE_GLOSSES = {
    "span": "a span counts only if start, end and label are all exact",
    "token": "each entity token counts if its full BIO tag (B-PER, I-PER...) is right",
    "boundary": "like span, but the label is ignored: a span counts if its "
                "start and end are exact",
}


def gloss(dimension: str) -> str:
    """One-line meaning of a dimension (empty if none is registered)."""
    return DIMENSION_GLOSSES.get(dimension, "")


#: How to read the shape lists; shown under every Shape section.
SHAPE_NOTE = ("Prototypicality: how much a span's form looks like the other "
              "spans of its label in training (one character language model "
              "per label).")

#: ``(key, title)`` of the per-label shape lists, in display order.
SHAPE_LISTS = (("most_prototypical", "most prototypical"),
               ("least_prototypical", "least prototypical"))


def identity_rows(identity: dict) -> List[tuple]:
    """``(name, count, denominator, share)`` rows of an identity summary:
    the three radar axes first, then the two shares of new spans."""
    return [
        ("new (never seen in training)", identity["n_unseen"], identity["n_spans"],
         identity["unseen_share"]),
        ("multi-label (of seen)", identity["n_multi_label"], identity["n_seen"],
         identity["multi_label_share"]),
        ("unseen label (of seen)", identity["n_unseen_label"], identity["n_seen"],
         identity["unseen_label_share"]),
        ("new, word seen in a span", identity["n_unseen_familiar_word"],
         identity["n_unseen"], identity["unseen_familiar_word_share"]),
        ("new, seen as spelling variant", identity["n_unseen_variant"],
         identity["n_unseen"], identity["unseen_variant_share"]),
    ]


def _identity_lines(identity: dict) -> List[str]:
    """Identity against training: the three radar axes, then extras."""
    lines = ["Identity (test spans vs training)"]
    lines += [f"  {name:<32}{part:>6} / {whole:<6}{_share(share):>7}"
              for name, part, whole, share in identity_rows(identity)]
    lines.append("")
    return lines


def _shape_lines(shape: dict) -> List[str]:
    """Shape: the least and most prototypical spans, without scores."""
    lines = ["Shape", f"  {SHAPE_NOTE}"]
    for label, lists in shape.items():
        lines.append(f"  {label}")
        for key, title in SHAPE_LISTS:
            lines.append(f"    {title + ':':<20}{', '.join(lists[key])}")
    lines.append("")
    return lines


def build_summary(casc) -> str:
    """Render the report as an aligned plain-text summary."""
    report = build_report(casc)
    lines: List[str] = ["cascabel report", "=" * 15, ""]

    dataset = report["dataset"]
    labels = ", ".join(f"{k}: {v}" for k, v in sorted(dataset["labels"].items()))
    lines.append(
        f"Dataset: {dataset['tokens']} tokens, {dataset['sentences']} sentences, "
        f"{dataset['entities']} entities ({labels or 'no labels'})"
    )
    lines.append("")

    if "scores" in report:
        header = f"  {'':<22}{'P':>8}{'R':>8}{'F1':>8}"
        lines.append("Scores")
        lines.append(header)
        for name in ("span", "token", "boundary"):
            lines.append(_score_row(name, report["scores"][name]))
        lines += [f"  {name}: {text}" for name, text in SCORE_GLOSSES.items()]
        lines.append("")

        if report["scores_by_label"]:
            lines.append("Scores by label")
            lines.append(header)
            for label, data in report["scores_by_label"].items():
                lines.append(_score_row(label, data))
            lines.append("")

        total_errors = sum(report["error_counts"].values())
        lines.append(f"Errors ({total_errors})")
        if report["error_counts"]:
            for error_type, count in sorted(
                report["error_counts"].items(), key=lambda item: -item[1]
            ):
                lines.append(f"  {error_type:<26}{count:>5}")
        else:
            lines.append("  none")
        lines.append("")

        lines.append("Diagnostics")
        for name, value in report["diagnostics"].items():
            lines.append(f"  {name:<28}{value:>8.4f}")
        lines.append("")

    if "identity" in report:
        lines.extend(_identity_lines(report["identity"]))
    if "shape" in report:
        lines.extend(_shape_lines(report["shape"]))

    if report["dimensions"]:
        by_dimension = report.get("scores_by_dimension", {})
        width = max(len(gloss(name)) for name in report["dimensions"]) + 2
        if by_dimension:
            lines.append("Dimensions")
            lines.append(f"  {'':<22}{'':<{width}}{'n':>5}{'%':>7}{'P':>8}{'R':>8}{'F1':>8}"
                         "  example")
            for name, stats in report["dimensions"].items():
                row = (f"  {name:<22}{gloss(name):<{width}}"
                       f"{stats['count']:>5}{stats['percentage']:>7.1f}")
                if name in by_dimension:
                    data = by_dimension[name]
                    row += (
                        f"{data['precision']:>8.4f}{data['recall']:>8.4f}"
                        f"{data['f1']:>8.4f}"
                    )
                else:
                    row += " " * 24
                lines.append(f"{row}  {_example_text(stats.get('example'))}".rstrip())
        else:
            lines.append("Dimensions")
            lines.append(f"  {'':<22}{'':<{width}}{'n':>5}{'%':>7}  example")
            for name, stats in report["dimensions"].items():
                lines.append(
                    f"  {name:<22}{gloss(name):<{width}}"
                    f"{stats['count']:>5}{stats['percentage']:>7.1f}"
                    f"  {_example_text(stats.get('example'))}".rstrip()
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

#: One colour per error type. Chosen so that every pair differs by at least
#: 15 (OKLab x100) for normal colour vision and every colour differs from the
#: green of correct spans; nine hues cannot all stay apart under colour
#: blindness, so the error name is always shown too (chips, hover text).
_ERROR_COLORS = {
    "missed": "#c2187a",
    "spurious": "#f77f00",
    "type": "#7b2cbf",
    "fused": "#1d63d6",
    "split": "#5bc0eb",
    "missegmented": "#7f4f24",
    "overlap_missed": "#ff8fc7",
    "overlap_spurious": "#f2c200",
    "overlap_spurious_missed": "#1f1f1f",
}

#: Colour of correctly predicted spans in the sentence view.
_CORRECT_COLOR = "#2e9e5b"


def _ink(color: str) -> str:
    """Dark or white text, whichever reads better on ``color``."""
    def channel(value: int) -> float:
        value /= 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(int(color[i:i + 2], 16)) for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#222" if luminance > 0.25 else "#fff"


def _chip(error: str) -> str:
    color = _ERROR_COLORS.get(error, "#7f8c8d")
    return (f'<span class="chip" style="background:{color};color:{_ink(color)}">'
            f"{error}</span>")


def _highlight_style(color: str) -> str:
    """Light tint of a colour with the full colour underneath."""
    return f"background:{color}40;border-bottom:3px solid {color}"

_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       margin: 2rem auto; max-width: 60rem; color: #222; padding: 0 1rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
.cards { display: flex; gap: 1rem; flex-wrap: wrap; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: .8rem 1.2rem; }
.card b { display: block; font-size: 1.3rem; }
.card small { color: #666; }
table { border-collapse: collapse; margin-top: .5rem; }
th, td { text-align: left; padding: .25rem .8rem; border-bottom: 1px solid #eee; }
th { color: #666; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.ent { padding: .1rem .25rem; border-radius: 4px; }
.sentence { margin: 1rem 0; padding: .8rem 1rem; border: 1px solid #eee;
            border-radius: 8px; line-height: 2; }
.chips { margin-top: .4rem; line-height: 1.8; }
.chip { display: inline-block; color: #fff; border-radius: 10px;
        padding: 0 .6rem; font-size: .78rem; margin-right: .4rem; }
.detail { color: #555; font-size: .85rem; margin-right: 1rem; }
.legend span { margin-right: 1rem; }
footer { margin-top: 2rem; color: #999; font-size: .8rem; }
"""


def _spans_label(spans) -> str:
    return ", ".join(f"{s.text} ({s.label_})" for s in spans) or "nothing"


def _highlights(alignments) -> Dict[int, Tuple[int, str, str]]:
    """Start token -> (end token, colour, hover text) of every highlighted
    span. The *predicted* spans are highlighted, as the model output them
    (so gold *Ann Battelle* predicted as *Ann* highlights *Ann* only); a
    missed span, with no prediction, highlights the gold span."""
    marks: Dict[int, Tuple[int, str, str]] = {}
    for alignment in alignments:
        error = alignment.error_type.value if alignment.is_error else "correct"
        color = _ERROR_COLORS.get(error, _CORRECT_COLOR if error == "correct" else "#7f8c8d")
        title = (f"{error} | gold: {_spans_label(alignment.gold)} | "
                 f"predicted: {_spans_label(alignment.predicted)}")
        for span in alignment.predicted or alignment.gold:
            marks[span.start] = (span.end, color, title)
    return marks


def _render_sentence(sentence, alignments) -> str:
    parts = []
    marks = _highlights(alignments)
    i = sentence.start
    while i < sentence.end:
        if i in marks:
            end, color, title = marks[i]
            end = min(end, sentence.end)
            text = html.escape(" ".join(t.text for t in sentence.doc[i:end]))
            parts.append(f'<span class="ent" style="{_highlight_style(color)}" '
                         f'title="{html.escape(title)}">{text}</span>')
            i = end
        else:
            parts.append(html.escape(sentence.doc[i].text))
            i += 1
    chips = []
    for alignment in alignments:
        if not alignment.is_error:
            continue
        error = alignment.error_type.value
        # Each span with its label, so type errors show both labels.
        gold_texts = ", ".join(
            f"“{html.escape(g.text)}” <b>{html.escape(g.label_)}</b>" for g in alignment.gold
        ) or "∅"
        pred_texts = ", ".join(
            f"“{html.escape(p.text)}” <b>{html.escape(p.label_)}</b>" for p in alignment.predicted
        ) or "∅"
        chips.append(
            _chip(error)
            + f'<span class="detail">gold {gold_texts} → predicted {pred_texts}</span>'
        )
    chips_html = f'<div class="chips">{"".join(chips)}</div>' if chips else ""
    return f'<div class="sentence">{" ".join(parts)}{chips_html}</div>'


def build_html(casc, only_errors: bool = True) -> str:
    """Render a self-contained HTML error report.

    ``only_errors`` limits the sentence view to sentences containing at
    least one alignment error (the summary always covers everything).
    """
    report = build_report(casc)
    parts: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>cascabel report</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>cascabel report 🔔</h1>",
    ]
    dataset = report["dataset"]
    labels = ", ".join(f"{k}: {v}" for k, v in sorted(dataset["labels"].items()))
    parts.append(
        f"<p>{dataset['tokens']} tokens · {dataset['sentences']} sentences · "
        f"{dataset['entities']} entities ({html.escape(labels or 'no labels')})</p>"
    )

    if "scores" in report:
        parts.append('<div class="cards">')
        for name in ("span", "token", "boundary"):
            data = report["scores"][name]
            parts.append(
                f'<div class="card"><small>{name} F1</small><b>{data["f1"]:.4f}</b>'
                f'<small>P {data["precision"]:.4f} · R {data["recall"]:.4f}</small></div>'
            )
        parts.append("</div>")
        parts.append(
            "<p class='legend'>"
            + "<br>".join(f"<b>{name}</b>: {html.escape(text)}"
                          for name, text in SCORE_GLOSSES.items())
            + "</p>"
        )

        if report["error_counts"]:
            parts.append("<h2>Errors</h2><table><tr><th>type</th>"
                         "<th class='num'>count</th></tr>")
            for error_type, count in sorted(
                report["error_counts"].items(), key=lambda item: -item[1]
            ):
                parts.append(
                    f"<tr><td>{_chip(error_type)}</td><td class='num'>{count}</td></tr>"
                )
            parts.append("</table>")

        if report["scores_by_label"]:
            parts.append(
                "<h2>Scores by label</h2><table><tr><th>label</th>"
                "<th class='num'>P</th><th class='num'>R</th><th class='num'>F1</th>"
                "<th class='num'>tp</th><th class='num'>fp</th><th class='num'>fn</th></tr>"
            )
            for label, data in report["scores_by_label"].items():
                parts.append(
                    f"<tr><td>{html.escape(label)}</td>"
                    f"<td class='num'>{data['precision']:.4f}</td>"
                    f"<td class='num'>{data['recall']:.4f}</td>"
                    f"<td class='num'>{data['f1']:.4f}</td>"
                    f"<td class='num'>{data['tp']}</td><td class='num'>{data['fp']}</td>"
                    f"<td class='num'>{data['fn']}</td></tr>"
                )
            parts.append("</table>")

    if "identity" in report:
        parts.append(
            "<h2>Identity (test spans vs training)</h2><table><tr><th>measure</th>"
            "<th class='num'>count</th><th class='num'>out of</th>"
            "<th class='num'>share</th></tr>"
        )
        for name, part, whole, share in identity_rows(report["identity"]):
            parts.append(
                f"<tr><td>{name}</td><td class='num'>{part}</td>"
                f"<td class='num'>{whole}</td><td class='num'>{_share(share)}</td></tr>"
            )
        parts.append("</table>")

    if "shape" in report:
        shape = report["shape"]
        parts.append(
            f"<h2>Shape</h2><p class='legend'>{html.escape(SHAPE_NOTE)}</p>"
            "<table><tr><th>label</th>"
            + "".join(f"<th>{title}</th>" for _, title in SHAPE_LISTS)
            + "</tr>"
        )
        for label, lists in shape.items():
            cells = "".join(
                "<td>" + ", ".join(f"<i>{html.escape(t)}</i>" for t in lists[key]) + "</td>"
                for key, _ in SHAPE_LISTS
            )
            parts.append(f"<tr><td>{html.escape(label)}</td>{cells}</tr>")
        parts.append("</table>")

    if report["dimensions"]:
        by_dimension = report.get("scores_by_dimension", {})
        parts.append(
            "<h2>Dimensions</h2><table><tr><th>dimension</th><th>meaning</th>"
            "<th>example</th><th class='num'>n</th>"
            "<th class='num'>%</th><th class='num'>P</th><th class='num'>R</th>"
            "<th class='num'>F1</th></tr>"
        )
        for name, stats in report["dimensions"].items():
            if name in _HIDDEN_DIMENSIONS:
                continue
            # A barely-applicable dimension (e.g. casing on a caseless script
            # with a few embedded Latin names) is reported as n/a rather than
            # scored over an unrepresentative handful of entities.
            if stats["applicability"] < _APPLICABILITY_FLOOR:
                parts.append(
                    f"<tr><td>{name}</td><td>{html.escape(gloss(name))}</td>"
                    f"<td>{_example_html(stats.get('example'))}</td>"
                    f"<td class='num'>{stats['count']}</td>"
                    "<td class='num'>n/a</td><td class='num'>n/a</td>"
                    "<td class='num'>n/a</td><td class='num'>n/a</td></tr>"
                )
                continue
            row = (
                f"<tr><td>{name}</td><td>{html.escape(gloss(name))}</td>"
                f"<td>{_example_html(stats.get('example'))}</td>"
                f"<td class='num'>{stats['count']}</td>"
                f"<td class='num'>{stats['percentage']:.1f}</td>"
            )
            if name in by_dimension:
                data = by_dimension[name]
                row += (
                    f"<td class='num'>{data['precision']:.4f}</td>"
                    f"<td class='num'>{data['recall']:.4f}</td>"
                    f"<td class='num'>{data['f1']:.4f}</td>"
                )
            else:
                row += "<td class='num'>—</td><td class='num'>—</td><td class='num'>—</td>"
            parts.append(row + "</tr>")
        parts.append("</table>")

    if casc.prediction is not None:
        parts.append("<h2>Sentences</h2>")
        parts.append(
            '<p class="legend">Highlights show the spans as predicted '
            "(missed spans: the gold span); hover for gold and prediction. "
            f'<span class="ent" style="{_highlight_style(_CORRECT_COLOR)}">correct</span>'
            + "".join(
                f'<span class="ent" style="{_highlight_style(color)}">{error}</span>'
                for error, color in _ERROR_COLORS.items()
            )
            + "</p>"
        )
        alignments_by_sentence: Dict[int, list] = {}
        for alignment in casc.alignments:
            alignments_by_sentence.setdefault(alignment.sentence.start, []).append(
                alignment
            )
        shown = 0
        for sentence in casc.test.sents:
            sentence_alignments = alignments_by_sentence.get(sentence.start, [])
            has_error = any(a.is_error for a in sentence_alignments)
            if only_errors and not has_error:
                continue
            parts.append(_render_sentence(sentence, sentence_alignments))
            shown += 1
        if not shown:
            parts.append("<p>No sentences with errors 🎉</p>")

    from cascabel import __version__

    parts.append(f"<footer>generated by cascabel {__version__}</footer>")
    parts.append("</body></html>")
    return "".join(parts)


def write_html(
    casc, path: Union[str, Path], only_errors: bool = True
) -> Path:
    """Write the HTML report to ``path`` and return it."""
    path = Path(path)
    path.write_text(build_html(casc, only_errors=only_errors), encoding="utf-8")
    return path
