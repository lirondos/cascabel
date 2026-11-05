"""Sentence-level bootstrap confidence intervals for cascabel scores.

The evaluation unit is the sentence: to estimate the uncertainty of a score,
we resample sentences with replacement and recompute. All metrics in one
draw use the *same* resample, so the intervals are jointly valid. Reported
intervals are percentile bootstrap intervals; the point estimate is the
statistic on the full data, not the bootstrap mean.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from cascabel.dimensions import ENTITY_DIMENSIONS, SpanPredicate
from cascabel.doc import bio_tag


def _f1(tp: float, fp: float, fn: float) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _recall(tp: float, fp: float, fn: float) -> float:
    return tp / (tp + fn) if tp + fn else 0.0


def bootstrap_scores(
    casc,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    dimensions: Optional[Mapping[str, SpanPredicate]] = None,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs over resampled sentences.

    Returns a dict keyed by ``"span"``, ``"token"``, ``"boundary"`` (F1) and
    by dimension name (``"<dim>.f1"`` and ``"<dim>.recall"``); each value is
    ``{"point": .., "lo": .., "hi": ..}`` at the ``1 - alpha`` level. Only
    dimensions that are available (predicate not ``None`` for every gold
    span) are included, matching :meth:`Cascabel.scores_by_dimension`.
    """
    import numpy as np

    casc._require_prediction()
    sents = list(casc.test.sents)
    n_sents = len(sents)
    if n_sents == 0:
        return {}

    align_by_sent: Dict[int, list] = {s.start: [] for s in sents}
    for alignment in casc.alignments:
        align_by_sent[alignment.sentence.start].append(alignment)

    dims = dict(dimensions) if dimensions is not None else dict(ENTITY_DIMENSIONS)

    span = np.zeros((n_sents, 3))
    boundary = np.zeros((n_sents, 3))
    token = np.zeros((n_sents, 3))
    dim_arrays = {name: np.zeros((n_sents, 3)) for name in dims}
    dim_available = {name: False for name in dims}

    for i, sent in enumerate(sents):
        for alignment in align_by_sent[sent.start]:
            sc = alignment.span_score
            span[i] += (sc.tp, sc.fp, sc.fn)
            bc = alignment.boundary_score
            boundary[i] += (bc.tp, bc.fp, bc.fn)
            if alignment.gold:
                for name, predicate in dims.items():
                    flags = [predicate(g) for g in alignment.gold]
                    if all(flag is None for flag in flags):
                        continue
                    dim_available[name] = True
                    if any(flags):
                        dim_arrays[name][i] += (sc.tp, sc.fp, sc.fn)
        for tok in sent:
            gold, pred = bio_tag(tok), tok._.iob_pred
            if gold == "O" and pred == "O":
                continue
            if pred == gold:
                if gold != "O":
                    token[i, 0] += 1
            else:
                if gold != "O":
                    token[i, 2] += 1
                if pred != "O":
                    token[i, 1] += 1

    available = {name: arr for name, arr in dim_arrays.items()
                 if dim_available[name]}

    rng = np.random.default_rng(seed)
    # Draw all resamples up front; loop per draw to bound memory on large sets.
    draws = rng.integers(0, n_sents, size=(n, n_sents))
    view_f1_b = {"span": [], "token": [], "boundary": []}
    view_recall_b = {"span": [], "token": [], "boundary": []}
    view_arr = {"span": span, "token": token, "boundary": boundary}
    dim_f1_b = {name: [] for name in available}
    dim_recall_b = {name: [] for name in available}
    for draw in draws:
        for view, arr in view_arr.items():
            v = arr[draw].sum(0)
            view_f1_b[view].append(_f1(*v))
            view_recall_b[view].append(_recall(*v))
        for name, arr in available.items():
            d = arr[draw].sum(0)
            dim_f1_b[name].append(_f1(*d))
            dim_recall_b[name].append(_recall(*d))

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def interval(samples, point):
        return {"point": point,
                "lo": float(np.percentile(samples, lo_q)),
                "hi": float(np.percentile(samples, hi_q))}

    points = {"span": casc.span_scores(), "token": casc.token_scores(),
              "boundary": casc.boundary_scores()}
    result: Dict[str, Dict[str, float]] = {}
    for view, score in points.items():
        result[view] = interval(view_f1_b[view], score.f1)
        result[f"{view}.recall"] = interval(view_recall_b[view], score.recall)
    by_dim = casc.scores_by_dimension({name: dims[name] for name in available})
    for name in available:
        score = by_dim.get(name)
        if score is None:
            continue
        result[f"{name}.f1"] = interval(dim_f1_b[name], score.f1)
        result[f"{name}.recall"] = interval(dim_recall_b[name], score.recall)
    return result
