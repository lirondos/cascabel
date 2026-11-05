"""Precision/recall/F1 containers and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class Score:
    """A count of true positives, false positives and false negatives.

    Scores are immutable and can be added together to aggregate results
    across evaluation units (entities, sentences, buckets, ...).
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Score") -> "Score":
        if not isinstance(other, Score):
            return NotImplemented
        return Score(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    __radd__ = __add__

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if not denominator:
            return 0.0
        return 2 * self.precision * self.recall / denominator

    @classmethod
    def aggregate(cls, scores: Iterable["Score"]) -> "Score":
        total = cls()
        for score in scores:
            total = total + score
        return total

    def as_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} "
            f"(tp={self.tp} fp={self.fp} fn={self.fn})"
        )
