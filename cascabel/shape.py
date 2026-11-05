"""Token shape templates.

- :func:`word_shape` — a token's shape template (``"Fitbit"`` -> ``"Xxx"``).

Span-level surprisal and prototypicality scores come from the neural
character language models in :mod:`cascabel.charlm` (since 0.7.0; the
character n-gram scorer and the corpus-level Jensen–Shannon divergence were
removed).
"""

from __future__ import annotations

from typing import List

__all__ = ["word_shape"]


def word_shape(text: str, max_run: int = 2) -> str:
    """Map a token to its shape template, e.g. ``"Fitbit"`` -> ``"Xxx"``.

    Uppercase letters become ``X``, lowercase ``x``, digits ``d``; any other
    character is kept as is. Runs of the same class longer than ``max_run``
    are collapsed.
    """
    shape: List[str] = []
    run_char = ""
    run_length = 0
    for char in text:
        if char.isalpha():
            mapped = "X" if char.isupper() else "x"
        elif char.isdigit():
            mapped = "d"
        else:
            mapped = char
        if mapped == run_char:
            run_length += 1
        else:
            run_char, run_length = mapped, 1
        if run_length <= max_run:
            shape.append(mapped)
    return "".join(shape)
