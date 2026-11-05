"""Character-level neural language models for span *shape*.

Two questions are answered with the same kind of model, a character LSTM
language model that reads a string letter by letter (spaces included, case
preserved):

- **Surprisal** — how unlike the ordinary (non-entity, ``O``) tokens of the
  training set is a token? One model is trained on the training ``O``
  tokens; a token's surprisal is its bits per character under that model.
- **Prototypicality** — how much does a span look like the training spans of
  its label? One model is trained per label on the training spans of that
  label (tokens joined by single spaces). Because the model reads the whole
  span, it sees both odd character combinations (*Bigotes*) and odd word
  patterns (*El Bigotes*: article + noun for a ``PER``).

  - *Form typicality* (``span._.prototypicality``) is the span's bits per
    character under its own label's model: low = typical.
  - *Label fit* (``span._.label_fit``) is how much better the span fits its
    own label than the best-fitting other label, in bits per character:
    negative = it looks more like another label (*Imperio Argentina* as
    ``PER`` looking like a ``LOC``).
- **Span typicality** — does the string look like a span at all, whatever
  its label? One model is trained on every training span, labels ignored
  (``span._.span_typicality``, bits per character: low = span-like). It
  separates "a span, just not a typical ``PER``" (low span typicality, high
  prototypicality) from "does not look like a span" (both high).

Scores are relative to one training set: they are not comparable across
datasets in absolute terms.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = ["CharLM", "CharLMConfig", "ShapeModels"]

_PAD, _BOUNDARY, _UNK = 0, 1, 2  # padding, begin/end of string, unknown char
_FIRST_CHAR_ID = 3


@dataclass
class CharLMConfig:
    """Hyperparameters shared by every character language model.

    The defaults are provisional (chosen from a small benchmark on CoNLL-03
    ``PER`` spans) and still to be validated, e.g. with a seed-stability
    check.

    Parameters
    ----------
    embedding_size : int
        Size of the character embeddings.
    hidden_size : int
        Size of the LSTM hidden state.
    epochs : int
        Passes over the training strings (fixed; no early stopping, so every
        training string is seen by the final model).
    batch_size : int
        Strings per optimization step.
    learning_rate : float
        Adam learning rate.
    seed : int
        Random seed for initialization and shuffling.
    min_label_spans : int
        Labels with fewer training spans get no prototypicality model
        (their spans' prototypicality is ``None``).
    device : str
        Torch device, e.g. ``"cpu"`` or ``"cuda"``.
    """

    embedding_size: int = 32
    hidden_size: int = 128
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.003
    seed: int = 0
    min_label_spans: int = 20
    device: str = "cpu"


class _LSTMNetwork(nn.Module):
    def __init__(self, vocabulary_size: int, config: CharLMConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, config.embedding_size, padding_idx=_PAD
        )
        self.lstm = nn.LSTM(
            config.embedding_size, config.hidden_size, batch_first=True
        )
        self.output = nn.Linear(config.hidden_size, vocabulary_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(self.embedding(ids))
        return self.output(hidden)


class CharLM:
    """A character LSTM language model over whole strings.

    Each string is scored with a begin marker before it and an end marker
    after it, so the end of the string is one more predicted event.

    Parameters
    ----------
    config : CharLMConfig, optional
        Hyperparameters; defaults to ``CharLMConfig()``.
    """

    def __init__(self, config: Optional[CharLMConfig] = None):
        self.config = config or CharLMConfig()
        self._char_ids: Dict[str, int] = {}
        self._network: Optional[_LSTMNetwork] = None

    @property
    def is_fitted(self) -> bool:
        return self._network is not None

    def _encode(self, text: str) -> List[int]:
        ids = [self._char_ids.get(char, _UNK) for char in text]
        return [_BOUNDARY] + ids + [_BOUNDARY]

    def _batch(self, encoded: Sequence[List[int]]) -> torch.Tensor:
        width = max(len(ids) for ids in encoded)
        batch = torch.full((len(encoded), width), _PAD, dtype=torch.long)
        for row, ids in enumerate(encoded):
            batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return batch.to(self.config.device)

    def fit(self, texts: Sequence[str]) -> "CharLM":
        """Train the model on a collection of strings.

        Parameters
        ----------
        texts : sequence of str
            Training strings; empty strings are ignored. Repeated strings
            count once per occurrence.

        Returns
        -------
        CharLM
            ``self``.
        """
        texts = [text for text in texts if text]
        if not texts:
            raise ValueError("Cannot fit a CharLM on an empty collection")
        config = self.config
        chars = sorted({char for text in texts for char in text})
        self._char_ids = {char: i + _FIRST_CHAR_ID for i, char in enumerate(chars)}
        torch.manual_seed(config.seed)
        shuffler = random.Random(config.seed)
        network = _LSTMNetwork(len(chars) + _FIRST_CHAR_ID, config).to(config.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        loss_function = nn.CrossEntropyLoss(ignore_index=_PAD)
        encoded = [self._encode(text) for text in texts]
        network.train()
        for _ in range(config.epochs):
            shuffler.shuffle(encoded)
            for start in range(0, len(encoded), config.batch_size):
                batch = self._batch(encoded[start : start + config.batch_size])
                logits = network(batch[:, :-1])
                loss = loss_function(logits.transpose(1, 2), batch[:, 1:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        network.eval()
        self._network = network
        return self

    def log2_probabilities(
        self, texts: Sequence[str], batch_size: int = 512
    ) -> List[float]:
        """Total log2 probability of each string (end event included).

        Parameters
        ----------
        texts : sequence of str
            Strings to score.
        batch_size : int
            Strings scored per forward pass.

        Returns
        -------
        list of float
            One value per string, in input order.
        """
        if self._network is None:
            raise RuntimeError("CharLM must be fitted before scoring")
        results: List[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = [self._encode(text) for text in texts[start : start + batch_size]]
                batch = self._batch(chunk)
                targets = batch[:, 1:]
                log_probs = torch.log_softmax(self._network(batch[:, :-1]), dim=-1)
                picked = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                picked = picked.masked_fill(targets == _PAD, 0.0)
                results.extend((picked.sum(dim=1) / math.log(2)).tolist())
        return results

    def bits_per_char(self, texts: Sequence[str]) -> List[float]:
        """Mean negative log2 probability per character event.

        A string of ``n`` characters has ``n + 1`` events (each character
        plus the end of the string).
        """
        totals = self.log2_probabilities(texts)
        return [-total / (len(text) + 1) for total, text in zip(totals, texts)]


def span_text(tokens: Sequence[str]) -> str:
    """The string a span is scored as: its tokens joined by single spaces."""
    return " ".join(tokens)


class ShapeModels:
    """The surprisal model, the any-span model and the per-label
    prototypicality models of one training set.

    Parameters
    ----------
    background_tokens : sequence of str
        Texts of the training ``O`` tokens. The model is trained on the
        distinct texts (the shape of the ordinary *vocabulary*, so that very
        frequent words do not dominate).
    label_spans : sequence of (str, str)
        ``(label, span_text)`` for every training span, one per occurrence
        (frequent spans weigh more: prototypicality is familiarity). All of
        them, labels ignored, train the any-span model (rare labels without
        a model of their own included).
    config : CharLMConfig, optional
        Hyperparameters shared by every model.
    """

    def __init__(
        self,
        background_tokens: Sequence[str],
        label_spans: Sequence[Tuple[str, str]] = (),
        config: Optional[CharLMConfig] = None,
    ):
        self.config = config or CharLMConfig()
        vocabulary = sorted({token for token in background_tokens if token})
        if not vocabulary:
            raise ValueError("background_tokens must not be empty")
        self.background_model = CharLM(self.config).fit(vocabulary)
        by_label: Dict[str, List[str]] = {}
        for label, text in label_spans:
            if text:
                by_label.setdefault(label, []).append(text)
        all_spans = [text for _, text in label_spans if text]
        self.span_model: Optional[CharLM] = (
            CharLM(self.config).fit(all_spans) if all_spans else None
        )
        self.label_models: Dict[str, CharLM] = {
            label: CharLM(self.config).fit(texts)
            for label, texts in sorted(by_label.items())
            if len(texts) >= self.config.min_label_spans
        }

    def surprisal(self, tokens: Sequence[str]) -> List[float]:
        """Bits per character of each token under the ``O``-token model."""
        return self.background_model.bits_per_char(tokens)

    def span_typicality(self, texts: Sequence[str]) -> List[Optional[float]]:
        """Bits per character of each span text under the any-span model.

        Parameters
        ----------
        texts : sequence of str
            Span texts (see :func:`span_text`).

        Returns
        -------
        list of float or None
            One value per text (low = span-like); ``None`` for every text
            when the training set has no spans.
        """
        if self.span_model is None:
            return [None] * len(texts)
        return self.span_model.bits_per_char(texts)

    def prototypicality(
        self, spans: Sequence[Tuple[str, str]]
    ) -> List[Tuple[Optional[float], Optional[float], Optional[str]]]:
        """Form typicality and label fit of spans.

        Parameters
        ----------
        spans : sequence of (str, str)
            ``(label, span_text)`` pairs.

        Returns
        -------
        list of (float or None, float or None, str or None)
            Per span: bits per character under its own label's model
            (``None`` if that label has no model), label fit in bits per
            character (own label minus best other label; ``None`` if either
            is missing) and the best-fitting other label.
        """
        texts = [text for _, text in spans]
        scores = {
            label: model.log2_probabilities(texts)
            for label, model in self.label_models.items()
        }
        results: List[Tuple[Optional[float], Optional[float], Optional[str]]] = []
        for i, (label, text) in enumerate(spans):
            events = len(text) + 1
            own = scores[label][i] if label in scores else None
            others = [(scores[other][i], other) for other in scores if other != label]
            best_other = max(others) if others else None
            typicality = None if own is None else -own / events
            fit = (
                None if own is None or best_other is None
                else (own - best_other[0]) / events
            )
            results.append(
                (typicality, fit, best_other[1] if best_other else None)
            )
        return results
