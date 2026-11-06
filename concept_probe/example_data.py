"""A dataset that ships with the library, to see it work before writing texts of your own.

:func:`load_example` returns it. It is an example, not a default: nothing in the library uses it on
its own.
"""
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Dict, List

#: The example datasets: name -> (file in ``concept_probe/data``, positive label, negative label, description).
_DATASETS = {
    "valence": (
        "valence.jsonl", "positive", "negative",
        "224 minimal pairs of English texts: each pair is one situation, told once with a good outcome and "
        "once with a bad one, in 16 topics (work, health, travel, ...) and 8 genres (chat messages, diary "
        "entries, emails, reviews, ...). Written with an LLM, audited so that surface features (length, "
        "function words) do not tell the poles apart, and red-teamed. 141 pairs are for training and 83 for "
        "evaluation, split by a hash of each pair's id.",
    ),
}


@dataclass
class ExampleDataset:
    """Pairs of texts of two poles, split into training and evaluation pairs.

    ``train_pos[i]`` and ``train_neg[i]`` are the same situation with the concept turned around, and
    the same holds for ``eval_pos`` and ``eval_neg``, so an eval set built from them is paired
    (``EvalSet(..., paired=True)``). ``pairs`` holds every pair with all its fields: ``id``,
    ``split``, ``pos``, ``neg``, and what the dataset adds (for valence, ``topic``, ``genre`` and
    ``explicit``: whether a keyword carries the valence, or only the situation does).
    """

    name: str
    pos_label: str
    neg_label: str
    description: str
    pairs: List[Dict[str, Any]]

    def _texts(self, split: str, side: str) -> List[str]:
        return [p[side] for p in self.pairs if p["split"] == split]

    @property
    def train_pos(self) -> List[str]:
        return self._texts("train", "pos")

    @property
    def train_neg(self) -> List[str]:
        return self._texts("train", "neg")

    @property
    def eval_pos(self) -> List[str]:
        return self._texts("eval", "pos")

    @property
    def eval_neg(self) -> List[str]:
        return self._texts("eval", "neg")


def load_example(name: str = "valence") -> ExampleDataset:
    """An example dataset that ships with the library, to try it out with (``"valence"``).

    ``"valence"``: 224 minimal pairs of positive and negative English texts (141 for training, 83 for
    evaluation); :class:`ExampleDataset` says how they are laid out, and its ``description`` how they
    were made. It is there to see the library work, not to measure a model against: for a concept of
    your own, write its texts (the ``datasets`` guide says how: ``concept_probe.guide("datasets")``).
    """
    if name not in _DATASETS:
        raise ValueError(f"no example dataset called {name!r}; the examples are {sorted(_DATASETS)}")
    filename, pos_label, neg_label, description = _DATASETS[name]
    text = files("concept_probe").joinpath("data").joinpath(filename).read_text(encoding="utf-8")
    pairs = [json.loads(line) for line in text.splitlines() if line.strip()]
    return ExampleDataset(name, pos_label, neg_label, description, pairs)
