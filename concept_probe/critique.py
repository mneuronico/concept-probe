"""Having a model, or an agent, read a two-pole dataset and say what is wrong with it.

The audit of :mod:`concept_probe.datasets` measures what it is told to measure; a reader can notice
what nobody thought of. A critique asks, about a sample of each pole:

- what differs systematically between the poles **besides the concept** (length, topic, tense,
  person, register, punctuation, particular words or structures...), with examples and, when it
  can be detected by a regular expression, that expression;
- what is **formulaic** (repeated templates, openings, structures);
- which items look **mislabeled** or ambiguous;
- what would make the dataset better.

Two ways to get it, with the same instructions:

- :func:`critique_task` writes the task to a folder (``TASK.md``, the sample, and the exact format of
  ``critique.json``) for an agent (or a subagent, or a person) to answer, and :func:`read_critique`
  reads and checks the answer;
- :func:`critique_dataset` asks a judge directly (:class:`~concept_probe.judges.APIJudge`, which sends
  the sample out, or :class:`~concept_probe.judges.LocalJudge`).

A critique is a list of hypotheses: :meth:`Critique.checks` turns the differences that come with a
regular expression into checks for :func:`concept_probe.audit_dataset`, which measures them on every
text, and each one can also become an attack for :func:`concept_probe.red_team`.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .datasets import count_matches

FORMAT = """{
  "differences": [
    {"name": "<short_snake_case_name>",
     "description": "<what differs between the poles, besides the concept>",
     "more_in": "<pos or neg: the pole that has more of it>",
     "examples": ["<ids of items that show it, from both poles if possible>"],
     "regex": "<a regular expression that detects it in a text, or null>"}
  ],
  "formulaic": [
    {"description": "<a repeated template, opening or structure>", "examples": ["<ids>"]}
  ],
  "mislabeled": [
    {"id": "<id of an item that looks mislabeled or ambiguous>", "reason": "<why>"}
  ],
  "suggestions": ["<how to make the dataset better>"]
}"""


def _sample(pos: Sequence[str], neg: Sequence[str], n: int, paired: bool, seed: int) -> Dict[str, List[Dict[str, str]]]:
    rng = np.random.default_rng(seed)
    if paired:
        if len(pos) != len(neg):
            raise ValueError("paired poles need the same number of texts")
        idx = sorted(rng.choice(len(pos), min(n, len(pos)), replace=False).tolist())
        return {"pairs": [{"pos": {"id": f"pos:{i}", "text": str(pos[i])}, "neg": {"id": f"neg:{i}", "text": str(neg[i])}}
                          for i in idx]}
    ip = sorted(rng.choice(len(pos), min(n, len(pos)), replace=False).tolist())
    i_n = sorted(rng.choice(len(neg), min(n, len(neg)), replace=False).tolist())
    return {"pos": [{"id": f"pos:{i}", "text": str(pos[i])} for i in ip],
            "neg": [{"id": f"neg:{i}", "text": str(neg[i])} for i in i_n]}


def _instructions(concept: str, paired: bool, n_pos: int, n_neg: int) -> str:
    sample = (f"{n_pos} pairs: each pair is a positive and a negative text meant to differ only in the concept"
              if paired else f"{n_pos} positive and {n_neg} negative texts")
    return f"""You are auditing a dataset for training a concept probe: texts of two poles that should differ
only in one concept: {concept}. A probe learns whatever separates the poles, so any other systematic
difference between them (length, topic, tense, grammatical person, register, punctuation, particular
words, sentence structures...) is a confound the probe can learn instead of the concept.

You get a random sample of the dataset ({sample}); each text has an id like "pos:3" or "neg:3".
Read them all and report:

1. differences: every systematic difference between the poles besides the concept. For each, a short
   name, a description, the pole that has more of it ("pos" or "neg"), the ids of items that show it,
   and a regular expression that detects it in a single text (Python syntax, case-insensitive), or
   null when no expression can capture it. Report only differences you can point to in the texts.
2. formulaic: repeated templates, openings or structures that make the texts less varied, with ids.
3. mislabeled: items that look like the wrong pole, or that do not clearly belong to either, with why.
4. suggestions: how to make the dataset better (variety, balance, pairing, removing a confound).

Answer with one JSON object exactly in this format, and nothing else:

{FORMAT}
"""


@dataclass
class Critique:
    """A reader's critique of a dataset (see the module docstring)."""

    differences: List[Dict[str, Any]]
    formulaic: List[Dict[str, Any]]
    mislabeled: List[Dict[str, Any]]
    suggestions: List[str]
    source: Dict[str, Any] = field(default_factory=dict)

    def checks(self) -> Dict[str, Callable[[str], float]]:
        """The differences that came with a regular expression, as checks for
        :func:`concept_probe.audit_dataset` (each counts the expression's matches in a text)."""
        return {d["name"]: count_matches(d["regex"]) for d in self.differences if d.get("regex")}

    def to_dict(self) -> Dict[str, Any]:
        return {"differences": self.differences, "formulaic": self.formulaic, "mislabeled": self.mislabeled,
                "suggestions": self.suggestions, "source": self.source}


def _parse(data: Any, ids: Sequence[str]) -> Critique:
    """Check a critique's JSON against the format; raises saying what is wrong."""
    if isinstance(data, str):
        start, end = data.find("{"), data.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("the critique is not a JSON object")
        data = json.loads(data[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the critique must be a JSON object")
    known = set(ids)
    problems: List[str] = []
    out: Dict[str, Any] = {}
    for key in ("differences", "formulaic", "mislabeled", "suggestions"):
        value = data.get(key, [])
        if not isinstance(value, list):
            problems.append(f"{key!r} must be a list")
            value = []
        out[key] = value
    names = set()
    for k, d in enumerate(out["differences"]):
        if not isinstance(d, dict) or not d.get("name") or not d.get("description"):
            problems.append(f"differences[{k}] needs a name and a description")
            continue
        if d["name"] in names:
            problems.append(f"differences[{k}]: the name {d['name']!r} is used twice")
        names.add(d["name"])
        if d.get("more_in") not in ("pos", "neg"):
            problems.append(f"differences[{k}] ({d['name']}): more_in must be 'pos' or 'neg'")
        unknown = [i for i in d.get("examples", []) if i not in known]
        if unknown:
            problems.append(f"differences[{k}] ({d['name']}): unknown ids {unknown[:3]}")
        if d.get("regex"):
            try:
                re.compile(d["regex"])
            except re.error as exc:
                problems.append(f"differences[{k}] ({d['name']}): the regex does not compile ({exc})")
    for k, m in enumerate(out["mislabeled"]):
        if not isinstance(m, dict) or m.get("id") not in known:
            problems.append(f"mislabeled[{k}] needs the id of an item of the sample")
    if problems:
        raise ValueError("the critique does not follow the format: " + "; ".join(problems[:6]))
    return Critique(out["differences"], out["formulaic"], out["mislabeled"], [str(x) for x in out["suggestions"]])


def critique_task(pos: Sequence[str], neg: Sequence[str], folder: str, *, concept: str = "the concept of the poles",
                  paired: bool = False, n: int = 40, seed: int = 0) -> str:
    """Write a critique task to ``folder`` for an agent (or anyone) to answer: ``TASK.md`` (what to
    do and the exact format of the answer), ``sample.json`` (``n`` items of each pole, or ``n``
    pairs) and where to write ``critique.json``. Returns the folder; read the answer with
    :func:`read_critique`."""
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    sample = _sample(pos, neg, n, paired, seed)
    n_pos = len(sample.get("pairs", sample.get("pos", [])))
    n_neg = len(sample.get("pairs", sample.get("neg", [])))
    (path / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    layout = ('`pairs`, a list of `{"pos": {"id": ..., "text": ...}, "neg": {"id": ..., "text": ...}}`' if paired
              else '`pos` and `neg`, lists of `{"id": ..., "text": ...}`')
    task = (f"# Critique task\n\n{_instructions(concept, paired, n_pos, n_neg)}\n## Files\n\n"
            f"- Input: `sample.json` in this folder: {layout}.\n"
            "- Output: write the JSON object to `critique.json` in this folder (only the object, UTF-8).\n\n"
            "The code that asked for the critique reads it with `read_critique` and checks it against the format\n"
            "(names, poles, ids of the sample, regular expressions that compile); anything wrong is reported so\n"
            "the file can be fixed.\n")
    (path / "TASK.md").write_text(task, encoding="utf-8")
    (path / "task.json").write_text(json.dumps({"concept": concept, "paired": paired, "n": n, "seed": seed},
                                               indent=1), encoding="utf-8")
    return str(path)


def _ids(sample: Dict[str, Any]) -> List[str]:
    if "pairs" in sample:
        return [p[k]["id"] for p in sample["pairs"] for k in ("pos", "neg")]
    return [x["id"] for k in ("pos", "neg") for x in sample[k]]


def read_critique(folder: str) -> Critique:
    """The critique an agent wrote for :func:`critique_task`, checked against the format."""
    path = Path(folder)
    answer = path / "critique.json"
    if not answer.exists():
        raise FileNotFoundError(f"No critique.json in {folder} yet: answer {path / 'TASK.md'} first")
    sample = json.loads((path / "sample.json").read_text(encoding="utf-8"))
    critique = _parse(json.loads(answer.read_text(encoding="utf-8")), _ids(sample))
    critique.source = {"by": "agent", "folder": str(path)}
    return critique


def critique_dataset(pos: Sequence[str], neg: Sequence[str], judge: Any, *, concept: str = "the concept of the poles",
                     paired: bool = False, n: int = 40, seed: int = 0) -> Critique:
    """Ask ``judge`` (an :class:`~concept_probe.judges.APIJudge`, which sends the sample out, or a
    :class:`~concept_probe.judges.LocalJudge`) for the critique directly, with the instructions of
    :func:`critique_task`. For an agent, use :func:`critique_task` and :func:`read_critique`."""
    sample = _sample(pos, neg, n, paired, seed)
    n_pos = len(sample.get("pairs", sample.get("pos", [])))
    n_neg = len(sample.get("pairs", sample.get("neg", [])))
    reply = judge.complete(_instructions(concept, paired, n_pos, n_neg),
                           "The sample:\n" + json.dumps(sample, ensure_ascii=False))
    critique = _parse(reply, _ids(sample))
    critique.source = {"by": judge.describe()}
    return critique
