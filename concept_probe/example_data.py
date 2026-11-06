"""Datasets that ship with the library, to see it work before writing texts of your own.

:func:`load_example` returns one; ``EXAMPLES`` lists them. They are examples, not defaults: nothing
in the library uses them on its own. All were written with an LLM and checked: the two-pole sets
for surface differences between the poles, and the test sets for whether each item says what its
label says.
"""
import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, Dict, List, Optional

_CONCEPT_SPEC = "a concept written as two system prompts: ``extra`` has ``pos_system``, ``neg_system``, " \
                "``train_questions`` and ``eval_questions`` for the model to answer, and the records are 20 pairs of " \
                "eval texts written by hand."

#: name -> (file in ``concept_probe/data``, positive label, negative label, description).
EXAMPLES: Dict[str, tuple] = {
    "valence": (
        "valence.jsonl", "positive", "negative",
        "224 minimal pairs of English texts: each pair is one situation, told once with a good outcome and once "
        "with a bad one, in 16 topics (work, health, travel, ...) and 8 genres (chat messages, diary entries, "
        "emails, reviews, ...). Audited so that surface features (length, function words) do not tell the poles "
        "apart, and red-teamed. 141 pairs are for training and 83 for evaluation (``split``, by a hash of each "
        "pair's id). ``explicit`` says whether a keyword carries the valence or only the situation does.",
    ),
    "valence_paraphrase": ("valence_paraphrase.jsonl", "positive", "negative",
                           "32 pairs of the evaluation split of ``valence``, paraphrased: the same situations in "
                           "other words."),
    "valence_genre": ("valence_genre.jsonl", "positive", "negative",
                      "32 minimal pairs in genres ``valence`` lacks (``genre``: headlines and verse)."),
    "valence_spanish": ("valence_spanish.jsonl", "positive", "negative",
                        "32 minimal pairs in Spanish, in the genres of ``valence``."),
    "valence_attacks": (
        "valence_attacks.jsonl", "positive", "negative",
        "54 attacks on a valence probe, written after the last revision of ``valence`` and never used to fix it, "
        "nine kinds (``attack``: negation, exclamation marks, a medical topic, ...). Records of ``kind`` "
        "``concept`` are a ``text`` with its valence (``label``: pos or neg) and the attacked feature; records of "
        "``kind`` ``neutral_pair`` are two neutral texts ``a`` and ``b`` that differ only in that feature, between "
        "which a probe of valence should not move.",
    ),
    "neutral": ("neutral.jsonl", None, None,
                "150 neutral texts (no valence), in the genres of ``valence``: a baseline, and texts without the "
                "concept to place on a probe."),
    "valence_needles": (
        "valence_needles.jsonl", "positive", "negative",
        "40 minimal pairs of neutral paragraphs (an email, minutes, a chat, ...) with one sentence of valence, "
        "the needle, at the beginning, the middle or the end (``position``); only the needle differs. A test set, "
        "to compare ways of summarizing token scores.",
    ),
    "valence_confounds": (
        "valence_confounds.jsonl", "positive", "negative",
        "Valence pairs in the four cells of five confounds (``confound``: length, tense, person, language, "
        "exclamation). ``pos_1`` and ``neg_1`` carry the marked level (a neutral continuation, the past tense, the "
        "first person, Spanish, a final '!'), ``pos_0`` and ``neg_0`` the other. Trained on ``pos_1`` and "
        "``neg_0`` of the train split, a probe sees the confound fully correlated with valence; the four cells of "
        "the eval split decouple them.",
    ),
    "valence_negated": ("valence_negated.jsonl", None, None,
                        "20 valence sentences and their negation (``affirmative``, ``negated``: 'I'm happy with ...' / "
                        "'I'm not happy with ...'); ``sign`` is the affirmative's valence."),
    "neutral_negation": ("neutral_negation.jsonl", None, None,
                         "40 neutral sentences with and without a negation (``affirmative``, ``negated``: 'The meeting "
                         "is on the second floor.' / '... is not on ...'), split into train and eval: to fit a "
                         "negation direction, and erase it."),
    "valence_synonyms": ("valence_synonyms.jsonl", "positive", "negative",
                         "20 explicit pairs of ``valence`` with their keywords replaced by rarer synonyms ('best' -> "
                         "'finest'), in both poles."),
    "valence_in_names": ("valence_in_names.jsonl", None, None,
                         "24 neutral sentences with a valence word inside a proper name (``with_word``: 'The Happy "
                         "Valley bus ...') and the same sentence without it (``without_word``: 'The Green Valley bus "
                         "...'); ``sign`` is the word's valence."),
    "truth": ("truth.jsonl", "true", "false",
              "145 minimal pairs of simple facts (``pos`` true, ``neg`` false): the false statement changes an entity "
              "or a number of the true one. ``form`` is affirmative for most, and negated, hedged, passive, obscure "
              "and other forms for the rest; ``split`` is train or eval. They end with a period."),
    "topics": ("topics.jsonl", None, None,
               "240 neutral texts in six sets (``set``): food, cars, cats and teachers (40 each), and French and "
               "Spanish translations of 40 texts of ``neutral``. Concepts without a natural opposite."),
    "topic_valence": ("topic_valence.jsonl", None, None,
                      "60 texts that cross valence with a setting (``setting``: medical or party; ``valence``: pos, "
                      "neg or neutral): positive hospital stories, negative party stories, and neutral ones of both."),
    "days_months": ("days_months.jsonl", None, None,
                    "166 sentences that end with a day of the week or a month (``kind``: day or month; ``index`` "
                    "from 0; ``name``), each name in the same carrier sentences (``carrier``)."),
    "emotions": ("emotions.jsonl", None, None, "120 short texts of four emotions (``emotion``: joy, sadness, anger, "
                                               "fear; 30 each)."),
    "ordinal": ("ordinal.jsonl", None, None,
                "390 template sentences with a number (``variable``: year, temperature or quantity; ``value``), "
                "each value in the same carrier sentences."),
    "bored_vs_interested": ("bored_vs_interested.json", None, None, "Being interested or bored, " + _CONCEPT_SPEC),
    "distracted_vs_focused": ("distracted_vs_focused.json", None, None, "Being focused or distracted, " + _CONCEPT_SPEC),
    "impulsive_vs_planning": ("impulsive_vs_planning.json", None, None, "Planning or acting on impulse, " + _CONCEPT_SPEC),
}


@dataclass
class ExampleDataset:
    """An example dataset: its ``records`` (one dict per item or pair), and how they are laid out.

    Two-pole sets have ``pos`` and ``neg`` in each record: ``pos`` and ``neg`` give them all, and
    ``train_pos`` / ``train_neg`` / ``eval_pos`` / ``eval_neg`` the ones of each ``split``. In a minimal
    pair ``pos[i]`` and ``neg[i]`` are the same situation, so an eval set built from them is paired
    (``EvalSet(..., paired=True)``). :meth:`column` gives any field, of every record or of those
    that match; ``texts`` the ``text`` field. ``extra`` holds what is not records (the system prompts
    of a concept written as two, for example).
    """

    name: str
    description: str
    records: List[Dict[str, Any]]
    pos_label: Optional[str] = None
    neg_label: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def column(self, key: str, **where: Any) -> List[Any]:
        """``key`` of every record, or of the records whose fields equal ``where`` (e.g.
        ``column("text", set="food")``)."""
        out = []
        for r in self.records:
            if all(r.get(k) == v for k, v in where.items()):
                if key not in r:
                    raise KeyError(f"{self.name} records have no {key!r}; they have {sorted(r)}")
                out.append(r[key])
        return out

    @property
    def texts(self) -> List[str]:
        return self.column("text")

    @property
    def pos(self) -> List[str]:
        return self.column("pos")

    @property
    def neg(self) -> List[str]:
        return self.column("neg")

    @property
    def train_pos(self) -> List[str]:
        return self.column("pos", split="train")

    @property
    def train_neg(self) -> List[str]:
        return self.column("neg", split="train")

    @property
    def eval_pos(self) -> List[str]:
        return self.column("pos", split="eval")

    @property
    def eval_neg(self) -> List[str]:
        return self.column("neg", split="eval")


def load_example(name: str = "valence") -> ExampleDataset:
    """An example dataset that ships with the library (``EXAMPLES`` lists them with what each is).

    ``"valence"`` is 224 minimal pairs of positive and negative English texts (141 for training, 83
    for evaluation); the others test a valence probe (paraphrases, other genres, Spanish, attacks,
    needles, confounds, negation), give baselines (neutral texts, topics), other concepts (truth,
    concepts written as system prompts) and targets beyond two poles (days and months, emotions,
    numbers). :class:`ExampleDataset` says how they are laid out. They are there to see the library
    work, not to measure a model against: for a concept of your own, write its texts (the
    ``datasets`` guide says how: ``concept_probe.guide("datasets")``).
    """
    if name not in EXAMPLES:
        raise ValueError(f"no example dataset called {name!r}; the examples are {sorted(EXAMPLES)}")
    filename, pos_label, neg_label, description = EXAMPLES[name]
    text = files("concept_probe").joinpath("data").joinpath(filename).read_text(encoding="utf-8")
    if filename.endswith(".json"):
        spec = json.loads(text)
        concept, prompts = spec["concept"], spec["prompts"]
        records = [{"pos": p, "neg": n, "split": "eval"}
                   for p, n in zip(concept["eval_pos_texts"], concept["eval_neg_texts"])]
        extra = {"pos_system": concept["pos_system"], "neg_system": concept["neg_system"],
                 "train_questions": list(prompts["train_questions"]), "eval_questions": list(prompts["eval_questions"])}
        return ExampleDataset(name, description, records, concept["pos_label"], concept["neg_label"], extra)
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return ExampleDataset(name, description, records, pos_label, neg_label)
