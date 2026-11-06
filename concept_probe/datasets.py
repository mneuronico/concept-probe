"""Auditing a two-pole dataset before trusting a probe trained on it.

Model-free checks for differences between the poles other than the concept, and for how varied
the texts are. By default it looks at what any two-pole dataset can be checked for, in any
language: length, and which words each pole uses. Anything else you suspect (negation, first
person, a topic, exclamation marks...) is a **check**: a function ``text -> number``, compared
between the poles like length; :func:`count_matches` builds one from a regular expression.

- :func:`compare_poles`: per feature (length, and each check), the poles' means, the standardized
  difference and its p;
- :func:`surface_baseline`: a cross-validated classifier that sees only words, only length, only
  your checks, or any representation you give (embeddings): what a probe has to beat to show it
  reads more than the surface;
- :func:`distinctive_words`: the words much more frequent in one pole than in the other (weighted
  log-odds with an informative prior);
- :func:`formulaicity` and :func:`diversity`: repeated openings, near duplicates, vocabulary and
  length spread, and how spread out the texts are in an embedding you give;
- :func:`audit_dataset` runs them all and writes a report (saved by default).

The classifiers need scikit-learn (``pip install concept-probe[stats]``). For a reading of the texts
by a model (differences, formulaic writing, mislabeled items), see :mod:`concept_probe.critique`.
"""
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .settings import settings
from .stats import _welch_p_columns
from .utils import ensure_dir, json_dump, now_tag

WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+", re.UNICODE)

Check = Callable[[str], float]


def _words(text: str) -> List[str]:
    return [w.lower() for w in WORD.findall(str(text))]


def count_matches(pattern: str, flags: int = re.IGNORECASE) -> Check:
    """A check that counts the matches of a regular expression in each text (case-insensitive by
    default), e.g. ``count_matches(r"\\b(not|no|never)\\b|n't\\b")`` for negation in English."""
    rx = re.compile(pattern, flags)

    def count(text: str) -> float:
        return float(len(rx.findall(str(text))))

    count.__name__ = f"count_matches({pattern!r})"
    return count


def text_features(text: str, checks: Optional[Dict[str, Check]] = None,
                  count_tokens: Optional[Callable[[str], int]] = None) -> Dict[str, float]:
    """A text's length in characters and words (and tokens, with ``count_tokens``, e.g. a tokenizer's
    count), and the value of each check."""
    text = str(text)
    feats: Dict[str, float] = {"chars": float(len(text)), "words": float(len(_words(text)))}
    if count_tokens is not None:
        feats["tokens"] = float(count_tokens(text))
    for name, fn in (checks or {}).items():
        if name in feats:
            raise ValueError(f"a check cannot be called {name!r}; that name is a built-in feature")
        feats[name] = float(fn(text))
    return feats


def _sign_test_p(n_plus: int, n_minus: int) -> float:
    n = n_plus + n_minus
    if n == 0:
        return 1.0
    k = min(n_plus, n_minus)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def compare_poles(pos: Sequence[str], neg: Sequence[str], *, paired: bool = False,
                  checks: Optional[Dict[str, Check]] = None, count_tokens: Optional[Callable[[str], int]] = None
                  ) -> List[Dict[str, Any]]:
    """Per feature (length, and each check): each pole's mean, the difference in pooled SDs
    (``std_diff``; infinite when the feature does not vary within the poles but differs between
    them) and its p (a sign test on the pairs when ``paired``, Welch's t otherwise), sorted by
    ``|std_diff|``. A feature the poles differ on is a cue a probe can learn instead of the concept."""
    if paired and len(pos) != len(neg):
        raise ValueError("paired poles need the same number of texts")
    fp = [text_features(t, checks, count_tokens) for t in pos]
    fn = [text_features(t, checks, count_tokens) for t in neg]
    rows = []
    for k in fp[0]:
        a = np.array([f[k] for f in fp], dtype=np.float64)
        b = np.array([f[k] for f in fn], dtype=np.float64)
        sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
        diff = a.mean() - b.mean()
        # no spread within the poles: any difference separates them completely
        std_diff = float(diff / sd) if sd > 0 else (0.0 if diff == 0 else math.copysign(math.inf, diff))
        row = {"feature": k, "mean_pos": float(a.mean()), "mean_neg": float(b.mean()), "std_diff": std_diff}
        if paired:
            d = a - b
            row.update(more_pos=int((d > 0).sum()), more_neg=int((d < 0).sum()),
                       p=_sign_test_p(int((d > 0).sum()), int((d < 0).sum())))
        else:
            row["p"] = float(_welch_p_columns(a[:, None], b[:, None])[0]) if sd > 0 else (1.0 if diff == 0 else 0.0)
        rows.append(row)
    return sorted(rows, key=lambda r: -abs(r["std_diff"]))


# ------------------------------------------------------------------------------------ classifiers
def _features(texts: Sequence[str], features: Any, vocabulary: List[str]) -> np.ndarray:
    if isinstance(features, dict):
        return np.array([[float(fn(t)) for fn in features.values()] for t in texts], dtype=np.float64)
    if callable(features):
        return np.asarray(features(list(texts)), dtype=np.float64)
    if features == "length":
        rows = [text_features(t) for t in texts]
        return np.array([[r["chars"], r["words"]] for r in rows], dtype=np.float64)
    index = {w: i for i, w in enumerate(vocabulary)}
    X = np.zeros((len(texts), len(vocabulary)))
    for i, t in enumerate(texts):
        for w in _words(t):
            if w in index:
                X[i, index[w]] += 1
    return X


def surface_baseline(
    pos: Sequence[str],
    neg: Sequence[str],
    features: Any = "words",
    *,
    vocabulary: Optional[Sequence[str]] = None,
    paired: bool = False,
    folds: int = 8,
    C: float = 1.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """How well a classifier that sees only ``features`` tells the poles apart, cross-validated:

    - ``"words"``: word counts (a bag of words), of every word or only of ``vocabulary`` (e.g. a
      list of function words);
    - ``"length"``: the length in characters and words;
    - ``{name: check}``: your checks (functions ``text -> number``);
    - a function ``texts -> array (texts, dim)``: any representation, e.g. a sentence embedding.

    A logistic regression on standardized features (``C``), ``folds``-fold. With ``paired``, it
    learns from each pair which member is positive (the two members of a pair are never split
    between training and test), so what the members of a pair share cancels out. Compare its
    ``accuracy`` with a probe's: a probe that does not beat the words did not need the model.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise ImportError("surface_baseline needs scikit-learn: pip install concept-probe[stats]") from exc
    pos, neg = [str(t) for t in pos], [str(t) for t in neg]
    if paired and len(pos) != len(neg):
        raise ValueError("paired poles need the same number of texts")
    words: List[str] = []
    if features == "words":
        words = sorted({w.lower() for w in vocabulary}) if vocabulary is not None else \
            sorted({w for t in pos + neg for w in _words(t)})
    elif not (callable(features) or isinstance(features, dict) or features == "length"):
        raise ValueError(f"features must be 'words', 'length', a dict of checks or a function; got {features!r}")
    P, N = _features(pos, features, words), _features(neg, features, words)
    if paired:
        diff = P - N
        X = np.vstack([diff, -diff])
        y = np.r_[np.ones(len(diff)), np.zeros(len(diff))]
        group = np.r_[np.arange(len(diff)), np.arange(len(diff))]
        k = int(min(folds, len(diff)))
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=C, max_iter=3000,
                                                                                  fit_intercept=False))
        scores = cross_val_score(model, X, y, groups=group, cv=GroupKFold(k))
    else:
        X = np.vstack([P, N])
        y = np.r_[np.ones(len(P)), np.zeros(len(N))]
        k = int(min(folds, len(P), len(N)))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    name = (features if isinstance(features, str) else "checks: " + ", ".join(features) if isinstance(features, dict)
            else getattr(features, "__name__", "a function"))
    return {"features": name, "accuracy": float(np.mean(scores)), "sd": float(np.std(scores)), "folds": k,
            "paired": bool(paired), "chance": 0.5 if paired else float(max(len(P), len(N)) / (len(P) + len(N))),
            "n_features": int(X.shape[1])}


# ----------------------------------------------------------------------------- words and variety
def distinctive_words(pos: Sequence[str], neg: Sequence[str], *, prior: float = 0.01, top: int = 25
                      ) -> Dict[str, List[Dict[str, Any]]]:
    """The words that most separate the poles: weighted log-odds with an informative Dirichlet
    prior (Monroe et al., 2008), as a z score per word, with its count in each pole."""
    cp = Counter(w for t in pos for w in _words(t))
    cn = Counter(w for t in neg for w in _words(t))
    allc = cp + cn
    n_p, n_n, n_all = sum(cp.values()), sum(cn.values()), sum(allc.values())
    a0 = prior * n_all
    rows = []
    for w, c in allc.items():
        aw = prior * c
        lp = math.log((cp[w] + aw) / (n_p + a0 - cp[w] - aw))
        ln = math.log((cn[w] + aw) / (n_n + a0 - cn[w] - aw))
        z = (lp - ln) / math.sqrt(1 / (cp[w] + aw) + 1 / (cn[w] + aw))
        rows.append({"word": w, "z": float(z), "n_pos": int(cp[w]), "n_neg": int(cn[w])})
    rows.sort(key=lambda r: r["z"])
    return {"pos": rows[::-1][:top], "neg": rows[:top]}


def formulaicity(texts: Sequence[str], ids: Optional[Sequence[str]] = None, *, jaccard_threshold: float = 0.6
                 ) -> Dict[str, Any]:
    """Signs of formulaic writing: the most repeated two-word openings, the share of distinct word
    bigrams, and near duplicates (word sets with Jaccard similarity ``>= jaccard_threshold``).
    ``ids`` of the form ``"<pair>:<pole>"`` skip the two members of a pair, which are meant to be
    close."""
    toks = [_words(t) for t in texts]
    ids = [str(i) for i in (ids or range(len(texts)))]
    openings = Counter(" ".join(t[:2]) for t in toks if len(t) >= 2)
    bigrams = [tuple(t[i:i + 2]) for t in toks for i in range(len(t) - 1)]
    sets = [set(t) for t in toks]
    near = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if ":" in ids[i] and ids[i].split(":")[0] == ids[j].split(":")[0]:
                continue
            jac = len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
            if jac >= jaccard_threshold:
                near.append({"a": ids[i], "b": ids[j], "jaccard": round(jac, 3)})
    return {"top_openings": openings.most_common(12), "distinct_bigrams": len(set(bigrams)) / max(1, len(bigrams)),
            "near_duplicates": near}


def diversity(texts: Sequence[str], embeddings: Optional[Any] = None) -> Dict[str, float]:
    """How varied a set of texts is: the type-token ratio, the share of distinct word bigrams, the
    mean and SD of the length in words, and, with ``embeddings`` (one row per text, from any model,
    e.g. pooled representations from :func:`concept_probe.read_texts`), their dispersion: one minus
    the mean cosine between different texts (0: all the same; near 1: unrelated)."""
    toks = [_words(t) for t in texts]
    words = [w for t in toks for w in t]
    bigrams = [tuple(t[i:i + 2]) for t in toks for i in range(len(t) - 1)]
    lens = [len(t) for t in toks]
    out = {"type_token": len(set(words)) / max(1, len(words)), "distinct_bigrams": len(set(bigrams)) / max(1, len(bigrams)),
           "mean_words": float(np.mean(lens)) if lens else 0.0, "length_sd": float(np.std(lens)) if lens else 0.0}
    if embeddings is not None:
        E = np.asarray(embeddings, dtype=np.float64).reshape(len(texts), -1)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
        n = len(E)
        sim = E @ E.T
        out["embedding_dispersion"] = float(1 - (sim.sum() - n) / max(1, n * (n - 1)))
    return out


# ---------------------------------------------------------------------------------------- audit
@dataclass
class DatasetAudit:
    """What :func:`audit_dataset` found; :meth:`markdown` writes it up."""

    name: str
    n_pos: int
    n_neg: int
    paired: bool
    features: List[Dict[str, Any]]
    baselines: Dict[str, Dict[str, Any]]
    words: Dict[str, List[Dict[str, Any]]]
    formulaicity: Dict[str, Any]
    diversity: Dict[str, Dict[str, float]]
    out_dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "n_pos": self.n_pos, "n_neg": self.n_neg, "paired": self.paired,
                "features": self.features, "baselines": self.baselines, "words": self.words,
                "formulaicity": self.formulaicity, "diversity": self.diversity, "notes": self.notes}

    def markdown(self) -> str:
        lines = [f"# Audit: {self.name}", "", f"{self.n_pos} positive and {self.n_neg} negative texts"
                 + (", in pairs." if self.paired else "."), ""]
        lines += ["## Classifiers that see only part of the texts", "",
                  "| features | accuracy | chance |", "|---|---|---|"]
        for name, b in self.baselines.items():
            lines.append(f"| {name} | {b['accuracy']:.2f} +/- {b['sd']:.2f} | {b['chance']:.2f} |")
        lines += ["", "Words are expected to tell the poles apart (the concept is in the words): a probe that "
                  "does not read better than them did not need the model. Length and your checks should "
                  "stay near chance.", "",
                  "## Features (largest differences first)", "",
                  "| feature | positive | negative | difference (SDs) | p |", "|---|---|---|---|---|"]
        for r in self.features[:12]:
            lines.append(f"| {r['feature']} | {r['mean_pos']:.2f} | {r['mean_neg']:.2f} | {r['std_diff']:+.2f} | "
                         f"{r['p']:.3g} |")
        for pole, label in (("pos", "positive"), ("neg", "negative")):
            lines += ["", f"## Words much more frequent in the {label} pole", "",
                      ", ".join(f"{w['word']} ({w['z']:+.1f})" for w in self.words[pole][:20])]
        f = self.formulaicity
        lines += ["", "## Variety", "",
                  f"- Distinct word bigrams: {f['distinct_bigrams']:.2f}; near duplicates across pairs: "
                  f"{len(f['near_duplicates'])}.",
                  "- Most repeated openings: " + ", ".join(f"\"{o}\" ({c})" for o, c in f["top_openings"][:6]) + "."]
        for pole, d in self.diversity.items():
            lines.append(f"- {pole}: type-token {d['type_token']:.2f}, {d['mean_words']:.1f} +/- {d['length_sd']:.1f} "
                         "words" + (f", embedding dispersion {d['embedding_dispersion']:.2f}"
                                    if "embedding_dispersion" in d else "") + ".")
        lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines).rstrip() + "\n"

    def save(self, out_dir: str) -> str:
        ensure_dir(out_dir)
        json_dump(os.path.join(out_dir, "audit.json"), self.to_dict())
        with open(os.path.join(out_dir, "audit.md"), "w", encoding="utf-8") as f:
            f.write(self.markdown())
        self.out_dir = out_dir
        return out_dir


def audit_dataset(
    pos: Sequence[str],
    neg: Sequence[str],
    *,
    paired: bool = False,
    name: str = "dataset",
    checks: Optional[Dict[str, Check]] = None,
    ids: Optional[Sequence[str]] = None,
    embed: Optional[Callable[[List[str]], Any]] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
    folds: int = 8,
    save: Optional[bool] = None,
    out_dir: Optional[str] = None,
) -> DatasetAudit:
    """Every check of this module on a two-pole dataset: length and your ``checks`` compared
    between the poles; classifiers that see only words, only length, only the checks (and only
    ``embed(texts)`` when given); the words much more frequent in each pole; formulaic writing and
    variety. Saved by default (``audit.json`` and ``audit.md``) to ``out_dir`` or
    ``audits/<name>-<time>`` under the session's ``output_root``."""
    pos, neg = [str(t) for t in pos], [str(t) for t in neg]
    baselines = {"words": surface_baseline(pos, neg, "words", paired=paired, folds=folds),
                 "length": surface_baseline(pos, neg, "length", paired=paired, folds=folds)}
    if checks:
        baselines["checks"] = surface_baseline(pos, neg, dict(checks), paired=paired, folds=folds)
    if embed is not None:
        baselines["embeddings"] = surface_baseline(pos, neg, embed, paired=paired, folds=folds)
    notes = []
    for key in ("length", "checks"):
        b = baselines.get(key)
        if b and b["accuracy"] > b["chance"] + 2 * max(b["sd"], 0.02):
            notes.append(f"The {key} classifier tells the poles apart ({b['accuracy']:.2f}): the poles differ in "
                         "more than the concept.")
    if ids is None and paired:
        ids = [f"{i}:pos" for i in range(len(pos))] + [f"{i}:neg" for i in range(len(neg))]
    audit = DatasetAudit(
        name=name, n_pos=len(pos), n_neg=len(neg), paired=paired,
        features=compare_poles(pos, neg, paired=paired, checks=checks, count_tokens=count_tokens),
        baselines=baselines, words=distinctive_words(pos, neg),
        formulaicity=formulaicity(pos + neg, ids),
        diversity={"positive": diversity(pos, None if embed is None else embed(pos)),
                   "negative": diversity(neg, None if embed is None else embed(neg))},
        notes=notes)
    if settings().save if save is None else save:
        audit.save(out_dir or os.path.join(settings().output_root, "audits", f"{name}-{now_tag()}"))
    return audit
