"""Auditing a two-pole dataset before trusting a probe trained on it (layer 6d of docs/DESIGN.md).

Model-free checks for differences between the poles other than the concept, and for how varied
the texts are:

- :func:`compare_poles`: per surface feature (length, punctuation, word groups such as negation or
  first person), the poles' means, the standardized difference and its p;
- :func:`surface_baseline`: a cross-validated classifier that sees only surface features, only
  words, only function words, or any representation you give (embeddings): what a probe has to
  beat to show it reads more than the surface;
- :func:`distinctive_words`: the words that most separate the poles (weighted log-odds with an
  informative prior), which should be concept words, not function words or repeated phrasings;
- :func:`formulaicity` and :func:`diversity`: repeated openings, near duplicates, vocabulary and
  length spread, and how spread out the texts are in an embedding you give;
- :func:`audit_dataset` runs them all and writes a report (saved by default).

The word groups (:data:`WORD_GROUPS`) are English and Spanish lists, good enough to catch a skew;
pass your own for other languages or concepts. The classifiers need scikit-learn
(``pip install concept-probe[stats]``).
"""
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .settings import settings
from .stats import _welch_p_columns
from .utils import ensure_dir, json_dump, now_tag

WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+", re.UNICODE)

#: Word groups that should not separate the poles of most concepts (English and Spanish).
WORD_GROUPS: Dict[str, "re.Pattern[str]"] = {
    "negation": re.compile(r"\b(not|no|never|nobody|none|nothing|nowhere|neither|nor|without|cannot|no one|"
                           r"no longer|nunca|nadie|nada|ning[uú]n[oa]?|ni|sin|jam[aá]s|tampoco)\b|n't\b", re.I),
    "first_person": re.compile(r"\b(i|me|my|mine|we|us|our|ours|i'm|i've|i'd|we're|we've|yo|me|mi|mis|nosotros|"
                               r"nosotras|nuestro|nuestra)\b", re.I),
    "second_person": re.compile(r"\b(you|your|yours|you're|you'll|you've|tú|tu|tus|vos|usted|ustedes)\b", re.I),
    "third_person": re.compile(r"\b(he|she|they|him|her|them|his|their|he's|she's|they're|él|ella|ellos|ellas|"
                               r"su|sus)\b", re.I),
    "contrast": re.compile(r"\b(but|though|although|despite|however|yet|pero|aunque|sin embargo)\b", re.I),
    "intensifier": re.compile(r"\b(so|very|really|totally|absolutely|completely|muy|tan|re|súper|totalmente)\b", re.I),
    "future": re.compile(r"\b(will|won't|'ll|going to|tomorrow|next|mañana|próximo|próxima|va a|vamos a)\b", re.I),
    "number": re.compile(r"\b\d+\b"),
}

FUNCTION_WORDS = sorted(set("""
a an the and or but if of to in on at by for with from as into over under about after before since until
while than then so too very just only even still again also already yet ever never not no nor is are was were
be been being am do does did have has had will would shall should can could may might must it its this that
these those there here what which who whom whose when where why how all any both each every some such more
most other another own same i me my we us our you your he him his she her they them their one up down out
el la los las un una unos unas y o pero si de del a al en con por para sin sobre entre desde hasta que como
cuando donde quien cual muy más menos ya también no ni es son era fue ser estar está están hay me te se nos
lo le les mi tu su sus este esta estos estas ese esa eso
""".split()))


def _words(text: str) -> List[str]:
    return [w.lower() for w in WORD.findall(str(text))]


def surface_features(text: str, groups: Optional[Dict[str, Any]] = None,
                     n_tokens: Optional[Callable[[str], int]] = None) -> Dict[str, float]:
    """Surface statistics of one text, and how often each word group (``groups``, default
    :data:`WORD_GROUPS`) occurs in it; ``n_tokens`` adds a token count (e.g. a tokenizer's)."""
    text = str(text)
    words = _words(text)
    feats: Dict[str, float] = {
        "chars": len(text), "words": len(words),
        "sentences": max(1, len(re.findall(r"[.!?]+(\s|$)", text))),
        "exclamations": text.count("!"), "questions": text.count("?"), "commas": text.count(","),
        "uppercase_ratio": sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text)),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
    }
    for name, rx in (WORD_GROUPS if groups is None else groups).items():
        feats[name] = len(re.findall(rx, text)) if isinstance(rx, str) else len(rx.findall(text))
    if n_tokens is not None:
        feats["tokens"] = float(n_tokens(text))
    return feats


def _sign_test_p(n_plus: int, n_minus: int) -> float:
    n = n_plus + n_minus
    if n == 0:
        return 1.0
    k = min(n_plus, n_minus)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def compare_poles(pos: Sequence[str], neg: Sequence[str], *, paired: bool = False,
                  groups: Optional[Dict[str, Any]] = None, n_tokens: Optional[Callable[[str], int]] = None
                  ) -> List[Dict[str, Any]]:
    """Per surface feature: each pole's mean, the difference in pooled SDs (``std_diff``) and its p
    (a sign test on the pairs when ``paired``, Welch's t otherwise), sorted by ``|std_diff|``. A
    feature the poles differ on is a cue a probe can learn instead of the concept."""
    if paired and len(pos) != len(neg):
        raise ValueError("paired poles need the same number of texts")
    fp = [surface_features(t, groups, n_tokens) for t in pos]
    fn = [surface_features(t, groups, n_tokens) for t in neg]
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
def _features(texts: Sequence[str], features: Union[str, Callable[[List[str]], Any]], vocabulary: List[str],
              groups: Optional[Dict[str, Any]]) -> np.ndarray:
    if callable(features):
        return np.asarray(features(list(texts)), dtype=np.float64)
    if features == "surface":
        rows = [surface_features(t, groups) for t in texts]
        return np.array([[r[k] for k in rows[0]] for r in rows], dtype=np.float64)
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
    features: Union[str, Callable[[List[str]], Any]] = "words",
    *,
    paired: bool = False,
    folds: int = 8,
    C: float = 1.0,
    groups: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """How well a classifier that sees only ``features`` tells the poles apart, cross-validated:

    - ``"words"``: word counts (a bag of words); ``"function_words"``: counts of
      :data:`FUNCTION_WORDS` only; ``"surface"``: :func:`surface_features`;
    - a function ``texts -> array (texts, dim)``: any representation, e.g. a sentence embedding.

    A logistic regression on standardized features (``C``), ``folds``-fold. With ``paired``, it
    learns from each pair which member is positive (the two members of a pair are never split
    between training and test), so pair-level differences such as topic cancel out. Compare its
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
    vocabulary: List[str] = []
    if features == "words":
        vocabulary = sorted({w for t in pos + neg for w in _words(t)})
    elif features == "function_words":
        vocabulary = list(FUNCTION_WORDS)
    elif not callable(features) and features != "surface":
        raise ValueError(f"features must be 'words', 'function_words', 'surface' or a function; got {features!r}")
    P, N = _features(pos, features, vocabulary, groups), _features(neg, features, vocabulary, groups)
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
    return {"features": features if isinstance(features, str) else getattr(features, "__name__", "a function"),
            "accuracy": float(np.mean(scores)), "sd": float(np.std(scores)), "folds": k, "paired": bool(paired),
            "chance": 0.5 if paired else float(max(len(P), len(N)) / (len(P) + len(N))), "n_features": int(X.shape[1])}


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
    dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "n_pos": self.n_pos, "n_neg": self.n_neg, "paired": self.paired,
                "features": self.features, "baselines": self.baselines, "words": self.words,
                "formulaicity": self.formulaicity, "diversity": self.diversity, "notes": self.notes}

    def markdown(self) -> str:
        lines = [f"# Audit: {self.name}", "", f"{self.n_pos} positive and {self.n_neg} negative texts"
                 + (", in pairs." if self.paired else "."), ""]
        lines += ["## Classifiers that should stay near chance", "",
                  "| features | accuracy | chance |", "|---|---|---|"]
        for name, b in self.baselines.items():
            lines.append(f"| {name} | {b['accuracy']:.2f} ± {b['sd']:.2f} | {b['chance']:.2f} |")
        lines += ["", "Words alone are expected to tell the poles apart (the concept is in the words); surface "
                  "features and function words are not.", "",
                  "## Surface features (largest differences first)", "",
                  "| feature | positive | negative | difference (SDs) | p |", "|---|---|---|---|---|"]
        for r in self.features[:12]:
            lines.append(f"| {r['feature']} | {r['mean_pos']:.2f} | {r['mean_neg']:.2f} | {r['std_diff']:+.2f} | "
                         f"{r['p']:.3g} |")
        for pole, label in (("pos", "positive"), ("neg", "negative")):
            lines += ["", f"## Words that most mark the {label} pole", "",
                      ", ".join(f"{w['word']} ({w['z']:+.1f})" for w in self.words[pole][:20])]
        f = self.formulaicity
        lines += ["", "## Variety", "",
                  f"- Distinct word bigrams: {f['distinct_bigrams']:.2f}; near duplicates across pairs: "
                  f"{len(f['near_duplicates'])}.",
                  "- Most repeated openings: " + ", ".join(f"\"{o}\" ({c})" for o, c in f["top_openings"][:6]) + "."]
        for pole, d in self.diversity.items():
            lines.append(f"- {pole}: type-token {d['type_token']:.2f}, {d['mean_words']:.1f} ± {d['length_sd']:.1f} "
                         "words" + (f", embedding dispersion {d['embedding_dispersion']:.2f}"
                                    if "embedding_dispersion" in d else "") + ".")
        lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines).rstrip() + "\n"

    def save(self, out_dir: str) -> str:
        ensure_dir(out_dir)
        json_dump(os.path.join(out_dir, "audit.json"), self.to_dict())
        with open(os.path.join(out_dir, "audit.md"), "w", encoding="utf-8") as f:
            f.write(self.markdown())
        self.dir = out_dir
        return out_dir


def audit_dataset(
    pos: Sequence[str],
    neg: Sequence[str],
    *,
    paired: bool = False,
    name: str = "dataset",
    ids: Optional[Sequence[str]] = None,
    embed: Optional[Callable[[List[str]], Any]] = None,
    groups: Optional[Dict[str, Any]] = None,
    n_tokens: Optional[Callable[[str], int]] = None,
    folds: int = 8,
    save: Optional[bool] = None,
    out_dir: Optional[str] = None,
) -> DatasetAudit:
    """Every check of this module on a two-pole dataset: the surface features, the classifiers on
    surface features, function words and words (and on ``embed(texts)`` when given), the words that
    mark each pole, formulaic writing and variety. Saved by default (``audit.json`` and
    ``audit.md``) to ``out_dir`` or ``audits/<name>-<time>`` under the session's ``output_root``."""
    pos, neg = [str(t) for t in pos], [str(t) for t in neg]
    baselines = {}
    notes = []
    for feats in ("surface", "function_words", "words"):
        baselines[feats] = surface_baseline(pos, neg, feats, paired=paired, folds=folds, groups=groups)
    if embed is not None:
        baselines["embeddings"] = surface_baseline(pos, neg, embed, paired=paired, folds=folds)
    for feats in ("surface", "function_words"):
        b = baselines[feats]
        if b["accuracy"] > b["chance"] + 2 * max(b["sd"], 0.02):
            notes.append(f"The {feats.replace('_', ' ')} classifier tells the poles apart ({b['accuracy']:.2f}): "
                         "the poles differ in more than the concept.")
    if ids is None and paired:
        ids = [f"{i}:pos" for i in range(len(pos))] + [f"{i}:neg" for i in range(len(neg))]
    audit = DatasetAudit(
        name=name, n_pos=len(pos), n_neg=len(neg), paired=paired,
        features=compare_poles(pos, neg, paired=paired, groups=groups, n_tokens=n_tokens),
        baselines=baselines, words=distinctive_words(pos, neg),
        formulaicity=formulaicity(pos + neg, ids),
        diversity={"positive": diversity(pos, None if embed is None else embed(pos)),
                   "negative": diversity(neg, None if embed is None else embed(neg))},
        notes=notes)
    if settings().save if save is None else save:
        audit.save(out_dir or os.path.join(settings().output_root, "audits", f"{name}-{now_tag()}"))
    return audit
