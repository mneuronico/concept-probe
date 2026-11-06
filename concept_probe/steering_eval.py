"""Evaluating steering: a dose sweep with random controls and
measures you choose, and the largest dose that keeps the text coherent.

:func:`evaluate_steering` generates every prompt at each dose of an intervention (a probe's
steering, or any :class:`~concept_probe.interventions.Add`) and, at each non-zero dose, under
``n_random`` random controls: the same intervention with a random direction of the same length at
each layer (control ``k`` is the same direction at every dose). Every completion is measured:

- **coherence**, locally: its perplexity under the model without the intervention, as the reply to
  its prompt, and its share of repeated word 3-grams (:func:`repetition_rate`);
- **effect**: the score of each probe given, read again without the intervention (the steering
  probe's own by default, which is informative but not independent of it), and any function of
  the texts (a classifier, an embedding model, a judge's ``measure(rubric)``) or of the texts and
  their items (whether a completion gives an item's expected answer: :func:`exact_match`).

For each dose and measure it reports the mean change from dose 0, paired by prompt, with a
bootstrap interval; the same change under each control and the 95th percentile of their absolute
values (what a direction without the concept does); and whether the dose is coherent, by a rule you
can replace (:class:`CoherenceRule`). The largest coherent dose on each side is how far one can push
before the text breaks. Measures can be added after the sweep (:meth:`SteeringEvaluation.add_measure`),
for example when an agent rates the completions, and :meth:`SteeringEvaluation.relate` tells whether
one measure (a probe's score) predicts another (being right) at a given dose.
"""
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .generation import _continuation_logprobs, _pad_id
from .interventions import Add, Intervention
from .record import Choice, announce, progress_line
from .scoring import Condition, Scores, generate_and_score
from .settings import settings
from .utils import ensure_dir, json_dump, now_tag

BUILT_IN = ("perplexity", "repetition", "n_tokens", "dose", "control", "dose_index", "prompt", "intervention")


# -------------------------------------------------------------------------------------- measures
def repetition_rate(text: str, n: int = 3) -> float:
    """The share of repeated word ``n``-grams: near 0 in ordinary text, high when a model loops."""
    words = str(text).split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def _answer_matches(candidate: str, expected: Any, case_sensitive: bool) -> bool:
    def norm(value: Any) -> str:
        text = str(value).strip()
        text = text[:-1].rstrip() if text.endswith(".") else text
        return text if case_sensitive else text.casefold()

    a, b = norm(candidate), norm(expected)
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def exact_match(key: str = "expected", *, marker: Optional[str] = None, marker_position: str = "after",
                case_sensitive: bool = True) -> Callable[[List[str], List[Any]], np.ndarray]:
    """A measure of the texts and their items: 1 when a completion gives its item's
    ``item[key]`` (a value, or a list of accepted values), 0 otherwise.

    The answer is the whole completion or, with ``marker``, the first line after it
    (``marker_position="after"``) or the last line before it (``"before"``); a completion without the
    marker is wrong. Answers match after stripping whitespace and a trailing period, as numbers when
    both read as numbers, and case-sensitively unless ``case_sensitive=False``. Under strong steering,
    format drift (a missing marker, extra words) counts as wrong: read some completions before taking
    a drop as the model knowing less.
    """
    if marker_position not in ("after", "before"):
        raise ValueError(f"marker_position must be 'after' or 'before'; got {marker_position!r}")

    def correct(texts: List[str], items: List[Any]) -> np.ndarray:
        out = []
        for text, item in zip(texts, items):
            expected = item.get(key) if isinstance(item, dict) else None
            if expected is None:
                raise ValueError(f"exact_match needs item[{key!r}] in every item")
            candidate: Optional[str] = str(text).strip()
            if marker:
                i = str(text).find(marker)
                if i == -1:
                    candidate = None
                elif marker_position == "before":
                    lines = str(text)[:i].strip().splitlines()
                    candidate = lines[-1].strip() if lines else ""
                else:
                    lines = str(text)[i + len(marker):].strip().splitlines()
                    candidate = lines[0].strip() if lines else ""
            accepted = list(expected) if isinstance(expected, (list, tuple, set, frozenset)) else [expected]
            out.append(float(candidate is not None and any(_answer_matches(candidate, v, case_sensitive)
                                                           for v in accepted)))
        return np.array(out)

    correct.__name__ = f"exact_match({key!r})"
    return correct


def _takes_items(fn: Callable) -> bool:
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2


@dataclass
class CoherenceRule:
    """When a dose counts as coherent, against dose 0: the median perplexity of its completions
    under the unsteered model stays within ``perplexity_ratio`` times that of dose 0, and their mean
    repetition rate rises by at most ``repetition_increase``. The default numbers are a heuristic
    (the ones that separated readable from broken text in one model the authors tried), not a
    standard: look at completions near the limit, and change them, or pass any function
    ``rule(at_dose, at_zero) -> bool``; both dicts have ``perplexity`` (median), ``repetition`` and
    ``n_tokens`` (means) and the mean of every measure."""

    perplexity_ratio: Tuple[float, float] = (0.5, 2.0)
    repetition_increase: float = 0.2

    def __call__(self, at_dose: Dict[str, float], at_zero: Dict[str, float]) -> bool:
        ppl, base = at_dose["perplexity"], at_zero["perplexity"]
        if not (np.isfinite(ppl) and np.isfinite(base) and base > 0):
            return False
        lo, hi = self.perplexity_ratio
        return bool(lo * base <= ppl <= hi * base
                    and at_dose["repetition"] <= at_zero["repetition"] + self.repetition_increase)

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": "perplexity and repetition against dose 0", "perplexity_ratio": list(self.perplexity_ratio),
                "repetition_increase": self.repetition_increase}


# ------------------------------------------------------------------------------------------- table
def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> Optional[List[float]]:
    v = values[np.isfinite(values)]
    if len(v) < 2 or n_bootstrap <= 0:
        return None
    means = v[rng.integers(0, len(v), (int(n_bootstrap), len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summary(items: List[Any], measures: List[str]) -> Dict[str, float]:
    ppl = np.array([it.info["perplexity"] for it in items], dtype=np.float64)
    out = {"perplexity": float(np.nanmedian(ppl)) if np.isfinite(ppl).any() else float("nan"),
           "repetition": float(np.mean([it.info["repetition"] for it in items])),
           "n_tokens": float(np.mean([it.info["n_tokens"] for it in items]))}
    for m in measures:
        v = np.array([it.info[m] for it in items], dtype=np.float64)
        out[m] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def _largest_coherent(table: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for side, sign in (("positive", 1), ("negative", -1)):
        rows = sorted((r for r in table if sign * r["dose"] > 0), key=lambda r: abs(r["dose"]))
        best = None
        for r in rows:
            if not r["coherence"]["coherent"]:
                break
            best = r["dose"]
        out[side] = best
    return out


def _table(items: List[Any], doses: List[float], n_random: int, names: List[str], rule: Callable,
           n_bootstrap: int, seed: int) -> List[Dict[str, Any]]:
    """Per dose: the steering and its controls against dose 0, paired by prompt."""
    by: Dict[Tuple[float, Any], Dict[int, Any]] = {}
    for it in items:
        by.setdefault((it.info["dose"], it.info["control"]), {})[it.index] = it
    base = by[(0.0, None)]
    prompt_ids = sorted(base)
    zero = _summary([base[p] for p in prompt_ids], names)
    rng = np.random.default_rng(int(seed))

    def changes(group: Dict[int, Any], name: str) -> np.ndarray:
        return np.array([group[p].info[name] - base[p].info[name] for p in prompt_ids], dtype=np.float64)

    table = []
    for d in doses:
        steered = by[(d, None)]
        stats = _summary([steered[p] for p in prompt_ids], names)
        controls = [by[(d, k)] for k in range(int(n_random)) if (d, k) in by]
        control_stats = [_summary([c[p] for p in prompt_ids], names) for c in controls]
        row: Dict[str, Any] = {"dose": d, "coherence": {
            "perplexity": stats["perplexity"], "repetition": stats["repetition"], "n_tokens": stats["n_tokens"],
            "coherent": True if d == 0.0 else bool(rule(stats, zero)),
            "controls": {"perplexity": [c["perplexity"] for c in control_stats],
                         "repetition": [c["repetition"] for c in control_stats],
                         "coherent": [bool(rule(c, zero)) for c in control_stats]}}, "effects": {}}
        for name in names:
            delta = changes(steered, name)
            control_deltas = [changes(c, name) for c in controls]
            control_means = [float(np.nanmean(x)) if np.isfinite(x).any() else float("nan") for x in control_deltas]
            finite = [abs(m) for m in control_means if np.isfinite(m)]
            p95 = float(np.percentile(finite, 95)) if finite else None
            mean_change = float(np.nanmean(delta)) if np.isfinite(delta).any() else float("nan")
            row["effects"][name] = {
                "mean": stats[name], "change": mean_change,
                "ci95": None if d == 0.0 else _bootstrap_ci(delta, n_bootstrap, rng),
                "n_prompts": int(np.isfinite(delta).sum()), "controls": control_means, "null_abs_change95": p95,
                "beyond_controls": None if p95 is None else bool(abs(mean_change) > p95)}
        table.append(row)
    return table


# ------------------------------------------------------------------------------------------ result
@dataclass
class SteeringEvaluation:
    """What :func:`evaluate_steering` gives.

    ``scores`` has every completion (``info``: ``dose``, ``control`` — None for the steering itself
    — ``perplexity``, ``repetition``, ``n_tokens`` and every measure); ``measures`` says what each
    effect measure is; ``table`` has one row per dose (see :meth:`row`); ``largest_coherent_dose``
    the largest coherent dose on each side (None when even the smallest is not coherent, or there
    is no dose on that side); ``items`` the prompts' items, when given.
    """

    scores: Scores
    doses: List[float]
    measures: Dict[str, str]
    table: List[Dict[str, Any]]
    largest_coherent_dose: Dict[str, Optional[float]]
    settings: Dict[str, Any] = field(default_factory=dict)
    items: Optional[List[Any]] = None
    out_dir: Optional[str] = None
    coherence_rule: Callable[[Dict[str, float], Dict[str, float]], bool] = field(default_factory=CoherenceRule, repr=False)

    def row(self, dose: float) -> Dict[str, Any]:
        """The row of ``dose``: ``coherence`` (the steering's median perplexity, mean repetition and
        length, whether it is coherent, and the same for each control) and ``effects`` (per measure:
        the mean, the mean change from dose 0 with its interval, each control's change, the 95th
        percentile of their absolute values, and whether the steering's change goes beyond it)."""
        for r in self.table:
            if r["dose"] == float(dose):
                return r
        raise KeyError(f"no dose {dose}; the doses are {self.doses}")

    def _refresh(self) -> None:
        self.table = _table(self.scores.items, self.doses, int(self.settings.get("n_random", 0)), list(self.measures),
                            self.coherence_rule, int(self.settings.get("n_bootstrap", 1000)),
                            int(self.settings.get("control_seed", 0)))
        self.largest_coherent_dose = _largest_coherent(self.table)

    def add_measure(self, name: str, measure: Any, *, kind: str = "function of the texts") -> "SteeringEvaluation":
        """Measure every completion with ``measure`` and recompute the table: a function of the
        texts (``texts -> numbers``), of the texts and their items (``(texts, items) -> numbers``), or
        the numbers themselves, one per completion in order. A judge's ``measure(rubric)`` works; an
        :class:`~concept_probe.judges.AgentJudge` raises ``AnswersPending`` until its task is answered,
        and then this call, repeated, adds the ratings. Saved again when the evaluation was saved."""
        if name in self.measures or name in BUILT_IN:
            raise ValueError(f"there is already a measure called {name!r}")
        texts = [it.text for it in self.scores]
        if callable(measure):
            if _takes_items(measure):
                if self.items is None:
                    raise ValueError(f"measure {name!r} needs the items: pass items= to evaluate_steering")
                values = measure(texts, [self.items[it.index] for it in self.scores])
            else:
                values = measure(texts)
        else:
            values = measure
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != len(texts):
            raise ValueError(f"measure {name!r} gave {len(values)} values for {len(texts)} completions")
        for it, v in zip(self.scores, values):
            it.info[name] = float(v)
        self.measures[name] = kind
        self._refresh()
        if self.out_dir is not None:
            self.save(self.out_dir)
        return self

    def relate(self, predictor: str, outcome: str, *, include_random_controls: bool = False) -> Dict[str, Any]:
        """Whether ``predictor`` (a measure, e.g. a probe's score) predicts ``outcome`` (another, e.g.
        ``exact_match``) across completions, with the dose as a covariate: a GEE with each prompt as a
        cluster (it recurs at every dose), logistic when the outcome is 0/1 and linear otherwise,
        with robust standard errors corrected for few clusters. The predictor is in SDs within a dose,
        since steering itself moves it. Random controls are left out unless ``include_random_controls``. Needs
        statsmodels."""
        for name in (predictor, outcome):
            if name not in self.measures and name not in ("perplexity", "repetition", "n_tokens"):
                raise KeyError(f"no measure called {name!r}; the measures are {list(self.measures)}")
        rows = [it for it in self.scores if include_random_controls or it.info["control"] is None]
        x = np.array([it.info[predictor] for it in rows], dtype=np.float64)
        y = np.array([it.info[outcome] for it in rows], dtype=np.float64)
        dose = np.array([it.info["dose"] for it in rows], dtype=np.float64)
        prompt = np.array([it.index for it in rows])
        keep = np.isfinite(x) & np.isfinite(y)
        x, y, dose, prompt = x[keep], y[keep], dose[keep], prompt[keep]
        binary = bool(np.all(np.isin(y, (0.0, 1.0))))
        out: Dict[str, Any] = {"predictor": predictor, "outcome": outcome, "n_rows": int(len(y)),
                               "n_prompts": int(len(np.unique(prompt))),
                               "model": ("logistic" if binary else "linear") + f" GEE: {outcome} ~ {predictor} (SDs "
                                        "within a dose) + dose, exchangeable within prompt, bias-reduced robust SE",
                               "note": None}
        if len(y) < 4 or (binary and y.min() == y.max()):
            out["note"] = "needs at least 4 rows, and both outcomes when it is 0/1"
            return out
        try:
            import statsmodels.api as sm
        except Exception:
            out["note"] = "statsmodels is not installed (pip install concept-probe[stats])"
            return out
        levels = sorted(set(dose.tolist()))
        within = x - np.array([x[dose == d].mean() for d in dose])
        dof = len(x) - len(levels)
        sd = float(np.sqrt((within ** 2).sum() / dof)) if dof > 0 else 0.0
        if sd == 0:
            out["note"] = "the predictor does not vary within a dose"
            return out
        X = np.column_stack([np.ones(len(x)), (x - x.mean()) / sd] + [(dose == d).astype(float) for d in levels[1:]])
        family = sm.families.Binomial() if binary else sm.families.Gaussian()
        try:
            fit = sm.GEE(y, X, groups=prompt, family=family, cov_struct=sm.cov_struct.Exchangeable()).fit(
                cov_type="bias_reduced")
        except Exception as exc:  # e.g. perfect separation
            out["note"] = f"the model could not be fit: {exc}"
            return out
        ci = np.asarray(fit.conf_int())
        out.update({"sd_within_dose": sd, "coef_per_sd": float(fit.params[1]), "ci95": [float(ci[1, 0]), float(ci[1, 1])],
                    "p_value": float(fit.pvalues[1])})
        if binary:
            out["odds_ratio_per_sd"] = float(np.exp(fit.params[1]))
        return out

    def save(self, out_dir: str, *, html: bool = True) -> str:
        """The scores' files, ``steering.json`` (the table, the largest coherent doses, the measures,
        the items and the settings) and ``dose_response.png`` in ``out_dir``."""
        self.scores.save(out_dir, html=html)
        items = self.items
        try:
            json.dumps(items)
        except TypeError:
            import warnings

            warnings.warn("the items cannot be written as JSON, so steering.json saves them as null; the measures "
                          "that took them were computed and saved", stacklevel=2)
            items = None
        json_dump(os.path.join(out_dir, "steering.json"), {
            "doses": self.doses, "measures": self.measures, "largest_coherent_dose": self.largest_coherent_dose,
            "table": self.table, "settings": self.settings, "items": items})
        self.plot(os.path.join(out_dir, "dose_response.png"))
        self.out_dir = out_dir
        return out_dir

    @classmethod
    def load(cls, out_dir: str, coherence_rule: Optional[Callable] = None) -> "SteeringEvaluation":
        """An evaluation saved by :meth:`save`, to add measures to it or relate them. ``coherence_rule`` is the
        rule that judges coherence (by default the one saved, when it was a :class:`CoherenceRule`)."""
        data = json.loads(open(os.path.join(out_dir, "steering.json"), encoding="utf-8").read())
        saved = data["settings"].get("coherence_rule")
        if coherence_rule is None:
            if isinstance(saved, dict) and "perplexity_ratio" in saved:
                coherence_rule = CoherenceRule(tuple(saved["perplexity_ratio"]), saved["repetition_increase"])
            else:
                import warnings

                warnings.warn(f"the evaluation was judged with its own coherence rule ({saved}), which cannot be "
                              "loaded; measures added now use the default CoherenceRule unless coherence_rule= is given",
                              stacklevel=2)
                coherence_rule = CoherenceRule()
        ev = cls(Scores.load(out_dir), [float(d) for d in data["doses"]], dict(data["measures"]), data["table"],
                 data["largest_coherent_dose"], data["settings"], data.get("items"), out_dir, coherence_rule)
        return ev

    def plot(self, out_dir: str) -> bool:
        from .visuals import plot_dose_response

        return plot_dose_response(self, out_dir)


# ----------------------------------------------------------------------------------------- the sweep
def _is_probe(obj: Any) -> bool:
    from .probe import ConceptProbe

    return isinstance(obj, ConceptProbe)


def _eval_sd(probe: Any) -> Optional[float]:
    """The SD of a probe's score on its own eval items (its first calibration), or None."""
    if not getattr(probe, "calibrations", None):
        return None
    cal = probe.calibrations[0]
    layers = probe._score_layers()
    sd = float(np.mean(np.asarray(cal.scale)[layers]))
    return sd if np.isfinite(sd) and sd > 0 else None


def _prompt_norms(model, tokenizer, prompts: Sequence[Any], system: Optional[str], batch_size: int) -> np.ndarray:
    """The mean residual norm per layer of the prompts' tokens, read as generation feeds them."""
    from .reading import read_texts

    reps = read_texts(model, tokenizer, list(prompts), "text_mean", system=system, add_generation_prompt=True,
                      batch_size=batch_size, cache=False, console=False)
    return np.asarray(reps.norms, dtype=np.float64).mean(axis=0)


def _resolve_steer(steer: Any, unit: str, layers: Any, window: Optional[int], distribute: bool, positions: Any,
                   norms: Optional[np.ndarray]) -> Tuple[Callable[[float], Intervention], str]:
    """``dose -> intervention``, and a label."""
    if _is_probe(steer):
        steer_layers = steer.steer_layer_list(layers, window)
        if norms is None:
            steer._check_unit(unit)
            return (lambda d: steer.steering(d, unit, layers=steer_layers, distribute=distribute, positions=positions),
                    f"{steer.name} ({unit})")
        share = 1.0 / len(steer_layers) if distribute else 1.0
        vectors = {l: np.asarray(steer.steer_directions[l], dtype=np.float32) for l in steer_layers}
        return (lambda d: Add(vectors, alpha={l: float(d) * share * float(norms[l]) for l in steer_layers},
                              positions=positions,
                              label=f"steering with {steer.name!r}: alpha {float(d):g} (resid_norm of the prompts)"),
                f"{steer.name} (resid_norm of the prompts)")
    if isinstance(steer, Add):
        return steer.scaled, steer.label or "an addition"
    if callable(steer):
        return steer, getattr(steer, "__name__", "a function of the dose")
    raise TypeError("steer must be a ConceptProbe, an Add (at dose 1) or a function dose -> intervention")


@torch.no_grad()
def evaluate_steering(
    steer: Any,
    prompts: Sequence[Any],
    doses: Sequence[float],
    *,
    items: Optional[Sequence[Any]] = None,
    unit: str = "resid_norm",
    layers: Any = None,
    window: Optional[int] = None,
    distribute: bool = False,
    positions: Any = None,
    model: Any = None,
    tokenizer: Any = None,
    probes: Sequence[Any] = (),
    measures: Optional[Dict[str, Any]] = None,
    n_random: int = 10,
    control_seed: int = 0,
    coherence_rule: Callable[[Dict[str, float], Dict[str, float]], bool] = CoherenceRule(),
    system: Optional[str] = None,
    max_new_tokens: int = 48,
    greedy: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: int = 16,
    read_batch_size: int = 16,
    n_bootstrap: int = 1000,
    save: Optional[bool] = None,
    out_dir: Optional[str] = None,
    html: bool = True,
    console: Optional[bool] = None,
    allow_model_mismatch: bool = False,
) -> SteeringEvaluation:
    """Sweep the doses of a steering intervention, with random controls, and measure every completion.

    ``steer`` is a probe (its :meth:`~concept_probe.ConceptProbe.steering` at each dose, in ``unit``,
    at ``layers`` or within ``window`` of its layer, with ``distribute`` and ``positions``), an
    ``Add`` (scaled by each dose: give it at dose 1) or a function ``dose -> Add``. A probe's dose is
    by default in residual-stream norms of each layer (0.25 adds a quarter of a typical token's
    norm), which means the same in any model; a probe without the norms of its training texts gets
    them measured on the prompts (announced). Dose 0 is the baseline every change is measured from,
    and is added (and announced) when missing. ``prompts`` are user turns after ``system``, generated ``batch_size`` at a
    time, greedily by default; ``items`` (one per prompt, any objects, e.g. dicts with an expected
    answer) go to the measures that take them; ``model`` and ``tokenizer`` are the probe's unless
    given.

    Measures of effect: each probe in ``probes`` (and the steering probe), read without the
    intervention, as its raw score (``name``) and, when it has a calibration, in SDs of its own eval
    items (``name (eval SD)``); and each ``measures[name]``: ``texts -> numbers`` (a classifier, a judge's
    ``measure(rubric)``) or ``(texts, items) -> numbers`` (:func:`exact_match`). Measures of
    coherence: perplexity under the model without the intervention, repetition and length, judged by
    ``coherence_rule`` (:class:`CoherenceRule` by default).

    ``n_random`` random controls at each non-zero dose (``control_seed`` seeds them) give, per measure,
    the 95th percentile of the absolute change a direction of the same length produces. The result
    (see :class:`SteeringEvaluation`) is saved by default to ``out_dir``, or to ``steering/<time>``
    under the probe's run folder or the session's ``output_root``.
    """
    steer_probe = steer if _is_probe(steer) else None
    if steer_probe is not None:
        steer_probe._require_model("evaluate_steering")
        model = model or steer_probe.model_bundle.model
        tokenizer = tokenizer or steer_probe.model_bundle.tokenizer
    run_choices: List[Choice] = []
    norms = None
    if steer_probe is not None and unit == "resid_norm" and steer_probe.resid_norm is None:
        norms = _prompt_norms(model, tokenizer, prompts, system, read_batch_size)
        run_choices.append(Choice("resid_norm", "the norms of the probe's training texts", "measured on the prompts",
                                  "the probe has none (it was built from directions)"))
    make, label = _resolve_steer(steer, unit, layers, window, distribute, positions, norms)
    if model is None or tokenizer is None:
        raise ValueError("evaluate_steering needs the model and its tokenizer (or a probe that has them).")
    if items is not None and len(items) != len(prompts):
        raise ValueError(f"{len(items)} items for {len(prompts)} prompts")
    show = settings().console if console is None else bool(console)
    status = (lambda message: print(f"[concept-probe] {message}", flush=True)) if show else (lambda message: None)
    if 0.0 not in {float(d) for d in doses}:
        run_choices.append(Choice("doses", sorted({float(d) for d in doses}), "dose 0 added",
                                  "every change is measured from the completions without steering"))
    doses = sorted({float(d) for d in doses} | {0.0})
    if len(doses) < 2:
        raise ValueError("doses needs at least one non-zero dose")
    announce(run_choices, console=console)
    readers = list(probes)
    if steer_probe is not None and all(steer_probe is not p for p in readers):
        readers.insert(0, steer_probe)
    if not allow_model_mismatch:
        from .modeling import identity_of, identity_mismatches

        bundle = getattr(steer_probe, "model_bundle", None) if steer_probe is not None else None
        here = bundle.identity if bundle is not None and model is bundle.model else identity_of(model, tokenizer)
        for p in readers:
            mismatched = identity_mismatches(p.identity or {}, here) if p.identity else []
            if mismatched:
                raise ValueError(f"the probe {p.name!r} belongs to another model than the one generating ("
                                 + "; ".join(mismatched) + "); its scores of these completions would mean nothing. Pass "
                                 "allow_model_mismatch=True to read them anyway.")
    scorers = [p._token_scorer() for p in readers]
    eval_sd = {p.name: _eval_sd(p) for p in readers}
    measures = dict(measures or {})
    names = {s.name for s in scorers} | {f"{n} (eval SD)" for n, sd in eval_sd.items() if sd}
    clash = (names & set(measures)) | ((names | set(measures)) & set(BUILT_IN))
    if clash:
        raise ValueError(f"measure names must differ from the probes' and from the built-in ones; clash: {sorted(clash)}")

    conditions: List[Condition] = []
    for j, d in enumerate(doses):
        intervention = None if d == 0.0 else make(d)
        conditions.append(Condition({"dose": d, "dose_index": j, "control": None}, intervention))
        if intervention is not None and n_random:
            if not isinstance(intervention, Add):
                raise ValueError("random controls need an Add intervention; pass n_random=0 for others")
            for k in range(int(n_random)):
                conditions.append(Condition({"dose": d, "dose_index": j, "control": k},
                                            intervention.random_like([int(control_seed), k])))
    status(f"Steering evaluation: {len(prompts)} prompts x {len(conditions)} conditions "
           f"({len(doses)} doses, {n_random} random controls at each non-zero one)")
    completions = generate_and_score(model, tokenizer, prompts, conditions, scorers, system=system,
                                     max_new_tokens=max_new_tokens, greedy=greedy, temperature=temperature,
                                     top_p=top_p, seed=seed, batch_size=batch_size, read_batch_size=read_batch_size,
                                     status=status, progress=progress_line("Generated condition", show))

    status("Measuring perplexity under the model without the intervention")
    pairs = [(torch.as_tensor(it.token_ids[:it.span[0]]), torch.as_tensor(it.token_ids[it.span[0]:it.span[1]]))
             for it in completions]
    logprobs = _continuation_logprobs(model, pairs, pad=_pad_id(tokenizer), batch_size=read_batch_size)
    for it, lp in zip(completions, logprobs):
        it.info["perplexity"] = float(np.exp(-lp.mean())) if len(lp) else float("nan")
        it.info["repetition"] = repetition_rate(it.text)
        it.info["n_tokens"] = int(it.span[1] - it.span[0])
    kinds: Dict[str, str] = {}
    for s in scorers:
        kinds[s.name] = "probe raw score"
        sd = eval_sd[s.name]
        if sd:
            kinds[f"{s.name} (eval SD)"] = "probe score, in SDs of the probe's own eval items"
        for it in completions:
            v = it.score_mean[s.name]
            it.info[s.name] = float("nan") if v is None else float(v)
            if sd:
                it.info[f"{s.name} (eval SD)"] = it.info[s.name] / sd

    run_settings = {
        "steer": label, "unit": unit if steer_probe is not None else None, "doses": doses, "n_random": int(n_random),
        "choices": [c.to_dict() for c in run_choices],
        "control_seed": int(control_seed), "n_bootstrap": int(n_bootstrap), "n_prompts": len(prompts), "system": system,
        "generation": {"greedy": bool(greedy), "max_new_tokens": int(max_new_tokens),
                       "temperature": None if greedy else temperature, "top_p": None if greedy else top_p,
                       "seed": None if greedy else seed, "batch_size": int(batch_size)},
        "coherence_rule": coherence_rule.to_dict() if hasattr(coherence_rule, "to_dict") else getattr(
            coherence_rule, "__name__", "a function"),
        "intervention_at_dose_1": make(1.0).to_dict(),
    }
    scores = Scores("generations", [s.name for s in scorers], completions, run_settings,
                    {s.name: "probes are measured in raw scores and in SDs of their eval items" for s in scorers})
    result = SteeringEvaluation(scores, doses, kinds, [], {}, run_settings, None if items is None else list(items),
                                None, coherence_rule)
    result._refresh()
    for name, fn in measures.items():
        result.add_measure(name, fn)
    if settings().save if save is None else save:
        if out_dir is None:
            root = steer_probe.out_dir if steer_probe is not None and steer_probe.out_dir else settings().output_root
            out_dir = os.path.join(root, "steering", now_tag())
        ensure_dir(out_dir)
        result.save(out_dir, html=html)
        status(f"Saved the steering evaluation to {out_dir}")
    return result
