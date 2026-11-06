"""Evaluating how a probe reads (layer 6b of docs/DESIGN.md).

:func:`evaluate_reading` takes a probe's directions and named sets of evaluation items (each the
representations of its positive and negative items, with how they were read and whether they come
in pairs). The primary set chooses the layer, on part of its items, and gives the numbers reported
for the probe, on the rest; every other set is measured at that layer. For every set it reports
Cohen's d and its p, a bootstrap interval, the AUROC, and what d is without a concept on the same
items (random directions); for the primary set also the accuracy with a calibrated threshold.
:func:`label_shuffle_control` retrains with shuffled labels, and :func:`cross_reading` reads several
probes on each other's sets. Everything works on arrays.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .directions import fit_directions
from .inputs import ReadContext
from .linear_probe import fit_threshold
from .stats import (
    SWEEP_SELECTS,
    _bootstrap_d_ci,
    _cohen_d_columns,
    _max_d_permutation_p,
    _select_best_layer,
    _split_indices,
    _welch_p_columns,
    accuracy,
    auroc,
    bootstrap_ci,
    cohen_d_np,
    random_directions_null,
)


@dataclass
class EvalSet:
    """Evaluation items: the representations ``(items, layers, dim)`` of the positive and negative
    ones, how they were read (``context``), and whether they come in ``paired`` rows (row i of both
    is one pair: the two poles' answers to one question, or a minimal pair of texts). Pairs keep
    together when the items are split and resampled, and get paired tests."""

    name: str
    pos: np.ndarray
    neg: np.ndarray
    context: Optional[ReadContext] = None
    paired: bool = False
    description: str = ""

    def __post_init__(self):
        self.pos, self.neg = np.asarray(self.pos), np.asarray(self.neg)
        if self.paired and len(self.pos) != len(self.neg):
            raise ValueError(f"eval set {self.name!r} is paired but has {len(self.pos)} positive and "
                             f"{len(self.neg)} negative items.")


def _opt(x: Any) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)


@dataclass
class SetResult:
    """One set's numbers. Curves have one value per layer on all the set's items; the values at the
    probe's layer are measured on ``measured_on`` (the held-out part of the primary set, all items
    of the others)."""

    name: str
    description: str
    context: Optional[Dict[str, Any]]
    paired: bool
    n_pos: int
    n_neg: int
    pos_scores: np.ndarray
    neg_scores: np.ndarray
    d_curve: np.ndarray
    p_curve: np.ndarray
    null_d95_curve: np.ndarray
    measured_on: str
    d: Optional[float] = None
    p: Optional[float] = None
    d_ci95: Optional[List[float]] = None
    auroc: Optional[float] = None
    auroc_ci95: Optional[List[float]] = None
    null_d95: Optional[float] = None
    accuracy: Optional[float] = None
    accuracy_ci95: Optional[List[float]] = None

    def to_metrics(self) -> Dict[str, Any]:
        return {"description": self.description, "context": self.context, "paired": self.paired,
                "n_pos": self.n_pos, "n_neg": self.n_neg, "measured_on": self.measured_on,
                "d": self.d, "p": self.p, "d_ci95": self.d_ci95, "auroc": self.auroc, "auroc_ci95": self.auroc_ci95,
                "null_d95": self.null_d95, "accuracy": self.accuracy, "accuracy_ci95": self.accuracy_ci95}


@dataclass
class ReadingEvaluation:
    """The result of :func:`evaluate_reading`. ``selection`` says how the layer was chosen:
    ``"split"`` (on the selection part of the primary set's items, numbers on the rest),
    ``"all_eval"`` (on the same items that measure it: optimistic), or ``"fixed_layer"``."""

    layer: int
    primary: str
    selection: str
    sets: Dict[str, SetResult]
    selection_d_curve: Optional[np.ndarray] = None
    report_d_curve: Optional[np.ndarray] = None
    report_null_d95_curve: Optional[np.ndarray] = None
    selection_idx: Tuple[np.ndarray, np.ndarray] = (np.zeros(0, int), np.zeros(0, int))
    report_idx: Tuple[np.ndarray, np.ndarray] = (np.zeros(0, int), np.zeros(0, int))
    d_selection: Optional[float] = None
    sweep_max_d: Optional[float] = None
    sweep_max_d_perm_p: Optional[float] = None
    n_permutations: Optional[int] = None
    search_layers: Optional[List[int]] = None
    threshold: Optional[Dict[str, Any]] = None
    n_random: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def main(self) -> SetResult:
        return self.sets[self.primary]

    def to_metrics(self) -> Dict[str, Any]:
        notes = {
            "split": "the layer was chosen on the selection part of the primary set's items; its numbers are "
                     "measured on the held-out rest (split by pair when the items are paired).",
            "all_eval": "the layer was chosen on the same items that measure it, so its numbers are optimistic.",
            "fixed_layer": "the layer was fixed in advance; the numbers use all of the primary set's items.",
        }
        return {"primary": self.primary, "selection": self.selection, "selection_note": notes[self.selection],
                "layer": self.layer, "d_selection": self.d_selection, "sweep_max_d": self.sweep_max_d,
                "sweep_max_d_perm_p": self.sweep_max_d_perm_p, "n_permutations": self.n_permutations,
                "n_random_directions": self.n_random, "threshold": self.threshold,
                "sets": {name: r.to_metrics() for name, r in self.sets.items()}}


def _project(reps: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.einsum("nld,ld->nl", np.asarray(reps, dtype=np.float64), np.asarray(directions, dtype=np.float64))


def evaluate_reading(
    directions: np.ndarray,
    sets: Sequence[EvalSet],
    *,
    primary: Optional[str] = None,
    layer: Optional[int] = None,
    selection_split: Optional[float] = 0.5,
    search_layers: Optional[Sequence[int]] = None,
    sweep_select: str = "effect_size",
    n_random: int = 50,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    seed: int = 0,
) -> ReadingEvaluation:
    """Choose the probe's layer on the ``primary`` set (the first by default) and measure every set
    there. See the module docstring.

    With ``layer`` given, it is used as is. Otherwise the layer with the largest d (or smallest p
    among layers with d > 0, ``sweep_select="p_value"``) within ``search_layers`` is chosen on a
    ``selection_split`` part of the primary items, and its numbers come from the rest; a
    permutation p for the largest d corrects for having searched over the layers.
    """
    if not sets:
        raise ValueError("evaluate_reading needs at least one eval set.")
    if sweep_select not in SWEEP_SELECTS:
        raise ValueError(f"sweep_select must be one of {SWEEP_SELECTS}; got {sweep_select!r}")
    by_name = {s.name: s for s in sets}
    if len(by_name) != len(sets):
        raise ValueError("eval sets need distinct names.")
    primary = primary or sets[0].name
    if primary not in by_name:
        raise ValueError(f"no eval set named {primary!r}; the sets are {sorted(by_name)}.")
    main = by_name[primary]
    num_layers = int(np.asarray(directions).shape[0])
    search = list(range(num_layers)) if search_layers is None else list(search_layers)
    warnings: List[str] = []

    P, N = _project(main.pos, directions), _project(main.neg, directions)
    nP, nN = len(P), len(N)
    sweep_d = _cohen_d_columns(P, N)
    # float64: p-values of well-separated sets underflow to 0 in float32.
    sweep_p = _welch_p_columns(P, N, paired=main.paired)
    pos_sel = neg_sel = np.zeros((0,), dtype=np.int64)
    pos_est, neg_est = np.arange(nP), np.arange(nN)
    selection_d = d_selection = perm = max_d = None
    if layer is not None:
        selection, best = "fixed_layer", int(layer)
    else:
        selection = "all_eval"
        if selection_split is not None:
            split_rng = np.random.default_rng([seed, 1])
            ps, pe = _split_indices(nP, selection_split, split_rng)
            # Both items of a pair go to the same side: split independently, a pair could help choose
            # the layer through one pole and then be measured through the other.
            ns, ne = (ps, pe) if main.paired else _split_indices(nN, selection_split, split_rng)
            if min(len(ps), len(pe), len(ns), len(ne)) < 2:
                warnings.append("selection_split needs at least 2 items per pole on each side of the split; the layer "
                                "is chosen and measured on all eval items.")
            else:
                selection, pos_sel, pos_est, neg_sel, neg_est = "split", ps, pe, ns, ne
        if selection == "split":
            selection_d = _cohen_d_columns(P[pos_sel], N[neg_sel])
            best = _select_best_layer(selection_d, _welch_p_columns(P[pos_sel], N[neg_sel], paired=main.paired),
                                      search, sweep_select)
            d_selection = float(selection_d[best])
        else:
            best = _select_best_layer(sweep_d, sweep_p, search, sweep_select)
            warnings.append(f"the layer was chosen over {len(search)} layers on the same eval items that measure it, "
                            "so its numbers are optimistic; set a selection split for a held-out estimate.")
        if n_permutations and int(n_permutations) > 0:
            max_d, perm = _max_d_permutation_p(P[:, search], N[:, search], int(n_permutations),
                                               np.random.default_rng([seed, 2]), paired=main.paired)

    # The primary set's numbers, on the held-out items.
    ep, en = P[pos_est], N[neg_est]
    report_d = _cohen_d_columns(ep, en)
    report_null = random_directions_null(main.pos[pos_est], main.neg[neg_est], n_random, seed=seed) if n_random else None
    threshold = None
    if len(P) and len(N):
        tp, tn = (P[pos_sel, best], N[neg_sel, best]) if selection == "split" else (P[:, best], N[:, best])
        threshold = dict(fit_threshold(tp, tn), layer=int(best),
                         fit_on="the selection part of the primary set" if selection == "split" else "all its items")
    results: Dict[str, SetResult] = {}
    p_best = _welch_p_columns(ep[:, [best]], en[:, [best]], paired=main.paired)[0]
    acc = None if threshold is None else accuracy(ep[:, best], en[:, best], threshold["threshold"])
    results[primary] = SetResult(
        name=primary, description=main.description, context=None if main.context is None else main.context.to_dict(),
        paired=main.paired, n_pos=nP, n_neg=nN, pos_scores=P, neg_scores=N, d_curve=sweep_d, p_curve=sweep_p,
        null_d95_curve=report_null if report_null is not None else np.full(num_layers, np.nan),
        measured_on="the held-out part" if selection == "split" else "all items",
        d=_opt(cohen_d_np(ep[:, best], en[:, best])), p=_opt(p_best),
        d_ci95=_bootstrap_d_ci(ep[:, best], en[:, best], n_bootstrap, np.random.default_rng([seed, 3]), paired=main.paired),
        auroc=_opt(auroc(ep[:, best], en[:, best])),
        auroc_ci95=bootstrap_ci(auroc, ep[:, best], en[:, best], n_bootstrap, np.random.default_rng([seed, 5]),
                                paired=main.paired),
        null_d95=None if report_null is None else _opt(report_null[best]),
        accuracy=_opt(acc),
        accuracy_ci95=None if threshold is None else bootstrap_ci(
            lambda a, b: accuracy(a, b, threshold["threshold"]), ep[:, best], en[:, best], n_bootstrap,
            np.random.default_rng([seed, 6]), paired=main.paired),
    )

    # Every other set, at that layer, on all its items.
    for k, s in enumerate([s for s in sets if s.name != primary]):
        TP, TN = _project(s.pos, directions), _project(s.neg, directions)
        if len(TP) == 0 or len(TN) == 0:
            continue
        null = random_directions_null(s.pos, s.neg, n_random, seed=seed) if n_random else np.full(num_layers, np.nan)
        d_curve, p_curve = _cohen_d_columns(TP, TN), _welch_p_columns(TP, TN, paired=s.paired)
        results[s.name] = SetResult(
            name=s.name, description=s.description, context=None if s.context is None else s.context.to_dict(),
            paired=s.paired, n_pos=len(TP), n_neg=len(TN), pos_scores=TP, neg_scores=TN, d_curve=d_curve,
            p_curve=p_curve, null_d95_curve=null, measured_on="all items",
            d=_opt(d_curve[best]), p=_opt(p_curve[best]),
            d_ci95=_bootstrap_d_ci(TP[:, best], TN[:, best], n_bootstrap, np.random.default_rng([seed, 4 + 10 * k]),
                                   paired=s.paired),
            auroc=_opt(auroc(TP[:, best], TN[:, best])),
            auroc_ci95=bootstrap_ci(auroc, TP[:, best], TN[:, best], n_bootstrap,
                                    np.random.default_rng([seed, 5 + 10 * k + 10]), paired=s.paired),
            null_d95=_opt(null[best]) if n_random else None,
        )
    return ReadingEvaluation(
        layer=int(best), primary=primary, selection=selection, sets=results, selection_d_curve=selection_d,
        report_d_curve=report_d, report_null_d95_curve=report_null, selection_idx=(pos_sel, neg_sel),
        report_idx=(pos_est, neg_est), d_selection=d_selection, sweep_max_d=_opt(max_d), sweep_max_d_perm_p=_opt(perm),
        n_permutations=int(n_permutations) if perm is not None else None, search_layers=search, threshold=threshold,
        n_random=int(n_random), warnings=warnings,
    )


def label_shuffle_control(
    estimator: Any,
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    eval_pos: np.ndarray,
    eval_neg: np.ndarray,
    layer: int,
    n_shuffles: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    """What the whole training procedure finds when the labels mean nothing: the training items'
    labels are shuffled, directions are fitted again at ``layer``, and Cohen's d is measured on the
    eval items. A probe whose d is not well above these did not learn the labels."""
    train = np.concatenate([np.asarray(train_pos)[:, [layer]], np.asarray(train_neg)[:, [layer]]])
    n_pos = len(train_pos)
    rng = np.random.default_rng([seed, 8])
    ds = []
    for _ in range(int(n_shuffles)):
        order = rng.permutation(len(train))
        v = fit_directions(estimator, train[order[:n_pos]], train[order[n_pos:]])[0]
        a = np.asarray(eval_pos)[:, layer].astype(np.float64) @ v.astype(np.float64)
        b = np.asarray(eval_neg)[:, layer].astype(np.float64) @ v.astype(np.float64)
        ds.append(abs(cohen_d_np(a, b)))
    ds = np.asarray(ds)
    return {"n_shuffles": int(n_shuffles), "layer": int(layer), "abs_d_mean": _opt(float(ds.mean())) if len(ds) else None,
            "abs_d_p95": _opt(float(np.percentile(ds, 95))) if len(ds) else None, "abs_d": ds.tolist()}


def cross_reading(
    probes: Sequence[Any],
    sets: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    n_random: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    """How much each probe reads each set: Cohen's d of the probe's scores (at its layer) between the
    set's positive and negative items, next to what random directions give on the same items at that
    layer, and the cosine between the probes' directions. Probes are read on each other's training
    or eval sets to see whether one also encodes the other's concept.

    ``probes`` have ``directions``, ``layer`` and ``concept.name``; ``sets`` map a name to the
    representations ``(pos, neg)``, each ``(items, layers, dim)``.
    """
    names = [str(p.concept.name) for p in probes]
    table: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for probe, name in zip(probes, names):
        v = np.asarray(probe.directions)[probe.layer].astype(np.float64)
        row = {}
        for set_name, (pos, neg) in sets.items():
            pos, neg = np.asarray(pos), np.asarray(neg)
            d = cohen_d_np(pos[:, probe.layer].astype(np.float64) @ v, neg[:, probe.layer].astype(np.float64) @ v)
            null = random_directions_null(pos[:, [probe.layer]], neg[:, [probe.layer]], n_random, seed=seed)[0]
            row[set_name] = {"d": _opt(d), "null_d95": _opt(null)}
        table[name] = row
    cosines = {}
    for a, na in zip(probes, names):
        cosines[na] = {}
        for b, nb in zip(probes, names):
            va = np.asarray(a.directions)[a.layer].astype(np.float64)
            vb = np.asarray(b.directions)[a.layer].astype(np.float64)
            cosines[na][nb] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return {"d": table, "cosine_at_row_probe_layer": cosines}


# ------------------------------------------------------------------------------- learning curves
def learning_curve(
    pos: np.ndarray,
    neg: np.ndarray,
    eval_sets: Sequence[EvalSet],
    sizes: Sequence[int],
    *,
    n_repeats: int = 20,
    n_null: int = 20,
    estimator: Any = "mean_diff",
    paired: bool = False,
    layer: Optional[int] = None,
    selection_split: Optional[float] = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """How a probe's reading grows with the number of training items, on fixed eval sets.

    For each size ``n``, ``n_repeats`` times: ``n`` training items of each pole drawn at random (the
    same pairs of both poles when ``paired``), directions fitted, and :func:`evaluate_reading` on
    ``eval_sets`` (the layer chosen on part of the primary set, or ``layer``; the primary set is split
    once, the same way for every draw, so the spread across draws is that of the training items
    drawn). The control, ``n_null``
    times per size: the ``2n`` items of a draw split into two random groups, fitted and evaluated
    the same way, which says what d a probe of that size reaches with no concept at all. Each draw
    also gives the cosine between its direction and the one fitted on every training item, at the
    chosen layer.

    Returns ``rows`` (one per draw: ``n``, ``kind`` ``"probe"`` or ``"null"``, ``d``, ``auroc``, the
    layer, the cosine with the full direction, and every other set's d) and ``summary`` per kind and
    size: mean, SD and 95th percentile of each number, and the share of draws that chose the most
    common layer.
    """
    P, N = np.asarray(pos), np.asarray(neg)
    if paired and len(P) != len(N):
        raise ValueError("paired training items need as many positive as negative items")
    full = fit_directions(estimator, P, N).astype(np.float64)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    def evaluate(dirs: np.ndarray, kind: str, n: int, rep: int) -> None:
        # The same seed for every draw: the eval items are split the same way each time, so the
        # spread across draws comes from the training items drawn, not from which items measure.
        ev = evaluate_reading(dirs, eval_sets, layer=layer, selection_split=selection_split, n_random=0,
                              n_bootstrap=0, n_permutations=0, seed=seed)
        v = dirs[ev.layer]
        row = {"n": int(n), "kind": kind, "repeat": rep, "layer": int(ev.layer), "d": ev.main.d, "auroc": ev.main.auroc,
               "cos_full": float(v @ full[ev.layer] / (np.linalg.norm(v) * np.linalg.norm(full[ev.layer]) + 1e-12))}
        row.update({f"d_{name}": s.d for name, s in ev.sets.items() if name != ev.primary})
        rows.append(row)

    for n in sizes:
        n = int(n)
        if n > min(len(P), len(N)):
            raise ValueError(f"size {n} is larger than the training items ({min(len(P), len(N))} per pole)")
        for rep in range(int(n_repeats)):
            ip = rng.choice(len(P), n, replace=False)
            i_n = ip if paired else rng.choice(len(N), n, replace=False)
            evaluate(fit_directions(estimator, P[ip], N[i_n]).astype(np.float64), "probe", n, rep)
        for rep in range(int(n_null)):
            ip = rng.choice(len(P), n, replace=False)
            i_n = ip if paired else rng.choice(len(N), n, replace=False)
            X = np.concatenate([P[ip], N[i_n]])
            perm = rng.permutation(len(X))
            evaluate(fit_directions(estimator, X[perm[:n]], X[perm[n:]]).astype(np.float64), "null", n, rep)

    summary: Dict[str, Dict[int, Dict[str, float]]] = {}
    for kind in ("probe", "null"):
        summary[kind] = {}
        for n in sizes:
            sel = [r for r in rows if r["kind"] == kind and r["n"] == int(n)]
            if not sel:
                continue
            s: Dict[str, float] = {}
            for key in [k for k in sel[0] if k not in ("n", "kind", "repeat", "layer")]:
                vals = np.array([np.nan if r[key] is None else r[key] for r in sel], dtype=np.float64)
                if np.isfinite(vals).any():
                    s[key] = float(np.nanmean(vals))
                    s[f"{key}_sd"] = float(np.nanstd(vals))
                    s[f"{key}_p95"] = float(np.nanpercentile(vals, 95))
            layers = np.array([r["layer"] for r in sel])
            s["layer_mode_share"] = float(np.bincount(layers).max() / len(layers))
            summary[kind][int(n)] = s
    return {"rows": rows, "summary": summary, "sizes": [int(n) for n in sizes], "n_repeats": int(n_repeats),
            "n_null": int(n_null)}


# -------------------------------------------------------------------------------------- confounds
def confound_check(
    directions: np.ndarray,
    layer: int,
    confounds: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    *,
    n_random: int = 50,
    selection_split: float = 0.5,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Does a probe read something it should not? For each confound, given the representations of
    items that have it and items that do not (``{name: (with, without)}``, ideally items that
    differ only in it):

    - ``probe_d``: Cohen's d of the probe's scores (at ``layer``) between the two groups, next to
      ``null_d95``, what random directions give on the same items: a probe that reads the confound
      separates them beyond it (``reads_confound``);
    - ``confound_direction_d``: how separable the confound is at all: a difference of means fitted
      on part of the items (``selection_split``) and measured on the rest, at ``layer``;
    - ``cosine``: per layer, the cosine between that confound's direction (fitted on every item) and
      the probe's, and ``cosine_at_layer``.

    To train a probe without a confound, see :class:`concept_probe.directions.Eraser`.
    """
    V = np.asarray(directions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Any]] = {}
    for name, (a, b) in confounds.items():
        A, B = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        probe_d = cohen_d_np(A[:, layer] @ V[layer], B[:, layer] @ V[layer])
        null = random_directions_null(A[:, [layer]], B[:, [layer]], n_random, seed=seed)[0]
        conf = fit_directions("mean_diff", A, B).astype(np.float64)
        cos = np.einsum("ld,ld->l", conf, V) / (np.linalg.norm(conf, axis=1) * np.linalg.norm(V, axis=1) + 1e-12)
        sa, ra = _split_indices(len(A), selection_split, rng)
        sb, rb = _split_indices(len(B), selection_split, rng)
        held = None
        if min(len(sa), len(sb), len(ra), len(rb)) >= 2:
            c = fit_directions("mean_diff", A[sa][:, [layer]], B[sb][:, [layer]])[0].astype(np.float64)
            held = _opt(cohen_d_np(A[ra, layer] @ c, B[rb, layer] @ c))
        out[str(name)] = {"n_with": len(A), "n_without": len(B), "probe_d": _opt(probe_d), "null_d95": _opt(null),
                          "reads_confound": bool(abs(probe_d) > null) if np.isfinite(null) else None,
                          "confound_direction_d": held, "cosine": cos.tolist(), "cosine_at_layer": float(cos[layer])}
    return out


# ------------------------------------------------------------------------------------ red teaming
def red_team(probe: Any, attacks: Mapping[str, Mapping[str, Any]], *, role: str = "raw",
             system: Optional[str] = None, prompt: Optional[str] = None, batch_size: int = 16) -> Dict[str, Any]:
    """Attack a probe with items written to exploit a suspected confound of its dataset.

    ``attacks`` maps a name to ``{"pos": [...], "neg": [...], "neutral": [(a, b), ...]}`` (any of
    them): items of each pole that carry the other pole's surface features, and pairs of texts
    without the concept that differ only in the feature (``a`` with the feature suspected to push
    toward the positive pole). The texts are read with the probe's model as ``role``, ``system`` and
    ``prompt`` say (default: raw text) and scored in z with the probe's calibration for that reading.
    Per attack: the share of each pole's items on the right side (the calibrated threshold when the
    calibration has one, else z = 0) and each item's z; for the neutral pairs, the shift
    ``z(a) - z(b)``: how far the feature alone moves the probe, in within-pole SDs (0: not at all).
    """
    from .reading import read_texts  # reading needs the model; the rest of the module works on arrays

    probe._require_model("red_team")
    model, tok = probe.model_bundle.model, probe.model_bundle.tokenizer
    ctx = ReadContext(kind="text", role=role, system=system, prompt=prompt)
    cal = probe.calibration_for(ctx)
    if cal is None:
        raise ValueError(probe._missing_z_reason(ctx))

    def read(texts: Sequence[Any]) -> Any:
        return read_texts(model, tok, list(texts), role=role, system=system, prompt=prompt, batch_size=batch_size,
                          cache=False, console=False)

    out: Dict[str, Any] = {}
    for name, attack in attacks.items():
        row: Dict[str, Any] = {}
        for pole in ("pos", "neg"):
            texts = list(attack.get(pole, []))
            if not texts:
                continue
            reps = read(texts)
            z = probe.z(reps, ctx)
            if cal.threshold:
                positive, _ = probe.classify(reps, ctx)
            else:
                positive = z > 0
            right = positive if pole == "pos" else ~positive
            row[pole] = {"accuracy": float(np.mean(right)), "n": len(texts), "z": z.tolist(),
                         "rule": "calibrated threshold" if cal.threshold else "z = 0"}
        neutral = list(attack.get("neutral", []))
        if neutral:
            shift = probe.z(read([a for a, _ in neutral]), ctx) - probe.z(read([b for _, b in neutral]), ctx)
            row["neutral"] = {"shift_mean": float(shift.mean()), "shift": shift.tolist(), "n": len(neutral)}
        out[str(name)] = row
    return out
