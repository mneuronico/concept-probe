"""Statistics on scores: effect sizes, tests, splits, permutation and
bootstrap. Pure functions on arrays."""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import ttest_ind, ttest_rel
except Exception:
    ttest_ind = ttest_rel = None


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d of scores ``x`` against ``y``: the difference of means over the pooled SD (``ddof=1``);
    NaN with fewer than two items on a side."""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / (np.sqrt(pooled) + 1e-12))


def _cohen_d_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cohen's d (pooled SD, ddof=1) of each column of ``x`` ``(n_x, k)`` against ``y`` ``(n_y, k)``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = x.shape[0], y.shape[0]
    if nx < 2 or ny < 2:
        return np.full(x.shape[1], np.nan)
    pooled = ((nx - 1) * x.var(axis=0, ddof=1) + (ny - 1) * y.var(axis=0, ddof=1)) / (nx + ny - 2)
    return (x.mean(axis=0) - y.mean(axis=0)) / (np.sqrt(pooled) + 1e-12)


def _pooled_sd_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pooled within-group SD (the denominator of Cohen's d) of each column."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = x.shape[0], y.shape[0]
    if nx < 2 or ny < 2:
        return np.full(x.shape[1], np.nan)
    return np.sqrt(((nx - 1) * x.var(axis=0, ddof=1) + (ny - 1) * y.var(axis=0, ddof=1)) / (nx + ny - 2))


def _welch_p_columns(x: np.ndarray, y: np.ndarray, paired: bool = False) -> np.ndarray:
    """Two-sided t-test p-value of each column (NaN without scipy or with < 2 per group).

    Welch's test for independent groups; with ``paired`` (row i of ``x`` and of ``y`` come from
    the same question), the paired t-test.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if ttest_ind is None or x.shape[0] < 2 or y.shape[0] < 2:
        return np.full(x.shape[1], np.nan)
    if paired:
        return np.asarray(ttest_rel(x, y, axis=0).pvalue, dtype=np.float64)
    return np.asarray(ttest_ind(x, y, axis=0, equal_var=False).pvalue, dtype=np.float64)


def _split_indices(n: int, frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A random split of ``range(n)`` into ``round(frac * n)`` and the rest, each sorted."""
    order = rng.permutation(n)
    k = int(round(float(frac) * n))
    return np.sort(order[:k]), np.sort(order[k:])


def _max_d_permutation_p(
    pos: np.ndarray, neg: np.ndarray, n_permutations: int, rng: np.random.Generator, paired: bool = False
) -> Tuple[float, float]:
    """Largest Cohen's d over the columns (layers), and its permutation p-value.

    Each permutation reassigns the pos/neg labels and takes the largest d over the same columns,
    so the p-value accounts for having searched over all of them. Independent items are shuffled
    across poles; ``paired`` items (row i of both poles comes from one question) swap poles within
    each pair, which is what is exchangeable under the null.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    observed = float(np.nanmax(_cohen_d_columns(pos, neg)))
    pooled = np.concatenate([pos, neg], axis=0)
    n_pos = pos.shape[0]
    exceed = 0
    for _ in range(int(n_permutations)):
        if paired:
            swap = rng.random(n_pos) < 0.5
            a = np.where(swap[:, None], neg, pos)
            b = np.where(swap[:, None], pos, neg)
        else:
            order = rng.permutation(pooled.shape[0])
            a, b = pooled[order[:n_pos]], pooled[order[n_pos:]]
        exceed += np.nanmax(_cohen_d_columns(a, b)) >= observed
    return observed, (exceed + 1) / (int(n_permutations) + 1)


def _bootstrap_d_ci(
    pos: np.ndarray, neg: np.ndarray, n_bootstrap: int, rng: np.random.Generator, paired: bool = False
) -> Optional[List[float]]:
    """95% percentile bootstrap interval of Cohen's d. Independent items are resampled within each
    pole; ``paired`` items are resampled by pair, so both poles of a question stay together."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if not n_bootstrap or pos.size < 2 or neg.size < 2:
        return None
    ds = []
    for _ in range(int(n_bootstrap)):
        if paired:
            idx = rng.integers(0, pos.size, pos.size)
            ds.append(cohen_d(pos[idx], neg[idx]))
        else:
            ds.append(cohen_d(rng.choice(pos, pos.size), rng.choice(neg, neg.size)))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return [float(lo), float(hi)]


LAYER_SELECTIONS = ("effect_size", "p_value")


def _select_best_layer(
    d_by_layer: np.ndarray, p_by_layer: np.ndarray, search_layers: Sequence[int], select_layer_by: str
) -> int:
    """Best layer of the sweep among ``search_layers``.

    ``"effect_size"``: the largest Cohen's d. ``"p_value"``: the smallest p among layers that
    separate the eval sets in the concept's direction (d > 0), since the t-test is two-sided;
    ties in p (it underflows to 0 for well-separated sets) go to the larger d.
    """
    if select_layer_by not in LAYER_SELECTIONS:
        raise ValueError(f"select_layer_by must be one of {LAYER_SELECTIONS}; got {select_layer_by!r}")
    d = np.asarray(d_by_layer, dtype=np.float64)
    p = np.asarray(p_by_layer, dtype=np.float64)
    allowed = np.zeros(d.shape, dtype=bool)
    allowed[list(search_layers)] = True
    if select_layer_by == "effect_size":
        masked = np.where(allowed & np.isfinite(d), d, -np.inf)
        best = int(np.argmax(masked))
        if not np.isfinite(masked[best]):
            raise ValueError(
                "No finite Cohen's d in search_layers: d needs at least 2 eval items per pole (on each side of "
                "selection_split, if set); add eval items, or give layer= instead of searching for one."
            )
        return best
    candidates = np.flatnonzero(allowed & (d > 0) & np.isfinite(p))
    if candidates.size == 0:
        raise ValueError(
            "No layer in search_layers has d > 0 and a finite p-value. "
            "Install scipy, add eval data, or use select_layer_by='effect_size'."
        )
    order = np.lexsort((-d[candidates], p[candidates]))
    return int(candidates[order[0]])


# ------------------------------------------------------------------------------- comparisons
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under the ROC curve of scores separating ``pos`` from ``neg``: the chance that a random
    positive item scores above a random negative one (ties count half)."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):  # average ranks over ties
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def accuracy(pos: np.ndarray, neg: np.ndarray, threshold: float) -> float:
    """Share of items on the right side of ``threshold`` (positives above, negatives below)."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    n = pos.size + neg.size
    return float(((pos > threshold).sum() + (neg <= threshold).sum()) / n) if n else float("nan")


def bootstrap_ci(stat, pos: np.ndarray, neg: np.ndarray, n_bootstrap: int, rng: np.random.Generator,
                 paired: bool = False, level: float = 0.95) -> Optional[List[float]]:
    """Percentile bootstrap interval of ``stat(pos, neg)``. Independent items are resampled within
    each group; ``paired`` items (row i of both groups is one pair) are resampled by pair."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if not n_bootstrap or len(pos) < 2 or len(neg) < 2:
        return None
    values = []
    for _ in range(int(n_bootstrap)):
        if paired:
            idx = rng.integers(0, len(pos), len(pos))
            values.append(stat(pos[idx], neg[idx]))
        else:
            values.append(stat(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]))
    lo, hi = np.nanpercentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def compare_scorers(stat, a_pos: np.ndarray, a_neg: np.ndarray, b_pos: np.ndarray, b_neg: np.ndarray,
                    n_bootstrap: int = 1000, seed: int = 0, paired: bool = False) -> Dict[str, Any]:
    """How much better scorer ``a`` does than scorer ``b`` on the same items, by ``stat(pos, neg)``
    (for example :func:`auroc`, or Cohen's d), with a paired bootstrap interval: every resample
    takes the same items for both scorers, so what the items have in common cancels out.

    ``a_pos[i]`` and ``b_pos[i]`` are two scores of the same item (two probes, or a probe and a
    judge); ``paired`` says row i of the positive and negative items is one pair.
    """
    a_pos, a_neg, b_pos, b_neg = (np.asarray(x, dtype=np.float64) for x in (a_pos, a_neg, b_pos, b_neg))
    if a_pos.shape != b_pos.shape or a_neg.shape != b_neg.shape:
        raise ValueError("Both scorers must score the same items: a_pos/b_pos and a_neg/b_neg need equal shapes.")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(n_bootstrap)):
        ip = rng.integers(0, len(a_pos), len(a_pos))
        ineg = ip if paired else rng.integers(0, len(a_neg), len(a_neg))
        diffs.append(stat(a_pos[ip], a_neg[ineg]) - stat(b_pos[ip], b_neg[ineg]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    a, b = stat(a_pos, a_neg), stat(b_pos, b_neg)
    return {"a": float(a), "b": float(b), "difference": float(a - b), "ci95": [float(lo), float(hi)],
            "n_bootstrap": int(n_bootstrap)}


def random_directions_null(pos: np.ndarray, neg: np.ndarray, n_random: int = 50, seed: int = 0,
                           percentile: float = 95.0) -> np.ndarray:
    """What Cohen's d is without a concept, on the same items: per layer, the ``percentile`` of |d|
    between ``pos`` and ``neg`` representations ``(n, layers, dim)`` projected on ``n_random``
    random unit directions (the same directions at every layer). A direction's length does not
    change d, so random directions of any norm give the same null."""
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    w = np.random.default_rng(seed).standard_normal((pos.shape[2], int(n_random)))
    w /= np.linalg.norm(w, axis=0, keepdims=True)
    out = np.full(pos.shape[1], np.nan)
    for l in range(pos.shape[1]):
        a = pos[:, l, :].astype(np.float64) @ w
        b = neg[:, l, :].astype(np.float64) @ w
        out[l] = float(np.percentile(np.abs(_cohen_d_columns(a, b)), percentile))
    return out


# ----------------------------------------------------------------------------------- token scores
#: How the scores of a text's tokens are summarized into one per text (see :func:`summarize_tokens`);
#: also ``"p<k>"`` for the k-th percentile.
TOKEN_SUMMARIES = ("mean", "max", "min", "median", "extreme")


def summarize_tokens(values: np.ndarray, items: np.ndarray, how: str = "mean", center: float = 0.0
                     ) -> Tuple[np.ndarray, List[int]]:
    """One value per item from per-token ``values`` (rows, or ``(rows, k)``) and the item of each row:
    ``"mean"``, ``"max"``, ``"min"``, ``"median"``, ``"p<k>"`` (the k-th percentile) or
    ``"extreme"`` (the token farthest from ``center``, with its sign). Returns the values, in the
    order the items first appear, and those items."""
    values = np.asarray(values, dtype=np.float64)
    items = np.asarray(items)
    order = [int(i) for i in dict.fromkeys(items.tolist())]
    out = []
    for i in order:
        v = values[items == i]
        if how == "mean":
            out.append(v.mean(axis=0))
        elif how == "max":
            out.append(v.max(axis=0))
        elif how == "min":
            out.append(v.min(axis=0))
        elif how == "median":
            out.append(np.median(v, axis=0))
        elif how == "extreme":
            idx = np.abs(v - center).argmax(axis=0)
            out.append(np.take_along_axis(v, np.asarray(idx)[None, ...], axis=0)[0] if v.ndim > 1 else v[idx])
        elif how.startswith("p") and how[1:].replace(".", "", 1).isdigit():
            out.append(np.percentile(v, float(how[1:]), axis=0))
        else:
            raise ValueError(f"unknown summary {how!r}; expected one of {TOKEN_SUMMARIES} or 'p<k>'")
    return np.asarray(out), order
