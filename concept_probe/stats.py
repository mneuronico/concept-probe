"""Statistics on scores (layer 6a of docs/DESIGN.md): effect sizes, tests, splits, permutation and
bootstrap. Pure functions on arrays."""
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import ttest_ind, ttest_rel
except Exception:
    ttest_ind = ttest_rel = None


def cohen_d_np(x: np.ndarray, y: np.ndarray) -> float:
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
            ds.append(cohen_d_np(pos[idx], neg[idx]))
        else:
            ds.append(cohen_d_np(rng.choice(pos, pos.size), rng.choice(neg, neg.size)))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return [float(lo), float(hi)]


SWEEP_SELECTS = ("effect_size", "p_value")


def _select_best_layer(
    sweep_d: np.ndarray, sweep_p: np.ndarray, search_layers: Sequence[int], sweep_select: str
) -> int:
    """Best layer of the sweep among ``search_layers``.

    ``"effect_size"``: the largest Cohen's d. ``"p_value"``: the smallest p among layers that
    separate the eval sets in the concept's direction (d > 0), since the t-test is two-sided;
    ties in p (it underflows to 0 for well-separated sets) go to the larger d.
    """
    if sweep_select not in SWEEP_SELECTS:
        raise ValueError(f"training.sweep_select must be one of {SWEEP_SELECTS}; got {sweep_select!r}")
    d = np.asarray(sweep_d, dtype=np.float64)
    p = np.asarray(sweep_p, dtype=np.float64)
    allowed = np.zeros(d.shape, dtype=bool)
    allowed[list(search_layers)] = True
    if sweep_select == "effect_size":
        masked = np.where(allowed & np.isfinite(d), d, -np.inf)
        best = int(np.argmax(masked))
        if not np.isfinite(masked[best]):
            raise ValueError(
                "No finite Cohen's d in the best_layer_search range: d needs at least 2 eval items per pole "
                "(on each side of evaluation.selection_split, if set); add eval data or disable the layer sweep."
            )
        return best
    candidates = np.flatnonzero(allowed & (d > 0) & np.isfinite(p))
    if candidates.size == 0:
        raise ValueError(
            "No layer in best_layer_search has d > 0 and a finite p-value. "
            "Install scipy, add eval data, or use sweep_select='effect_size'."
        )
    order = np.lexsort((-d[candidates], p[candidates]))
    return int(candidates[order[0]])
