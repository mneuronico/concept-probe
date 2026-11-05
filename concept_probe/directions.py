"""Directions (layer 2 of docs/DESIGN.md): an estimator takes the representations of two groups at
one layer and returns a unit direction in the raw activation space, oriented so that the first
group projects higher. A new estimator is added by registering it, without touching anything else.
"""
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np


def normed_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


DIRECTION_NORMALIZATIONS = ("none", "var", "sd")


def concept_direction(pos_reps: np.ndarray, neg_reps: np.ndarray, normalization: str = "none") -> np.ndarray:
    """Unit concept direction at one layer from pole reps of shape ``(n, dim)``.

    - ``"none"``: difference of means, ``mu_pos - mu_neg``.
    - ``"var"``: difference of means divided coordinate-wise by the pooled within-pole
      variance, ``0.5 * (var_pos + var_neg)``. The diagonal form of the covariance
      correction of mass-mean probing: it keeps a few high-variance coordinates (the
      massive activations of e.g. Gemma 3) from dominating the direction.
    - ``"sd"``: the same, divided by the pooled standard deviation instead.

    Variances are population variances (``ddof=0``), computed in float64. The result is
    normalized to unit length, so scoring and steering use it exactly like an unnormalized
    direction. ``"none"`` is computed in the input dtype, as training always did.

    In a trained probe this is the :class:`MeanDiff` estimator. It sets the direction to read with;
    steering uses the raw difference of means unless asked otherwise (``steering.steer_direction``),
    since ``"var"`` tends to read better and steer less (see the README).
    """
    if normalization not in DIRECTION_NORMALIZATIONS:
        raise ValueError(
            f"Unknown direction normalization {normalization!r}; expected one of {DIRECTION_NORMALIZATIONS}."
        )
    if normalization == "none":
        return normed_np(np.asarray(pos_reps).mean(axis=0) - np.asarray(neg_reps).mean(axis=0))
    pos = np.asarray(pos_reps, dtype=np.float64)
    neg = np.asarray(neg_reps, dtype=np.float64)
    pooled_var = 0.5 * (pos.var(axis=0) + neg.var(axis=0)) + 1e-12
    gap = pos.mean(axis=0) - neg.mean(axis=0)
    gap = gap / (pooled_var if normalization == "var" else np.sqrt(pooled_var))
    return normed_np(gap).astype(np.float32)


# ------------------------------------------------------------------------------------- estimators
def _drop_outliers_direction(pos: np.ndarray, neg: np.ndarray, sd_ratio: float, t_max: float) -> np.ndarray:
    """The raw difference of means with the "massive" coordinates that the concept does not move
    set to zero: a coordinate is dropped when its pooled within-pole SD is more than ``sd_ratio``
    times the layer's median SD and the poles' means do not differ measurably on it (|t| below
    ``t_max``). Every other coordinate keeps its raw weight; unlike ``"var"``, nothing is rescaled."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    gap = pos.mean(axis=0) - neg.mean(axis=0)
    var = 0.5 * (pos.var(axis=0) + neg.var(axis=0))
    sd = np.sqrt(var)
    t = np.abs(gap) / np.sqrt(var / len(pos) + var / len(neg) + 1e-24)
    gap[(sd > sd_ratio * np.median(sd)) & (t < t_max)] = 0.0
    return normed_np(gap).astype(np.float32)


class MeanDiff:
    """Difference of the poles' means, ``normalization``: ``"none"`` (raw), ``"var"`` or ``"sd"``
    (divided coordinate-wise by the pooled within-pole variance or SD; see
    :func:`concept_direction`), or ``"drop_outliers"`` (raw, without the massive coordinates the
    concept does not move; ``sd_ratio`` and ``t_max`` set which)."""

    kind = "mean_diff"
    NORMALIZATIONS = DIRECTION_NORMALIZATIONS + ("drop_outliers",)

    def __init__(self, normalization: str = "none", sd_ratio: float = 10.0, t_max: float = 2.0):
        if normalization not in self.NORMALIZATIONS:
            raise ValueError(f"mean_diff normalization must be one of {self.NORMALIZATIONS}; got {normalization!r}")
        self.normalization, self.sd_ratio, self.t_max = normalization, float(sd_ratio), float(t_max)

    def fit(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        if self.normalization == "drop_outliers":
            return _drop_outliers_direction(pos, neg, self.sd_ratio, self.t_max)
        return concept_direction(pos, neg, self.normalization)

    def spec(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "normalization": self.normalization}
        if self.normalization == "drop_outliers":
            out.update(sd_ratio=self.sd_ratio, t_max=self.t_max)
        return out


def _sklearn(what: str) -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise ImportError(f"The {what} estimator needs scikit-learn: pip install concept-probe[stats].") from exc


class Logistic:
    """L2-regularized logistic regression on standardized coordinates (``C`` is the inverse
    regularization strength), mapped back to the raw activation space."""

    kind = "logistic"

    def __init__(self, C: float = 0.05, max_iter: int = 3000):
        self.C, self.max_iter = float(C), int(max_iter)

    def fit(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        _sklearn("logistic")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        x = np.concatenate([pos, neg]).astype(np.float64)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        scaler = StandardScaler().fit(x)
        w = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(scaler.transform(x), y).coef_[0]
        return normed_np(w / scaler.scale_).astype(np.float32)

    def spec(self) -> Dict[str, Any]:
        return {"kind": self.kind, "C": self.C, "max_iter": self.max_iter}


class LDA:
    """Linear discriminant analysis with a shrunk covariance (``shrinkage``: ``"auto"`` for
    Ledoit-Wolf, or a number in [0, 1])."""

    kind = "lda"

    def __init__(self, shrinkage: Union[str, float] = "auto"):
        self.shrinkage = shrinkage

    def fit(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        _sklearn("lda")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        x = np.concatenate([pos, neg]).astype(np.float64)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        w = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage).fit(x, y).coef_[0]
        return normed_np(w).astype(np.float32)

    def spec(self) -> Dict[str, Any]:
        return {"kind": self.kind, "shrinkage": self.shrinkage}


ESTIMATORS: Dict[str, Callable[..., Any]] = {"mean_diff": MeanDiff, "logistic": Logistic, "lda": LDA}
#: Shorthand names, as the config takes them.
SHORTHANDS: Dict[str, Dict[str, Any]] = {
    "mean_diff": {"kind": "mean_diff"},
    "mean_diff_var": {"kind": "mean_diff", "normalization": "var"},
    "mean_diff_sd": {"kind": "mean_diff", "normalization": "sd"},
    "mean_diff_drop_outliers": {"kind": "mean_diff", "normalization": "drop_outliers"},
    "logistic": {"kind": "logistic"},
    "lda": {"kind": "lda"},
}


def register_estimator(kind: str, factory: Callable[..., Any]) -> None:
    """Make ``{"kind": kind, ...}`` build ``factory(**params)``: an object with ``fit(pos, neg)``
    returning a direction, and ``spec()`` returning its settings."""
    ESTIMATORS[str(kind)] = factory


def make_estimator(spec: Any) -> Any:
    """An estimator from a shorthand name (:data:`SHORTHANDS`), a dict ``{"kind": ..., **params}``,
    or an object that already has ``fit``."""
    if hasattr(spec, "fit"):
        return spec
    if isinstance(spec, str):
        if spec not in SHORTHANDS:
            raise ValueError(f"Unknown estimator {spec!r}; expected one of {sorted(SHORTHANDS)}, a dict with "
                             f"'kind' ({sorted(ESTIMATORS)}), or an object with fit(pos, neg).")
        spec = SHORTHANDS[spec]
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(f"An estimator is a name, a dict with 'kind', or an object with fit(pos, neg); got {spec!r}")
    params = {k: v for k, v in spec.items() if k != "kind"}
    if spec["kind"] not in ESTIMATORS:
        raise ValueError(f"Unknown estimator kind {spec['kind']!r}; registered: {sorted(ESTIMATORS)}")
    return ESTIMATORS[spec["kind"]](**params)


def estimator_spec(estimator: Any) -> Dict[str, Any]:
    """The settings of an estimator, as recorded in metrics (its ``spec()``, or its class name)."""
    est = make_estimator(estimator)
    return est.spec() if hasattr(est, "spec") else {"kind": type(est).__name__}


def fit_directions(
    estimator: Any, pos: np.ndarray, neg: np.ndarray, layers: Optional[Sequence[int]] = None
) -> np.ndarray:
    """One unit direction per layer, ``(layers, dim)`` float32, from reps ``(n, layers, dim)``.

    ``layers`` limits which layers are fitted (the others are NaN rows). Every direction is
    oriented so the first group's mean projects higher than the second's.
    """
    est = make_estimator(estimator)
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    out = np.full(pos.shape[1:], np.nan, dtype=np.float32)
    for l in range(pos.shape[1]) if layers is None else layers:
        v = np.asarray(est.fit(pos[:, l, :], neg[:, l, :]), dtype=np.float32)
        gap = pos[:, l, :].astype(np.float64).mean(axis=0) - neg[:, l, :].astype(np.float64).mean(axis=0)
        if float(gap @ v.astype(np.float64)) < 0:
            v = -v
        out[l] = v
    return out
