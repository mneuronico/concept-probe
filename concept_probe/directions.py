"""Directions: an estimator takes the representations of two groups at
one layer and returns a unit direction in the raw activation space, oriented so that the first
group projects higher. A new estimator is added by registering it, without touching anything else.
"""
import importlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .record import Choice


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
    steering uses the raw difference of means unless asked otherwise (``ConceptProbe.fit(steer_estimator=...)``),
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
def _drop_outliers_direction(pos: np.ndarray, neg: np.ndarray, sd_ratio: float, t_max: float,
                             dropped: Optional[List[int]] = None) -> np.ndarray:
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
    drop = (sd > sd_ratio * np.median(sd)) & (t < t_max)
    gap[drop] = 0.0
    if dropped is not None:
        dropped.append(int(drop.sum()))
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
        #: with ``"drop_outliers"``, how many coordinates each fit left out (one entry per layer fitted)
        self.dropped: List[int] = []

    def fit(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        if self.normalization == "drop_outliers":
            return _drop_outliers_direction(pos, neg, self.sd_ratio, self.t_max, self.dropped)
        return concept_direction(pos, neg, self.normalization)

    def spec(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "normalization": self.normalization}
        if self.normalization == "drop_outliers":
            out.update(sd_ratio=self.sd_ratio, t_max=self.t_max)
        return out


def _sklearn(what: str) -> None:
    try:
        importlib.import_module("sklearn")
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
#: Shorthand names of the estimators.
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
    estimator: Any, pos: np.ndarray, neg: np.ndarray, layers: Optional[Sequence[int]] = None,
    erase: Optional["Eraser"] = None, flips: Optional[List[int]] = None,
) -> np.ndarray:
    """One unit direction per layer, ``(layers, dim)`` float32, from reps ``(n, layers, dim)``.

    ``layers`` limits which layers are fitted (the others are NaN rows). Every direction is
    oriented so the first group's mean projects higher than the second's. With ``erase`` (an
    :class:`Eraser`), the estimator is fitted on the reps with the confound erased, and each
    direction is turned into the one that reads the same thing from raw reps
    (:meth:`Eraser.to_raw_directions`): projections then do not depend on the confound.
    """
    est = make_estimator(estimator)
    if hasattr(est, "dropped"):
        est.dropped = []
    pos = np.asarray(getattr(pos, "data", pos))
    neg = np.asarray(getattr(neg, "data", neg))
    if erase is not None:
        pos, neg = erase.apply(pos), erase.apply(neg)
    out = np.full(pos.shape[1:], np.nan, dtype=np.float32)
    fitted = range(pos.shape[1]) if layers is None else layers
    for l in fitted:
        v = np.asarray(est.fit(pos[:, l, :], neg[:, l, :]), dtype=np.float32)
        gap = pos[:, l, :].astype(np.float64).mean(axis=0) - neg[:, l, :].astype(np.float64).mean(axis=0)
        if float(gap @ v.astype(np.float64)) < 0:
            v = -v
            if flips is not None:
                flips.append(int(l))
        out[l] = v
    if erase is not None:
        out = erase.to_raw_directions(out)
    return out


# ------------------------------------------------------------------------------- erasing confounds
ERASE_METHODS = ("leace", "project")


def _one_hot(labels: Any) -> np.ndarray:
    z = np.asarray(labels)
    if z.ndim == 2:
        return z.astype(np.float64)
    values = np.unique(z)
    if len(values) < 2:
        raise ValueError("the confound labels have a single value; there is nothing to erase")
    return (z[:, None] == values[None, :]).astype(np.float64)[:, 1:]  # one column fewer than classes


class Eraser:
    """Erase what separates the groups of a confound from representations, layer by layer.

    Fit it with :meth:`fit` on reps ``(n, layers, dim)`` and each item's confound ``labels`` (a
    category per item, or a matrix with one column per numeric confound):

    - ``"leace"`` (least-squares concept erasure, Belrose et al. 2023): the smallest change, in the
      reps' own covariance, after which no linear classifier does better than chance at telling
      the confound on those reps;
    - ``"project"``: the orthogonal projection that removes the confound's directions (the
      difference of the groups' means, or of each group's mean from the overall mean).

    :meth:`apply` erases reps; :meth:`to_raw_directions` turns a direction fitted on erased reps into
    the direction that gives the same projections (up to a constant) on raw reps, which is what a
    probe stores. Both work in the space the fitting reps span: what lies outside it is left alone.
    """

    def __init__(self, method: str, layers: Dict[int, Dict[str, np.ndarray]], n_items: int):
        self.method, self._layers, self.n_items = method, layers, int(n_items)

    @classmethod
    def fit(cls, reps: np.ndarray, labels: Any, method: str = "leace", layers: Optional[Sequence[int]] = None,
            tol: float = 1e-8) -> "Eraser":
        if method not in ERASE_METHODS:
            raise ValueError(f"method must be one of {ERASE_METHODS}; got {method!r}")
        X = np.asarray(reps, dtype=np.float64)
        Z = _one_hot(labels)
        if len(Z) != len(X):
            raise ValueError(f"{len(Z)} labels for {len(X)} items")
        Zc = Z - Z.mean(axis=0)
        per: Dict[int, Dict[str, np.ndarray]] = {}
        for l in range(X.shape[1]) if layers is None else layers:
            x = X[:, l, :]
            mu = x.mean(axis=0)
            xc = x - mu
            if method == "project":
                # the confound's directions: each column's covariance with the reps
                c = xc.T @ Zc
                q, r = np.linalg.qr(c)
                keep = np.abs(np.diag(r)) > tol * max(1.0, float(np.abs(r).max()))
                per[l] = {"mu": mu, "left": q[:, keep], "right": q[:, keep]}
                continue
            # LEACE in the reps' own span: xc = U S Vt, whitening there is Vt / sigma
            _, s, vt = np.linalg.svd(xc, full_matrices=False)
            keep = s > tol * max(1.0, float(s.max()))
            vt, sigma = vt[keep], s[keep] / np.sqrt(len(x))
            cov_xz = (vt @ (xc.T @ Zc)) / len(x)  # the covariance with the confound, in that basis
            q, r = np.linalg.qr(cov_xz / sigma[:, None])
            q = q[:, np.abs(np.diag(r)) > tol * max(1.0, float(np.abs(r).max()))]
            # erased x = x - left @ right.T @ (x - mu)
            per[l] = {"mu": mu, "left": vt.T @ (sigma[:, None] * q), "right": vt.T @ (q / sigma[:, None])}
        return cls(method, per, len(X))

    @property
    def layers(self) -> List[int]:
        return sorted(self._layers)

    def apply(self, reps: np.ndarray) -> np.ndarray:
        """``reps`` ``(n, layers, dim)`` with the confound erased at the fitted layers."""
        out = np.array(reps, dtype=np.float64, copy=True)
        for l, e in self._layers.items():
            out[:, l, :] -= ((out[:, l, :] - e["mu"]) @ e["right"]) @ e["left"].T
        return out.astype(np.asarray(reps).dtype, copy=False)

    def to_raw_directions(self, directions: np.ndarray) -> np.ndarray:
        """Directions fitted on erased reps, as directions on raw reps: ``v - right @ left.T @ v``
        (normalized), so that ``raw @ result`` equals ``erased @ v`` up to a constant and a scale."""
        out = np.array(directions, dtype=np.float64, copy=True)
        for l, e in self._layers.items():
            if not np.all(np.isfinite(out[l])):
                continue
            v = out[l] - e["right"] @ (e["left"].T @ out[l])
            out[l] = v / (np.linalg.norm(v) + 1e-12)
        return out.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "layers": self.layers, "n_items": self.n_items,
                "dimensions_erased": {l: int(e["left"].shape[1]) for l, e in self._layers.items()}}


# ---------------------------------------------------------------------------- steering direction
def steer_directions_for(estimator: Any, steer: Any, pos: np.ndarray, neg: np.ndarray,
                         to_raw_directions: np.ndarray, erase: Any = None) -> Tuple[np.ndarray, Dict[str, Any], Optional[Choice]]:
    """The directions to steer with, their estimator's settings, and the choice to
    announce when they differ from the reading directions.

    ``steer`` is ``"read"`` (the reading directions, returned as the same array), or an estimator
    spec (default ``"mean_diff"``: the raw difference of means, which pushes further than
    variance-normalized directions do).
    """
    read_spec = estimator_spec(estimator)
    if steer in (None, "read"):
        return to_raw_directions, read_spec, None
    steer_spec = estimator_spec(steer)
    if steer_spec == read_spec:
        return to_raw_directions, steer_spec, None
    vectors = fit_directions(make_estimator(steer), pos, neg, erase=erase)
    return vectors, steer_spec, Choice(
        "steering direction", steer if isinstance(steer, (str, dict)) else steer_spec, steer_spec,
        f"steering uses its own direction, not the reading one ({read_spec}): the raw difference of means "
        "pushes further than normalized directions; pass steer_estimator='read' to steer with the reading direction",
    )
