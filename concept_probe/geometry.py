"""Beyond two poles (layers 2 and 3 of docs/DESIGN.md): several classes, a continuous variable, the
plane of the classes' centroids and a circle on it. Everything works on representations
``(items, layers, dim)`` (arrays, or what :func:`concept_probe.read_texts` returns), layer by layer.

- :func:`fit_classes`: one direction per class against the rest, with a cut per class, to tell
  several classes apart (:class:`ClassDirections`).
- :func:`fit_regression`: a ridge direction whose projection predicts a number per item
  (:class:`RegressionDirections`).
- :func:`centroid_subspace`: the few directions along which the classes' centroids spread.
- :func:`fit_circle`: for classes in a cyclic order (days, months), the plane of their centroids
  and the map that puts them on a circle, how well it fits, what shuffled orders give, and which
  class a new item falls nearest to (:class:`CircleFit`).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .directions import fit_directions


def _reps(x: Any) -> np.ndarray:
    data = getattr(x, "data", x)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"representations must be (items, layers, dim); got shape {arr.shape}")
    return arr


def _groups(groups: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out = {str(k): _reps(v) for k, v in groups.items()}
    if len(out) < 2:
        raise ValueError("needs at least two classes")
    shapes = {v.shape[1:] for v in out.values()}
    if len(shapes) != 1:
        raise ValueError(f"every class needs reps of the same (layers, dim); got {sorted(shapes)}")
    return out


# ------------------------------------------------------------------------------------- classes
@dataclass
class ClassDirections:
    """One direction per class (that class against all the others) and layer, and a cut per class:
    the midpoint between the class's mean projection and the others'. An item belongs to the class
    whose projection is furthest past its cut."""

    classes: List[str]
    directions: np.ndarray  # (classes, layers, dim)
    cuts: np.ndarray        # (classes, layers)

    def scores(self, reps: Any, layer: int) -> np.ndarray:
        """``(items, classes)``: each item's projection on each class's direction minus its cut."""
        X = _reps(reps)[:, layer, :]
        return X @ self.directions[:, layer, :].T.astype(np.float64) - self.cuts[:, layer]

    def predict(self, reps: Any, layer: int) -> List[str]:
        return [self.classes[k] for k in self.scores(reps, layer).argmax(axis=1)]

    def accuracy(self, groups: Dict[str, Any]) -> Dict[str, Any]:
        """On items of known class (``{class: reps}``, ideally not those it was fitted on): the
        accuracy per layer, the best layer, its confusion matrix (rows: true class) and chance."""
        g = _groups(groups)
        n_layers = self.directions.shape[1]
        acc = np.zeros(n_layers)
        total = sum(len(v) for v in g.values())
        for l in range(n_layers):
            acc[l] = sum(int((self.scores(v, l).argmax(axis=1) == self.classes.index(c)).sum())
                         for c, v in g.items()) / total
        best = int(np.argmax(acc))
        confusion = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for c, v in g.items():
            for k in self.scores(v, best).argmax(axis=1):
                confusion[self.classes.index(c), k] += 1
        return {"accuracy": acc, "best_layer": best, "best_accuracy": float(acc[best]), "confusion": confusion,
                "chance": float(max(len(v) for v in g.values()) / total)}


def fit_classes(groups: Dict[str, Any], estimator: Any = "mean_diff") -> ClassDirections:
    """A direction per class against the rest, at every layer, with any two-group estimator
    (see :mod:`concept_probe.directions`), from ``{class name: reps}``."""
    g = _groups(groups)
    names = list(g)
    dirs, cuts = [], []
    for c in names:
        rest = np.concatenate([g[o] for o in names if o != c])
        v = fit_directions(estimator, g[c], rest).astype(np.float64)
        dirs.append(v)
        cuts.append(0.5 * (np.einsum("nld,ld->l", g[c], v) / len(g[c]) + np.einsum("nld,ld->l", rest, v) / len(rest)))
    return ClassDirections(names, np.stack(dirs).astype(np.float32), np.stack(cuts))


def centroid_subspace(groups: Dict[str, Any], k: int = 2) -> Dict[str, np.ndarray]:
    """Per layer, the ``k`` directions along which the classes' centroids spread the most (PCA of
    the centroids): ``basis`` ``(layers, k, dim)``, the centroids' coordinates in it
    ``(layers, classes, k)``, and the share of their spread those ``k`` keep, per layer."""
    g = _groups(groups)
    C = np.stack([v.mean(axis=0) for v in g.values()])  # (classes, layers, dim)
    k = int(min(k, len(g) - 1))
    basis, coords, share = [], [], []
    for l in range(C.shape[1]):
        centered = C[:, l, :] - C[:, l, :].mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        basis.append(vt[:k])
        coords.append(centered @ vt[:k].T)
        share.append(float((s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-24)))
    return {"classes": list(g), "basis": np.stack(basis).astype(np.float32), "coords": np.stack(coords),
            "share": np.array(share)}


# ---------------------------------------------------------------------------------- regression
@dataclass
class RegressionDirections:
    """A ridge regression per layer: ``directions`` (unit, ``(layers, dim)``), and the slope and
    intercept that turn a projection on them into the predicted value."""

    directions: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, reps: Any) -> np.ndarray:
        """``(items, layers)``: the predicted value at every layer."""
        proj = np.einsum("nld,ld->nl", _reps(reps), self.directions.astype(np.float64))
        return proj * self.slope + self.intercept

    def evaluate(self, reps: Any, y: Sequence[float]) -> Dict[str, Any]:
        """Per layer on items with known values: Pearson's r and R² (1 - residual / total
        variance; it can be negative on new items), and the best layer by R²."""
        pred = self.predict(reps)
        y = np.asarray(y, dtype=np.float64)
        resid = ((pred - y[:, None]) ** 2).sum(axis=0)
        r2 = 1 - resid / max(((y - y.mean()) ** 2).sum(), 1e-24)
        r = np.array([np.corrcoef(pred[:, l], y)[0, 1] if np.std(pred[:, l]) > 0 else np.nan
                      for l in range(pred.shape[1])])
        best = int(np.nanargmax(r2))
        return {"r": r, "r2": r2, "best_layer": best, "best_r2": float(r2[best])}


def fit_regression(reps: Any, y: Sequence[float], alpha: float = 1.0) -> RegressionDirections:
    """A ridge regression of ``y`` on the reps at each layer. ``alpha`` is relative to the reps'
    mean squared norm, so the same value means the same regularization at every layer and model.
    Solved in the items' space, so it is cheap when there are fewer items than dimensions."""
    X = _reps(reps)
    y = np.asarray(y, dtype=np.float64)
    if len(y) != len(X):
        raise ValueError(f"{len(y)} values for {len(X)} items")
    yc = y - y.mean()
    dirs, slope, intercept = [], [], []
    for l in range(X.shape[1]):
        x = X[:, l, :]
        mu = x.mean(axis=0)
        xc = x - mu
        K = xc @ xc.T
        lam = float(alpha) * float(np.trace(K)) / len(x)
        w = xc.T @ np.linalg.solve(K + lam * np.eye(len(x)), yc)
        norm = float(np.linalg.norm(w)) or 1.0
        dirs.append(w / norm)
        slope.append(norm)
        intercept.append(float(y.mean() - mu @ w))
    return RegressionDirections(np.stack(dirs).astype(np.float32), np.array(slope), np.array(intercept), float(alpha))


# -------------------------------------------------------------------------------------- circles
def _circle_targets(n: int) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.c_[np.cos(a), np.sin(a)]


def _plane_and_map(C: np.ndarray):
    """The PCA-2 plane of centroids ``C`` (classes, dim) in their cyclic order, the affine map from
    it to the circle, and how much of the circle's positions the map explains (R²)."""
    n = len(C)
    mu = C.mean(axis=0)
    _, _, vt = np.linalg.svd(C - mu, full_matrices=False)
    A = np.c_[(C - mu) @ vt[:2].T, np.ones(n)]
    target = _circle_targets(n)
    W, *_ = np.linalg.lstsq(A, target, rcond=None)
    r2 = 1 - ((A @ W - target) ** 2).sum() / (target ** 2).sum()
    return mu, vt[:2], W, float(r2)


@dataclass
class CircleFit:
    """Classes in a cyclic order, put on a circle at each layer: the plane of their centroids
    (``mu``, ``plane``), the affine map from it to positions ``2 pi k / n`` (``maps``), how well
    the centroids fit (``r2``) and the 95th percentile of the same fit with the classes in shuffled
    orders (``r2_null_p95``): a circle the order does not explain."""

    classes: List[str]
    mu: np.ndarray
    plane: np.ndarray
    maps: np.ndarray
    r2: np.ndarray
    r2_null_p95: np.ndarray
    n_shuffles: int = 0

    def position(self, reps: Any, layer: int) -> np.ndarray:
        """Each item's angle on the circle, in radians from the first class."""
        X = _reps(reps)[:, layer, :]
        P = np.c_[(X - self.mu[layer]) @ self.plane[layer].T, np.ones(len(X))] @ self.maps[layer]
        return np.arctan2(P[:, 1], P[:, 0]) % (2 * np.pi)

    def predict(self, reps: Any, layer: int) -> List[str]:
        """The class whose position on the circle each item falls nearest to."""
        n = len(self.classes)
        idx = np.round(self.position(reps, layer) / (2 * np.pi / n)).astype(int) % n
        return [self.classes[k] for k in idx]

    def accuracy(self, groups: Dict[str, Any]) -> np.ndarray:
        """Per layer, the share of items (``{class: reps}``) that fall nearest their own class."""
        g = _groups(groups)
        total = sum(len(v) for v in g.values())
        return np.array([sum(sum(p == c for p in self.predict(v, l)) for c, v in g.items()) / total
                         for l in range(self.mu.shape[0])])


def fit_circle(groups: Dict[str, Any], n_shuffles: int = 200, seed: int = 0) -> CircleFit:
    """Fit a circle to classes given in their cyclic order (``{class: reps}``, e.g. Monday to
    Sunday), at every layer; ``n_shuffles`` shuffled orders give the null of the fit."""
    g = _groups(groups)
    if len(g) < 3:
        raise ValueError("a circle needs at least three classes")
    C = np.stack([v.mean(axis=0) for v in g.values()])  # (classes, layers, dim)
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(g)) for _ in range(int(n_shuffles))]
    mus, planes, maps, r2, null = [], [], [], [], []
    for l in range(C.shape[1]):
        mu, plane, W, fit = _plane_and_map(C[:, l, :])
        mus.append(mu)
        planes.append(plane)
        maps.append(W)
        r2.append(fit)
        shuffled = [_plane_and_map(C[o, l, :])[3] for o in orders]
        null.append(float(np.percentile(shuffled, 95)) if shuffled else np.nan)
    return CircleFit(list(g), np.stack(mus), np.stack(planes), np.stack(maps), np.array(r2), np.array(null),
                     int(n_shuffles))
