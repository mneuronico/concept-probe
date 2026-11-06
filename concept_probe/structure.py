"""Reading any target, and the shape of representations (beyond two poles).

A :class:`Target` says what each item is: a category (``Target.categories``), a number
(``Target.number``), a place in an order (``Target.order``) or in a cycle (``Target.cycle``, e.g.
days of the week), or any places you give (``Target.layout``, e.g. the nodes of a tree laid out in
a plane). Two questions,
each layer by layer, each with its control:

- :func:`read_target` — **can the target be read** from the representations? A map from the
  representations to the target is fitted on some items and tested on others. The map is the
  nearest class mean (``"centroid"``, the several-class form of the difference of means), a ridge
  map (``"linear"``, optionally of low ``rank``), the plane of the class centroids (``"shape"``),
  or any model with ``fit(X, Y)`` and ``predict(X)``, nonlinear ones included. Control: the same, fitted with the target shuffled
  (between items, or between classes: which class gets which place).
- :func:`target_shape` — **do the representations have the target's shape**? The classes'
  centroids are compared with the target's places: the correlation between their distances
  (representational similarity), and how well the centroids' own top plane (or line) matches the
  places up to rotation, scale and shift (Procrustes). Control: the same with the classes' places
  shuffled.

With many dimensions and few classes, almost any arrangement of the classes can be read; that the
classes *lie* on a circle, in order, is what :func:`target_shape` tests. The two-pole
:class:`~concept_probe.ConceptProbe` does not use this module and is not changed by it.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .settings import settings
from .utils import ensure_dir, json_dump, now_tag


def _reps(x: Any) -> np.ndarray:
    arr = np.asarray(getattr(x, "data", x), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"representations must be (items, layers, dim); got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------------------- targets
@dataclass
class Target:
    """What each item is. ``values`` are each item's coordinates ``(items, k)``. A target defined by
    classes also has each item's ``labels``, the ``classes`` in order and each class's ``places``
    ``(classes, k)``; ``kind`` is ``"categories"``, ``"number"``, ``"order"``, ``"cycle"`` or
    ``"layout"``."""

    kind: str
    values: np.ndarray
    labels: Optional[np.ndarray] = None
    classes: Optional[List[Any]] = None
    places: Optional[np.ndarray] = None

    @property
    def by_class(self) -> bool:
        return self.classes is not None

    @property
    def structured(self) -> bool:
        """Whether the classes' places say something beyond which class is which (an order, a cycle,
        a layout), so that their shape can be tested."""
        return self.kind in ("order", "cycle", "layout")

    def subset(self, idx: np.ndarray) -> "Target":
        return Target(self.kind, self.values[idx], None if self.labels is None else self.labels[idx],
                      self.classes, self.places)

    def with_places(self, places: np.ndarray) -> "Target":
        """The same items, with the classes moved to ``places`` (a shuffled control)."""
        index = {c: i for i, c in enumerate(self.classes)}
        values = places[[index[l] for l in self.labels]]
        return Target(self.kind, values, self.labels, self.classes, places)

    # ---- building targets
    @classmethod
    def categories(cls, labels: Sequence[Any], classes: Optional[Sequence[Any]] = None) -> "Target":
        """Unordered classes: each class is a corner of a simplex (one-hot), all equally far apart."""
        classes = list(dict.fromkeys(labels)) if classes is None else list(classes)
        return cls._of_classes("categories", labels, classes, np.eye(len(classes)))

    @classmethod
    def number(cls, values: Sequence[float]) -> "Target":
        """A number per item."""
        return cls("number", np.asarray(values, dtype=np.float64).reshape(-1, 1))

    @classmethod
    def order(cls, labels: Sequence[Any], classes: Sequence[Any]) -> "Target":
        """Classes in an order (``classes`` from first to last): places 0, 1, 2, ... on a line."""
        return cls._of_classes("order", labels, classes, np.arange(len(classes), dtype=np.float64)[:, None])

    @classmethod
    def cycle(cls, labels: Sequence[Any], classes: Sequence[Any]) -> "Target":
        """Classes in a cyclic order (``classes`` around once, e.g. Monday to Sunday): evenly spaced
        places on a circle."""
        a = 2 * np.pi * np.arange(len(classes)) / len(classes)
        return cls._of_classes("cycle", labels, classes, np.c_[np.cos(a), np.sin(a)])

    @classmethod
    def layout(cls, labels: Sequence[Any], places: Dict[Any, Sequence[float]]) -> "Target":
        """Classes at places you give (``{class: coordinates}``): a grid, a tree drawn in a plane..."""
        classes = list(places)
        return cls._of_classes("layout", labels, classes, np.array([places[c] for c in classes], dtype=np.float64))

    @classmethod
    def _of_classes(cls, kind: str, labels: Sequence[Any], classes: Sequence[Any], places: np.ndarray) -> "Target":
        labels = np.asarray(list(labels), dtype=object)
        classes = list(classes)
        missing = sorted({str(l) for l in labels} - {str(c) for c in classes})
        if missing:
            raise ValueError(f"labels not among the classes: {missing[:5]}")
        index = {c: i for i, c in enumerate(classes)}
        places = np.asarray(places, dtype=np.float64)
        return cls(kind, places[[index[l] for l in labels]], labels, classes, places)



# ----------------------------------------------------------------------------------------- maps
class _Map:
    """A fitted map from representations at one layer to a target's coordinates."""

    kind = ""

    def predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> Dict[str, np.ndarray]:
        return {}


class _LinearMap(_Map):
    """A ridge map (optionally of low rank): ``(Z - mu) @ W + ymu``."""

    kind = "linear"

    def __init__(self, mu: np.ndarray, W: np.ndarray, ymu: np.ndarray):
        self.mu, self.W, self.ymu = mu, W, ymu

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=np.float64) - self.mu) @ self.W + self.ymu

    def params(self) -> Dict[str, np.ndarray]:
        return {"mu": self.mu, "W": self.W, "ymu": self.ymu}


class _ShapeMap(_Map):
    """Through the shape: the coordinates in the class centroids' own top directions ``V`` (about
    their mean ``mu``), mapped to the places by an affine ``W``."""

    kind = "shape"

    def __init__(self, mu: np.ndarray, V: np.ndarray, W: np.ndarray):
        self.mu, self.V, self.W = mu, V, W

    def plane(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=np.float64) - self.mu) @ self.V.T

    def predict(self, Z: np.ndarray) -> np.ndarray:
        c = self.plane(Z)
        return np.c_[c, np.ones(len(c))] @ self.W

    def params(self) -> Dict[str, np.ndarray]:
        return {"mu": self.mu, "V": self.V, "W": self.W}


class _CentroidMap(_Map):
    """The place of the nearest class centroid."""

    kind = "centroid"

    def __init__(self, centroids: np.ndarray, places: np.ndarray):
        self.centroids, self.places = centroids, places

    def predict(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        return self.places[np.argmin(((Z[:, None, :] - self.centroids[None]) ** 2).sum(-1), axis=1)]

    def params(self) -> Dict[str, np.ndarray]:
        return {"centroids": self.centroids, "places": self.places}


class _ModelMap(_Map):
    """Any model with ``fit``/``predict``."""

    kind = "model"

    def __init__(self, model: Any):
        self.model = model

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(Z, dtype=np.float64)), dtype=np.float64).reshape(len(Z), -1)


def _map_from_params(kind: str, params: Dict[str, np.ndarray]) -> _Map:
    return {"linear": lambda p: _LinearMap(p["mu"], p["W"], p["ymu"]),
            "shape": lambda p: _ShapeMap(p["mu"], p["V"], p["W"]),
            "centroid": lambda p: _CentroidMap(p["centroids"], p["places"])}[kind](params)


def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float, rank: Optional[int]) -> _LinearMap:
    """A ridge map ``X -> Y`` (alpha relative to the reps' mean squared norm), solved in the items'
    space, optionally restricted to the ``rank`` directions its fitted values use the most."""
    mu, ymu = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mu, Y - ymu
    K = Xc @ Xc.T
    lam = float(alpha) * float(np.trace(K)) / len(X)
    W = Xc.T @ np.linalg.solve(K + lam * np.eye(len(X)), Yc)  # (dim, k)
    if rank is not None and rank < W.shape[1]:
        _, _, vt = np.linalg.svd(Xc @ W, full_matrices=False)
        W = W @ vt[:rank].T @ vt[:rank]
    return _LinearMap(mu, W, ymu)


def _fit_map(model: Any, X: np.ndarray, target: Target, alpha: float, rank: Optional[int]) -> _Map:
    X = np.asarray(X, dtype=np.float64)
    if model == "shape":
        if not target.structured:
            raise ValueError("the 'shape' map needs an order, a cycle or a layout")
        cents = np.stack([X[target.labels == c].mean(axis=0) for c in target.classes])
        mu = cents.mean(axis=0)
        _, _, vt = np.linalg.svd(cents - mu, full_matrices=False)
        V = vt[:target.places.shape[1]]
        A = np.c_[(cents - mu) @ V.T, np.ones(len(cents))]
        W, *_ = np.linalg.lstsq(A, target.places, rcond=None)
        return _ShapeMap(mu, V, W)
    if model == "centroid":
        if not target.by_class:
            raise ValueError("the 'centroid' map needs a target defined by classes; use 'linear' for numbers")
        present = [c for c in target.classes if np.any(target.labels == c)]
        cents = np.stack([X[target.labels == c].mean(axis=0) for c in present])
        return _CentroidMap(cents, target.places[[target.classes.index(c) for c in present]])
    if model == "linear":
        return _ridge(X, target.values, alpha, rank)
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise ValueError("model must be 'centroid', 'linear', 'shape' or an object with fit(X, Y) and predict(X)")
    from copy import deepcopy

    m = deepcopy(model)
    m.fit(X, target.values if target.values.shape[1] > 1 else target.values[:, 0])
    return _ModelMap(m)


def _score(pred: np.ndarray, target: Target) -> float:
    """Accuracy (nearest place is the item's own class) for class targets; R² for numbers."""
    if target.by_class:
        nearest = np.argmin(((pred[:, None, :] - target.places[None]) ** 2).sum(-1), axis=1)
        own = np.array([target.classes.index(l) for l in target.labels])
        return float((nearest == own).mean())
    y = target.values
    return float(1 - ((pred - y) ** 2).sum() / max(((y - y.mean(axis=0)) ** 2).sum(), 1e-24))


def _rank_corr(pred: np.ndarray, target: Target) -> float:
    """Spearman's rho between a number and its prediction (the order, whatever the scale)."""
    return _spearman(pred[:, 0], target.values[:, 0])


def _split(target: Target, frac: float, rng: np.random.Generator):
    """Items to fit and to test on, splitting each class (or all items) at ``frac``."""
    n = len(target.values)
    if not target.by_class:
        perm = rng.permutation(n)
        k = int(round(n * frac))
        return perm[:k], perm[k:]
    fit, test = [], []
    for c in target.classes:
        idx = rng.permutation(np.flatnonzero(target.labels == c))
        k = int(round(len(idx) * frac))
        fit.extend(idx[:k])
        test.extend(idx[k:])
    return np.array(fit, dtype=int), np.array(test, dtype=int)


@dataclass
class TargetReading:
    """What :func:`read_target` measured, per layer: ``score`` (accuracy for class targets, R² for
    numbers) on the test items, and ``null_p95`` (the 95th percentile of the same with the target
    shuffled, as ``shuffle`` says); the best layer and its numbers; ``chance`` for class targets."""

    target_kind: str
    model: str
    score: np.ndarray
    null_p95: np.ndarray
    best_layer: int
    best_score: float
    shuffle: str
    n_fit: int
    n_test: int
    chance: Optional[float] = None
    rank_correlation: Optional[np.ndarray] = None
    maps: Dict[int, Any] = field(default_factory=dict, repr=False)

    def predict(self, reps: Any, layer: Optional[int] = None) -> np.ndarray:
        """Each item's predicted coordinates at ``layer`` (the best by default)."""
        layer = self.best_layer if layer is None else int(layer)
        return self.maps[layer].predict(_reps(reps)[:, layer, :])

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target_kind, "model": self.model, "score": self.score.tolist(),
                "null_p95": self.null_p95.tolist(), "best_layer": self.best_layer, "best_score": self.best_score,
                "shuffle": self.shuffle, "n_fit": self.n_fit, "n_test": self.n_test, "chance": self.chance,
                "rank_correlation": None if self.rank_correlation is None else self.rank_correlation.tolist()}


def read_target(
    reps: Any,
    target: Target,
    *,
    model: Any = "linear",
    test: Optional[Any] = None,
    split: float = 0.5,
    rank: Optional[int] = None,
    alpha: float = 1.0,
    shuffle: Optional[str] = None,
    n_shuffles: int = 20,
    layers: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> TargetReading:
    """Whether ``target`` can be read from ``reps`` ``(items, layers, dim)``, at each layer.

    The map (``model``: ``"linear"``, ``"centroid"``, ``"shape"`` or an object with
    ``fit``/``predict``) is
    fitted on ``split`` of each class's items and tested on the rest, or fitted on all of ``reps``
    and tested on ``test = (reps, target)``. A class target is read as the class whose place is
    nearest to the prediction (accuracy); a number, by R² and by its rank correlation (the order,
    whatever the scale). ``"shape"`` reads through the shape: the plane (or line) of the fitting
    items' class centroids and the map from it to the places, so only the classes' arrangement can
    help. ``rank`` limits a linear map to that many
    directions (2: "the target lives in a plane"); ``alpha`` is its ridge, relative to the reps'
    scale.

    The control fits the same map ``n_shuffles`` times with the target shuffled: between items
    (``"items"``, the default for categories and numbers: is there anything to read?) or between
    classes (``"classes"``, the default for orders, cycles and layouts: does the arrangement
    matter, or would any assignment of places to classes be read as well?).
    """
    X = _reps(reps)
    rng = np.random.default_rng(seed)
    if test is None:
        fit_idx, test_idx = _split(target, split, rng)
        Xf, Xt, tf, tt = X[fit_idx], X[test_idx], target.subset(fit_idx), target.subset(test_idx)
    else:
        Xf, tf = X, target
        Xt, tt = _reps(test[0]), test[1]
    if len(Xt) == 0 or len(Xf) == 0:
        raise ValueError("no items to fit or to test on")
    shuffle = shuffle or ("classes" if target.structured else "items")
    if shuffle not in ("items", "classes"):
        raise ValueError(f"shuffle must be 'items' or 'classes'; got {shuffle!r}")
    if shuffle == "classes" and not target.by_class:
        raise ValueError("shuffling between classes needs a target defined by classes")
    perms = [rng.permutation(len(tf.values) if shuffle == "items" else len(target.classes)) for _ in range(n_shuffles)]
    chosen = list(range(X.shape[1])) if layers is None else [int(l) for l in layers]
    score = np.full(X.shape[1], np.nan)
    null = np.full(X.shape[1], np.nan)
    rho = None if target.by_class else np.full(X.shape[1], np.nan)
    maps: Dict[int, Any] = {}
    for l in chosen:
        f = _fit_map(model, Xf[:, l, :], tf, alpha, rank)
        maps[l] = f
        pred = f.predict(Xt[:, l, :])
        score[l] = _score(pred, tt)
        if rho is not None:
            rho[l] = _rank_corr(pred, tt)
        shuffled = []
        for p in perms:
            if shuffle == "items":
                ts = Target(tf.kind, tf.values[p], None if tf.labels is None else tf.labels[p], tf.classes, tf.places)
                shuffled.append(_score(_fit_map(model, Xf[:, l, :], ts, alpha, rank).predict(Xt[:, l, :]), tt))
            else:
                moved = target.places[p]
                fs = _fit_map(model, Xf[:, l, :], tf.with_places(moved), alpha, rank)
                shuffled.append(_score(fs.predict(Xt[:, l, :]), tt.with_places(moved)))
        null[l] = float(np.percentile(shuffled, 95)) if shuffled else np.nan
    best = int(np.nanargmax(score))
    chance = None
    if target.by_class:
        counts = [int(np.sum(tt.labels == c)) for c in target.classes]
        chance = float(max(counts) / max(1, sum(counts)))
    name = model if isinstance(model, str) else type(model).__name__
    return TargetReading(target.kind, name, score, null, best, float(score[best]), shuffle, len(Xf), len(Xt),
                         chance, rho, maps)


# ----------------------------------------------------------------------------------------- shape
def _pdist(P: np.ndarray) -> np.ndarray:
    d = np.sqrt(((P[:, None, :] - P[None]) ** 2).sum(-1))
    return d[np.triu_indices(len(P), 1)]


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # ties get their mean rank
        hit = x == v
        ranks[hit] = ranks[hit].mean()
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _shape_fit(coords: np.ndarray, places: np.ndarray, fit: str) -> float:
    """R² of the places predicted from the centroids' coordinates: by a similarity transform
    (rotation or reflection, one scale, a shift) or by any affine map."""
    P = places - places.mean(axis=0)
    C = coords - coords.mean(axis=0)
    total = max((P ** 2).sum(), 1e-24)
    if fit == "affine":
        A = np.c_[C, np.ones(len(C))]
        W, *_ = np.linalg.lstsq(A, P, rcond=None)
        return float(1 - ((A @ W - P) ** 2).sum() / total)
    u, s, vt = np.linalg.svd(C.T @ P, full_matrices=False)
    R = u @ vt
    scale = s.sum() / max((C ** 2).sum(), 1e-24)
    return float(1 - ((scale * C @ R - P) ** 2).sum() / total)


@dataclass
class TargetShape:
    """What :func:`target_shape` measured, per layer: ``similarity`` (Spearman correlation between
    the centroids' distances and the places' distances), ``fit`` (R² of the places from the
    centroids' own top ``dims`` directions, by ``fit_kind``), each with the 95th percentile of the
    same with the classes' places shuffled, and ``share`` (how much of the centroids' spread those
    directions keep). ``coords`` are the centroids in those directions, to draw them."""

    target_kind: str
    classes: List[Any]
    dims: int
    fit_kind: str
    similarity: np.ndarray
    similarity_null_p95: np.ndarray
    fit: np.ndarray
    fit_null_p95: np.ndarray
    share: np.ndarray
    coords: np.ndarray
    n_shuffles: int

    @property
    def best_layer(self) -> int:
        return int(np.nanargmax(self.fit))

    def plot(self, path: str, layer: Optional[int] = None, title: Optional[str] = None) -> bool:
        """Draw the centroids at ``layer`` (the best by default), joined in the target's order."""
        from .visuals import plot_shape

        return plot_shape(self, path, layer, title=title)

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target_kind, "classes": [str(c) for c in self.classes], "dims": self.dims,
                "fit_kind": self.fit_kind, "similarity": self.similarity.tolist(),
                "similarity_null_p95": self.similarity_null_p95.tolist(), "fit": self.fit.tolist(),
                "fit_null_p95": self.fit_null_p95.tolist(), "share": self.share.tolist(),
                "best_layer": self.best_layer, "n_shuffles": self.n_shuffles}


def target_shape(
    reps: Any,
    target: Target,
    *,
    dims: Optional[int] = None,
    fit: str = "similarity",
    n_shuffles: int = 200,
    layers: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> TargetShape:
    """Whether the classes' centroids have the shape of ``target``'s places, at each layer.

    ``dims`` (by default the places' own: 1 for an order, 2 for a cycle) is how many of the
    centroids' principal directions are compared with the places; ``fit`` is ``"similarity"``
    (rotation or reflection, one scale and a shift: the shape itself) or ``"affine"`` (any linear
    map: an ellipse counts as a circle). The control shuffles which class gets which place,
    ``n_shuffles`` times. Needs an order, a cycle or a layout with at least three classes.
    """
    if not target.structured:
        raise ValueError("a shape needs a target whose places mean something: order, cycle or layout "
                         "(categories are all equally far apart)")
    if len(target.classes) < 3:
        raise ValueError("a shape needs at least three classes")
    if fit not in ("similarity", "affine"):
        raise ValueError(f"fit must be 'similarity' or 'affine'; got {fit!r}")
    X = _reps(reps)
    present = [c for c in target.classes if np.any(target.labels == c)]
    if len(present) != len(target.classes):
        raise ValueError("every class needs at least one item")
    dims = int(target.places.shape[1] if dims is None else dims)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(target.classes)) for _ in range(int(n_shuffles))]
    place_d = _pdist(target.places)
    n_layers = X.shape[1]
    chosen = list(range(n_layers)) if layers is None else [int(l) for l in layers]
    sim, sim_null, fitv, fit_null, share = (np.full(n_layers, np.nan) for _ in range(5))
    coords = np.full((n_layers, len(target.classes), dims), np.nan)
    for l in chosen:
        C = np.stack([X[target.labels == c, l, :].mean(axis=0) for c in target.classes])
        Cc = C - C.mean(axis=0)
        _, s, vt = np.linalg.svd(Cc, full_matrices=False)
        k = min(dims, len(s))
        top = Cc @ vt[:k].T
        coords[l, :, :k] = top
        share[l] = float((s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-24))
        cent_d = _pdist(C)
        sim[l] = _spearman(cent_d, place_d)
        fitv[l] = _shape_fit(top, target.places, fit)
        sim_null[l] = float(np.percentile([_spearman(cent_d, _pdist(target.places[p])) for p in perms], 95)) \
            if perms else np.nan
        fit_null[l] = float(np.percentile([_shape_fit(top, target.places[p], fit) for p in perms], 95)) \
            if perms else np.nan
    return TargetShape(target.kind, list(target.classes), dims, fit, sim, sim_null, fitv, fit_null, share, coords,
                       int(n_shuffles))


# ------------------------------------------------------------------------------------ the probe
@dataclass
class TargetScores:
    """What a :class:`TargetProbe` says about each activation.

    ``coordinates`` are where the map puts it ``(items, k)``. ``position`` is the score along the
    target: for a cycle, the place on the circle in class steps from the first class (``2.98`` is
    just before the fourth class), continuous and in ``[0, n)``; for an order, the place on the line
    in class steps; for a number, the value. ``nearest`` is the nearest class. ``fit`` says how well
    an item sits on the shape: for a cycle, its distance from the center (1 is on the circle).
    ``class_scores`` are the map's outputs per class, for categories.
    """

    kind: str
    coordinates: np.ndarray
    position: Optional[np.ndarray] = None
    nearest: Optional[List[Any]] = None
    fit: Optional[np.ndarray] = None
    class_scores: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.coordinates)

    def to_dict(self) -> Dict[str, Any]:
        def plain(x):
            return None if x is None else np.asarray(x).tolist()

        return {"kind": self.kind, "coordinates": plain(self.coordinates), "position": plain(self.position),
                "nearest": None if self.nearest is None else [str(c) for c in self.nearest], "fit": plain(self.fit),
                "class_scores": plain(self.class_scores)}


class TargetProbe:
    """A probe toward a target (beyond two poles): fitted once, it scores any activation.

    Build it with :meth:`fit` on representations and a :class:`Target`. It keeps the map at one
    layer (``"shape"`` by default for orders, cycles and layouts, ``"centroid"`` for categories,
    ``"linear"`` for numbers, or any of :func:`read_target`'s), how well it read held-out items
    (``reading``) and, for a structured target, how well the centroids have its shape (``shape``).

    - :meth:`score` gives each activation its :class:`TargetScores` (a place on the cycle, a value,
      the nearest class, how well it fits);
    - :meth:`read` and :meth:`score_tokens` read texts with the attached model and score them, whole
      or token by token;
    - :meth:`steering` moves activations along the target's shape: for a cycle, a rotation in the
      centroids' plane about the cycle's center (one step is one class); for an order or a number, an
      addition along the map's direction. :meth:`toward` moves them from one class to another by the
      whole difference between the two classes' centroids, in every dimension. They need not do the
      same: in Llama 3.2 3B and Qwen 2.5 0.5B, rotating "Monday" within the days' circle did not
      change what the model said the next day was, and the whole step between centroids did (in
      Llama, at layers 4 to 12). Both work at any layer, not only the probe's;
    - :meth:`save` and :meth:`load`; saved by default when fitted.

    The two-pole :class:`~concept_probe.ConceptProbe` is a separate object and is not changed by it.
    """

    def __init__(self, kind: str, classes: Optional[List[Any]], places: Optional[np.ndarray], layer: int,
                 mapping: _Map, *, name: str = "target", readout: Optional[str] = None,
                 identity: Optional[Dict[str, Any]] = None, reading: Optional[Dict[str, Any]] = None,
                 shape: Optional[Dict[str, Any]] = None):
        self.kind, self.classes, self.places, self.layer, self.mapping = kind, classes, places, int(layer), mapping
        self.name, self.readout, self.identity = name, readout, identity
        self.reading, self.shape = reading, shape
        #: each class's mean representation at every layer ``(classes, layers, dim)``, for class targets
        self.centroids: Optional[np.ndarray] = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.dir: Optional[str] = None

    # ---- building
    @classmethod
    def fit(
        cls,
        reps: Any,
        target: Target,
        *,
        model: Any = None,
        layer: Optional[int] = None,
        test: Optional[Any] = None,
        split: float = 0.5,
        rank: Optional[int] = None,
        alpha: float = 1.0,
        n_shuffles: int = 20,
        seed: int = 0,
        name: str = "target",
        identity: Optional[Dict[str, Any]] = None,
        save: Optional[bool] = None,
        root_dir: Optional[str] = None,
    ) -> "TargetProbe":
        """Fit a probe toward ``target`` on ``reps`` ``(items, layers, dim)``.

        The layer is ``layer``, or the one that reads ``target`` best on held-out items
        (:func:`read_target`, on ``split`` of each class's items, or on ``test = (reps, target)``);
        among layers that read equally well, the one whose centroids fit the target's shape best (for
        orders, cycles and layouts) or that is furthest above its shuffled control. The map is then
        refitted on every item of ``reps`` at that layer. ``reading`` keeps the
        held-out numbers with their shuffled control, and ``shape`` the centroids' fit to a structured
        target at that layer. Saved by default to ``<root_dir or output_root>/<name>/<time>``.
        """
        X = _reps(reps)
        if model is None:
            model = "shape" if target.structured else ("centroid" if target.by_class else "linear")
        reading = read_target(reps, target, model=model, test=test, split=split, rank=rank, alpha=alpha,
                              n_shuffles=n_shuffles, layers=None if layer is None else [layer], seed=seed)
        ts = None
        if target.structured and len(target.classes) >= 3:
            ts = target_shape(reps, target, layers=None if layer is None else [layer], n_shuffles=200, seed=seed)
        if layer is None:
            best = np.nanmax(reading.score)
            ties = np.flatnonzero(reading.score >= best - 1e-9)
            tie_break = ts.fit[ties] if ts is not None else (reading.score - np.nan_to_num(reading.null_p95))[ties]
            layer = int(ties[int(np.nanargmax(tie_break))])
        mapping = _fit_map(model, X[:, layer, :], target, alpha, rank)
        shape = None
        if ts is not None:
            shape = {"layer": layer, "fit": float(ts.fit[layer]), "fit_null_p95": float(ts.fit_null_p95[layer]),
                     "similarity": float(ts.similarity[layer]),
                     "similarity_null_p95": float(ts.similarity_null_p95[layer]), "share": float(ts.share[layer])}
        summary = {"layer": layer, "score": float(reading.score[layer]), "null_p95": float(reading.null_p95[layer]),
                   "metric": "accuracy" if target.by_class else "R2", "shuffle": reading.shuffle,
                   "chance": reading.chance, "n_fit": reading.n_fit, "n_test": reading.n_test, "model": reading.model}
        readout = getattr(getattr(reps, "readout", None), "name", None)
        probe = cls(target.kind, target.classes, target.places, layer, mapping, name=name, readout=readout,
                    identity=identity or getattr(reps, "identity", None), reading=summary, shape=shape)
        if target.by_class:
            probe.centroids = np.stack([X[target.labels == c].mean(axis=0) for c in target.classes]).astype(np.float32)
        if settings().save if save is None else save:
            probe.save(os.path.join(root_dir or settings().output_root, name, now_tag()))
        return probe

    # ---- scoring
    def _at_layer(self, reps: Any) -> np.ndarray:
        arr = np.asarray(getattr(reps, "data", reps), dtype=np.float64)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[:, self.layer, :] if arr.shape[1] > self.layer else arr[:, 0, :]
        raise ValueError(f"representations must be (items, layers, dim) or (items, dim); got {arr.shape}")

    def coordinates(self, reps: Any) -> np.ndarray:
        """Where the map puts each activation ``(items, k)``: representations ``(items, layers, dim)``
        (read at the probe's layer) or ``(items, dim)`` (already at it)."""
        return self.mapping.predict(self._at_layer(reps))

    def score(self, reps: Any) -> TargetScores:
        """Each activation's :class:`TargetScores`."""
        C = self.coordinates(reps)
        out = TargetScores(self.kind, C)
        if self.classes is not None:
            nearest = np.argmin(((C[:, None, :] - self.places[None]) ** 2).sum(-1), axis=1)
            out.nearest = [self.classes[k] for k in nearest]
        if self.kind == "cycle":
            n = len(self.classes)
            out.position = (np.arctan2(C[:, 1], C[:, 0]) % (2 * np.pi)) / (2 * np.pi / n)
            out.nearest = [self.classes[int(k)] for k in np.round(out.position).astype(int) % n]
            out.fit = np.linalg.norm(C, axis=1)
        elif self.kind in ("order", "number"):
            out.position = C[:, 0]
        elif self.kind == "categories":
            out.class_scores = C
        return out

    # ---- reading texts
    def attach(self, model: Any, tokenizer: Any, *, allow_model_mismatch: bool = False) -> "TargetProbe":
        """Give the probe a loaded model to read texts with; one of another identity raises unless
        ``allow_model_mismatch``."""
        from .modeling import identity_mismatches, identity_of

        if self.identity and not allow_model_mismatch:
            mismatched = identity_mismatches(identity_of(model, tokenizer), self.identity)
            if mismatched:
                raise ValueError(f"the model is not the one the probe was fitted on ({'; '.join(mismatched)}); "
                                 "pass allow_model_mismatch=True to use it anyway")
        self.model, self.tokenizer = model, tokenizer
        return self

    def _read(self, texts: Sequence[Any], readout: Any, role: str, system: Optional[str], prompt: Optional[str],
              batch_size: int) -> Any:
        from .reading import read_texts

        if self.model is None:
            raise ValueError("reading texts needs the model: probe.attach(model, tokenizer)")
        return read_texts(self.model, self.tokenizer, list(texts), readout, role=role, system=system, prompt=prompt,
                          layers=[self.layer], batch_size=batch_size, console=False)

    def read(self, texts: Sequence[Any], *, role: str = "raw", system: Optional[str] = None,
             prompt: Optional[str] = None, batch_size: int = 16) -> TargetScores:
        """Read each text (as ``role``, ``system`` and ``prompt`` say, pooled as the probe's training
        representations were, by default the mean over the text's tokens) and score it."""
        return self.score(self._read(texts, self.readout or "assistant_all_mean", role, system, prompt,
                                     batch_size).data[:, 0, :])

    def score_tokens(self, texts: Sequence[Any], *, role: str = "raw", system: Optional[str] = None,
                     prompt: Optional[str] = None, batch_size: int = 16) -> List[TargetScores]:
        """Each text's tokens scored one by one (the text's own tokens): one :class:`TargetScores`
        per text, with a row per token."""
        reps = self._read(texts, "assistant_tokens", role, system, prompt, batch_size)
        return [self.score(reps.data[reps.items == i, 0, :]) for i in range(len(texts))]

    # ---- steering
    def _shape_map(self, layer: int) -> "_ShapeMap":
        if layer == self.layer and isinstance(self.mapping, _ShapeMap):
            return self.mapping
        if self.centroids is None:
            raise ValueError("steering at another layer needs the class centroids, which this probe does not have")
        C = self.centroids[:, layer, :].astype(np.float64)
        mu = C.mean(axis=0)
        _, _, vt = np.linalg.svd(C - mu, full_matrices=False)
        V = vt[:self.places.shape[1]]
        W, *_ = np.linalg.lstsq(np.c_[(C - mu) @ V.T, np.ones(len(C))], self.places, rcond=None)
        return _ShapeMap(mu, V, W)

    def steering(self, steps: float, *, layer: Optional[int] = None, positions: Any = None) -> Any:
        """The intervention that moves activations ``steps`` along the target's shape, at ``layer``
        (the probe's by default; any layer for cycles and orders, from the class centroids there).

        For a cycle: a rotation by ``steps`` classes (``2 pi steps / n``) in the plane of the class
        centroids, about the cycle's center, in the direction of the classes' order; exact when the
        centroids have the cycle's shape, approximate otherwise. For an order: an addition along the
        centroids' line, ``steps`` classes long. For an order or a number with a ``"linear"`` map (at
        the probe's layer): the smallest addition that moves the predicted value by ``steps``. Moving
        within the shape need not change what the model does; :meth:`toward` moves by the whole
        difference between classes. Use either with :func:`concept_probe.intervene`, ``generate`` or
        ``evaluate_steering``.
        """
        from .interventions import Add, Rotate

        layer = self.layer if layer is None else int(layer)
        if self.kind in ("cycle", "order") and (isinstance(self.mapping, _ShapeMap) or self.centroids is not None):
            m = self._shape_map(layer)
            if self.kind == "cycle":
                A, b = m.W[:2], m.W[2]                      # places = plane @ A + b
                center = m.mu + np.linalg.solve(A.T, -b) @ m.V  # the point that maps to the circle's center
                sign = 1.0 if np.linalg.det(A) > 0 else -1.0  # the map may mirror the plane
                angle = sign * float(steps) * 2 * np.pi / len(self.classes)
                return Rotate(m.V[0], m.V[1], angle, layers=[layer], center=center, positions=positions,
                              label=f"{self.name}: {float(steps):g} steps around the cycle")
            return Add({layer: m.V[0] / float(m.W[0, 0])}, alpha=float(steps), positions=positions,
                       label=f"{self.name}: {float(steps):g} steps along the order")
        if self.kind in ("order", "number") and isinstance(self.mapping, _LinearMap):
            if layer != self.layer:
                raise ValueError("a linear map exists only at the probe's layer; fit another probe for another layer")
            w = self.mapping.W[:, 0]
            return Add({layer: w / float(w @ w)}, alpha=float(steps), positions=positions,
                       label=f"{self.name}: {float(steps):g} along the value")
        raise ValueError(f"steering along a {self.kind} target with a {self.mapping.kind!r} map is not defined; it is "
                         "for cycles and orders, and for numbers with a linear map. To move from one class to "
                         "another, use toward().")

    def toward(self, to: Any = None, *, source: Any, steps: Optional[int] = None, scale: float = 1.0,
               layer: Optional[int] = None, positions: Any = None) -> Any:
        """The intervention that moves activations from class ``source`` to class ``to`` (or, in a cycle
        or an order, ``steps`` classes on from ``source``): the difference between their centroids
        (times ``scale``) added at ``layer`` (the probe's by default), in every dimension. Any target
        with classes."""
        from .interventions import Add

        if self.centroids is None:
            raise ValueError("toward() needs a target with classes")
        if (to is None) == (steps is None):
            raise ValueError("give either to= (a class) or steps= (classes on from source)")
        if source not in self.classes:
            raise ValueError(f"{source!r} is not one of the classes {self.classes}")
        if steps is not None:
            if self.kind not in ("cycle", "order"):
                raise ValueError("steps= needs a cycle or an order; give to= (a class)")
            k = self.classes.index(source) + int(steps)
            if self.kind == "order" and not 0 <= k < len(self.classes):
                raise ValueError(f"{steps} steps from {source!r} is outside the order")
            to = self.classes[k % len(self.classes)]
        for c in (to, source):
            if c not in self.classes:
                raise ValueError(f"{c!r} is not one of the classes {self.classes}")
        layer = self.layer if layer is None else int(layer)
        step = (self.centroids[self.classes.index(to), layer] - self.centroids[self.classes.index(source), layer])
        return Add({layer: step.astype(np.float32)}, alpha=float(scale), positions=positions,
                   label=f"{self.name}: from {source} to {to}")

    # ---- saving
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "classes": None if self.classes is None else
                [str(c) for c in self.classes], "layer": self.layer, "map": self.mapping.kind, "readout": self.readout,
                "identity": self.identity, "reading": self.reading, "shape": self.shape}

    def save(self, path: str) -> str:
        """``target_probe.json`` (what it is, its numbers) and ``target_probe.npz`` (the map) in
        ``path``; a map made by another library's model is pickled to ``model.pkl``."""
        import pickle

        ensure_dir(path)
        meta = self.describe()
        meta["classes_raw"] = None if self.classes is None else list(self.classes)
        json_dump(os.path.join(path, "target_probe.json"), meta)
        arrays = {"places": np.zeros((0, 0)) if self.places is None else self.places,
                  "centroids": np.zeros((0, 0, 0)) if self.centroids is None else self.centroids}
        arrays.update({f"map_{k}": v for k, v in self.mapping.params().items()})
        np.savez_compressed(os.path.join(path, "target_probe.npz"), **arrays)
        if isinstance(self.mapping, _ModelMap):
            with open(os.path.join(path, "model.pkl"), "wb") as f:
                pickle.dump(self.mapping.model, f)
        self.dir = path
        return path

    @classmethod
    def load(cls, path: str, model: Any = None, tokenizer: Any = None) -> "TargetProbe":
        """A probe saved by :meth:`save` (attached to ``model`` when given)."""
        import pickle

        meta = json.loads(open(os.path.join(path, "target_probe.json"), encoding="utf-8").read())
        data = np.load(os.path.join(path, "target_probe.npz"))
        params = {k[len("map_"):]: data[k] for k in data.files if k.startswith("map_")}
        if meta["map"] == "model":
            with open(os.path.join(path, "model.pkl"), "rb") as f:
                mapping: _Map = _ModelMap(pickle.load(f))
        else:
            mapping = _map_from_params(meta["map"], params)
        places = data["places"] if data["places"].size else None
        probe = cls(meta["kind"], meta.get("classes_raw"), places, meta["layer"], mapping, name=meta["name"],
                    readout=meta.get("readout"), identity=meta.get("identity"), reading=meta.get("reading"),
                    shape=meta.get("shape"))
        probe.centroids = data["centroids"] if data["centroids"].size else None
        probe.dir = path
        if model is not None:
            probe.attach(model, tokenizer)
        return probe
