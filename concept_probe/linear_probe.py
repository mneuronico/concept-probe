"""What turns a probe's projections into numbers someone can read (layer 3 of docs/DESIGN.md).

A :class:`Calibration` holds the zero and unit of z scores, measured on items read in a given way
(a :class:`~concept_probe.inputs.ReadContext`), and optionally where texts without the concept
fall and a calibrated threshold to classify with. Every z score of the library is computed here.
:class:`~concept_probe.probe.ConceptProbe` holds a list of them, one per reading context.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .directions import estimator_spec, fit_directions, make_estimator
from .inputs import ReadContext
from .record import Choice
from .stats import _pooled_sd_columns


def _token_references(pos_tokens: List[np.ndarray], neg_tokens: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-layer zero and unit for single-token scores: the midpoint between the mean token score of
    each pole, and the pooled within-pole SD of single tokens (wider than that of per-item means,
    which average much of the token-to-token variation away)."""
    pos = np.concatenate(pos_tokens, axis=0)
    neg = np.concatenate(neg_tokens, axis=0)
    return 0.5 * (pos.mean(axis=0) + neg.mean(axis=0)), _pooled_sd_columns(pos, neg)


# ------------------------------------------------------------------------------------- threshold
def fit_threshold(pos: np.ndarray, neg: np.ndarray, l2: float = 1e-2, iters: int = 100) -> Dict[str, float]:
    """A one-variable logistic regression of the pole on a score.

    Returns ``slope`` and ``intercept`` (P(positive) = sigmoid(slope * score + intercept)) and
    ``threshold``, the score where that probability is 0.5. The fit is on standardized scores with
    a small L2 penalty, so perfectly separated poles give a finite slope.
    """
    x = np.r_[np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    design = np.c_[np.ones_like(x), (x - mu) / sd]
    penalty = np.diag([0.0, l2 * len(x)])
    w = np.zeros(2)  # intercept, slope on standardized scores
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(design @ w)))
        step = np.linalg.solve(design.T @ (design * (p * (1 - p))[:, None]) + penalty + 1e-9 * np.eye(2),
                               design.T @ (p - y) + penalty @ w)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    slope = w[1] / sd
    intercept = w[0] - w[1] * mu / sd
    return {"slope": float(slope), "intercept": float(intercept),
            "threshold": float(-intercept / slope) if slope != 0 else float("nan"),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ----------------------------------------------------------------------------------- calibration
def _arr(x: Any) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=np.float64)


@dataclass
class Calibration:
    """How to read a probe's projections for items read in ``context``.

    Per layer: ``center`` (the zero of item z scores) and ``scale`` (their unit), and the same for
    single tokens. ``zero`` says what the center is: ``"midpoint"`` between the two poles' mean item
    scores (the default; it is not "without the concept"), or ``"baseline"``, the second group's
    mean, for a concept read against a baseline instead of an opposite pole. ``neutral`` is the mean
    item score of texts without the concept, when some were given: where they fall is reported, not
    used as the zero. ``threshold`` is a calibrated classifier on the item score at
    ``threshold["layer"]``. ``source`` says where it was measured (``"evaluation"``,
    ``"transfer check"``, ``"calibrate()"``, ...).
    """

    context: ReadContext
    center: np.ndarray
    scale: np.ndarray
    token_center: Optional[np.ndarray] = None
    token_scale: Optional[np.ndarray] = None
    zero: str = "midpoint"
    neutral: Optional[np.ndarray] = None
    n_neutral: int = 0
    threshold: Optional[Dict[str, Any]] = None
    n_pos: int = 0
    n_neg: int = 0
    source: str = ""
    #: each pole's mean item score per layer (the poles' scale: +1 is the positive mean)
    pos_mean: Optional[np.ndarray] = None
    neg_mean: Optional[np.ndarray] = None

    @classmethod
    def from_scores(
        cls,
        pos: np.ndarray,
        neg: np.ndarray,
        context: ReadContext,
        *,
        pos_tokens: Optional[List[np.ndarray]] = None,
        neg_tokens: Optional[List[np.ndarray]] = None,
        zero: str = "midpoint",
        source: str = "",
    ) -> "Calibration":
        """From item scores ``(n, layers)`` of each group (and, optionally, each item's token scores
        ``(tokens, layers)``): the midpoint (or baseline) and the pooled within-group SD."""
        if zero not in ("midpoint", "baseline"):
            raise ValueError(f"zero must be 'midpoint' or 'baseline'; got {zero!r}")
        P = np.asarray(pos, dtype=np.float64)
        N = np.asarray(neg, dtype=np.float64)
        center = 0.5 * (P.mean(axis=0) + N.mean(axis=0)) if zero == "midpoint" else N.mean(axis=0)
        token_center = token_scale = None
        if pos_tokens and neg_tokens:
            token_center, token_scale = _token_references(pos_tokens, neg_tokens)
            if zero == "baseline":
                token_center = np.concatenate(neg_tokens, axis=0).mean(axis=0)
        return cls(context=context, center=center, scale=_pooled_sd_columns(P, N), token_center=token_center,
                   token_scale=token_scale, zero=zero, n_pos=int(P.shape[0]), n_neg=int(N.shape[0]), source=source,
                   pos_mean=P.mean(axis=0), neg_mean=N.mean(axis=0))

    def usable(self, layers: Sequence[int], tokens: bool = False) -> bool:
        """Whether the references at ``layers`` exist and are finite, with a positive unit."""
        c, u = (self.token_center, self.token_scale) if tokens else (self.center, self.scale)
        if c is None or u is None:
            return False
        c, u = np.asarray(c)[list(layers)], np.asarray(u)[list(layers)]
        return bool(np.all(np.isfinite(c)) and np.all(np.isfinite(u)) and np.all(u > 0))

    def items_z(self, scores: np.ndarray, layers: Sequence[int], aggregate: str = "mean",
                unit: str = "sd") -> Optional[np.ndarray]:
        """Item z scores from item scores ``(items, len(layers))`` at ``layers`` (an item's score is
        its mean projection over its tokens): standardized per layer, then aggregated over the
        layers (``"mean"`` or ``"sum"``). ``unit="sd"`` divides by the within-pole SD; ``"poles"``
        by the distance from the zero to the positive pole's mean (+1 is that mean). None when the
        references at those layers are unusable."""
        layers = list(layers)
        if not self.usable(layers):
            return None
        if unit == "poles":
            if self.pos_mean is None:
                return None
            unit_size = np.asarray(self.pos_mean)[layers] - np.asarray(self.center)[layers]
            if not np.all(np.abs(unit_size) > 0):
                return None
        else:
            unit_size = self.scale[layers]
        z = (np.asarray(scores, dtype=np.float64) - self.center[layers]) / unit_size
        return z.mean(axis=1) if aggregate == "mean" else z.sum(axis=1)

    def tokens_z(self, scores: np.ndarray, layers: Sequence[int], aggregate: str = "mean"
                 ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Single-token z scores from token scores ``(tokens, len(layers))``: per layer, and
        aggregated over the layers. None when this calibration has no usable token references."""
        layers = list(layers)
        if not self.usable(layers, tokens=True):
            return None
        z = ((np.asarray(scores, dtype=np.float64) - self.token_center[layers]) / self.token_scale[layers])
        agg = z.mean(axis=1) if aggregate == "mean" else z.sum(axis=1)
        return z.astype(np.float32), agg.astype(np.float32)

    def add_neutral(self, scores: np.ndarray) -> None:
        """Record where texts without the concept fall: their mean item score per layer."""
        s = np.asarray(scores, dtype=np.float64)
        self.neutral, self.n_neutral = s.mean(axis=0), int(s.shape[0])

    def add_threshold(self, pos: np.ndarray, neg: np.ndarray, layer: int, fit_on: str,
                      l2: float = 1e-2) -> Dict[str, Any]:
        """Fit :func:`fit_threshold` on item scores at ``layer`` and keep it."""
        self.threshold = dict(fit_threshold(pos, neg, l2=l2), layer=int(layer), fit_on=fit_on, l2=float(l2))
        return self.threshold

    def to_dict(self) -> Dict[str, Any]:
        def plain(x):
            return None if x is None else np.asarray(x, dtype=np.float64).tolist()

        return {"context": self.context.to_dict(), "center": plain(self.center), "scale": plain(self.scale),
                "token_center": plain(self.token_center), "token_scale": plain(self.token_scale), "zero": self.zero,
                "neutral": plain(self.neutral), "n_neutral": self.n_neutral, "threshold": self.threshold,
                "n_pos": self.n_pos, "n_neg": self.n_neg, "source": self.source,
                "pos_mean": plain(self.pos_mean), "neg_mean": plain(self.neg_mean)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Calibration":
        return cls(context=ReadContext.from_dict(d["context"]), center=_arr(d["center"]), scale=_arr(d["scale"]),
                   token_center=_arr(d.get("token_center")), token_scale=_arr(d.get("token_scale")),
                   zero=d.get("zero", "midpoint"), neutral=_arr(d.get("neutral")), n_neutral=int(d.get("n_neutral", 0)),
                   threshold=d.get("threshold"), n_pos=int(d.get("n_pos", 0)), n_neg=int(d.get("n_neg", 0)),
                   source=d.get("source", ""), pos_mean=_arr(d.get("pos_mean")), neg_mean=_arr(d.get("neg_mean")))


def calibrations_from_tensors(tensors: Dict[str, Any]) -> List[Calibration]:
    """The calibrations a run's ``tensors.npz`` holds: the full list when it was saved, otherwise the
    evaluation's and the transfer check's references (runs saved before calibrations had their own
    record)."""
    if "calibrations" in tensors:
        return [Calibration.from_dict(c) for c in json.loads(str(np.asarray(tensors["calibrations"])[0]))]
    out = []
    ref = json.loads(str(np.asarray(tensors["reference_context"])[0])) if "reference_context" in tensors else None
    if ref:
        out.append(Calibration(context=ReadContext.from_dict(ref), center=_arr(tensors["score_center"]),
                               scale=_arr(tensors["score_scale"]), token_center=_arr(tensors.get("token_center")),
                               token_scale=_arr(tensors.get("token_scale")), source="evaluation"))
    transfer = json.loads(str(np.asarray(tensors["transfer_references"])[0])) if "transfer_references" in tensors else None
    if transfer:
        out.append(Calibration(context=ReadContext.from_dict(transfer["context"]), center=_arr(transfer["score_center"]),
                               scale=_arr(transfer["score_scale"]), token_center=_arr(transfer.get("token_center")),
                               token_scale=_arr(transfer.get("token_scale")), source="transfer check"))
    return out


# --------------------------------------------------------------------------------- steering (D1)
def steer_directions_for(estimator: Any, steer: Any, pos: np.ndarray, neg: np.ndarray,
                         read_directions: np.ndarray, erase: Any = None) -> Tuple[np.ndarray, Dict[str, Any], Optional[Choice]]:
    """The directions to steer with (decision D1), their estimator's settings, and the choice to
    announce when they differ from the reading directions.

    ``steer`` is ``"read"`` (the reading directions, returned as the same array), or an estimator
    spec (default ``"mean_diff"``: the raw difference of means, which pushes further than
    variance-normalized directions do).
    """
    read_spec = estimator_spec(estimator)
    if steer in (None, "read"):
        return read_directions, read_spec, None
    steer_spec = estimator_spec(steer)
    if steer_spec == read_spec:
        return read_directions, steer_spec, None
    vectors = fit_directions(make_estimator(steer), pos, neg, erase=erase)
    return vectors, steer_spec, Choice(
        "steering direction", steer if isinstance(steer, (str, dict)) else steer_spec, steer_spec,
        f"steering uses its own direction, not the reading one ({read_spec}): the raw difference of means "
        "pushes further than normalized directions; pass steer='read' to steer with the reading direction",
    )
