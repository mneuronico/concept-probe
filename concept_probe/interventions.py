"""Interventions on the residual stream.

An intervention says what to change in the residual stream, at which layers and at which token
positions, with any vectors:

- :class:`Add` adds ``alpha * v_l`` (steering);
- :class:`Ablate` projects directions out, or sets the stream's coordinate along one to a value;
- :class:`Patch` overwrites positions with stored activations (activation patching);
- :class:`Rotate` rotates the stream in a plane;
- :class:`Edit` applies a function of your own;
- ``a + b`` (:class:`Compose`) applies several, in order.

``with intervene(model, intervention):`` applies one through forward hooks on the chosen blocks;
everything the model does inside the block (generating, reading, a plain forward pass) runs on the
edited stream. An intervention knows nothing about probes: ``probe.steering(...)`` builds an
:class:`Add` from a probe's steering directions and units.

Layer ``l`` is the output of block ``l``, the residual stream that ``directions[l]`` reads, as
everywhere in the library. **Positions** count each row's own tokens from 0, without padding, so an
intervention means the same in a batch padded on either side and while generating with a cache.
Negative positions count back from the end of the prompt (``-1`` is its last token): the prompt is
the input the model was given (when generating, without the new tokens). ``"prompt"`` and
``"generated"`` select those two parts; ``None`` is every position.

Reading under an intervention (:func:`concept_probe.read_texts` and the scorers inside the ``with``
block) reports at an edited layer the stream after the edit, which is what the next block receives.
"""
import math
import weakref
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .modeling import get_transformer_layers
from .utils import resolve_layer_index

PositionSpec = Union[None, int, Sequence[int], slice, str]
VectorSpec = Any  # {layer: vector}, an array (layers, dim), or one vector for every layer in ``layers``


# ------------------------------------------------------------------------------------- positions
def _normalize_positions(spec: PositionSpec, what: str = "positions") -> Any:
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in ("prompt", "generated"):
            raise ValueError(f"{what} must be None, 'prompt', 'generated', an int, a list of ints or a slice; "
                             f"got {spec!r}")
        return spec
    if isinstance(spec, slice):
        if spec.step not in (None, 1):
            raise ValueError(f"{what}: a slice with a step is not supported")
        return slice(None if spec.start is None else int(spec.start), None if spec.stop is None else int(spec.stop))
    if isinstance(spec, (int, np.integer)):
        return (int(spec),)
    out = tuple(int(p) for p in spec)
    if not out:
        raise ValueError(f"{what} is empty")
    return out


def _describe_positions(spec: Any) -> Any:
    if spec is None or isinstance(spec, str):
        return "all" if spec is None else spec
    if isinstance(spec, slice):
        return {"start": spec.start, "stop": spec.stop}
    return list(spec)


class Where:
    """Where the columns of the current forward pass sit in their rows.

    ``positions[r, c]`` is the position of column ``c`` in row ``r``'s own sequence (padding
    excluded), ``prompt_lens[r]`` the length of that row's prompt, and ``items[r]`` which input row
    ``r`` is (the prompt's index, as given to :func:`concept_probe.generate` or
    :func:`concept_probe.answer_logprobs`; None when the caller did not say). :class:`Edit`
    functions get one, to edit only some positions with :meth:`mask`.
    """

    def __init__(self, positions: torch.Tensor, prompt_lens: torch.Tensor, *, generating: bool = False,
                 fresh: bool = True, items: Optional[torch.Tensor] = None):
        self.positions = positions
        self.prompt_lens = prompt_lens
        self.generating = generating
        self.fresh = fresh
        self.items = items

    def resolve(self, position: int) -> torch.Tensor:
        """Each row's position for ``position`` (negative ones count back from the end of the prompt)."""
        if position >= 0:
            return torch.full_like(self.prompt_lens, int(position))
        resolved = self.prompt_lens + int(position)
        if bool((resolved < 0).any()):
            raise IndexError(f"position {position} is before the start of a prompt of {int(self.prompt_lens.min())} "
                             "tokens")
        return resolved

    def _check_inside(self, resolved: torch.Tensor, position: int) -> None:
        # Outside a generation the whole input is the prompt, so a position past it is a mistake.
        if not self.generating and self.fresh and bool((resolved >= self.prompt_lens).any()):
            raise IndexError(f"position {position} is past the end of an input of {int(self.prompt_lens.min())} tokens")

    def mask(self, spec: PositionSpec) -> Optional[torch.Tensor]:
        """A ``(batch, columns)`` boolean mask of the positions ``spec`` selects; None means all."""
        spec = _normalize_positions(spec)
        if spec is None:
            return None
        pos, plen = self.positions, self.prompt_lens[:, None]
        if spec == "prompt":
            return pos < plen
        if spec == "generated":
            return pos >= plen
        if isinstance(spec, slice):
            lo = torch.zeros_like(plen) if spec.start is None else self.resolve(spec.start)[:, None]
            out = pos >= lo
            if spec.stop is not None:
                out &= pos < self.resolve(spec.stop)[:, None]
            return out
        out = torch.zeros_like(pos, dtype=torch.bool)
        for p in spec:
            resolved = self.resolve(p)
            self._check_inside(resolved, p)
            out |= pos == resolved[:, None]
        return out


def _work(hidden: torch.Tensor) -> torch.Tensor:
    """``hidden`` in at least float32, for edits that project (half precision loses the small part)."""
    return hidden if hidden.dtype == torch.float64 else hidden.float()


def _apply_masked(hidden: torch.Tensor, new: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return new
    return torch.where(mask.to(hidden.device)[..., None], new, hidden)


# --------------------------------------------------------------------------------------- vectors
def _as_array(v: Any) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().to("cpu", torch.float32).numpy()
    return np.asarray(v, dtype=np.float32)


def _per_layer(vectors: VectorSpec, layers: Optional[Sequence[int]], what: str, sets: bool = False
               ) -> List[Tuple[int, np.ndarray]]:
    """``[(layer, vector)]`` from ``{layer: vector}``, an array with one vector per model layer
    (``layers`` picks some), or one vector for every layer in ``layers``. With ``sets``, a layer may
    have several vectors: the rows of a dict value, or an array ``(model layers, k, dim)``."""
    if isinstance(vectors, dict):
        if layers is not None:
            raise ValueError(f"{what}: give the layers as the dict's keys, not also as layers=")
        out = [(int(l), _as_array(v)) for l, v in vectors.items()]
    else:
        arr = _as_array(vectors)
        if arr.ndim == 1:  # one vector for every layer in ``layers``
            if layers is None:
                raise ValueError(f"{what}: one vector needs layers= (the layers to apply it at)")
            out = [(int(l), arr) for l in layers]
        elif arr.ndim == 2 or (sets and arr.ndim == 3):  # one per model layer
            chosen = range(arr.shape[0]) if layers is None else layers
            out = [(int(l), arr[int(l)]) for l in chosen]
        else:
            raise ValueError(f"{what} must be a dict, one vector, or one vector per model layer; got shape {arr.shape}")
    if not out:
        raise ValueError(f"{what}: no layers")
    for l, v in out:
        if v.ndim not in ((1, 2) if sets else (1,)) or not np.all(np.isfinite(v)):
            raise ValueError(f"{what} at layer {l}: expected finite vectors, got shape {v.shape}")
    return out


def _resolve(entries: Sequence[Tuple[int, Any]], n_blocks: int, what: str) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for l, v in entries:
        li = resolve_layer_index(l, n_blocks, f"{what} layer")
        if li in out:
            raise ValueError(f"{what}: layer {li} is given twice")
        out[li] = v
    return out


def _per_layer_value(value: Any, layers: Sequence[int], what: str) -> Dict[int, Any]:
    if isinstance(value, dict):
        missing = [l for l in layers if l not in value]
        if missing:
            raise ValueError(f"{what}: no value for layers {missing}")
        return {int(l): value[l] for l in layers}
    return {int(l): value for l in layers}


# ---------------------------------------------------------------------------------- interventions
class Intervention:
    """Base class: what to change in the residual stream. Subclasses define :meth:`layer_list`,
    :meth:`edit` and :meth:`to_dict`."""

    def layer_list(self, n_blocks: int) -> List[int]:
        raise NotImplementedError

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def prepare(self, n_blocks: int) -> None:
        """Resolve negative layers against the model; called when the intervention is applied."""

    def __add__(self, other: "Intervention") -> "Compose":
        return Compose(self, other)


class Add(Intervention):
    """Add ``alpha * v_l`` to the residual stream at each layer ``l`` (steering).

    ``vectors`` is ``{layer: vector}``, an array with one vector per model layer (with ``layers``,
    only those), or one vector for every layer in ``layers``. ``alpha`` is a number or
    ``{layer: number}``. The vector is added as given (not normalized), at ``positions``.
    """

    def __init__(self, vectors: VectorSpec, alpha: Union[float, Dict[int, float]] = 1.0, *,
                 layers: Optional[Sequence[int]] = None, positions: PositionSpec = None, label: Optional[str] = None):
        self._entries = _per_layer(vectors, layers, "Add vectors")
        self._alpha = alpha
        self.positions = _normalize_positions(positions)
        self.label = label
        self._by_layer: Optional[Dict[int, Tuple[float, torch.Tensor]]] = None
        self._devices: Dict[Tuple[int, Any, Any], torch.Tensor] = {}
        #: the seed of a random control (see :meth:`random_like`); None for anything else
        self.control_seed: Any = None

    def prepare(self, n_blocks: int) -> None:
        vectors = _resolve(self._entries, n_blocks, "Add")
        if isinstance(self._alpha, dict):
            alphas = _per_layer_value(_resolve(list(self._alpha.items()), n_blocks, "Add alpha"), list(vectors),
                                      "Add alpha")
        else:
            alphas = {l: self._alpha for l in vectors}
        self._by_layer = {l: (float(alphas[l]), torch.as_tensor(v, dtype=torch.float32)) for l, v in vectors.items()}
        self._devices = {}

    @property
    def alpha_per_layer(self) -> Dict[int, float]:
        """The alpha each layer adds its vector with (after :meth:`prepare`)."""
        if self._by_layer is None:
            return {int(l): float(self._alpha[l] if isinstance(self._alpha, dict) else self._alpha)
                    for l, _ in self._entries}
        return {l: a for l, (a, _) in self._by_layer.items()}

    def layer_list(self, n_blocks: int) -> List[int]:
        if self._by_layer is None:
            self.prepare(n_blocks)
        return sorted(self._by_layer)

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        alpha, v = self._by_layer[layer]
        if alpha == 0.0:
            return hidden
        key = (layer, hidden.device, hidden.dtype)
        offset = self._devices.get(key)
        if offset is None:
            # The device of the block's output, not of its parameters: with CPU offload (accelerate),
            # a block's parameters sit on the "meta" device outside its forward pass.
            offset = (alpha * v).view(1, 1, -1).to(device=hidden.device, dtype=hidden.dtype)
            self._devices[key] = offset
        mask = where.mask(self.positions)
        if mask is None:
            return hidden + offset
        return hidden + offset * mask.to(hidden.device, hidden.dtype)[..., None]

    def to_dict(self) -> Dict[str, Any]:
        if self._by_layer is not None:
            norms = {l: float(v.norm()) for l, (_, v) in self._by_layer.items()}
        else:
            norms = {int(l): float(np.linalg.norm(v)) for l, v in self._entries}
        out = {"type": "add", "label": self.label, "layers": sorted(norms), "alpha": self.alpha_per_layer,
               "vector_norm": norms, "positions": _describe_positions(self.positions)}
        if self.control_seed is not None:
            out["control_seed"] = self.control_seed
        return out

    def _alphas_as_given(self) -> Union[float, Dict[int, float]]:
        return dict(self._alpha) if isinstance(self._alpha, dict) else float(self._alpha)

    def scaled(self, factor: float) -> "Add":
        """The same intervention with every alpha multiplied by ``factor``: a dose of it."""
        alpha = ({l: float(a) * float(factor) for l, a in self._alpha.items()} if isinstance(self._alpha, dict)
                 else float(self._alpha) * float(factor))
        out = Add({l: v for l, v in self._entries}, alpha, positions=self.positions, label=self.label)
        out.control_seed = self.control_seed
        return out

    def random_like(self, seed: Any) -> "Add":
        """A control for this intervention: the same layers, alphas and positions, with each layer's
        vector replaced by one random direction (from ``seed``, an int or a sequence of ints), at
        that layer's vector norm.

        One direction for every layer keeps the control comparable to concept vectors, which point
        nearly the same way at adjacent layers: independent random directions would partly cancel
        when several layers are steered, and push the stream about half as far.
        """
        dim = self._entries[0][1].shape[-1]
        direction = np.random.default_rng(seed).standard_normal(dim)
        direction /= np.linalg.norm(direction)
        vectors = {l: (direction * np.linalg.norm(v)).astype(np.float32) for l, v in self._entries}
        out = Add(vectors, self._alphas_as_given(), positions=self.positions,
                  label=f"random control of {self.label or 'an addition'}")
        out.control_seed = list(seed) if isinstance(seed, (list, tuple)) else seed
        return out


def _orthonormal_rows(vectors: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    q, r = np.linalg.qr(m.T)
    keep = np.abs(np.diag(r)) > 1e-10 * max(1.0, float(np.abs(r).max()))
    if not keep.all():
        raise ValueError("the directions are linearly dependent (or one is zero)")
    # QR may flip a sign; keep each basis vector on the side of the direction it came from
    q = q * np.sign(np.diag(r))[None, :]
    return q.T


class Ablate(Intervention):
    """Remove directions from the residual stream, or set the stream's coordinate along one.

    ``directions`` per layer (as in :class:`Add`) are one vector, or several as the rows of an
    array (a subspace), and need not be unit vectors. With ``value=None`` the stream's component
    in their span is removed (projected out); with a number (or ``{layer: number}``), the
    coordinate along the unit direction is set to it, as in mean ablation with a mean you computed.
    Computed in at least float32 whatever the model's dtype.
    """

    def __init__(self, directions: VectorSpec, *, layers: Optional[Sequence[int]] = None,
                 value: Union[None, float, Dict[int, float]] = None, positions: PositionSpec = None,
                 label: Optional[str] = None):
        self._entries = _per_layer(directions, layers, "Ablate directions", sets=True)
        self._value = value
        self.positions = _normalize_positions(positions)
        self.label = label
        self._by_layer: Optional[Dict[int, Tuple[torch.Tensor, Optional[float]]]] = None

    def prepare(self, n_blocks: int) -> None:
        bases = _resolve(self._entries, n_blocks, "Ablate")
        value = _resolve(list(self._value.items()), n_blocks, "Ablate value") if isinstance(self._value, dict)             else self._value
        values = _per_layer_value(value, list(bases), "Ablate value")
        self._by_layer = {}
        for l, d in bases.items():
            basis = _orthonormal_rows(d)
            if values[l] is not None and basis.shape[0] != 1:
                raise ValueError("Ablate: value= sets the coordinate along one direction; give one per layer")
            self._by_layer[l] = (torch.as_tensor(basis, dtype=torch.float64), None if values[l] is None else float(values[l]))

    def layer_list(self, n_blocks: int) -> List[int]:
        if self._by_layer is None:
            self.prepare(n_blocks)
        return sorted(self._by_layer)

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        basis, value = self._by_layer[layer]
        h = _work(hidden)
        U = basis.to(h.device, h.dtype)
        coords = h @ U.T
        target = torch.zeros_like(coords) if value is None else torch.full_like(coords, value)
        new = (h + (target - coords) @ U).to(hidden.dtype)
        return _apply_masked(hidden, new, where.mask(self.positions))

    def to_dict(self) -> Dict[str, Any]:
        layers = sorted(int(l) for l, _ in self._entries)
        return {"type": "ablate", "label": self.label, "layers": layers,
                "n_directions": {int(l): int(np.atleast_2d(d).shape[0]) for l, d in self._entries},
                "value": self._value if not isinstance(self._value, dict) else {int(k): v for k, v in self._value.items()},
                "positions": _describe_positions(self.positions)}


class Patch(Intervention):
    """Overwrite the residual stream of ``layer`` at ``positions`` with ``values`` (activation patching).

    ``values`` has one row per position (or is one vector, for one position), for example a donor's
    activations from :func:`concept_probe.read_texts` with ``readout="tokens"``; every row of a batch
    gets them. Positions follow the module's rules (negative ones count back from the end of the
    prompt); outside a generation, a position past the end of an input raises.

    One donor per input: ``values`` with a leading axis of inputs, ``(inputs, positions, dim)``,
    gives input ``i`` its own ``values[i]`` at the same ``positions``; or ``positions`` as one list
    per input, with ``values`` a list of ``(len(positions[i]), dim)`` arrays, gives each input its
    own positions too. An input is a prompt, numbered as given to :func:`concept_probe.generate` or
    :func:`concept_probe.answer_logprobs`, which say which input each row of their batches is; a
    direct ``model(...)`` call under :func:`concept_probe.intervene` says it with ``items=``.
    """

    def __init__(self, layer: int, positions: Any, values: Any, *, label: Optional[str] = None):
        self._layer = int(layer)
        self.label = label
        self.layer: Optional[int] = None
        per_position_lists = (isinstance(positions, (list, tuple)) and len(positions) > 0
                              and all(isinstance(p, (list, tuple)) for p in positions))
        self.per_input: Optional[List[Tuple[Tuple[int, ...], torch.Tensor]]] = None
        if per_position_lists:
            if not isinstance(values, (list, tuple)) or len(values) != len(positions):
                raise ValueError("with one list of positions per input, values is a list with one array per input")
            self.per_input = [self._entry(p, v) for p, v in zip(positions, values)]
            self.positions = None
            return
        self.positions = _normalize_positions(positions)
        if not isinstance(self.positions, tuple):
            raise ValueError("Patch positions must be an int or a list of ints, one per row of values "
                             "(or one list per input)")
        vals = _as_array(values)
        if vals.ndim == 3:
            self.per_input = [self._entry(self.positions, v) for v in vals]
            return
        if vals.ndim == 1:
            vals = vals[None, :]
        if vals.shape[0] != len(self.positions):
            raise ValueError(f"values has {vals.shape[0]} rows but {len(self.positions)} positions are given")
        self._values = torch.as_tensor(vals)

    @staticmethod
    def _entry(positions: Any, values: Any) -> Tuple[Tuple[int, ...], torch.Tensor]:
        pos = _normalize_positions(list(positions) if isinstance(positions, (list, tuple)) else positions)
        if not isinstance(pos, tuple):
            raise ValueError("Patch positions per input must be ints")
        vals = _as_array(values)
        if vals.ndim == 1:
            vals = vals[None, :]
        if vals.shape[0] != len(pos):
            raise ValueError(f"an input's values have {vals.shape[0]} rows for {len(pos)} positions")
        return pos, torch.as_tensor(vals)

    def prepare(self, n_blocks: int) -> None:
        self.layer = resolve_layer_index(self._layer, n_blocks, "Patch layer")

    def layer_list(self, n_blocks: int) -> List[int]:
        if self.layer is None:
            self.prepare(n_blocks)
        return [self.layer]

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        if self.per_input is None:
            out = hidden
            vals = self._values.to(device=hidden.device, dtype=hidden.dtype)
            for j, p in enumerate(self.positions):
                mask = where.mask((p,))
                out = torch.where(mask.to(hidden.device)[..., None], vals[j], out)
            return out
        if where.items is None:
            raise RuntimeError("Patch with one donor per input needs to know which input each row is: run it "
                               "through generate or answer_logprobs, or call intervene(..., items=[...]).")
        out = hidden.clone()
        cols = where.positions.to(hidden.device)
        for r, item in enumerate(where.items.tolist()):
            if not 0 <= int(item) < len(self.per_input):
                raise IndexError(f"Patch has values for {len(self.per_input)} inputs; row {r} is input {item}")
            pos, vals = self.per_input[int(item)]
            vals = vals.to(device=hidden.device, dtype=hidden.dtype)
            for j, p in enumerate(pos):
                at = int(where.resolve(p)[r])
                if not where.generating and where.fresh and at >= int(where.prompt_lens[r]):
                    raise IndexError(f"position {p} is past the end of input {item} ({int(where.prompt_lens[r])} tokens)")
                hit = cols[r] == at
                out[r] = torch.where(hit[:, None], vals[j], out[r])
        return out

    def to_dict(self) -> Dict[str, Any]:
        out = {"type": "patch", "label": self.label, "layers": [self._layer if self.layer is None else self.layer]}
        if self.per_input is None:
            out["positions"] = list(self.positions)
        else:
            out.update(inputs=len(self.per_input), positions=[list(p) for p, _ in self.per_input])
        return out


class Rotate(Intervention):
    """Rotate the residual stream by ``angle`` radians in the plane of ``u`` and ``v`` (from ``u``
    toward ``v``), leaving the rest of it as it is. ``u`` and ``v`` per layer as in :class:`Add`;
    they need not be orthogonal or unit vectors. The rotation is about ``center`` (a point per
    layer, as the vectors; by default the origin): for a cycle of classes, the centroids' center.
    Computed in at least float32."""

    def __init__(self, u: VectorSpec, v: VectorSpec, angle: float, *, layers: Optional[Sequence[int]] = None,
                 center: Optional[VectorSpec] = None, positions: PositionSpec = None, label: Optional[str] = None):
        self._u = _per_layer(u, layers, "Rotate u")
        self._v = _per_layer(v, layers, "Rotate v")
        self._center = None if center is None else _per_layer(center, layers, "Rotate center")
        self.angle = float(angle)
        self.positions = _normalize_positions(positions)
        self.label = label
        self._by_layer: Optional[Dict[int, torch.Tensor]] = None
        self._centers: Dict[int, torch.Tensor] = {}

    def prepare(self, n_blocks: int) -> None:
        us, vs = _resolve(self._u, n_blocks, "Rotate u"), _resolve(self._v, n_blocks, "Rotate v")
        if set(us) != set(vs):
            raise ValueError("Rotate: u and v must be given at the same layers")
        self._by_layer = {l: torch.as_tensor(_orthonormal_rows(np.stack([us[l], vs[l]])), dtype=torch.float64)
                          for l in us}
        self._centers = {}
        if self._center is not None:
            centers = _resolve(self._center, n_blocks, "Rotate center")
            if set(centers) != set(us):
                raise ValueError("Rotate: center must be given at the same layers as u and v")
            self._centers = {l: torch.as_tensor(c, dtype=torch.float64) for l, c in centers.items()}

    def layer_list(self, n_blocks: int) -> List[int]:
        if self._by_layer is None:
            self.prepare(n_blocks)
        return sorted(self._by_layer)

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        h = _work(hidden)
        E = self._by_layer[layer].to(h.device, h.dtype)  # (2, dim): e1 along u, e2 in the plane, orthogonal to it
        rel = h - self._centers[layer].to(h.device, h.dtype) if layer in self._centers else h
        a, b = rel @ E[0], rel @ E[1]
        c, s = math.cos(self.angle), math.sin(self.angle)
        new = h + ((a * c - b * s) - a)[..., None] * E[0] + ((a * s + b * c) - b)[..., None] * E[1]
        return _apply_masked(hidden, new.to(hidden.dtype), where.mask(self.positions))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "rotate", "label": self.label, "layers": sorted(int(l) for l, _ in self._u),
                "angle": self.angle, "positions": _describe_positions(self.positions)}


class Edit(Intervention):
    """Any change of your own: ``fn(layer, hidden, where) -> hidden`` at each of ``layers``.

    ``hidden`` is the block's output ``(batch, columns, dim)``; return a tensor of the same shape
    (an edited copy, not the input changed in place). ``where`` (a :class:`Where`) says which
    position each column is, and ``where.mask(spec)`` selects positions as the other interventions do.
    """

    def __init__(self, fn: Callable[[int, torch.Tensor, Where], torch.Tensor], layers: Sequence[int], *,
                 label: Optional[str] = None):
        self.fn = fn
        self._layers = [int(l) for l in layers]
        self.label = label
        self._resolved: Optional[List[int]] = None

    def prepare(self, n_blocks: int) -> None:
        self._resolved = sorted(_resolve([(l, None) for l in self._layers], n_blocks, "Edit"))

    def layer_list(self, n_blocks: int) -> List[int]:
        if self._resolved is None:
            self.prepare(n_blocks)
        return self._resolved

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        return self.fn(layer, hidden, where)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "edit", "label": self.label or getattr(self.fn, "__name__", "function"),
                "layers": self._layers if self._resolved is None else self._resolved}


class Compose(Intervention):
    """Several interventions, applied in order at each layer they share."""

    def __init__(self, *parts: Intervention):
        flat: List[Intervention] = []
        for p in parts:
            if not isinstance(p, Intervention):
                raise TypeError(f"Compose takes interventions; got {type(p).__name__}")
            flat.extend(p.parts if isinstance(p, Compose) else [p])
        if not flat:
            raise ValueError("Compose needs at least one intervention")
        self.parts = flat
        self._layers: Dict[int, List[Intervention]] = {}

    def prepare(self, n_blocks: int) -> None:
        self._layers = {}
        for p in self.parts:
            p.prepare(n_blocks)
            for l in p.layer_list(n_blocks):
                self._layers.setdefault(l, []).append(p)

    def layer_list(self, n_blocks: int) -> List[int]:
        if not self._layers:
            self.prepare(n_blocks)
        return sorted(self._layers)

    def edit(self, layer: int, hidden: torch.Tensor, where: Where) -> torch.Tensor:
        for p in self._layers.get(layer, []):
            hidden = p.edit(layer, hidden, where)
        return hidden

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "compose", "parts": [p.to_dict() for p in self.parts]}


# ------------------------------------------------------------------------------------ applying them
# The interventions active on each model, so that reading can report the edited stream.
_ACTIVE: "weakref.WeakKeyDictionary[torch.nn.Module, List[_Applied]]" = weakref.WeakKeyDictionary()


def _first_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    for key in ("input_ids", "inputs_embeds"):
        if kwargs.get(key) is not None:
            return kwargs[key]
    return args[0] if args and isinstance(args[0], torch.Tensor) else None


def _past_length(kwargs: Dict[str, Any]) -> int:
    past = kwargs.get("past_key_values")
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    try:
        return int(past[0][0].shape[-2])
    except Exception:
        return 0


def _where_from_call(args: Tuple[Any, ...], kwargs: Dict[str, Any], prompt_lens: Optional[torch.Tensor],
                     latched: List[Optional[torch.Tensor]], generating: bool,
                     items: Optional[torch.Tensor] = None) -> Where:
    """Positions of the columns of one forward call, from its attention mask (or cache position)."""
    x = _first_tensor(args, kwargs)
    if x is None:
        raise RuntimeError("intervene: could not find the input of the model's forward call")
    batch, q = int(x.shape[0]), int(x.shape[1])
    mask = kwargs.get("attention_mask")
    if mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
        mask = args[1]
    if isinstance(mask, torch.Tensor) and mask.dim() == 2:
        mask = mask.to("cpu")
        total = int(mask.shape[1])
        # Left padding: the leading zeros; right padding leaves no leading zeros.
        lead = (mask.cumsum(dim=1) == 0).sum(dim=1)
        positions = torch.arange(total - q, total)[None, :] - lead[:, None]
        fresh = total == q
        real = mask[:, total - q:].sum(dim=1)
    else:
        start = kwargs.get("cache_position")
        offset = int(start.reshape(-1)[0]) if isinstance(start, torch.Tensor) and start.numel() else _past_length(kwargs)
        positions = (torch.arange(q) + offset)[None, :].expand(batch, q)
        fresh = offset == 0
        real = torch.full((batch,), q, dtype=torch.long)
    if prompt_lens is None:
        if fresh or latched[0] is None:
            latched[0] = real.to(torch.long)
        lens = latched[0]
    else:
        lens = prompt_lens
    if items is not None and int(items.shape[0]) != batch:
        raise ValueError(f"intervene was told {int(items.shape[0])} items for a batch of {batch} rows")
    return Where(positions.to(torch.long), lens.to(torch.long), generating=generating, fresh=fresh, items=items)


class _Applied:
    """The hooks of one intervention on one model (see :func:`intervene`)."""

    def __init__(self, model: torch.nn.Module, intervention: Intervention, prompt_lens: Optional[Sequence[int]],
                 items: Optional[Sequence[int]] = None):
        if not isinstance(intervention, Intervention):
            raise TypeError(f"intervene needs an Intervention (Add, Ablate, Patch, Rotate, Edit or a sum of them); "
                            f"got {type(intervention).__name__}")
        self.model = model
        self.intervention = intervention
        self.prompt_lens = None if prompt_lens is None else torch.as_tensor(list(prompt_lens), dtype=torch.long)
        self.items = None if items is None else torch.as_tensor(list(items), dtype=torch.long)
        self.handles: List[Any] = []
        self.where: Optional[Where] = None
        self._latched: List[Optional[torch.Tensor]] = [None]
        self.record: Optional[Dict[int, torch.Tensor]] = None

    def __enter__(self) -> "_Applied":
        blocks = get_transformer_layers(self.model)
        n = len(blocks)
        self.intervention.prepare(n)
        layers = self.intervention.layer_list(n)

        def pre(module, args, kwargs):
            self.where = _where_from_call(args, kwargs, self.prompt_lens, self._latched, self.prompt_lens is not None,
                                          self.items)

        self.handles.append(self.model.register_forward_pre_hook(pre, with_kwargs=True))
        for li in layers:
            self.handles.append(blocks[li].register_forward_hook(self._hook(li)))
        _ACTIVE.setdefault(self.model, []).append(self)
        return self

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            if self.where is None:
                raise RuntimeError("intervene: the block ran outside a call to the model it was registered on; "
                                   "call the model itself (model(...) or model.generate(...)).")
            hidden = output[0] if isinstance(output, tuple) else output
            new = self.intervention.edit(layer, hidden, self.where)
            if self.record is not None:
                self.record[layer] = new
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

        return hook

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        active = _ACTIVE.get(self.model, [])
        if self in active:
            active.remove(self)


def intervene(model: torch.nn.Module, intervention: Intervention, *, prompt_lens: Optional[Sequence[int]] = None,
              items: Optional[Sequence[int]] = None) -> _Applied:
    """Apply ``intervention`` to ``model`` while the ``with`` block runs.

    ``prompt_lens`` (one per row) says where each row's prompt ends, for negative positions and
    ``"prompt"``/``"generated"``; :func:`concept_probe.generate` and
    :func:`concept_probe.answer_logprobs` pass it. Without it, the prompt is the input of the last
    forward call that started a sequence (no cache): right for a plain forward pass and for
    ``model.generate`` with a cache. ``items`` (one per row) says which input each row is, for a
    :class:`Patch` with one donor per input; generate and answer_logprobs pass it too.
    """
    return _Applied(model, intervention, prompt_lens, items)


class _Recording:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.edited: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> Dict[int, torch.Tensor]:
        for applied in _ACTIVE.get(self.model, []):
            applied.record = self.edited
        return self.edited

    def __exit__(self, exc_type, exc, tb) -> None:
        for applied in _ACTIVE.get(self.model, []):
            applied.record = None


def edited_outputs(model: torch.nn.Module) -> _Recording:
    """While a forward pass runs inside this block, collect the edited output of every block that an
    active intervention changed, so a reader can report the stream after the edit (Hugging Face
    records a block's output before forward hooks change it)."""
    return _Recording(model)


def active_interventions(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """What is being applied to ``model`` right now (for the records of a reading)."""
    return [a.intervention.to_dict() for a in _ACTIVE.get(model, [])]
