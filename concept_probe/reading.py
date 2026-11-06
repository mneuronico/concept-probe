"""Reading: pass inputs through the model and return their representations (layer 1 of
docs/DESIGN.md).

Inputs go through the model in batches; a :class:`Readout` reduces each item's tokens to one
vector per layer (or keeps every token); the result is plain arrays (:class:`Reps`). Reading saves
what it reads in a cache by default, so an input goes through a model once (decision D4).
Everything after this layer takes arrays, not texts.
"""
import hashlib
import inspect
import io
import json
import os
import sqlite3
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .inputs import (
    RENAMED_READOUTS,
    Encoded,
    PromptLike,
    ReadContext,
    encode_text,
)
from .interventions import active_interventions, edited_outputs
from .modeling import _resolve_attr_path, get_transformer_layers, identity_of, safe_attention
from .record import Choice, announce
from .settings import default_rep_cache_dir, settings
from .utils import resolve_layer_index


# Mirrors the wrapper layouts handled by modeling.get_transformer_layers, so any model
# whose blocks we can find also has a findable final norm.
_FINAL_NORM_PATHS = (
    "model.norm",
    "model.final_layernorm",
    "model.ln_f",
    "model.language_model.norm",
    "language_model.norm",
    "model.model.norm",
    "model.model.language_model.norm",
    "base_model.model.norm",
    "base_model.model.language_model.norm",
    "transformer.ln_f",
    "norm",
    "final_layernorm",
    "ln_f",
)


def _get_final_norm_module(model):
    for path in _FINAL_NORM_PATHS:
        mod = _resolve_attr_path(model, path)
        if mod is not None and isinstance(mod, torch.nn.Module):
            return mod
    tried = ", ".join(_FINAL_NORM_PATHS)
    raise AttributeError(
        f"Could not locate the final norm module on {type(model).__name__}; tried: {tried}"
    )


class NonFiniteActivationsError(FloatingPointError):
    """The model produced NaN or inf hidden states, typically an overflow of half precision."""


def _model_dtype(model) -> Optional[torch.dtype]:
    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return next(model.parameters()).dtype
    except Exception:
        return None


def check_finite_hidden_states(hs_layers: Sequence[torch.Tensor], model,
                               labels: Optional[Sequence[Any]] = None) -> None:
    """Raise :class:`NonFiniteActivationsError` if any of the layers read holds NaN or inf
    (``labels`` names them; by default 0, 1, ...).

    Nothing downstream can tell such a layer from a real one: its directions, scores and d come
    out NaN and the layer sweep would skip it in silence. Only the layers read are checked, so
    finite layers can be read from a model whose later layers overflow. The message names the
    model's dtype, since the usual cause is half precision overflowing (Gemma 3 in float16).
    """
    names = list(range(len(hs_layers))) if labels is None else list(labels)
    bad = [names[l] for l, h in enumerate(hs_layers) if not bool(torch.isfinite(h).all())]
    if not bad:
        return
    where = f"layers {bad}" if len(bad) <= 8 else f"{len(bad)} layers ({bad[0]} to {bad[-1]})"
    quantized = getattr(getattr(model, "config", None), "quantization_config", None) is not None
    raise NonFiniteActivationsError(
        f"The model's hidden states are not finite (NaN or inf) at {where} of {len(hs_layers)}; the model "
        f"runs in {_model_dtype(model)}{' (quantized)' if quantized else ''}. This is usually half precision "
        "overflowing (Gemma 3 does in float16): load the model in bfloat16 if the GPU supports it, or in "
        "float32" + (", and use that as the quantization's compute dtype." if quantized else ".")
    )


# --------------------------------------------------------------------------------------- readouts
#: Readout names. ``text_*`` pool over the item's own tokens (a text, or a generated completion);
#: ``sequence_*`` over the input from its start to the end of the item (template tokens included).
#: ``*_last_k`` is the mean of the last ``last_k``; ``*_tokens`` keeps every token as a row.
READ_MODES = (
    "text_mean",
    "text_last",
    "text_last_k",
    "text_first",
    "text_max",
    "text_tokens",
    "sequence_mean",
    "sequence_last",
    "sequence_last_k",
    "sequence_max",
    "sequence_tokens",
)
SPANS = ("text", "sequence", "input")
AGGREGATES = ("mean", "last", "first", "max", "tokens")


class EmptySpanError(ValueError):
    """An item has no tokens of its own to pool: an empty completion or text, or a position or
    substring the item does not have."""


@dataclass(frozen=True)
class Readout:
    """Which tokens of an item to read, and how to reduce them to one vector per layer.

    ``span``: ``"text"``, the item's own tokens (a text, or a generated completion);
    ``"sequence"``, from the start of the input to the end of the item (template tokens included);
    or ``"input"``, the whole input, template tokens after the item too (for a prompt read with
    ``generation=True``, its last token is the one that predicts the reply).
    Within it, optionally, ``last_k`` (the last k tokens), ``position`` (one token, counted from the
    start of the text; negative from its end) or ``substring`` (the tokens that cover that substring
    of the text, its ``occurrence``-th appearance; a list gives one substring per item).
    ``agg``: ``"mean"``, ``"last"``, ``"first"``, ``"max"`` (per coordinate), or ``"tokens"``
    (every token is a row of the result, with the item it comes from).
    """

    span: str = "text"
    agg: str = "mean"
    last_k: Optional[int] = None
    position: Optional[int] = None
    substring: Optional[Union[str, Tuple[str, ...]]] = None
    occurrence: int = 0

    def __post_init__(self):
        if self.span not in SPANS:
            raise ValueError(f"Readout span must be one of {SPANS}; got {self.span!r}")
        if self.agg not in AGGREGATES:
            raise ValueError(f"Readout agg must be one of {AGGREGATES}; got {self.agg!r}")
        narrowings = [x for x in (self.last_k, self.position, self.substring) if x is not None]
        if len(narrowings) > 1:
            raise ValueError("A readout takes at most one of last_k, position and substring.")
        if (self.position is not None or self.substring is not None) and self.span != "text":
            raise ValueError("position and substring readouts read within the text: use span='text'.")
        if self.last_k is not None and int(self.last_k) < 1:
            raise ValueError(f"last_k must be >= 1; got {self.last_k}")
        if isinstance(self.substring, list):
            object.__setattr__(self, "substring", tuple(self.substring))

    @classmethod
    def parse(cls, mode: Union[str, "Readout"], last_k: int = 12) -> "Readout":
        """A readout from a name of :data:`READ_MODES` (``last_k`` applies to ``*_last_k``)."""
        if isinstance(mode, Readout):
            return mode
        if mode in RENAMED_READOUTS:
            raise ValueError(f"The readout {mode!r} is now called {RENAMED_READOUTS[mode]!r}.")
        if mode not in READ_MODES:
            raise ValueError(f"Unknown readout {mode!r}; expected one of {READ_MODES} or a Readout.")
        span, rest = mode.split("_", 1)
        if rest == "last_k":
            return cls(span, "mean", last_k=int(last_k))
        return cls(span, rest)

    @property
    def name(self) -> str:
        """The :data:`READ_MODES` name of this readout, or a description of it."""
        if self.span != "input" and self.position is None and self.substring is None:
            if self.last_k is not None and self.agg == "mean":
                return f"{self.span}_last_k"
            if self.last_k is None and f"{self.span}_{self.agg}" in READ_MODES:
                return f"{self.span}_{self.agg}"
        parts = [self.span]
        if self.last_k is not None:
            parts.append(f"last_{self.last_k}")
        if self.position is not None:
            parts.append(f"position_{self.position}")
        if self.substring is not None:
            parts.append("substring" if isinstance(self.substring, tuple) else f"substring_{self.substring}")
        return ":".join(parts + [self.agg])

    def key(self, item: int) -> str:
        """What identifies this readout of item ``item`` (for the cache)."""
        sub = self.substring[item] if isinstance(self.substring, tuple) else self.substring
        return json.dumps([self.span, self.agg, self.last_k, self.position, sub, self.occurrence])

    def token_indices(self, total: int, start: Optional[int], end: Optional[int], item: int = 0,
                      text: Optional[str] = None, offsets: Optional[Sequence[Tuple[int, int]]] = None) -> List[int]:
        """Positions of the tokens to reduce, in an input of ``total`` tokens whose item's own
        tokens are ``[start, end)``. Raises :class:`EmptySpanError` when there are none."""
        end = total if end is None else min(int(end), total)
        if self.span == "input":
            lo, hi = 0, total
        elif self.span == "sequence":
            lo, hi = 0, end
        else:
            if start is None:
                raise ValueError(
                    f"readout {self.name} pools over a text or completion span, but its start is unknown; "
                    "use a sequence_* readout to read the whole sequence."
                )
            lo, hi = int(start), end
            if lo >= hi:
                raise EmptySpanError(f"readout {self.name}: no tokens to pool (span [{lo}, {hi}) of a {total}-token input)")
        if lo >= hi:
            raise EmptySpanError(f"readout {self.name}: no tokens to pool (an empty input)")
        if self.last_k is not None:
            lo = max(lo, hi - int(self.last_k))
        if self.position is not None:
            n = hi - lo
            p = int(self.position)
            if not -n <= p < n:
                raise EmptySpanError(f"readout {self.name}: the text has {n} tokens, no position {p}")
            idx = lo + (p if p >= 0 else n + p)
            return [idx]
        if self.substring is not None:
            sub = self.substring[item] if isinstance(self.substring, tuple) else self.substring
            if text is None or offsets is None:
                raise ValueError(f"readout {self.name} needs the text's character offsets, which this input lacks "
                                 "(a generated completion, or a tokenizer without offsets).")
            at = -1
            for _ in range(int(self.occurrence) + 1):
                at = text.find(sub, at + 1)
                if at < 0:
                    raise EmptySpanError(f"readout {self.name}: {sub!r} does not occur "
                                         f"{int(self.occurrence) + 1} time(s) in {text!r}")
            stop = at + len(sub)
            hit = [lo + j for j, (a, b) in enumerate(offsets) if b > a and a < stop and b > at and lo + j < hi]
            if not hit:
                raise EmptySpanError(f"readout {self.name}: no token covers {sub!r} in {text!r}")
            return hit
        return list(range(lo, hi))


def _reduce(stacked: torch.Tensor, idx: List[int], agg: str) -> torch.Tensor:
    """``stacked`` is ``(layers, seq_len, dim)``; returns ``(layers, dim)``, or ``(n, layers, dim)``
    for ``agg="tokens"``."""
    sel = stacked[:, idx, :]
    if agg == "mean":
        return sel.mean(dim=1)
    if agg == "max":
        return sel.max(dim=1).values
    if agg == "first":
        return sel[:, 0, :]
    if agg == "last":
        return sel[:, -1, :]
    return sel.permute(1, 0, 2)


def reps_from_hs_layers(
    hs_layers: List[torch.Tensor],
    prompt_len: Optional[int],
    mode: Union[str, "Readout"],
    last_k: int,
    span_end: Optional[int] = None,
) -> np.ndarray:
    """Pool each layer's hidden states ``(seq_len, dim)`` into one rep per layer, with readout
    ``mode`` (see :class:`Readout`); the item's own tokens are ``[prompt_len, span_end)``.

    ``text_*`` readouts pool over that span, ``sequence_*`` readouts from the start of the
    input to ``span_end`` (template tokens included). An empty span raises
    :class:`EmptySpanError` rather than pooling something else (the prompt and system prompt
    would stand in for the missing text).
    """
    readout = Readout.parse(mode, last_k)
    stacked = torch.stack([h.float() for h in hs_layers])
    idx = readout.token_indices(int(stacked.shape[1]), prompt_len, span_end)
    return _reduce(stacked, idx, readout.agg).numpy().astype(np.float32)


def _span_bounds(total: int, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """``[lo, hi)`` of an item's own tokens; position 0 (the BOS) is left out when the span is unknown."""
    lo = 1 if start is None else int(start)
    hi = total if end is None else min(int(end), total)
    if lo >= hi:
        lo, hi = 0, total
    return lo, hi


# --------------------------------------------------------------------------------- forward passes
def _forward_kwargs(model) -> Dict[str, Any]:
    """Ask for the last position's logits only, where the model allows it: reading needs no logits,
    and a batch's full logits (batch x tokens x vocabulary) can take more memory than the model."""
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    if "logits_to_keep" in params:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in params:
        return {"num_logits_to_keep": 1}
    return {}


@torch.no_grad()
def _forward_batch(model, ids_list: Sequence[torch.Tensor], pad_id: int) -> Tuple[List[torch.Tensor], List[int]]:
    """One forward pass over several inputs, right-padded: the hidden states of every layer,
    embeddings first, each ``(batch, seq_len, dim)``, and the inputs' lengths.

    Right padding leaves every real token where it would be alone (positions from 0, and causal
    attention never reaches the padding after it). The last entry is the input of the final norm,
    on the scale of the other layers (Hugging Face returns it after the norm). Under an
    intervention (:func:`concept_probe.intervene`), an edited block's entry is its output after the
    edit, which is what the next block receives.
    """
    lens = [int(x.shape[-1]) for x in ids_list]
    width = max(lens)
    input_ids = torch.full((len(ids_list), width), int(pad_id), dtype=torch.long)
    attn = torch.zeros((len(ids_list), width), dtype=torch.long)
    for j, ids in enumerate(ids_list):
        input_ids[j, : lens[j]] = ids.reshape(-1).to(torch.long)
        attn[j, : lens[j]] = 1
    captured: Dict[str, torch.Tensor] = {}

    def capture(module, args):
        captured["pre_norm"] = (args[0] if isinstance(args, tuple) else args).detach()

    handle = _get_final_norm_module(model).register_forward_pre_hook(capture)
    try:
        with edited_outputs(model) as edited, safe_attention(model):
            out = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                        output_hidden_states=True, use_cache=False, return_dict=True, **_forward_kwargs(model))
    finally:
        handle.remove()
    if "pre_norm" not in captured:
        raise RuntimeError("Forward pre-hook on final norm did not fire; model layout is unexpected.")
    hs = list(out.hidden_states)
    for layer, tensor in edited.items():
        hs[layer + 1] = tensor.detach()
    hs[-1] = captured["pre_norm"]
    return hs, lens


# -------------------------------------------------------------------------------------- the cache
#: Bump when reading changes what a cached entry would hold, so old entries are not reused.
READ_VERSION = 1


class RepCache:
    """Representations on disk, so an input goes through a model once (entry 25 of
    LIBRARY_ISSUES). Entries are keyed by the model's identity (model, revision, dtype,
    quantization, chat template), the exact token ids, the item's span and the readout.

    One SQLite file per model identity (``<root>/<identity hash>/reps.sqlite``, next to an
    ``identity.json`` that says which model it is) rather than one file per input: opening
    thousands of small files is slow (about 25 ms each on Windows, more than reading a short text
    with a small model). SQLite comes with Python and handles several processes at once.

    ``dtype="float32"`` stores what reading computed, so a cached read equals a fresh one;
    ``"float16"`` halves the size and rounds to float16, falling back to float32 for an entry with
    a value float16 cannot hold. The rounding is not always negligible: models with large
    activations (Gemma 3's reach tens of thousands, rounded by whole units) can read differently;
    with Gemma 3 1B it moved a probe's d by about 1% (2.70 read, 2.73 from float16).
    """

    def __init__(self, root: str, identity: Dict[str, Any], dtype: str = "float32"):
        if dtype not in ("float32", "float16"):
            raise ValueError(f"cache dtype must be 'float32' or 'float16'; got {dtype!r}")
        self.root, self.dtype = root, dtype
        ident = json.dumps({"read_version": READ_VERSION, **identity}, sort_keys=True, default=str)
        self.dir = os.path.join(root, hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16])
        os.makedirs(self.dir, exist_ok=True)
        meta = os.path.join(self.dir, "identity.json")
        if not os.path.exists(meta):
            with open(meta, "w", encoding="utf-8") as f:
                f.write(ident)
        self.path = os.path.join(self.dir, "reps.sqlite")
        self._db = sqlite3.connect(self.path, timeout=120)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, data BLOB NOT NULL)")
        self._db.commit()
        self.hits = self.misses = 0
        self.fallbacks: List[str] = []

    def key(self, enc: Encoded, readout: Readout, item: int) -> str:
        h = hashlib.sha1()
        h.update(np.asarray(enc.input_ids.reshape(-1).tolist(), dtype=np.int64).tobytes())
        h.update(json.dumps([enc.start, enc.end, readout.key(item), self.dtype]).encode("utf-8"))
        return h.hexdigest()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        row = self._db.execute("SELECT data FROM entries WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        with np.load(io.BytesIO(row[0])) as data:
            return {k: (data[k].astype(np.float32) if data[k].dtype == np.float16 else data[k]) for k in data.files}

    def put(self, key: str, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Store ``arrays`` and return them as stored (rounded, with ``dtype="float16"``). Call
        :meth:`commit` to make the writes durable."""
        stored = dict(arrays)
        if self.dtype == "float16":
            reps = stored["reps"]
            if np.all(np.isfinite(reps)) and float(np.abs(reps).max(initial=0.0)) < 65504.0:
                stored["reps"] = reps.astype(np.float16)
            else:
                self.fallbacks.append(key)
        buf = io.BytesIO()
        np.savez(buf, **stored)
        self._db.execute("INSERT OR REPLACE INTO entries (key, data) VALUES (?, ?)", (key, buf.getvalue()))
        return {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in stored.items()}

    def commit(self) -> None:
        self._db.commit()

    def close(self) -> None:
        self._db.commit()
        self._db.close()


def _cache_identity(model, tokenizer, identity: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ident = dict(identity) if identity else identity_of(model, tokenizer)
    template = getattr(tokenizer, "chat_template", None)
    ident["chat_template_sha1"] = None if not template else hashlib.sha1(str(template).encode("utf-8")).hexdigest()
    return ident


def _open_cache(model, tokenizer, cache: Any, identity: Optional[Dict[str, Any]], dtype: str,
                choices: List[Choice]) -> Optional[RepCache]:
    """The cache to use: ``cache`` is False (none), a directory, or None (the session default, on
    when the session saves; see :mod:`concept_probe.settings`)."""
    if cache is False:
        return None
    if cache is None or cache is True:
        if cache is None and not settings().save:
            return None
        root = default_rep_cache_dir()
    else:
        root = str(cache)
    if active_interventions(model):
        choices.append(Choice("representations cache", root, None,
                              "an intervention is applied to the model, so its representations are neither "
                              "taken from the cache nor stored in it"))
        return None
    ident = _cache_identity(model, tokenizer, identity)
    if not ident.get("model_id"):
        choices.append(Choice("representations cache", root, None,
                              "the model has no name to key cached representations by, so nothing is cached"))
        return None
    choices.append(Choice("representations cache", "save by default", root,
                          "reading stores what it reads and reuses it; turn it off with cache=False, or "
                          "output.rep_cache=false in a workspace"))
    return RepCache(root, ident, dtype)


# ---------------------------------------------------------------------------------------- reading
@dataclass
class Reps:
    """Representations read from a model: ``data`` is ``(rows, layers, dim)``, one row per item,
    or per token with a ``tokens`` readout. ``layers`` names axis 1 (block indices, and
    ``"embeddings"`` for the embedding output); ``items`` is the input each row comes from, and
    ``positions`` the token position of each row (``tokens`` readouts only). Items that could not
    be read (an empty span, a missing substring) have no rows and are listed in ``skipped`` with
    the reason. ``norms`` (items, layers) and ``projections`` (per item, tokens x layers) are
    there when asked for, for the items that were read. ``identity`` is the model they were read
    with (see :func:`concept_probe.modeling.identity_of`): a probe trained on them takes it.
    ``context`` says how the items were read (see :class:`concept_probe.ReadContext`), when the
    reader knows: probes calibrate on it and check it when they score."""

    data: np.ndarray
    layers: List[Union[int, str]]
    readout: Readout
    items: np.ndarray
    positions: Optional[np.ndarray] = None
    norms: Optional[np.ndarray] = None
    projections: Optional[List[np.ndarray]] = None
    skipped: Dict[int, str] = field(default_factory=dict)
    batch_size: int = 1
    cache_dir: Optional[str] = None
    identity: Optional[Dict[str, Any]] = None
    from_cache: int = 0
    choices: List[Choice] = field(default_factory=list)
    context: Optional[ReadContext] = None

    @property
    def read_items(self) -> List[int]:
        """The items that have rows, in order."""
        return [int(i) for i in dict.fromkeys(self.items.tolist())]


def _layer_rows(layers: Optional[Sequence[Union[int, str]]], n_blocks: int) -> Tuple[List[int], List[Union[int, str]]]:
    """Rows of the stacked hidden states (0 = embeddings, l + 1 = block l) for ``layers``, and
    their labels. None means every block (no embeddings)."""
    if layers is None:
        return [l + 1 for l in range(n_blocks)], list(range(n_blocks))
    rows, labels = [], []
    for value in layers:
        if value == "embeddings":
            rows.append(0)
            labels.append("embeddings")
        else:
            l = resolve_layer_index(value, n_blocks, "layers entries")
            rows.append(l + 1)
            labels.append(l)
    if len(set(rows)) != len(rows):
        raise ValueError(f"layers repeats a layer: {labels}")
    return rows, labels


def _read(
    model,
    tokenizer,
    encodings: Sequence[Encoded],
    readouts: Sequence[Readout],
    *,
    layers: Optional[Sequence[Union[int, str]]] = None,
    batch_size: int = 16,
    norms: bool = False,
    project_on: Optional[np.ndarray] = None,
    cache: Any = None,
    cache_dtype: str = "float32",
    identity: Optional[Dict[str, Any]] = None,
    skip_empty: bool = False,
    label: str = "items",
) -> List[Reps]:
    """Read ``encodings`` with each of ``readouts`` from one forward pass per batch.

    ``norms`` adds each item's mean residual norm per block over its own tokens, and
    ``project_on`` (one vector per block) each of those tokens' projection on its block's vector.
    """
    n_blocks = len(get_transformer_layers(model))
    rows, labels = _layer_rows(layers, n_blocks)
    choices: List[Choice] = []
    store = _open_cache(model, tokenizer, cache, identity, cache_dtype, choices)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    vectors = None if project_on is None else torch.as_tensor(np.asarray(project_on), dtype=torch.float64)
    n = len(encodings)
    found: List[List[Optional[Dict[str, np.ndarray]]]] = [[None] * n for _ in readouts]
    errors: List[Dict[int, str]] = [{} for _ in readouts]
    extra: Dict[int, Dict[str, np.ndarray]] = {}

    todo = []
    for i, enc in enumerate(encodings):
        if store is not None and vectors is None:
            hits = [store.get(store.key(enc, r, i)) for r in readouts]
            if all(h is not None for h in hits) and (not norms or all("norms" in h for h in hits)):
                for k, h in enumerate(hits):
                    found[k][i] = h
                continue
        todo.append(i)

    step = max(1, int(batch_size))
    for b in range(0, len(todo), step):
        batch = todo[b:b + step]
        hs, lens = _forward_batch(model, [encodings[i].input_ids for i in batch], pad_id)
        home = hs[0].device
        for j, i in enumerate(batch):
            enc = encodings[i]
            stacked = torch.stack([h[j, : lens[j]].to(home) for h in hs]).float()  # (blocks + 1, seq, dim)
            check_finite_hidden_states([stacked[r] for r in rows], model, labels)
            total = int(stacked.shape[1])
            lo, hi = _span_bounds(total, enc.start, enc.end)
            item_norms = stacked[1:, lo:hi].norm(dim=-1).mean(dim=1).double().cpu().numpy()
            if vectors is not None:
                proj = torch.einsum("ltd,ld->tl", stacked[1:, lo:hi].double(), vectors.to(stacked.device))
                extra[i] = {"projections": proj.cpu().numpy()}
            for k, r in enumerate(readouts):
                try:
                    idx = r.token_indices(total, enc.start, enc.end, i, enc.text, enc.offsets)
                except EmptySpanError as exc:
                    errors[k][i] = str(exc)
                    continue
                entry = {"reps": _reduce(stacked, idx, r.agg).cpu().numpy().astype(np.float32),
                         "norms": item_norms}
                if r.agg == "tokens":
                    entry["positions"] = np.asarray(idx, dtype=np.int64)
                if store is not None:
                    entry = store.put(store.key(enc, r, i), entry)
                found[k][i] = entry
            del stacked
        del hs
        if store is not None:
            store.commit()

    if store is not None:
        store.close()
        if store.fallbacks:
            choices.append(Choice("cache dtype", "float16", "float32",
                                  f"{len(store.fallbacks)} entries hold values float16 cannot represent"))
    out = []
    for k, r in enumerate(readouts):
        if errors[k] and not skip_empty:
            first = min(errors[k])
            raise EmptySpanError(f"{label}[{first}]: {errors[k][first]}")
        kept = [i for i in range(n) if found[k][i] is not None]
        if r.agg == "tokens":
            parts = [found[k][i]["reps"][:, rows, :] for i in kept]
            data = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(rows), 0), np.float32)
            items = np.concatenate([np.full(len(p), i) for p, i in zip(parts, kept)]) if parts else np.zeros(0, int)
            positions = np.concatenate([found[k][i]["positions"] for i in kept]) if parts else np.zeros(0, int)
        else:
            data = (np.stack([found[k][i]["reps"][rows] for i in kept]).astype(np.float32) if kept
                    else np.zeros((0, len(rows), 0), np.float32))
            items, positions = np.asarray(kept, dtype=np.int64), None
        out.append(Reps(
            data=data, layers=labels, readout=r, items=items, positions=positions,
            norms=(np.stack([found[k][i]["norms"] for i in kept])[:, [r - 1 for r in rows if r > 0]]
                   if (norms and kept) else None),
            projections=[extra[i]["projections"] for i in kept] if vectors is not None else None,
            skipped=dict(errors[k]), batch_size=step, cache_dir=None if store is None else store.dir,
            identity=dict(identity) if identity else identity_of(model, tokenizer),
            from_cache=0 if store is None else store.hits // max(1, len(readouts)), choices=choices,
        ))
    return out


def read_encoded(
    model,
    tokenizer,
    encodings: Sequence[Encoded],
    readout: Union[str, Readout, Sequence[Union[str, Readout]]] = "text_mean",
    *,
    context: Optional[ReadContext] = None,
    last_k: int = 12,
    layers: Optional[Sequence[Union[int, str]]] = None,
    batch_size: int = 16,
    norms: bool = False,
    cache: Any = None,
    cache_dtype: str = "float32",
    skip_empty: bool = False,
    console: Optional[bool] = None,
) -> Union[Reps, Dict[str, Reps]]:
    """Read already-encoded inputs (see :class:`concept_probe.inputs.Encoded`), such as generated
    completions given by their token ids (:meth:`concept_probe.Generations.encodings`), with one
    ``readout`` or several; ``context`` says how they were encoded, if it matters (see
    :func:`read_texts`)."""
    names, readouts = _readouts(readout, last_k)
    out = _read(model, tokenizer, encodings, readouts, layers=layers, batch_size=batch_size, norms=norms,
                cache=cache, cache_dtype=cache_dtype, skip_empty=skip_empty)
    for r in out:
        r.context = None if context is None else replace(context, readout=r.readout.name)
    return _one_or_many(names, out, console)


def read_texts(
    model,
    tokenizer,
    texts: Sequence[PromptLike],
    readout: Union[str, Readout, Sequence[Union[str, Readout]]] = "text_mean",
    *,
    role: str = "raw",
    system: Optional[str] = None,
    prompt: Union[None, str, Sequence[str]] = None,
    generation: bool = False,
    last_k: int = 12,
    layers: Optional[Sequence[Union[int, str]]] = None,
    batch_size: int = 16,
    norms: bool = False,
    cache: Any = None,
    cache_dtype: str = "float32",
    skip_empty: bool = False,
    console: Optional[bool] = None,
) -> Union[Reps, Dict[str, Reps]]:
    """Read texts with ``model`` and return their representations (layer 1 of docs/DESIGN.md).

    Each text is encoded as a text to read (``role``, ``system``, ``prompt``; see
    :func:`concept_probe.inputs.encode_text` and :func:`preview_input`): by default alone (the BOS
    and the text); ``role="user"`` as a user turn; ``role="assistant"`` as the model's reply, after
    ``system`` and a user turn ``prompt``, which may be one per text (to read generated completions
    as replies to their own prompts). With ``generation=True`` a text is a prompt to generate from,
    exactly as :func:`concept_probe.generate` and :func:`concept_probe.answer_logprobs` feed it (a
    user turn and the assistant header; ``Readout("input", ...)`` reaches the token that predicts
    the reply). Whatever ``role`` would leave out (a system prompt with ``role="raw"``, a template
    the tokenizer lacks, a BOS it does not have) is warned about.

    The inputs go through the model ``batch_size`` at a time, and ``readout`` (a :data:`READ_MODES`
    name or a :class:`Readout`) reduces each text's tokens at every layer in ``layers`` (default:
    every block; add ``"embeddings"`` for the embedding output). A list of readouts reads them all
    from the same forward passes and returns ``{name: Reps}`` (the names as given, or each
    :class:`Readout`'s ``name``). ``norms=True`` adds each text's mean residual norm per layer
    (``Reps.norms``), which a probe fitted on them uses as the unit of ``unit="resid_norm"``.
    The result's ``context`` records how the texts were read. Batching changes the order of
    floating-point operations, so in half precision the results can differ from ``batch_size=1`` in
    the last digits.

    Reading saves what it reads by default (decision D4): ``cache=None`` uses the session's cache
    (``<output_root>/rep_cache``, on while the session saves), a path uses that directory, and
    ``False`` turns it off. An entry is reused only for the same model identity, token ids, span and
    readout. Automatic choices (where the cache is, a float16 fallback) are announced and listed in
    the result's ``choices``.
    """
    texts = list(texts)
    per_text = prompt is not None and not isinstance(prompt, str)
    prompts = list(prompt) if per_text else [prompt] * len(texts)
    if len(prompts) != len(texts):
        raise ValueError(f"prompt has {len(prompts)} turns for {len(texts)} texts")
    encodings = [encode_text(tokenizer, t, system, role=role, prompt=p, generation=generation, warn=_warn,
                             warn_prefix=f"texts[{i}]") for i, (t, p) in enumerate(zip(texts, prompts))]
    names, readouts = _readouts(readout, last_k)
    out = _read(model, tokenizer, encodings, readouts, layers=layers, batch_size=batch_size, norms=norms,
                cache=cache, cache_dtype=cache_dtype, skip_empty=skip_empty, label="texts")
    for r in out:
        if per_text:
            if role != "assistant":
                raise ValueError("one prompt per text is a user turn before each text: it needs role='assistant'")
            r.context = ReadContext.generated(r.readout.name, system)
        else:
            r.context = ReadContext(kind="text", role="user" if generation else role,
                                    system=None if role == "raw" and not generation else system,
                                    prompt=prompt if role == "assistant" else None, readout=r.readout.name)
    return _one_or_many(names, out, console)


def _warn(message: str) -> None:
    import warnings

    warnings.warn(message, stacklevel=3)


def _readouts(readout: Any, last_k: int) -> Tuple[Optional[List[str]], List[Readout]]:
    """``(names, readouts)``: names is None for a single readout, else one per readout of a list."""
    if isinstance(readout, (str, Readout)):
        return None, [Readout.parse(readout, last_k)]
    parsed = [Readout.parse(r, last_k) for r in readout]
    if not parsed:
        raise ValueError("readout is an empty list; give a readout or several.")
    names = [r if isinstance(r, str) else p.name for r, p in zip(readout, parsed)]
    if len(set(names)) != len(names):
        raise ValueError(f"the readouts need distinct names; got {names}")
    return names, parsed


def _one_or_many(names: Optional[List[str]], out: List[Reps], console: Optional[bool]) -> Union[Reps, Dict[str, Reps]]:
    announce(out[0].choices, console=console)  # one read: the same choices for every readout
    return out[0] if names is None else dict(zip(names, out))
