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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .inputs import (
    Encoded,
    PromptLike,
    _completion_input_ids,
    _normalize_messages,
    encode_text,
    token_char_offsets,
)
from .modeling import _resolve_attr_path, attention_mask_from_ids, get_transformer_layers, identity_of
from .record import Choice, announce
from .settings import default_rep_cache_dir, settings
from .utils import resolve_layer_index


def _slice_generation_step_logits(
    full_logits: torch.Tensor,
    *,
    prompt_len: int,
    total_tokens: int,
) -> torch.Tensor:
    if full_logits.ndim != 2:
        raise ValueError(f"Expected full logits shape (seq_len, vocab_size); got {tuple(full_logits.shape)}")

    generated_steps = max(0, int(total_tokens) - int(prompt_len))
    if generated_steps <= 0:
        return full_logits.new_empty((0, int(full_logits.shape[-1])))

    start = 0 if int(prompt_len) <= 0 else int(prompt_len) - 1
    end = start + generated_steps
    if end > int(full_logits.shape[0]):
        raise ValueError(
            f"Cannot slice generation logits: prompt_len={prompt_len}, total_tokens={total_tokens}, logits_seq_len={int(full_logits.shape[0])}"
        )
    return full_logits[start:end]


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


def check_finite_hidden_states(hs_layers: Sequence[torch.Tensor], model) -> None:
    """Raise :class:`NonFiniteActivationsError` if any layer holds NaN or inf.

    Nothing downstream can tell such a layer from a real one: its directions, scores and d come
    out NaN and the layer sweep would skip it in silence. The message names the model's dtype,
    since the usual cause is half precision overflowing (Gemma 3 in float16, for one).
    """
    bad = [l for l, h in enumerate(hs_layers) if not bool(torch.isfinite(h).all())]
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


@torch.no_grad()
def forward_hidden_states_all_layers(
    model,
    input_ids: torch.Tensor,
    *,
    return_generation_logits: bool = False,
    generation_prompt_len: Optional[int] = None,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
    """Per-layer hidden states ``(seq_len, hidden_dim)`` (CPU float32) of one input: entry ``l`` is
    the output of block ``l``. Raises :class:`NonFiniteActivationsError` on NaN or inf."""
    attn = attention_mask_from_ids(input_ids).to(model.device)

    final_norm = _get_final_norm_module(model)
    captured: Dict[str, torch.Tensor] = {}

    def _capture_pre_norm(module, args):
        x = args[0] if isinstance(args, tuple) else args
        captured["pre_norm"] = x.detach()

    handle = final_norm.register_forward_pre_hook(_capture_pre_norm)
    try:
        out = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()

    if "pre_norm" not in captured:
        raise RuntimeError(
            "Forward pre-hook on final norm did not fire; model layout is unexpected."
        )

    hs = list(out.hidden_states)
    # HF returns the post-final-norm tensor as hidden_states[-1] for decoder-only LMs,
    # collapsing the last layer to a much smaller scale than the rest of the residual
    # stream. Replace it with the pre-norm input so all layers are on the same scale.
    hs[-1] = captured["pre_norm"]

    num_layers = len(hs) - 1
    hs_layers = [hs[l + 1][0].detach().float().cpu() for l in range(num_layers)]
    check_finite_hidden_states(hs_layers, model)
    if not return_generation_logits:
        return hs_layers

    logits = getattr(out, "logits", None)
    if logits is None:
        raise ValueError("Model output does not include logits; cannot save generation logits.")
    logits_2d = logits[0]
    seq_len = int(input_ids.shape[-1])
    prompt_len = seq_len if generation_prompt_len is None else int(generation_prompt_len)
    step_logits = _slice_generation_step_logits(
        logits_2d,
        prompt_len=prompt_len,
        total_tokens=seq_len,
    )
    return hs_layers, step_logits.detach().to(dtype=torch.float32, device="cpu")


# --------------------------------------------------------------------------------------- readouts
#: Readout names, as the config takes them. ``assistant_*`` pool over the item's own tokens (a text
#: or a generated completion); ``sequence_*`` from the start of the input to the end of the item.
#: ``*_tokens`` keeps every token as a row instead of pooling.
READ_MODES = (
    "sequence_last",
    "sequence_last_k_mean",
    "sequence_all_mean",
    "sequence_max",
    "assistant_last",
    "assistant_last_k_mean",
    "assistant_all_mean",
    "assistant_max",
    "assistant_first",
    "assistant_tokens",
)
SPANS = ("text", "sequence")
AGGREGATES = ("mean", "last", "first", "max", "tokens")


class EmptySpanError(ValueError):
    """An item has no tokens of its own to pool: an empty completion or text, or a position or
    substring the item does not have."""


@dataclass(frozen=True)
class Readout:
    """Which tokens of an item to read, and how to reduce them to one vector per layer.

    ``span``: ``"text"``, the item's own tokens (a text, or a generated completion), or
    ``"sequence"``, from the start of the input to the end of the item (template tokens included).
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
        """A readout from a name of :data:`READ_MODES` (``last_k`` applies to ``*_last_k_mean``)."""
        if isinstance(mode, Readout):
            return mode
        if mode not in READ_MODES:
            raise ValueError(f"Unknown readout mode {mode!r}; expected one of {READ_MODES} or a Readout.")
        span, rest = mode.split("_", 1)
        span = "text" if span == "assistant" else "sequence"
        if rest == "last_k_mean":
            return cls(span, "mean", last_k=int(last_k))
        return cls(span, {"all_mean": "mean", "last": "last", "first": "first", "max": "max",
                          "tokens": "tokens"}[rest])

    @property
    def name(self) -> str:
        """The :data:`READ_MODES` name of this readout, or a description of it."""
        prefix = "assistant" if self.span == "text" else "sequence"
        if self.position is None and self.substring is None:
            if self.last_k is not None and self.agg == "mean":
                return f"{prefix}_last_k_mean"
            if self.last_k is None:
                legacy = f"{prefix}_{'all_mean' if self.agg == 'mean' else self.agg}"
                if legacy in READ_MODES:
                    return legacy
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
        if self.span == "sequence":
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

    ``assistant_*`` readouts pool over that span, ``sequence_*`` readouts from the start of the
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
    on the scale of the other layers (Hugging Face returns it after the norm).
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
        out = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                    output_hidden_states=True, use_cache=False, return_dict=True, **_forward_kwargs(model))
    finally:
        handle.remove()
    if "pre_norm" not in captured:
        raise RuntimeError("Forward pre-hook on final norm did not fire; model layout is unexpected.")
    hs = list(out.hidden_states)
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
    a value float16 cannot hold.
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
    with (see :func:`concept_probe.modeling.identity_of`): a probe trained on them takes it."""

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
            check_finite_hidden_states(list(stacked[1:]), model)
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
            norms=np.stack([found[k][i]["norms"] for i in kept]) if (norms and kept) else None,
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
    readout: Union[str, Readout] = "assistant_all_mean",
    *,
    last_k: int = 12,
    layers: Optional[Sequence[Union[int, str]]] = None,
    batch_size: int = 16,
    cache: Any = None,
    cache_dtype: str = "float32",
    skip_empty: bool = False,
    console: Optional[bool] = None,
) -> Reps:
    """Read already-encoded inputs (see :class:`concept_probe.inputs.Encoded`), such as generated
    completions given by their token ids, with one ``readout``. See :func:`read_texts`."""
    reps = _read(model, tokenizer, encodings, [Readout.parse(readout, last_k)], layers=layers,
                 batch_size=batch_size, cache=cache, cache_dtype=cache_dtype, skip_empty=skip_empty)[0]
    announce(reps.choices, console=console)
    return reps


def read_texts(
    model,
    tokenizer,
    texts: Sequence[PromptLike],
    readout: Union[str, Readout] = "assistant_all_mean",
    *,
    role: str = "raw",
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    last_k: int = 12,
    layers: Optional[Sequence[Union[int, str]]] = None,
    batch_size: int = 16,
    cache: Any = None,
    cache_dtype: str = "float32",
    skip_empty: bool = False,
    console: Optional[bool] = None,
) -> Reps:
    """Read texts with ``model`` and return their representations (layer 1 of docs/DESIGN.md).

    Each text is encoded as training encodes texts (``role``, ``system``, ``prompt``; see
    :func:`concept_probe.inputs.encode_text` and :func:`preview_input`), the inputs go through the
    model ``batch_size`` at a time, and ``readout`` (a :data:`READ_MODES` name or a
    :class:`Readout`) reduces each text's tokens at every layer in ``layers`` (default: every block;
    add ``"embeddings"`` for the embedding output). Batching changes the order of floating-point
    operations, so in half precision the results can differ from ``batch_size=1`` in the last
    digits.

    Reading saves what it reads by default (decision D4): ``cache=None`` uses the session's cache
    (``<output_root>/rep_cache``, on while the session saves), a path uses that directory, and
    ``False`` turns it off. An entry is reused only for the same model identity, token ids, span and
    readout. Automatic choices (where the cache is, a float16 fallback) are announced and listed in
    the result's ``choices``.
    """
    encodings = [encode_text(tokenizer, t, system, role=role, prompt=prompt, warn_prefix=f"texts[{i}]")
                 for i, t in enumerate(texts)]
    reps = _read(model, tokenizer, encodings, [Readout.parse(readout, last_k)], layers=layers,
                 batch_size=batch_size, cache=cache, cache_dtype=cache_dtype, skip_empty=skip_empty,
                 label="texts")[0]
    announce(reps.choices, console=console)
    return reps


@torch.no_grad()
def extract_completion_reps(
    model,
    tokenizer,
    pairs: List[Tuple[PromptLike, str]],
    *,
    pool: str = "assistant_all_mean",
    last_k: int = 12,
    system: Optional[str] = None,
    batch_size: int = 16,
    cache: Any = None,
) -> np.ndarray:
    """Per-layer reps pooled over the completion tokens, for ``(prompt, completion)`` pairs.

    Reconstructs the readout used during training: each prompt is chat-formatted with an
    assistant generation prompt, the completion tokens are appended, and the hidden states are
    pooled over the completion span per layer (``pool='assistant_all_mean'`` by default). Returns
    an array of shape ``(len(pairs), num_layers, hidden_dim)``.

    Use it to score externally-provided text -- e.g. another model's generations -- against
    trained concept vectors, or to compare how different models represent the same text.
    Prompts follow the rules of every other path: a string is a user turn, a message list is used
    as given, and ``system`` (default None: no system message) is added unless the list has its own.
    The completion is tokenized from its text, so its tokens can differ slightly from the ids a
    generation produced. Reading goes ``batch_size`` pairs at a time and uses the cache as
    :func:`read_texts` does.
    """
    encodings = []
    for prompt, completion in pairs:
        messages, _ = _normalize_messages(prompt, system)
        full_ids, plen = _completion_input_ids(tokenizer, messages, completion)
        encodings.append(Encoded(full_ids[0], plen, int(full_ids.shape[-1]), str(completion),
                                 token_char_offsets(tokenizer, str(completion))))
    reps = _read(model, tokenizer, encodings, [Readout.parse(pool, last_k)], batch_size=batch_size,
                 cache=cache, label="pairs")[0]
    announce(reps.choices)
    return reps.data


@torch.no_grad()
def extract_text_reps(
    model,
    tokenizer,
    texts: List[str],
    *,
    role: str = "raw",
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    pool: Union[str, Sequence[str]] = "assistant_all_mean",
    last_k: int = 12,
    batch_size: int = 16,
    cache: Any = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Per-layer reps of texts, encoded exactly as texts-mode training encodes them.

    ``role``, ``prompt`` and ``system`` mean what ``training.train_text_role``,
    ``training.train_text_prompt`` and the training system prompt mean, and the defaults are the
    training defaults: the raw text with no template or system prompt, pooled over its own
    tokens. ``pool`` is one readout mode, returning ``(len(texts), num_layers, hidden_dim)``, or
    several, returning a dict of such arrays from a single forward pass per batch. Use
    :func:`preview_input` to see the exact input, and :func:`read_texts` for the readouts that are
    not a mode name (a position, a substring) and for the layer of embeddings.
    """
    pools = [pool] if isinstance(pool, str) else list(pool)
    encodings = [encode_text(tokenizer, t, system, role=role, prompt=prompt, warn_prefix=f"texts[{i}]")
                 for i, t in enumerate(texts)]
    readouts = [Readout.parse(p, last_k) for p in pools]
    if any(r.agg == "tokens" for r in readouts):
        raise ValueError("extract_text_reps returns one row per text; use read_texts for a tokens readout.")
    results = _read(model, tokenizer, encodings, readouts, batch_size=batch_size, cache=cache, label="texts")
    announce(results[0].choices)
    arrays = {p: r.data for p, r in zip(pools, results)}
    return arrays[pool] if isinstance(pool, str) else arrays
