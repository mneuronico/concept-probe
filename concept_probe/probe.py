import json
import os
import shutil
import sys
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from .config import ConceptSpec, check_config_keys, load_defaults, resolve_config
from .logger import JsonlLogger
from .modeling import ModelBundle, _chat_template_kwargs, _resolve_attr_path, apply_chat, attention_mask_from_ids
from .steering import MultiLayerSteererLayerwise
from .utils import (
    deep_merge,
    ensure_dir,
    json_dump,
    jsonl_to_pretty,
    now_tag,
    resolve_layer_index,
    safe_slug,
    set_seed,
)
from .visuals import plot_score_hist, plot_sweep, render_batch_heatmap, render_token_heatmap, segment_token_scores
from .console import ConsoleLogger

try:
    from scipy.stats import ttest_ind
except Exception:
    ttest_ind = None


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


def _resolve_random_control_seed(explicit_seed: Optional[int], config_seed: Any) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    if config_seed is None:
        return 0
    try:
        return int(config_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"random.seed in config must be an integer; got {config_seed!r}") from exc


def _mix_random_control_seed(base_seed: int, salt: int) -> int:
    x = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    s = int(salt) & 0xFFFFFFFFFFFFFFFF
    x ^= (s + 0x9E3779B97F4A7C15 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)) & 0xFFFFFFFFFFFFFFFF
    return int(x & 0x7FFFFFFFFFFFFFFF)


def _random_vectors_same_norm(base_vectors: np.ndarray, seed: int) -> np.ndarray:
    src = np.asarray(base_vectors, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"base_vectors must be 2D [num_layers, hidden_dim], got shape={src.shape}")
    rng = np.random.default_rng(int(seed))
    out = np.zeros_like(src, dtype=np.float32)
    for li in range(src.shape[0]):
        target_norm = float(np.linalg.norm(src[li]))
        if not np.isfinite(target_norm) or target_norm <= 0.0:
            continue
        rv = rng.standard_normal(src.shape[1]).astype(np.float32)
        rv_norm = float(np.linalg.norm(rv))
        if not np.isfinite(rv_norm) or rv_norm <= 0.0:
            continue
        out[li] = (rv / rv_norm) * target_norm
    return out


def _render_progress(current: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0"
    frac = min(1.0, max(0.0, current / total))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _progress_print(current: int, total: int, *, label: str, enabled: bool, last: bool = False) -> None:
    if not enabled:
        return
    inline_env = os.environ.get("CP_PROGRESS_INLINE", "").strip().lower()
    if inline_env in {"1", "true", "yes"}:
        inline = True
    elif inline_env in {"0", "false", "no"}:
        inline = False
    else:
        inline = sys.stdout.isatty()
    bar = _render_progress(current, total)
    end = "\n" if (last or not inline) else "\r"
    print(f"{label} {bar}", end=end, flush=True)


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


def _welch_p_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-sided Welch t-test p-value of each column (NaN without scipy or with < 2 per group)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if ttest_ind is None or x.shape[0] < 2 or y.shape[0] < 2:
        return np.full(x.shape[1], np.nan)
    return np.asarray(ttest_ind(x, y, axis=0, equal_var=False).pvalue, dtype=np.float64)


def _split_indices(n: int, frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A random split of ``range(n)`` into ``round(frac * n)`` and the rest, each sorted."""
    order = rng.permutation(n)
    k = int(round(float(frac) * n))
    return np.sort(order[:k]), np.sort(order[k:])


def _max_d_permutation_p(
    pos: np.ndarray, neg: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> Tuple[float, float]:
    """Largest Cohen's d over the columns (layers), and its permutation p-value.

    Each permutation shuffles the pos/neg labels and takes the largest d over the same
    columns, so the p-value accounts for having searched over all of them.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    observed = float(np.nanmax(_cohen_d_columns(pos, neg)))
    pooled = np.concatenate([pos, neg], axis=0)
    n_pos = pos.shape[0]
    exceed = 0
    for _ in range(int(n_permutations)):
        order = rng.permutation(pooled.shape[0])
        exceed += np.nanmax(_cohen_d_columns(pooled[order[:n_pos]], pooled[order[n_pos:]])) >= observed
    return observed, (exceed + 1) / (int(n_permutations) + 1)


def _bootstrap_d_ci(
    pos: np.ndarray, neg: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> Optional[List[float]]:
    """95% percentile bootstrap interval of Cohen's d, resampling each group with replacement."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if not n_bootstrap or pos.size < 2 or neg.size < 2:
        return None
    ds = [cohen_d_np(rng.choice(pos, pos.size), rng.choice(neg, neg.size)) for _ in range(int(n_bootstrap))]
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return [float(lo), float(hi)]


def _span_bounds(total: int, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """``[lo, hi)`` of an item's own tokens; position 0 (the BOS) is left out when the span is unknown."""
    lo = 1 if start is None else int(start)
    hi = total if end is None else min(int(end), total)
    if lo >= hi:
        lo, hi = 0, total
    return lo, hi


def _mean_token_norms(hs_layers: List[torch.Tensor], start: Optional[int], end: Optional[int]) -> np.ndarray:
    """Mean L2 norm per layer of the hidden states of the span ``[start, end)``."""
    lo, hi = _span_bounds(int(hs_layers[0].shape[0]), start, end)
    return np.array([float(h[lo:hi].float().norm(dim=-1).mean()) for h in hs_layers], dtype=np.float64)


def _span_token_projections(
    hs_layers: List[torch.Tensor], vectors: np.ndarray, start: Optional[int], end: Optional[int]
) -> np.ndarray:
    """Each span token's projection on each layer's vector: ``(n_tokens, layers)``."""
    lo, hi = _span_bounds(int(hs_layers[0].shape[0]), start, end)
    return np.stack(
        [hs_layers[l][lo:hi].float().numpy().astype(np.float64) @ vectors[l].astype(np.float64)
         for l in range(len(hs_layers))],
        axis=1,
    )


def _token_references(pos_tokens: List[np.ndarray], neg_tokens: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-layer zero and unit for single-token scores: the midpoint between the mean token score of
    each pole, and the pooled within-pole SD of single tokens (wider than that of per-item means,
    which average much of the token-to-token variation away)."""
    pos = np.concatenate(pos_tokens, axis=0)
    neg = np.concatenate(neg_tokens, axis=0)
    return 0.5 * (pos.mean(axis=0) + neg.mean(axis=0)), _pooled_sd_columns(pos, neg)


def pick_layer(num_layers: int, layer_idx: Optional[int], layer_frac: float) -> int:
    if layer_idx is not None:
        return resolve_layer_index(layer_idx, num_layers, "training.layer_idx")
    return resolve_layer_index({"frac": layer_frac}, num_layers, "training.layer_frac")


def _normalize_best_layer_search(
    search_spec: Optional[Any], num_layers: int
) -> Dict[str, Any]:
    """Layers the sweep picks the best layer from.

    ``[lo, hi]`` or ``[[lo, hi], ...]`` with layer indices (negative count from the end), or
    ``{"frac": [lo, hi]}`` / ``{"frac": [[lo, hi], ...]}`` with fractions of depth. Bounds are
    inclusive; indices outside the model raise.
    """
    if search_spec is None:
        return {
            "intervals": [(0, num_layers - 1)],
            "layers": list(range(num_layers)),
            "is_default": True,
        }

    as_frac = False
    if isinstance(search_spec, dict):
        if set(search_spec) != {"frac"}:
            raise ValueError("training.best_layer_search as a dict must be {'frac': [lo, hi]} or {'frac': [[lo, hi], ...]}.")
        as_frac = True
        search_spec = search_spec["frac"]

    if not isinstance(search_spec, (list, tuple)):
        raise TypeError("training.best_layer_search must be a list like [lo, hi] or [[lo, hi], ...].")

    if len(search_spec) == 0:
        raise ValueError("training.best_layer_search is empty; provide [lo, hi] or [[lo, hi], ...].")

    if len(search_spec) == 2 and not isinstance(search_spec[0], (list, tuple)):
        raw_intervals: List[Any] = [search_spec]
    else:
        raw_intervals = list(search_spec)

    intervals: List[Tuple[int, int]] = []
    for i, raw in enumerate(raw_intervals):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"training.best_layer_search[{i}] must be [lo, hi].")
        lo, hi = (
            resolve_layer_index({"frac": v} if as_frac else v, num_layers, f"training.best_layer_search[{i}]")
            for v in raw
        )
        intervals.append((min(lo, hi), max(lo, hi)))

    intervals.sort(key=lambda x: x[0])
    merged: List[List[int]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1] + 1:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    merged_intervals = [(int(lo), int(hi)) for lo, hi in merged]

    layers: List[int] = []
    for lo, hi in merged_intervals:
        layers.extend(range(lo, hi + 1))

    return {"intervals": merged_intervals, "layers": layers, "is_default": False}


def _window(center: int, radius: Any, num_layers: int, what: str) -> List[int]:
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"{what} must be >= 0; got {radius}")
    return list(range(max(0, center - radius), min(num_layers - 1, center + radius) + 1))


def _layer_list(values: Any, num_layers: int, what: str) -> List[int]:
    """A non-empty list of distinct layer indices (see :func:`resolve_layer_index`)."""
    if isinstance(values, (int, np.integer)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"{what} must be a non-empty list of layer indices; got {values!r}")
    layers = [resolve_layer_index(v, num_layers, f"{what} entries") for v in values]
    if len(set(layers)) != len(layers):
        raise ValueError(f"{what} repeats a layer: {layers}")
    return layers


def select_layers(selection: Dict[str, Any], best_layer: int, num_layers: int) -> List[int]:
    mode = selection.get("mode", "best")
    if mode == "best":
        return [best_layer]
    if mode == "window":
        return _window(best_layer, selection.get("window_radius", 0), num_layers, "layer_selection.window_radius")
    if mode == "explicit":
        return _layer_list(selection.get("layers", []), num_layers, "layer_selection.layers")
    if mode == "all":
        return list(range(num_layers))
    raise ValueError(f"Unknown layer selection mode: {mode}")


ALPHA_UNITS = ("raw", "sigma", "resid_norm")
# Below this many eval items per pole, training warns that the layer choice and d are unreliable.
MIN_EVAL_ITEMS_PER_POLE = 10
SWEEP_SELECTS = ("effect_size", "p_value")


def _validate_alpha_unit(alpha_unit: Any) -> str:
    if alpha_unit not in ALPHA_UNITS:
        raise ValueError(f"alpha_unit must be one of {ALPHA_UNITS}; got {alpha_unit!r}")
    return str(alpha_unit)


def _resolve_steer_layers(
    steer_layers: Any, best_layer: int, window_radius: Any, num_layers: int
) -> List[int]:
    """Layers to steer: ``"probe"`` (the best layer), ``"window"`` (best layer +/- radius),
    or explicit layer indices (an int or a list; negative indices count from the end)."""
    if isinstance(steer_layers, str):
        if steer_layers == "probe":
            return [int(best_layer)]
        if steer_layers == "window":
            return _window(int(best_layer), window_radius, num_layers, "steer_window_radius")
        raise ValueError(
            f"steer_layers must be 'probe', 'window' or a list of layer indices; got {steer_layers!r}"
        )
    return _layer_list(steer_layers, num_layers, "steer_layers")


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
                "No finite Cohen's d in the best_layer_search range. This typically means "
                "fewer than 2 eval samples per side; add eval data or disable the layer sweep."
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


def _decode_tokens(tokenizer, token_ids: Iterable[int]) -> List[str]:
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids]


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


_CHAT_TEMPLATE_FALLBACK_WARNED: set = set()


def _encode_for_read(
    tokenizer,
    prompt: "PromptLike",
    default_system: Optional[str],
    *,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "read",
) -> torch.Tensor:
    """Encode a chat-formatted 'read' input (no generation): ``prompt`` as a user turn, after
    ``default_system`` if one is set (see :func:`_normalize_messages`). Without a chat template
    it falls back to plain tokenization, and any system text is dropped.
    """
    if _has_chat_template(tokenizer):
        messages, _ = _normalize_messages(
            prompt, default_system, warn=warn, warn_prefix=warn_prefix
        )
        return apply_chat(tokenizer, messages, add_generation_prompt=False)

    text = prompt if isinstance(prompt, str) else _prompt_to_text(prompt)
    fallback_key = id(tokenizer)
    if warn is not None and fallback_key not in _CHAT_TEMPLATE_FALLBACK_WARNED:
        warn(
            f"{warn_prefix}: tokenizer has no chat_template; falling back to plain tokenization"
            + (" (the system prompt is dropped)." if default_system else ".")
        )
        _CHAT_TEMPLATE_FALLBACK_WARNED.add(fallback_key)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]


def _completion_input_ids(tokenizer, messages: "List[PromptMessage]", completion: str) -> Tuple[torch.Tensor, int]:
    """Chat-format ``messages`` with an assistant generation prompt and append ``completion``.

    Returns ``(input_ids, prompt_len)``: the completion occupies ``input_ids[0, prompt_len:]``,
    with no end-of-turn token after it.
    """
    prompt_ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
    plen = int(prompt_ids.shape[-1])
    comp_ids = tokenizer(str(completion), add_special_tokens=False, return_tensors="pt")["input_ids"]
    if comp_ids.dim() == 1:
        comp_ids = comp_ids.unsqueeze(0)
    return torch.cat([prompt_ids, comp_ids.to(prompt_ids.dtype)], dim=1), plen


TEXT_ROLES = ("user", "assistant", "raw")

_TEXT_SENTINEL = "CONCEPT_PROBE_TEXT_SENTINEL"


def _token_count(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(torch.as_tensor(ids).reshape(-1).shape[0])


def _user_text_span(
    tokenizer, text: "PromptLike", system: Optional[str], input_ids: torch.Tensor
) -> Optional[Tuple[int, int, bool]]:
    """Token span of a user-role text inside its encoding from :func:`_encode_for_read`.

    Returns ``(start, end, trimmed)``: ``[start, end)`` are the text's own tokens, leaving out
    the BOS, system block and role headers before the text and the end-of-turn tokens after it
    (for a message-list text, the last message's content). ``trimmed`` is True when the chat
    template stripped whitespace from the text. Returns None if the span cannot be located.
    """
    total = int(input_ids.shape[-1])
    if not _has_chat_template(tokenizer):
        bos = getattr(tokenizer, "bos_token_id", None)
        start = 1 if bos is not None and total > 0 and int(input_ids[0, 0]) == int(bos) else 0
        flat = text if isinstance(text, str) else _prompt_to_text(text)
        return start, min(total, start + _token_count(tokenizer, flat)), False

    messages, _ = _normalize_messages(text, system)
    content = messages[-1]["content"]
    try:
        kwargs = _chat_template_kwargs(tokenizer)
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, **kwargs)
        # The template's output before the content does not depend on the content, so a sentinel
        # in its place gives where the text starts.
        probe_messages = messages[:-1] + [dict(messages[-1], content=_TEXT_SENTINEL)]
        probe_rendered = tokenizer.apply_chat_template(
            probe_messages, add_generation_prompt=False, tokenize=False, **kwargs
        )
        enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    if not isinstance(rendered, str) or not isinstance(probe_rendered, str):
        return None
    if list(enc["input_ids"]) != input_ids[0].tolist():
        return None
    char_start = probe_rendered.rfind(_TEXT_SENTINEL)
    if char_start < 0:
        return None
    needle = next((n for n in (content, content.strip()) if n and rendered.startswith(n, char_start)), None)
    if needle is None:  # the template rewrote the text
        return None
    char_end = char_start + len(needle)
    covered = [
        i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < char_end and b > char_start
    ]
    if not covered:
        return None
    return covered[0], covered[-1] + 1, needle != content


_TEMPLATE_TRIM_WARNED: set = set()


def _encode_text_for_read(
    tokenizer,
    text: "PromptLike",
    system: Optional[str] = None,
    *,
    role: str = "raw",
    prompt: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "read",
) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    """Encode a text to read (no generation), as texts-mode training, eval texts,
    ``score_texts`` and ``extract_text_reps`` do.

    Returns ``(input_ids, text_start, text_end)``: ``input_ids[0, text_start:text_end]`` are the
    text's own tokens, the span the ``assistant_*`` readout modes pool over.

    - ``role="raw"`` (default): the text alone, with no chat template and no system prompt: the
      BOS token (if the tokenizer has one) and the text. ``system`` and ``prompt`` are ignored.
    - ``role="user"``: the text as a user turn of the chat template, after ``system`` if one is
      set. A message list is read as given. The span leaves out the template tokens around the
      text; it is None when it cannot be located (``sequence_*`` modes still work then).
    - ``role="assistant"``: the text as the model's own reply: ``system`` if set, an optional user
      turn ``prompt``, the assistant header, then the text, with no end-of-turn token after it.
    """
    if role not in TEXT_ROLES:
        raise ValueError(f"training.train_text_role must be one of {TEXT_ROLES}; got {role!r}")
    if role == "user":
        ids = _encode_for_read(tokenizer, text, system, warn=warn, warn_prefix=warn_prefix)
        span = _user_text_span(tokenizer, text, system, ids)
        if span is None:
            if warn is not None:
                warn(f"{warn_prefix}: could not locate the text's own tokens in the chat-formatted input.")
            return ids, None, None
        if span[2] and warn is not None and id(tokenizer) not in _TEMPLATE_TRIM_WARNED:
            warn(f"{warn_prefix}: the chat template strips leading/trailing whitespace from texts.")
            _TEMPLATE_TRIM_WARNED.add(id(tokenizer))
        return ids, span[0], span[1]
    if not isinstance(text, str):
        raise TypeError(
            f"{warn_prefix}: role={role!r} needs plain-string texts; read a message list with role='user'."
        )
    if role == "raw":
        if (prompt is not None or system) and warn is not None:
            warn(f"{warn_prefix}: role='raw' reads the text alone; system and prompt are ignored.")
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None:
            return ids, 0, int(ids.shape[-1])
        ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        return ids, 1, int(ids.shape[-1])
    if not _has_chat_template(tokenizer):
        raise ValueError("role='assistant' needs a tokenizer with a chat template.")
    messages: "List[PromptMessage]" = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    if prompt is not None:
        messages.append({"role": "user", "content": str(prompt)})
    if not messages:
        raise ValueError(
            f"{warn_prefix}: reading a text as the assistant's reply needs a system prompt or a user "
            "prompt before it; chat templates cannot format a conversation that starts with the reply."
        )
    ids, plen = _completion_input_ids(tokenizer, messages, text)
    return ids, plen, int(ids.shape[-1])


def preview_input(
    tokenizer,
    text: "PromptLike",
    *,
    role: str = "raw",
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    generation: bool = False,
) -> Dict[str, Any]:
    """Exactly what the library feeds the model for ``text``, before running anything.

    ``generation=False`` encodes ``text`` as a text to read, as texts-mode training, eval texts,
    ``score_texts`` and ``extract_text_reps`` do with the same ``role``/``system``/``prompt``.
    ``generation=True`` encodes it as a prompt to generate from, as ``score_prompts`` does: a user
    turn (after ``system`` if set) and the assistant header; ``role`` and ``prompt`` are ignored.

    Returns ``input_text`` (decoded with every special and template token), ``input_ids``,
    ``tokens``, and ``text_span`` / ``span_text``: the tokens the ``assistant_*`` readouts pool
    over (None for generation prompts).
    """
    if generation:
        messages, _ = _normalize_messages(text, system)
        ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
        start = end = None
    else:
        ids, start, end = _encode_text_for_read(tokenizer, text, system, role=role, prompt=prompt)
    token_ids = [int(t) for t in ids[0].tolist()]
    return {
        "input_text": tokenizer.decode(token_ids, skip_special_tokens=False),
        "input_ids": token_ids,
        "tokens": _decode_tokens(tokenizer, token_ids),
        "text_span": None if start is None else (int(start), int(end)),
        "span_text": None if start is None else tokenizer.decode(token_ids[start:end], skip_special_tokens=False),
    }


def _prompt_slug(text: str, max_words: int = 8, max_len: int = 48) -> str:
    words = text.strip().split()
    snippet = " ".join(words[:max_words])
    slug = safe_slug(snippet)[:max_len]
    return slug or "prompt"


PromptMessage = Dict[str, str]
PromptLike = Union[str, List[PromptMessage]]


def _prompt_to_text(prompt: PromptLike) -> str:
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)
    parts: List[str] = []
    for m in prompt:
        if not isinstance(m, dict):
            parts.append(str(m))
            continue
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return "\n".join(parts)


def _prompt_slug_any(prompt: PromptLike, max_words: int = 8, max_len: int = 48) -> str:
    return _prompt_slug(_prompt_to_text(prompt), max_words=max_words, max_len=max_len)


def _normalize_messages(
    prompt: PromptLike,
    default_system: Optional[str],
    *,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "prompt",
) -> Tuple[List[PromptMessage], bool]:
    """Return (messages, used_prompt_system).

    A system message is added only when ``default_system`` is set; None or "" add none.

    - A string prompt becomes a user turn, after the system message if any.
    - A message list is used as given. ``default_system`` is prepended only if the list has no
      system message; if it has one, that one wins (with a warning when ``default_system`` is set).
    """
    system_messages = [{"role": "system", "content": str(default_system)}] if default_system else []
    if isinstance(prompt, str):
        return system_messages + [{"role": "user", "content": prompt}], False

    if not isinstance(prompt, list):
        raise TypeError(f"{warn_prefix} must be str or list of messages; got {type(prompt)}")

    messages: List[PromptMessage] = []
    system_count = 0
    for i, m in enumerate(prompt):
        if not isinstance(m, dict):
            raise TypeError(f"{warn_prefix}[{i}] must be a dict with role/content; got {type(m)}")
        if "role" not in m or "content" not in m:
            raise ValueError(f"{warn_prefix}[{i}] must have 'role' and 'content' keys")
        role = str(m["role"])
        content = str(m["content"])
        if role == "system":
            system_count += 1
        messages.append({"role": role, "content": content})

    if system_count > 0:
        if default_system and warn is not None:
            warn(
                f"{warn_prefix} includes a system message; the provided default system prompt is ignored."
            )
        if system_count > 1 and warn is not None:
            warn(f"{warn_prefix} includes multiple system messages; using them as provided.")
        return (messages, True)

    return system_messages + messages, False


def _alpha_label(alpha: float, unit: str) -> str:
    if unit == "sigma":
        label = f"sigma{alpha:+.2f}"
    else:
        label = f"alpha{alpha:+.4f}"
    label = label.replace("+", "p").replace("-", "m").replace(".", "p")
    return safe_slug(label)


def _normalize_generation_logits_dtype(dtype_name: Optional[str]) -> Tuple[str, torch.dtype]:
    raw = "float16" if dtype_name is None else str(dtype_name).strip().lower()
    if raw in {"float16", "fp16", "half"}:
        return ("float16", torch.float16)
    if raw in {"float32", "fp32", "single"}:
        return ("float32", torch.float32)
    raise ValueError(
        f"Unsupported generation logits dtype '{dtype_name}'. Use 'float16' or 'float32'."
    )


def _pack_generation_logits_arrays(
    generation_step_logits: torch.Tensor,
    *,
    logits_top_k: Optional[int],
    logits_dtype_name: Optional[str],
    logits_source: str = "raw_model",
) -> Dict[str, np.ndarray]:
    if generation_step_logits.ndim != 2:
        raise ValueError(
            f"Expected generation_step_logits to have shape (num_steps, vocab_size); got {tuple(generation_step_logits.shape)}"
        )
    if generation_step_logits.shape[0] <= 0:
        return {}

    dtype_name, torch_dtype = _normalize_generation_logits_dtype(logits_dtype_name)
    logits = generation_step_logits.detach().to(dtype=torch.float32, device="cpu")
    out: Dict[str, np.ndarray] = {
        "generation_logits_steps": np.array([int(logits.shape[0])], dtype=np.int32),
        "generation_logits_dtype": np.array([dtype_name]),
        "generation_logits_source": np.array([str(logits_source)]),
    }

    if logits_top_k is not None and int(logits_top_k) > 0:
        k = int(max(1, min(int(logits_top_k), int(logits.shape[-1]))))
        top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
        out["generation_logits_topk_values"] = top_vals.to(dtype=torch_dtype).numpy()
        out["generation_logits_topk_indices"] = top_idx.to(dtype=torch.int32).numpy().astype(np.int32, copy=False)
        out["generation_logits_topk_k"] = np.array([k], dtype=np.int32)
    else:
        out["generation_logits"] = logits.to(dtype=torch_dtype).numpy()
        out["generation_logits_topk_k"] = np.array([0], dtype=np.int32)

    return out


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


@torch.no_grad()
def forward_hidden_states_all_layers(
    model,
    input_ids: torch.Tensor,
    *,
    return_generation_logits: bool = False,
    generation_prompt_len: Optional[int] = None,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
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


READ_MODES = (
    "sequence_last",
    "sequence_last_k_mean",
    "sequence_all_mean",
    "assistant_last",
    "assistant_last_k_mean",
    "assistant_all_mean",
)


class EmptySpanError(ValueError):
    """An item has no tokens of its own to pool: an empty completion or an empty text."""


def reps_from_hs_layers(
    hs_layers: List[torch.Tensor],
    prompt_len: Optional[int],
    mode: str,
    last_k: int,
    span_end: Optional[int] = None,
) -> np.ndarray:
    """Pool each layer's hidden states ``(seq_len, dim)`` into one rep per layer.

    ``sequence_*`` modes read the whole sequence, template tokens included. ``assistant_*``
    modes read the span ``[prompt_len, span_end)`` (``span_end`` defaults to the end of the
    sequence): a generated completion, or the own tokens of a read text. An empty span raises
    :class:`EmptySpanError` rather than pooling something else (the prompt and system prompt
    would stand in for the missing text).
    """
    if mode not in READ_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    num_layers = len(hs_layers)
    dim = int(hs_layers[0].shape[1])
    total = int(hs_layers[0].shape[0])
    reps = np.zeros((num_layers, dim), dtype=np.float32)

    start, end = 0, total
    if mode.startswith("assistant_"):
        if prompt_len is None:
            raise ValueError(
                f"mode={mode} pools over a text or completion span, but its start is unknown; "
                "use a sequence_* mode to read the whole sequence."
            )
        start = int(prompt_len)
        end = total if span_end is None else min(int(span_end), total)
        if start >= end:
            raise EmptySpanError(f"mode={mode}: no tokens to pool (span [{start}, {end}) of a {total}-token input)")

    for l in range(num_layers):
        hs = hs_layers[l]
        if mode == "sequence_last":
            reps[l] = hs[-1].numpy()
        elif mode == "sequence_last_k_mean":
            reps[l] = hs[-min(last_k, total):].mean(dim=0).numpy()
        elif mode == "sequence_all_mean":
            reps[l] = hs.mean(dim=0).numpy()
        else:
            span = hs[start:end]
            if mode == "assistant_last":
                reps[l] = span[-1].numpy()
            elif mode == "assistant_last_k_mean":
                reps[l] = span[-min(last_k, span.shape[0]):].mean(dim=0).numpy()
            else:
                reps[l] = span.mean(dim=0).numpy()

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
) -> np.ndarray:
    """Per-layer reps pooled over the completion tokens, for ``(prompt, completion)`` pairs.

    Reconstructs the readout used during training: each prompt is chat-formatted with an
    assistant generation prompt, the completion tokens are appended, one forward pass
    captures all layers, and the hidden states are pooled over the completion span per
    layer (``pool='assistant_all_mean'`` by default). Returns an array of shape
    ``(len(pairs), num_layers, hidden_dim)``.

    Use it to score externally-provided text -- e.g. another model's generations -- against
    trained concept vectors, or to compare how different models represent the same text.
    Pass ``system=None`` (default) to format prompts with no system message.
    """
    out_reps: List[np.ndarray] = []
    for prompt, completion in pairs:
        if isinstance(prompt, str):
            messages: List[PromptMessage] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = list(prompt)
        full_ids, plen = _completion_input_ids(tokenizer, messages, completion)
        hs_layers = forward_hidden_states_all_layers(model, full_ids)
        try:
            out_reps.append(reps_from_hs_layers(hs_layers, prompt_len=plen, mode=pool, last_k=last_k))
        except EmptySpanError as exc:
            raise EmptySpanError(f"pairs[{len(out_reps)}]: {exc}") from None
    return np.stack(out_reps, axis=0).astype(np.float32)


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
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Per-layer reps of texts, encoded exactly as texts-mode training encodes them.

    ``role``, ``prompt`` and ``system`` mean what ``training.train_text_role``,
    ``training.train_text_prompt`` and the training system prompt mean, and the defaults are the
    training defaults: the raw text with no template or system prompt, pooled over its own
    tokens. ``pool`` is one readout mode, returning ``(len(texts), num_layers, hidden_dim)``, or
    several, returning a dict of such arrays from a single forward pass per text. Use
    :func:`preview_input` to see the exact input.
    """
    pools = [pool] if isinstance(pool, str) else list(pool)
    out: Dict[str, List[np.ndarray]] = {p: [] for p in pools}
    for i, text in enumerate(texts):
        ids, start, end = _encode_text_for_read(tokenizer, text, system, role=role, prompt=prompt,
                                                warn_prefix=f"texts[{i}]")
        hs_layers = forward_hidden_states_all_layers(model, ids)
        for p in pools:
            try:
                out[p].append(reps_from_hs_layers(hs_layers, prompt_len=start, mode=p, last_k=last_k, span_end=end))
            except EmptySpanError as exc:
                raise EmptySpanError(f"texts[{i}]: {exc}") from None
    arrays = {p: np.stack(v, axis=0).astype(np.float32) for p, v in out.items()}
    return arrays[pool] if isinstance(pool, str) else arrays


def token_scores_from_hs_layers(
    hs_layers: List[torch.Tensor],
    concept_vectors: np.ndarray,
    layer_indices: List[int],
    aggregate: str,
    reference_layer: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project each selected layer's hidden states onto a concept direction.

    By default each layer ``li`` is scored with its own concept vector
    ``concept_vectors[li]`` (the usual per-layer probe). If ``reference_layer``
    is given, *every* selected layer is instead scored against the single
    direction ``concept_vectors[reference_layer]``. This is a "logit-lens"-style
    read: it measures how decodable a *fixed* direction (e.g. the one learned at
    a late layer) is from the activations at every other layer, holding the
    direction constant so that changes across layers reflect the representation
    rather than a change of basis. ``reference_layer`` accepts negative indices.
    """
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty")
    num_vectors = int(concept_vectors.shape[0])
    if reference_layer is not None:
        ref = resolve_layer_index(reference_layer, num_vectors, "reference_layer")
    total = int(hs_layers[0].shape[0])
    scores_per_layer = np.zeros((total, len(layer_indices)), dtype=np.float32)

    for i, li in enumerate(layer_indices):
        hs = hs_layers[li]
        vi = li if reference_layer is None else ref
        v = torch.tensor(concept_vectors[vi], dtype=hs.dtype)
        scores_per_layer[:, i] = (hs @ v).numpy()

    if aggregate == "mean":
        scores_agg = scores_per_layer.mean(axis=1)
    elif aggregate == "sum":
        scores_agg = scores_per_layer.sum(axis=1)
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

    return scores_per_layer, scores_agg


@torch.no_grad()
def score_text_tokens(
    model,
    tokenizer,
    concept_vectors: np.ndarray,
    text: str,
    *,
    layer_indices: Optional[List[int]] = None,
    reference_layer: Optional[int] = None,
    unit_norm: bool = False,
    add_bos: bool = True,
) -> Dict[str, Any]:
    """Score every token of a RAW text against the per-layer concept directions, in memory.

    Low-level read primitive for callers that need per-token, per-layer scores together with
    character offsets (e.g. to target a specific word/subtoken), without the artifact-writing of
    ``ConceptProbe.score_texts``. The text is read as ``role="raw"`` reads it everywhere else: the
    BOS token (if the tokenizer has one) and the text, with no chat template and no other special
    tokens. The forward pass uses the same all-layers capture as training (with the final-norm
    fix), so layer ``l`` of the result aligns with ``concept_vectors[l]`` (i.e. hidden_states[l+1]).

    Args:
        concept_vectors: (num_layers, hidden_dim) directions. Pass unit vectors, or set
            ``unit_norm=True`` to divide each row by its L2 norm before projecting (so the score is
            ``hs · v̂``, i.e. divided by ``||v_l||``).
        layer_indices: which layers to score (default: all). ``scores_per_layer`` has shape
            ``(num_tokens, len(layer_indices))``.
        reference_layer: if set, project every selected layer onto the single direction of that
            layer (logit-lens); see ``token_scores_from_hs_layers``.
        unit_norm: L2-normalize each concept vector before projecting (default False, to match
            ``token_scores_from_hs_layers`` semantics).
        add_bos: put the BOS token first (default True; it gets offset (0, 0)).

    Returns:
        dict with ``token_ids`` (np.int32 [num_tokens]), ``tokens`` (List[str]),
        ``offset_mapping`` (List[[start, end]] char spans), ``scores_per_layer``
        (np.float32 [num_tokens, len(layer_indices)]) and ``layer_indices`` (List[int]).
    """
    cv = np.asarray(concept_vectors, dtype=np.float32)
    if cv.ndim != 2:
        raise ValueError(f"concept_vectors must be 2D [num_layers, hidden_dim], got shape={cv.shape}")
    num_layers = int(cv.shape[0])
    if unit_norm:
        norms = np.linalg.norm(cv, axis=1, keepdims=True)
        cv = cv / np.clip(norms, 1e-12, None)
    layers = (
        list(range(num_layers))
        if layer_indices is None
        else _layer_list(list(layer_indices), num_layers, "layer_indices")
    )

    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    om = enc["offset_mapping"]
    if hasattr(om, "tolist"):
        om = om.tolist()
    offsets = [list(span) for span in om[0]]
    input_ids = enc["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    bos = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos is not None:
        input_ids = torch.cat([torch.tensor([[bos]], dtype=input_ids.dtype), input_ids], dim=1)
        offsets = [[0, 0]] + offsets

    hs_layers = forward_hidden_states_all_layers(model, input_ids)
    if len(hs_layers) != num_layers:
        raise ValueError(
            f"Layer mismatch: model returned {len(hs_layers)} layers, concept_vectors has {num_layers}."
        )
    scores_per_layer, _ = token_scores_from_hs_layers(
        hs_layers, cv, layers, aggregate="mean", reference_layer=reference_layer
    )
    token_ids = input_ids[0].detach().cpu().numpy().astype(np.int32)
    return {
        "token_ids": token_ids,
        "tokens": _decode_tokens(tokenizer, token_ids),
        "offset_mapping": offsets,
        "scores_per_layer": scores_per_layer.astype(np.float32),
        "layer_indices": layers,
    }


def _eos_token_ids(model, tokenizer) -> Set[int]:
    """Token ids that end a generation: the model's generation config, plus the tokenizer's EOS."""
    ids: Set[int] = set()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is not None:
        ids.update(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        ids.add(int(tok_eos))
    return ids


def _completion_end(token_ids: Any, prompt_len: int, eos_ids: Set[int]) -> int:
    """End of a generated completion: the first end-of-sequence token at or after ``prompt_len``
    (excluded from the completion), or the end of the sequence.

    Every generation path keeps that token in its ids and ends the completion before it, so
    pooling and ``score_mean`` never include it.
    """
    ids = [int(t) for t in token_ids]
    for i in range(int(prompt_len), len(ids)):
        if ids[i] in eos_ids:
            return i
    return len(ids)


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    system: str,
    prompt: PromptLike,
    max_new_tokens: int,
    greedy: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "prompt",
) -> Tuple[torch.Tensor, int]:
    messages, _ = _normalize_messages(prompt, system, warn=warn, warn_prefix=warn_prefix)
    if warn is not None and len(messages) > 0 and messages[-1].get("role") != "user":
        warn(
            f"{warn_prefix} ends with role='{messages[-1].get('role')}'; generation typically expects the last message to be a user turn."
        )

    input_ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
    prompt_len = int(input_ids.shape[-1])
    attn = attention_mask_from_ids(input_ids).to(model.device)

    gen_ids = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=temperature if not greedy else None,
        top_p=top_p if not greedy else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return gen_ids[0].detach().cpu(), prompt_len


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    system: str,
    prompts: List[PromptLike],
    max_new_tokens: int,
    greedy: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    batch_size: int = 16,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "prompt",
) -> List[Tuple[torch.Tensor, int]]:
    """Batched equivalent of ``generate_once`` over many prompts under one system.

    Returns one ``(ids, prompt_len)`` per prompt, where ``ids`` is a 1D CPU tensor of
    ``[prompt_tokens + completion_tokens (+ the end-of-sequence token)]`` -- the same contract
    as ``generate_once``, so callers can use the results interchangeably. Prompts are
    left-padded within each batch with an attention mask; left padding and the padding after
    the first end-of-sequence token are stripped per item.

    Batching changes the shapes of the matrix products, so in half precision a greedy
    completion can differ from the unbatched one. Use ``batch_size=1`` when exact
    reproducibility across batch sizes matters.
    """
    if not prompts:
        return []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    encoded: List[torch.Tensor] = []
    for i, p in enumerate(prompts):
        messages, _ = _normalize_messages(p, system, warn=warn, warn_prefix=f"{warn_prefix}[{i}]")
        if warn is not None and len(messages) > 0 and messages[-1].get("role") != "user":
            warn(
                f"{warn_prefix}[{i}] ends with role='{messages[-1].get('role')}'; "
                "generation typically expects the last message to be a user turn."
            )
        encoded.append(apply_chat(tokenizer, messages, add_generation_prompt=True)[0])

    eos_ids = _eos_token_ids(model, tokenizer)
    step = max(1, int(batch_size))
    results: List[Optional[Tuple[torch.Tensor, int]]] = [None] * len(encoded)
    for start in range(0, len(encoded), step):
        chunk = encoded[start : start + step]
        plens = [int(t.shape[-1]) for t in chunk]
        maxlen = max(plens)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, t in enumerate(chunk):
            input_ids[j, maxlen - plens[j] :] = t.to(input_ids.dtype)
            attn[j, maxlen - plens[j] :] = 1
        gen = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attn.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=temperature if not greedy else None,
            top_p=top_p if not greedy else None,
            pad_token_id=pad_id,
        )
        gen = gen.detach().cpu()
        for j in range(len(chunk)):
            plen = plens[j]
            row = gen[j][maxlen - plen :]  # drop left padding -> [prompt + completion (+ EOS + pad)]
            end = _completion_end(row, plen, eos_ids)
            keep = min(int(row.shape[-1]), end + 1)  # keep the EOS, drop the padding after it
            results[start + j] = (row[:keep].clone(), plen)
    return results  # type: ignore[return-value]


@dataclass
class ConceptProbe:
    model_bundle: ModelBundle
    config: Dict[str, Any]
    concept: ConceptSpec
    run_dir: str
    logger: JsonlLogger
    console: Optional[ConsoleLogger] = None

    concept_vectors: Optional[np.ndarray] = None
    best_layer: Optional[int] = None
    # The unit of alpha_unit="sigma": pooled within-pole SD of the eval scores at best_layer.
    sigma_best_layer: Optional[float] = None
    # Per-layer zero (midpoint of the eval poles) and unit (pooled within-pole SD) of per-item
    # scores (an eval item's mean over its tokens), and the same for single-token scores.
    score_center: Optional[np.ndarray] = None
    score_scale: Optional[np.ndarray] = None
    token_center: Optional[np.ndarray] = None
    token_scale: Optional[np.ndarray] = None
    # Per-layer mean residual norm of the training tokens: the unit of alpha_unit="resid_norm".
    resid_norm: Optional[np.ndarray] = None
    sweep_d: Optional[np.ndarray] = None
    sweep_p: Optional[np.ndarray] = None

    def train(self) -> "ConceptProbe":
        readout = self.config["readout"]
        train_cfg = self.config["training"]
        eval_cfg = self.config["evaluation"]
        out_cfg = self.config["output"]
        plot_cfg = self.config["plots"]
        prompts_cfg = self.config["prompts"]
        seed = int(self.config["random"]["seed"])
        train_prompt_mode = train_cfg.get("train_prompt_mode", "shared")
        train_prompts = list(prompts_cfg.get("train_questions", []))
        train_prompts_pos = list(prompts_cfg.get("train_questions_pos", []))
        train_prompts_neg = list(prompts_cfg.get("train_questions_neg", []))

        # ---- Validate everything before generating or writing anything.
        if train_prompt_mode not in ("shared", "opposed", "texts"):
            raise ValueError(f"Unknown train_prompt_mode: {train_prompt_mode}")
        direction_normalization = str(train_cfg.get("direction_normalization", "none"))
        if direction_normalization not in DIRECTION_NORMALIZATIONS:
            raise ValueError(
                f"training.direction_normalization must be one of {DIRECTION_NORMALIZATIONS}; "
                f"got {direction_normalization!r}"
            )
        sweep_select = train_cfg.get("sweep_select", "effect_size")
        if sweep_select not in SWEEP_SELECTS:
            raise ValueError(f"training.sweep_select must be one of {SWEEP_SELECTS}; got {sweep_select!r}")
        eval_mode = eval_cfg.get("eval_mode", "auto")
        if eval_mode not in ("auto", "read", "generate", "skip"):
            raise ValueError(f"Unknown evaluation.eval_mode: {eval_mode!r} (use 'auto', 'read', 'generate' or 'skip').")
        selection_split = eval_cfg.get("selection_split", 0.5)
        if selection_split is not None and not 0.0 < float(selection_split) < 1.0:
            raise ValueError(f"evaluation.selection_split must be in (0, 1) or null; got {selection_split!r}")
        if self.concept.pos_system == self.concept.neg_system and (
            train_prompt_mode in ("shared", "opposed") or (eval_cfg.get("do_eval", True) and eval_mode == "generate")
        ):
            raise ValueError(
                "concept.pos_system == concept.neg_system: contrasting generations under identical system "
                "prompts learns noise. Set the two system prompts that define the concept."
            )

        # Labelled texts (the training texts in texts mode, the eval texts in every mode) are read
        # with training.train_text_role: by default the raw text, with no chat template and no
        # system prompt. They are pooled like the vector's training data: with read_mode in texts
        # mode, and with train_mode (the readout of the model's own completions) otherwise.
        if train_prompt_mode == "texts":
            train_readout_mode, train_readout_last_k = readout["read_mode"], readout["read_last_k"]
        else:
            train_readout_mode, train_readout_last_k = readout["train_mode"], readout["train_last_k"]
        for key, mode in (("read_mode", readout["read_mode"]), ("train_mode", readout["train_mode"])):
            if mode not in READ_MODES:
                raise ValueError(f"readout.{key} must be one of {READ_MODES}; got {mode!r}")
        text_role = train_cfg.get("train_text_role") or "raw"
        if text_role not in TEXT_ROLES:
            raise ValueError(f"training.train_text_role must be one of {TEXT_ROLES}; got {text_role!r}")
        text_prompt = train_cfg.get("train_text_prompt")
        if text_role == "user" and text_prompt is not None:
            raise ValueError(
                "training.train_text_prompt is a user turn before the text, so it only applies to "
                "train_text_role='assistant'; unset it or change the role."
            )
        train_text_system = train_cfg.get("train_text_system")
        if train_text_system is None:
            train_text_system = prompts_cfg.get("neutral_system")
        eval_system = eval_cfg.get("eval_system")
        if eval_system is None:
            eval_system = train_text_system if train_prompt_mode == "texts" else prompts_cfg.get("neutral_system")
        if text_role == "raw":
            if text_prompt is not None or train_cfg.get("train_text_system") or eval_cfg.get("eval_system"):
                raise ValueError(
                    "train_text_role='raw' reads each text alone, so train_text_prompt, train_text_system and "
                    "evaluation.eval_system would be ignored; unset them or use train_text_role='user'/'assistant'."
                )
            train_text_system = eval_system = None
        eval_texts_given = len(self.concept.eval_pos_texts) > 0 and len(self.concept.eval_neg_texts) > 0
        reads_eval_texts = (
            eval_cfg.get("do_eval", True)
            and eval_texts_given
            and eval_mode != "skip"
            and not (train_prompt_mode == "texts" and eval_mode == "generate")
        )
        if reads_eval_texts and text_role == "assistant" and not eval_system and text_prompt is None:
            raise ValueError(
                "Eval texts are read as the assistant's reply, which needs a system prompt or a user prompt "
                "before them: set evaluation.eval_system or prompts.neutral_system, or training.train_text_prompt."
            )

        if ttest_ind is None:
            self._warn("scipy not available; sweep_p will be NaN.")
        if str(train_readout_mode).startswith("sequence_"):
            self._warn(
                f"readout {train_readout_mode} reads template tokens too: the BOS, system prompt and role "
                "headers are identical in every input, and under mean pooling their weight shrinks as "
                "1/length, which adds a length effect to the direction and the scores. Use an assistant_* "
                "mode to pool over the text's own tokens."
            )
        set_seed(seed)
        ensure_dir(self.run_dir)
        json_dump(os.path.join(self.run_dir, "config.json"), self.config)
        json_dump(os.path.join(self.run_dir, "concept.json"), self.concept.to_dict())
        self.logger.log("config", {"config": self.config, "concept": self.concept.to_dict()})

        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        num_layers = self.model_bundle.num_layers
        eos_ids = _eos_token_ids(model, tokenizer)
        gen_bs = int(train_cfg.get("gen_batch_size", 16))

        dropped_items: Dict[str, List[int]] = {}

        def pooled_reps(ids_list, starts, ends, mode, last_k, label, vectors=None):
            """Pooled reps ``(n, layers, dim)`` and mean token norms ``(n, layers)`` over each span, and,
            given ``vectors``, each item's per-token projections ``(n_tokens, layers)``.

            An item with no tokens of its own (an empty completion or text) is left out and listed
            under ``dropped_items[label]``; a side left with no items raises.
            """
            reps, norms, tokens = [], [], []
            for i, (ids, start, end) in enumerate(zip(ids_list, starts, ends)):
                hs_layers = forward_hidden_states_all_layers(model, ids.reshape(1, -1))
                try:
                    reps.append(reps_from_hs_layers(hs_layers, prompt_len=start, mode=mode, last_k=last_k, span_end=end))
                except EmptySpanError:
                    dropped_items.setdefault(label, []).append(i)
                    continue
                norms.append(_mean_token_norms(hs_layers, start, end))
                if vectors is not None:
                    tokens.append(_span_token_projections(hs_layers, vectors, start, end))
            if label in dropped_items:
                self._warn(
                    f"{label}: items {dropped_items[label]} have no tokens to pool (an empty completion or text) "
                    "and are left out."
                )
            if not reps:
                raise ValueError(f"{label}: no item has tokens to pool (every completion or text is empty).")
            return np.stack(reps, axis=0).astype(np.float32), np.stack(norms, axis=0), tokens

        self._status(f"Training concept '{self.concept.name}' with model {self.model_bundle.model_id}")
        phase1_action = "Reading training texts" if train_prompt_mode == "texts" else "Generating training completions"
        self._status(f"[Phase 1/4] {phase1_action} ({train_prompt_mode})")
        # Each training item is pooled over [start, end): a generated completion up to (not
        # including) its end-of-sequence token, or a text's own tokens.
        train_pos_ids, train_neg_ids = [], []
        train_pos_starts: List[Optional[int]] = []
        train_neg_starts: List[Optional[int]] = []
        train_pos_ends: List[Optional[int]] = []
        train_neg_ends: List[Optional[int]] = []

        if train_prompt_mode == "shared":
            total_train = len(train_prompts)
            progress_every = max(1, total_train // 5) if total_train else 1
            if not train_prompts:
                raise ValueError("prompts.train_questions is empty; provide training prompts.")
            if train_prompts_pos or train_prompts_neg:
                self._warn(
                    "train_prompt_mode=shared; ignoring prompts.train_questions_pos/train_questions_neg."
                )
            pos_results = generate_batch(
                model,
                tokenizer,
                self.concept.pos_system,
                train_prompts,
                train_cfg["train_max_new_tokens"],
                train_cfg["train_greedy"],
                temperature=train_cfg.get("train_temperature", 0.7),
                top_p=train_cfg.get("train_top_p", 0.9),
                batch_size=gen_bs,
                warn=self._warn,
                warn_prefix="train_questions (pos_system)",
            )
            neg_results = generate_batch(
                model,
                tokenizer,
                self.concept.neg_system,
                train_prompts,
                train_cfg["train_max_new_tokens"],
                train_cfg["train_greedy"],
                temperature=train_cfg.get("train_temperature", 0.7),
                top_p=train_cfg.get("train_top_p", 0.9),
                batch_size=gen_bs,
                warn=self._warn,
                warn_prefix="train_questions (neg_system)",
            )
            for i, q in enumerate(train_prompts):
                ids_pos, plen_pos = pos_results[i]
                ids_neg, plen_neg = neg_results[i]
                train_pos_ids.append(ids_pos)
                train_neg_ids.append(ids_neg)
                train_pos_starts.append(plen_pos)
                train_neg_starts.append(plen_neg)
                train_pos_ends.append(_completion_end(ids_pos, plen_pos, eos_ids))
                train_neg_ends.append(_completion_end(ids_neg, plen_neg, eos_ids))

                pos_completion = tokenizer.decode(ids_pos[plen_pos:], skip_special_tokens=True)
                neg_completion = tokenizer.decode(ids_neg[plen_neg:], skip_special_tokens=True)
                self.logger.log(
                    "train_gen",
                    {
                        "i": i,
                        "question": q,
                        "pos_label": self.concept.pos_label,
                        "neg_label": self.concept.neg_label,
                        "pos_system": self.concept.pos_system,
                        "neg_system": self.concept.neg_system,
                        "pos_text": tokenizer.decode(ids_pos, skip_special_tokens=False),
                        "neg_text": tokenizer.decode(ids_neg, skip_special_tokens=False),
                        "pos_completion": pos_completion,
                        "neg_completion": neg_completion,
                    },
                )
                if total_train and ((i + 1) % progress_every == 0 or (i + 1) == total_train):
                    self._status(f"Generated training completions {i + 1}/{total_train}")
        elif train_prompt_mode == "opposed":
            if train_prompts:
                self._warn("train_prompt_mode=opposed; ignoring prompts.train_questions.")
            if not train_prompts_pos or not train_prompts_neg:
                raise ValueError(
                    "train_prompt_mode=opposed requires prompts.train_questions_pos and prompts.train_questions_neg."
                )
            for side, questions, system, label in (
                ("pos", train_prompts_pos, self.concept.pos_system, self.concept.pos_label),
                ("neg", train_prompts_neg, self.concept.neg_system, self.concept.neg_label),
            ):
                ids_list, starts, ends = (
                    (train_pos_ids, train_pos_starts, train_pos_ends) if side == "pos"
                    else (train_neg_ids, train_neg_starts, train_neg_ends)
                )
                progress_every_side = max(1, len(questions) // 5)
                for i, q in enumerate(questions):
                    ids, plen = generate_once(
                        model,
                        tokenizer,
                        system,
                        q,
                        train_cfg["train_max_new_tokens"],
                        train_cfg["train_greedy"],
                        temperature=train_cfg.get("train_temperature", 0.7),
                        top_p=train_cfg.get("train_top_p", 0.9),
                        warn=self._warn,
                        warn_prefix=f"train_questions_{side}[{i}] ({side}_system)",
                    )
                    ids_list.append(ids)
                    starts.append(plen)
                    ends.append(_completion_end(ids, plen, eos_ids))
                    self.logger.log(
                        f"train_gen_{side}",
                        {
                            "i": i,
                            "question": q,
                            "label": label,
                            "system": system,
                            "text": tokenizer.decode(ids, skip_special_tokens=False),
                            "completion": tokenizer.decode(ids[plen:], skip_special_tokens=True),
                        },
                    )
                    if (i + 1) % progress_every_side == 0 or (i + 1) == len(questions):
                        self._status(f"Generated {side} training completions {i + 1}/{len(questions)}")
        else:
            if train_prompts:
                self._warn("train_prompt_mode=texts; ignoring prompts.train_questions.")
            if train_prompts_pos or train_prompts_neg:
                self._warn(
                    "train_prompt_mode=texts; ignoring prompts.train_questions_pos/train_questions_neg."
                )
            train_pos_texts = list(self.concept.train_pos_texts)
            train_neg_texts = list(self.concept.train_neg_texts)
            if not train_pos_texts or not train_neg_texts:
                raise ValueError(
                    "train_prompt_mode=texts requires concept.train_pos_texts and concept.train_neg_texts."
                )
            for side, texts, label in (
                ("pos", train_pos_texts, self.concept.pos_label),
                ("neg", train_neg_texts, self.concept.neg_label),
            ):
                ids_list, starts, ends = (
                    (train_pos_ids, train_pos_starts, train_pos_ends) if side == "pos"
                    else (train_neg_ids, train_neg_starts, train_neg_ends)
                )
                progress_every_side = max(1, len(texts) // 5)
                for i, s in enumerate(texts):
                    input_ids, text_start, text_end = _encode_text_for_read(
                        tokenizer,
                        s,
                        train_text_system,
                        role=text_role,
                        prompt=text_prompt,
                        warn=self._warn,
                        warn_prefix=f"train_{side}_texts[{i}]",
                    )
                    ids_1d = input_ids[0].detach().cpu()
                    ids_list.append(ids_1d)
                    starts.append(text_start)
                    ends.append(text_end)
                    self.logger.log(
                        f"train_read_{side}",
                        {
                            "i": i,
                            "label": label,
                            "system": train_text_system,
                            "text": tokenizer.decode(ids_1d, skip_special_tokens=False),
                        },
                    )
                    if (i + 1) % progress_every_side == 0 or (i + 1) == len(texts):
                        self._status(f"Read {side} training texts {i + 1}/{len(texts)}")

        self._status("[Phase 2/4] Forward passes for training reps")
        reps_pos, norms_pos, _ = pooled_reps(train_pos_ids, train_pos_starts, train_pos_ends,
                                             train_readout_mode, train_readout_last_k, "train_pos")
        reps_neg, norms_neg, _ = pooled_reps(train_neg_ids, train_neg_starts, train_neg_ends,
                                             train_readout_mode, train_readout_last_k, "train_neg")
        if reps_pos.shape[1] != num_layers:
            raise ValueError("Layer count mismatch between model and reps.")
        # Mean residual norm of the training tokens per layer: the unit of alpha_unit="resid_norm".
        resid_norm = np.concatenate([norms_pos, norms_neg], axis=0).mean(axis=0)

        concept_vectors = np.zeros((num_layers, reps_pos.shape[2]), dtype=np.float32)
        for l in range(num_layers):
            concept_vectors[l] = concept_direction(reps_pos[:, l, :], reps_neg[:, l, :], direction_normalization)

        # ---- Evaluation: one score per eval item and layer, on the same kind of data as training.
        # Texts mode evaluates on eval texts. Shared/opposed mode evaluates on completions generated
        # for eval_questions, read like the training completions; eval texts given with such a concept
        # are read too, but only to report how the vector transfers to them (not to pick the layer).
        eval_pos_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        eval_neg_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        transfer_pos_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        transfer_neg_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        eval_pos_tokens: List[np.ndarray] = []
        eval_neg_tokens: List[np.ndarray] = []
        eval_prompts = list(prompts_cfg["eval_questions"])
        did_eval = False
        eval_readout: Optional[Dict[str, Any]] = None
        transfer_readout: Optional[Dict[str, Any]] = None
        text_readout = {
            "role": text_role,
            "prompt": text_prompt,
            "system": eval_system,
            "mode": train_readout_mode,
            "last_k": train_readout_last_k,
        }

        def read_eval_texts(event: str) -> Dict[str, Tuple[np.ndarray, List[np.ndarray]]]:
            """Per-item scores ``(n, layers)`` and per-item token projections of the eval texts."""
            out = {}
            for side, texts in (("pos", self.concept.eval_pos_texts), ("neg", self.concept.eval_neg_texts)):
                ids_list, starts, ends = [], [], []
                for i, s in enumerate(texts):
                    input_ids, text_start, text_end = _encode_text_for_read(
                        tokenizer,
                        s,
                        eval_system,
                        role=text_role,
                        prompt=text_prompt,
                        warn=self._warn,
                        warn_prefix=f"eval_{side}_texts[{i}]",
                    )
                    ids_list.append(input_ids[0])
                    starts.append(text_start)
                    ends.append(text_end)
                    self.logger.log(
                        f"{event}_{side}", {"i": i, "text": tokenizer.decode(input_ids[0], skip_special_tokens=False)}
                    )
                reps, _, tokens = pooled_reps(ids_list, starts, ends, train_readout_mode, train_readout_last_k,
                                              f"{event.split('_')[0]}_{side}", vectors=concept_vectors)
                out[side] = (np.einsum("nld,ld->nl", reps, concept_vectors).astype(np.float32), tokens)
                self._status(f"Read {len(texts)} eval {side} texts")
            return out

        if eval_cfg.get("do_eval", True):
            if eval_mode == "auto":
                if train_prompt_mode != "texts":
                    eval_mode = "generate"
                elif eval_texts_given:
                    eval_mode = "read"
                else:
                    self._warn(
                        "eval_mode=auto with train_prompt_mode=texts but no eval_pos_texts/"
                        "eval_neg_texts; skipping evaluation (generate-mode eval would contrast "
                        "pos_system vs neg_system, which texts-mode concepts typically leave identical)."
                    )
                    eval_mode = "skip"

            if eval_mode == "read":
                if eval_texts_given:
                    self._status("[Phase 3/4] Reading eval texts")
                    if train_prompt_mode == "texts" and eval_system != train_text_system:
                        self._warn(
                            "evaluation.eval_system differs from the system prompt the training texts were "
                            "read with; the layer sweep then measures a different readout than training."
                        )
                    if train_prompt_mode != "texts":
                        self._warn(
                            f"eval_mode=read with train_prompt_mode={train_prompt_mode}: the layer is chosen on "
                            "eval texts read on their own, while the vector was trained on completions generated "
                            "in a chat; eval_mode=auto evaluates on generated completions instead."
                        )
                        if readout["read_mode"] != readout["train_mode"]:
                            self._warn(
                                f"readout.read_mode={readout['read_mode']} is not used in {train_prompt_mode} "
                                f"mode: eval texts are pooled with readout.train_mode={readout['train_mode']}, "
                                "the readout the vector was trained with."
                            )
                    read = read_eval_texts("eval_read")
                    eval_pos_scores_mat, eval_pos_tokens = read["pos"]
                    eval_neg_scores_mat, eval_neg_tokens = read["neg"]
                    did_eval = True
                    eval_readout = dict(text_readout)
                else:
                    self._warn("eval_mode=read but eval_pos_texts/eval_neg_texts are empty.")
            elif eval_mode == "generate":
                used_questions = train_prompts if train_prompt_mode == "shared" else train_prompts_pos + train_prompts_neg
                repeated = sorted({str(q) for q in eval_prompts} & {str(q) for q in used_questions})
                if repeated and train_prompt_mode != "texts":
                    self._warn(
                        f"{len(repeated)} eval_questions are also training questions, so the evaluation is not "
                        f"held out: {repeated[:3]}"
                    )
                if len(eval_prompts) > 0:
                    self._status("[Phase 3/4] Generating eval completions")
                    eval_bs = eval_cfg.get("gen_batch_size")
                    eval_bs = gen_bs if eval_bs is None else int(eval_bs)
                    side_scores = {}
                    for side, system in (("pos", self.concept.pos_system), ("neg", self.concept.neg_system)):
                        results = generate_batch(
                            model,
                            tokenizer,
                            system,
                            eval_prompts,
                            eval_cfg["eval_max_new_tokens"],
                            eval_cfg["eval_greedy"],
                            temperature=eval_cfg.get("eval_temperature", 0.7),
                            top_p=eval_cfg.get("eval_top_p", 0.9),
                            batch_size=eval_bs,
                            warn=self._warn,
                            warn_prefix=f"eval_questions ({side}_system)",
                        )
                        for i, (ids, plen) in enumerate(results):
                            self.logger.log(
                                f"eval_gen_{side}",
                                {
                                    "i": i,
                                    "question": eval_prompts[i],
                                    "text": tokenizer.decode(ids, skip_special_tokens=False),
                                    "completion": tokenizer.decode(ids[plen:], skip_special_tokens=True),
                                },
                            )
                        reps, _, tokens = pooled_reps(
                            [ids for ids, _ in results],
                            [plen for _, plen in results],
                            [_completion_end(ids, plen, eos_ids) for ids, plen in results],
                            readout["train_mode"],
                            readout["train_last_k"],
                            f"eval_{side}",
                            vectors=concept_vectors,
                        )
                        side_scores[side] = (np.einsum("nld,ld->nl", reps, concept_vectors).astype(np.float32), tokens)
                    eval_pos_scores_mat, eval_pos_tokens = side_scores["pos"]
                    eval_neg_scores_mat, eval_neg_tokens = side_scores["neg"]
                    did_eval = True
                    eval_readout = {
                        "role": "assistant",
                        "source": "completions generated for eval_questions under pos_system/neg_system",
                        "mode": readout["train_mode"],
                        "last_k": readout["train_last_k"],
                    }
                if eval_texts_given and train_prompt_mode != "texts":
                    self._status("Reading eval texts (transfer check)")
                    read = read_eval_texts("transfer_read")
                    transfer_pos_scores_mat, transfer_neg_scores_mat = read["pos"][0], read["neg"][0]
                    transfer_readout = dict(text_readout)

        n_eval_pos, n_eval_neg = eval_pos_scores_mat.shape[0], eval_neg_scores_mat.shape[0]
        if did_eval and min(n_eval_pos, n_eval_neg) < MIN_EVAL_ITEMS_PER_POLE:
            self._warn(
                f"Only {n_eval_pos} positive and {n_eval_neg} negative eval items: the layer choice and best_d "
                f"rest on very few items (with selection_split, about half of them each). Use at least "
                f"{MIN_EVAL_ITEMS_PER_POLE} per pole."
            )
        if not did_eval:
            reason = "evaluation disabled"
            if eval_cfg.get("do_eval", True):
                if eval_mode == "read":
                    reason = "missing eval_pos_texts/eval_neg_texts"
                elif eval_mode == "skip":
                    reason = "no eval texts provided for texts-mode concept"
                elif len(eval_prompts) == 0:
                    reason = "empty eval_questions"
            self._warn(reason)
            self.logger.log("eval_skipped", {"reason": reason})

        # ---- Layer sweep on all eval items (diagnostic curves and the plot).
        P = eval_pos_scores_mat.astype(np.float64)
        N = eval_neg_scores_mat.astype(np.float64)
        nan_layers = np.full((num_layers,), np.nan)
        if did_eval:
            eval_pos_mean, eval_neg_mean = P.mean(axis=0), N.mean(axis=0)
            eval_pos_std = P.std(axis=0, ddof=1) if n_eval_pos > 1 else nan_layers.copy()
            eval_neg_std = N.std(axis=0, ddof=1) if n_eval_neg > 1 else nan_layers.copy()
            sweep_d = _cohen_d_columns(P, N)
            # float64: p-values of well-separated sets underflow to 0 in float32.
            sweep_p = _welch_p_columns(P, N)
            # Scores are raw projections with an arbitrary offset. The midpoint between the eval
            # poles and the pooled within-pole SD give each layer's zero and unit (scores_z).
            score_center = 0.5 * (eval_pos_mean + eval_neg_mean)
            score_scale = _pooled_sd_columns(P, N)
            token_center, token_scale = _token_references(eval_pos_tokens, eval_neg_tokens)
        else:
            eval_pos_mean, eval_neg_mean, eval_pos_std, eval_neg_std = (nan_layers.copy() for _ in range(4))
            sweep_d, sweep_p, score_center, score_scale = (nan_layers.copy() for _ in range(4))
            token_center, token_scale = nan_layers.copy(), nan_layers.copy()

        # ---- Best layer, and the statistics reported for it.
        search_intervals = None
        search_layers = None
        search_default = None
        search_intervals_plot = None
        selection: Optional[str] = None
        best_d = best_p = best_d_selection = best_d_all_eval = sweep_max_d = sweep_max_d_perm_p = None
        best_d_ci95: Optional[List[float]] = None
        pos_sel = neg_sel = np.zeros((0,), dtype=np.int64)
        est_pos, est_neg = P, N
        if train_cfg["do_layer_sweep"] and did_eval:
            search_info = _normalize_best_layer_search(train_cfg.get("best_layer_search"), num_layers)
            search_intervals = search_info["intervals"]
            search_layers = search_info["layers"]
            search_default = search_info["is_default"]
            if not search_default:
                search_intervals_plot = search_intervals
            selection = "all_eval"
            if selection_split is not None:
                split_rng = np.random.default_rng([seed, 1])
                pos_sel, pos_est = _split_indices(n_eval_pos, selection_split, split_rng)
                neg_sel, neg_est = _split_indices(n_eval_neg, selection_split, split_rng)
                if min(len(pos_sel), len(pos_est), len(neg_sel), len(neg_est)) < 2:
                    self._warn(
                        "evaluation.selection_split needs at least 2 eval items per pole on each side of the "
                        "split; choosing the layer and reporting d on all eval items (post-selection)."
                    )
                    pos_sel = neg_sel = np.zeros((0,), dtype=np.int64)
                else:
                    selection = "split"
            if selection == "split":
                sel_d = _cohen_d_columns(P[pos_sel], N[neg_sel])
                best_layer = _select_best_layer(sel_d, _welch_p_columns(P[pos_sel], N[neg_sel]), search_layers, sweep_select)
                best_d_selection = float(sel_d[best_layer])
                est_pos, est_neg = P[pos_est], N[neg_est]
            else:
                best_layer = _select_best_layer(sweep_d, sweep_p, search_layers, sweep_select)
                self._warn(
                    f"best_layer was chosen over {len(search_layers)} layers on the same eval items that "
                    "measure best_d/best_p, so those are optimistic. Set evaluation.selection_split for a "
                    "held-out estimate."
                )
            n_permutations = int(eval_cfg.get("n_permutations") or 0)
            if n_permutations > 0:
                sweep_max_d, sweep_max_d_perm_p = _max_d_permutation_p(
                    P[:, search_layers], N[:, search_layers], n_permutations, np.random.default_rng([seed, 2])
                )
        else:
            best_layer = pick_layer(num_layers, train_cfg.get("layer_idx"), train_cfg.get("layer_frac", 0.75))
            if did_eval:
                selection = "fixed_layer"
        if did_eval:
            best_d = float(cohen_d_np(est_pos[:, best_layer], est_neg[:, best_layer]))
            best_p_arr = _welch_p_columns(est_pos[:, [best_layer]], est_neg[:, [best_layer]])
            best_p = None if np.isnan(best_p_arr[0]) else float(best_p_arr[0])
            best_d_all_eval = float(sweep_d[best_layer])
            best_d_ci95 = _bootstrap_d_ci(
                est_pos[:, best_layer], est_neg[:, best_layer],
                int(eval_cfg.get("n_bootstrap") or 0), np.random.default_rng([seed, 3]),
            )
        notes = {
            "split": "best_layer was chosen on the selection part of the eval items; best_d, best_p and "
                     "best_d_ci95 are measured on the held-out rest.",
            "all_eval": "best_d/best_p are post-selection values: best_layer was chosen on the same eval items. "
                        "Set evaluation.selection_split for a held-out estimate.",
            "fixed_layer": "best_layer was fixed in advance (training.layer_idx / layer_frac); best_d/best_p use "
                           "all eval items.",
        }

        def _opt(x: Any) -> Optional[float]:
            return None if x is None or not np.isfinite(x) else float(x)

        # ---- Transfer check: the eval texts of a shared/opposed concept, at the layer chosen above.
        TP = transfer_pos_scores_mat.astype(np.float64)
        TN = transfer_neg_scores_mat.astype(np.float64)
        transfer_d, transfer_p = nan_layers.copy(), nan_layers.copy()
        transfer_eval: Optional[Dict[str, Any]] = None
        if TP.shape[0] > 0 and TN.shape[0] > 0:
            transfer_d, transfer_p = _cohen_d_columns(TP, TN), _welch_p_columns(TP, TN)
            transfer_eval = {
                "readout": transfer_readout,
                "n_pos": int(TP.shape[0]),
                "n_neg": int(TN.shape[0]),
                "d_at_best_layer": _opt(transfer_d[best_layer]),
                "p_at_best_layer": _opt(transfer_p[best_layer]),
                "d_ci95_at_best_layer": _bootstrap_d_ci(
                    TP[:, best_layer], TN[:, best_layer],
                    int(eval_cfg.get("n_bootstrap") or 0), np.random.default_rng([seed, 4]),
                ),
                "note": "The concept's eval texts, read on their own and scored with the vector at best_layer. "
                        "They do not take part in choosing the layer. They test whether the vector finds the "
                        "concept outside the generation context it was trained in (where the system prompt "
                        "that defines the concept is in context).",
            }

        sigma = float(score_scale[best_layer])

        self.concept_vectors = concept_vectors
        self.best_layer = best_layer
        self.sigma_best_layer = sigma if np.isfinite(sigma) else None
        self.score_center = score_center
        self.score_scale = score_scale
        self.token_center = token_center
        self.token_scale = token_scale
        self.resid_norm = resid_norm
        self.sweep_d = sweep_d
        self.sweep_p = sweep_p

        self._status("[Phase 4/4] Sweep + artifact saving")
        self.logger.log(
            "sweep_full",
            {
                "best_layer": best_layer,
                "best_d": _opt(best_d),
                "best_p": _opt(best_p),
                "selection": selection,
                "best_layer_search": train_cfg.get("best_layer_search"),
                "best_layer_search_intervals": search_intervals,
                "best_layer_search_layers": search_layers,
                "best_layer_search_default": search_default,
                "sweep_d": sweep_d.tolist(),
                "sweep_p": [_opt(x) for x in sweep_p],
                "eval_pos_mean": eval_pos_mean.tolist(),
                "eval_neg_mean": eval_neg_mean.tolist(),
                "eval_pos_std": eval_pos_std.tolist(),
                "eval_neg_std": eval_neg_std.tolist(),
                "sigma_best_layer": _opt(sigma),
            },
        )

        plots_dir = os.path.join(self.run_dir, "plots")
        ensure_dir(plots_dir)
        if plot_cfg.get("enabled", True) and did_eval:
            if plot_cfg.get("sweep_plot", True):
                ok = plot_sweep(
                    sweep_d,
                    sweep_p,
                    os.path.join(plots_dir, "sweep.png"),
                    best_layer=best_layer,
                    search_intervals=search_intervals_plot,
                )
                self.logger.log("plot_sweep", {"ok": ok})
            if plot_cfg.get("score_hist", True):
                ok = plot_score_hist(
                    eval_pos_scores_mat[:, best_layer],
                    eval_neg_scores_mat[:, best_layer],
                    os.path.join(plots_dir, "score_hist.png"),
                )
                self.logger.log("plot_score_hist", {"ok": ok})
        if plot_cfg.get("enabled", True) and plot_cfg.get("sweep_plot", True) and transfer_eval is not None:
            ok = plot_sweep(transfer_d, transfer_p, os.path.join(plots_dir, "transfer_sweep.png"), best_layer=best_layer)
            self.logger.log("plot_transfer_sweep", {"ok": ok})

        tensors = {
            "best_layer": np.array([best_layer], dtype=np.int32),
            "best_v": concept_vectors[best_layer].astype(np.float32),
            "concept_vectors": concept_vectors.astype(np.float32),
            "sweep_d": sweep_d.astype(np.float32),
            "sweep_p": sweep_p.astype(np.float64),
            "eval_pos_mean": eval_pos_mean.astype(np.float32),
            "eval_neg_mean": eval_neg_mean.astype(np.float32),
            "eval_pos_std": eval_pos_std.astype(np.float32),
            "eval_neg_std": eval_neg_std.astype(np.float32),
            "eval_pos_scores_mat": eval_pos_scores_mat.astype(np.float32),
            "eval_neg_scores_mat": eval_neg_scores_mat.astype(np.float32),
            "eval_selection_pos_idx": np.asarray(pos_sel, dtype=np.int32),
            "eval_selection_neg_idx": np.asarray(neg_sel, dtype=np.int32),
            "score_center": score_center.astype(np.float64),
            "score_scale": score_scale.astype(np.float64),
            "token_center": token_center.astype(np.float64),
            "token_scale": token_scale.astype(np.float64),
            "resid_norm": resid_norm.astype(np.float64),
            "transfer_pos_scores_mat": transfer_pos_scores_mat.astype(np.float32),
            "transfer_neg_scores_mat": transfer_neg_scores_mat.astype(np.float32),
            "transfer_sweep_d": transfer_d.astype(np.float32),
            "transfer_sweep_p": transfer_p.astype(np.float64),
        }
        if out_cfg.get("save_reps_to_npz", True):
            tensors.update(
                {
                    "train_pos_reps": reps_pos.astype(np.float32),
                    "train_neg_reps": reps_neg.astype(np.float32),
                }
            )
        npz_path = os.path.join(self.run_dir, "tensors.npz")
        np.savez_compressed(npz_path, **tensors)

        n_train_pos = len(train_pos_ids)
        n_train_neg = len(train_neg_ids)
        if train_prompt_mode == "shared":
            n_train = n_train_pos
        else:
            n_train = n_train_pos + n_train_neg

        metrics = {
            "best_layer": best_layer,
            "best_d": _opt(best_d),
            "best_p": _opt(best_p),
            "best_d_ci95": best_d_ci95,
            "selection": selection,
            "best_layer_selection_note": notes.get(selection),
            "best_d_selection": _opt(best_d_selection),
            "best_d_all_eval": _opt(best_d_all_eval),
            "n_eval_selection": {"pos": int(len(pos_sel)), "neg": int(len(neg_sel))} if selection == "split" else None,
            "sweep_max_d": _opt(sweep_max_d),
            "sweep_max_d_perm_p": _opt(sweep_max_d_perm_p),
            "n_permutations": int(eval_cfg.get("n_permutations") or 0) if sweep_max_d is not None else None,
            "best_layer_search": train_cfg.get("best_layer_search"),
            "best_layer_search_intervals": search_intervals,
            "best_layer_search_layers": search_layers,
            "best_layer_search_default": search_default,
            "sigma_best_layer": _opt(sigma),
            "num_layers": num_layers,
            "n_train": n_train,
            "n_train_pos": n_train_pos,
            "n_train_neg": n_train_neg,
            "train_prompt_mode": train_prompt_mode,
            "direction_normalization": direction_normalization,
            "train_text_role": text_role,
            "gen_batch_size": gen_bs if train_prompt_mode == "shared" else None,
            "n_eval": n_eval_pos + n_eval_neg,
            "n_eval_pos": n_eval_pos,
            "n_eval_neg": n_eval_neg,
            "eval_mode_used": eval_mode if did_eval else None,
            "eval_readout": eval_readout,
            "transfer_eval": transfer_eval,
            "dropped_items": dropped_items or None,
        }
        json_dump(os.path.join(self.run_dir, "metrics.json"), metrics)
        self.logger.log("artifacts_saved", {"npz_path": npz_path})
        if self.config["output"].get("pretty_json_log", False) and self.config["output"].get("log_jsonl", True):
            pretty_path = os.path.join(self.run_dir, "log.pretty.json")
            jsonl_to_pretty(os.path.join(self.run_dir, "log.jsonl"), pretty_path)
        self._status(f"Saved artifacts to {self.run_dir}")
        return self

    def _require_trained(self) -> None:
        if self.concept_vectors is None or self.best_layer is None:
            raise ValueError("ConceptProbe is not trained or loaded.")

    def _status(self, message: str) -> None:
        if self.console is not None:
            self.console.info(message)

    def _warn(self, message: str) -> None:
        if self.console is not None:
            self.console.warn(message)
        self.logger.log("warning", {"message": message})

    def _layer_selection(self, override: Optional[Dict[str, Any]] = None) -> List[int]:
        self._require_trained()
        selection = dict(self.config["evaluation"]["layer_selection"])
        if override:
            selection = deep_merge(selection, override)
        return select_layers(selection, int(self.best_layer), self.model_bundle.num_layers)

    def _random_control_vectors(self, random_vector_seed: Optional[int]) -> Tuple[np.ndarray, int]:
        """Same-norm random vectors per layer: this probe's random-direction control.

        The seed combines ``random_vector_seed`` (or ``random.seed``) with the probe's identity
        (concept name and run directory), so score_texts, score_prompts (scoring and steering)
        and multi_probe_score_prompts all use the same control vectors for the same probe.
        """
        base = _resolve_random_control_seed(random_vector_seed, self.config.get("random", {}).get("seed", 0))
        identity = f"{self.concept.name}|{os.path.basename(os.path.normpath(self.run_dir))}"
        seed = _mix_random_control_seed(base, zlib.crc32(identity.encode("utf-8")))
        return _random_vectors_same_norm(self.concept_vectors, seed=seed), seed

    @staticmethod
    def _references(center: Optional[np.ndarray], scale: Optional[np.ndarray], layers: List[int]):
        if center is None or scale is None:
            return None
        c = np.asarray(center, dtype=np.float64)[layers]
        u = np.asarray(scale, dtype=np.float64)[layers]
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(u)) and np.all(u > 0)):
            return None
        return c, u

    def _token_z(
        self, scores_per_layer: np.ndarray, layers: List[int], aggregate: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Standardized single-token scores per selected layer, and their aggregate over layers.

        ``(score - token_center) / token_scale``: 0 is the midpoint between the mean token score of
        the eval poles, 1 the pooled within-pole SD of single tokens. None without eval references.
        """
        refs = self._references(self.token_center, self.token_scale, layers)
        if refs is None:
            return None
        z = ((np.asarray(scores_per_layer, dtype=np.float64) - refs[0]) / refs[1]).astype(np.float32)
        z_agg = z.mean(axis=1) if aggregate == "mean" else z.sum(axis=1)
        return z, z_agg.astype(np.float32)

    def _item_z(self, span_scores_per_layer: np.ndarray, layers: List[int], aggregate: str) -> Optional[float]:
        """Standardized score of an item: its mean over the span, per selected layer, as
        ``(mean - score_center) / score_scale`` (the per-item references), aggregated over layers."""
        refs = self._references(self.score_center, self.score_scale, layers)
        span = np.asarray(span_scores_per_layer, dtype=np.float64)
        if refs is None or span.shape[0] == 0:
            return None
        z = (span.mean(axis=0) - refs[0]) / refs[1]
        return float(z.mean() if aggregate == "mean" else z.sum())

    def _system_for(self, role: str, system_prompt: Optional[str], prompt: Optional[str]) -> Optional[str]:
        """The system prompt for reading texts with ``role``; raises on settings ``role`` would ignore."""
        if role not in TEXT_ROLES:
            raise ValueError(f"role must be one of {TEXT_ROLES}; got {role!r}")
        if role == "raw":
            if system_prompt or prompt is not None:
                raise ValueError("role='raw' reads each text alone; system_prompt and prompt would be ignored.")
            return None
        if role == "user" and prompt is not None:
            raise ValueError("prompt is a user turn before the text, so it only applies to role='assistant'.")
        return self.config["prompts"].get("neutral_system") if system_prompt is None else system_prompt

    def score_texts(
        self,
        texts: List[PromptLike],
        system_prompt: Optional[str] = None,
        layer_selection: Optional[Dict[str, Any]] = None,
        output_subdir: str = "scores",
        save_html: Optional[bool] = None,
        batch_subdir: Optional[str] = None,
        random_scoring_vector: bool = False,
        random_vector_seed: Optional[int] = None,
        reference_layer: Optional[int] = None,
        role: str = "raw",
        prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Score every token of each text with the probe (token-level scores, no pooling).

        By default each text is read raw: the BOS token and the text, with no chat template and
        no system prompt. ``role="user"`` reads it as a user turn (a message list is read as
        given) after ``system_prompt``, or ``prompts.neutral_system`` if that is set;
        ``role="assistant"`` reads it as the model's reply, after that system prompt and an
        optional user turn ``prompt``. :func:`preview_input` shows the exact input, and each
        record carries it as ``input_text`` together with ``text_span``, the text's own tokens.

        The npz holds raw projections (``scores_per_layer``, ``scores_agg``) and, when the probe
        has eval references, standardized single-token scores (``scores_z_per_layer``,
        ``scores_agg_z``: 0 is the midpoint between the eval poles, 1 the within-pole SD of single
        tokens), which the heatmaps then show. Each record has the text's mean score over its own
        tokens, ``score_mean``, and ``score_mean_z``, standardized with the per-item references.
        """
        self._require_trained()
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = self._system_for(role, system_prompt, prompt)
        save_html = self.config["plots"].get("heatmap_html", True) if save_html is None else save_html
        save_token_scores_npz = self.config["output"].get("save_token_scores_npz", True)
        scoring_seed: Optional[int] = None
        scoring_vectors = self.concept_vectors
        if random_scoring_vector:
            scoring_vectors, scoring_seed = self._random_control_vectors(random_vector_seed)
            self._warn(
                f"score_texts using random scoring vector control (same-norm per layer, seed={scoring_seed})."
            )

        base_dir = os.path.join(self.run_dir, output_subdir)
        ensure_dir(base_dir)
        if batch_subdir is None:
            batch_subdir = f"batch_{now_tag()}"
        out_dir = base_dir if batch_subdir == "" else os.path.join(base_dir, batch_subdir)
        ensure_dir(out_dir)
        results = []
        batch_entries = []
        total_texts = len(texts)
        progress_every = max(1, total_texts // 5) if total_texts else 1
        self._status(f"Scoring {total_texts} texts -> {out_dir}")

        for i, text in enumerate(texts):
            input_ids, text_start, text_end = _encode_text_for_read(
                tokenizer,
                text,
                sys_prompt,
                role=role,
                prompt=prompt,
                warn=self._warn,
                warn_prefix=f"score_texts.texts[{i}]",
            )
            used_prompt_system = isinstance(text, list) and any(
                isinstance(m, dict) and m.get("role") == "system" for m in text
            )
            hs_layers = forward_hidden_states_all_layers(model, input_ids)
            scores_per_layer, scores_agg = token_scores_from_hs_layers(
                hs_layers, scoring_vectors, layers, aggregate=aggregate,
                reference_layer=reference_layer,
            )
            calibrated = not (random_scoring_vector or reference_layer is not None)
            standardized = self._token_z(scores_per_layer, layers, aggregate) if calibrated else None
            span = slice(text_start, text_end) if text_start is not None else slice(0, 0)
            score_mean = float(np.mean(scores_agg[span])) if scores_agg[span].size else None
            score_mean_z = self._item_z(scores_per_layer[span], layers, aggregate) if calibrated else None
            token_ids = input_ids[0].detach().cpu().numpy().astype(np.int32)
            tokens = _decode_tokens(tokenizer, token_ids)
            text_span = None if text_start is None else [int(text_start), int(text_end)]

            slug = _prompt_slug_any(text)
            base = f"text_{i:03d}_{slug}"
            if save_token_scores_npz:
                npz_path = os.path.join(out_dir, f"{base}.npz")
                payload: Dict[str, Any] = dict(
                    token_ids=token_ids,
                    scores_per_layer=scores_per_layer,
                    scores_agg=scores_agg,
                    layer_indices=np.array(layers, dtype=np.int32),
                    text_span=np.array(text_span if text_span else [-1, -1], dtype=np.int32),
                )
                if standardized is not None:
                    payload.update(scores_z_per_layer=standardized[0], scores_agg_z=standardized[1])
                np.savez_compressed(npz_path, **payload)
            else:
                npz_path = None

            if save_html:
                shown = scores_agg if standardized is None else standardized[1]
                unit = "raw projection" if standardized is None else "token z: 0 = midpoint of the eval poles"
                html_path = os.path.join(out_dir, f"{base}.html")
                render_token_heatmap(tokens, shown.tolist(), html_path, title=f"{slug} ({unit})")
                batch_entries.append((f"text {i:03d}: {slug} ({unit})", tokens, shown.tolist()))
            else:
                html_path = None

            rec = {
                "text": text,
                "role": role,
                "system_prompt": sys_prompt,
                "system_prompt_overridden": bool(used_prompt_system),
                "input_text": tokenizer.decode(token_ids.tolist(), skip_special_tokens=False),
                "text_span": text_span,
                "layers": layers,
                "score_mean": score_mean,
                "score_mean_z": score_mean_z,
                "npz_path": npz_path,
                "html_path": html_path,
                "scores_standardized": standardized is not None,
                "random_scoring_vector": bool(random_scoring_vector),
                "random_scoring_vector_seed": scoring_seed,
                "batch_dir": out_dir,
            }
            results.append(rec)
            self.logger.log("score_text", rec)
            if total_texts and ((i + 1) % progress_every == 0 or (i + 1) == total_texts):
                self._status(f"Scored text {i + 1}/{total_texts}")

        if save_html and batch_entries:
            batch_path = os.path.join(out_dir, "batch_texts.html")
            render_batch_heatmap(batch_entries, batch_path, title="Batch text scores")
            self.logger.log("score_text_batch", {"html_path": batch_path, "count": len(batch_entries)})

        return results

    def _check_alpha_unit(self, alpha_unit: str) -> None:
        """Raise if the probe lacks the reference that ``alpha_unit`` scales by."""
        if alpha_unit == "sigma" and (self.sigma_best_layer is None or not np.isfinite(self.sigma_best_layer)):
            raise ValueError("sigma_best_layer is not available for alpha_unit='sigma' (the probe has no eval reference).")
        if alpha_unit == "resid_norm" and self.resid_norm is None:
            raise ValueError("resid_norm is not available for alpha_unit='resid_norm'; re-train the probe.")

    def _steering_alpha(
        self, alpha: float, alpha_unit: str, steer_layers: List[int]
    ) -> Tuple[float, Optional[Dict[int, float]]]:
        """The alpha to inject, and per-layer scales of it, for ``alpha`` in ``alpha_unit``."""
        value = float(alpha)
        if alpha_unit == "sigma" and abs(value) > 1e-12:
            return value * float(self.sigma_best_layer), None
        if alpha_unit == "resid_norm":
            return value, {li: float(self.resid_norm[li]) for li in steer_layers}
        return value, None

    def score_prompts(
        self,
        prompts: List[PromptLike],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        greedy: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        layer_selection: Optional[Dict[str, Any]] = None,
        output_subdir: str = "scores",
        alphas: Optional[List[float]] = None,
        alpha_unit: str = "raw",
        steer_layers: Optional[Union[str, List[int]]] = None,
        steer_window_radius: Optional[int] = None,
        steer_distribute: Optional[bool] = None,
        save_html: Optional[bool] = None,
        save_segments: Optional[bool] = None,
        save_generation_logits: Optional[bool] = None,
        generation_logits_top_k: Optional[int] = None,
        generation_logits_dtype: Optional[str] = None,
        batch_subdir: Optional[str] = None,
        random_scoring_vector: bool = False,
        random_steering_vector: bool = False,
        random_vector_seed: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate (optionally steered) completions and score them with the probe.

        Each prompt is a user turn of the chat template, after ``system_prompt`` (or
        ``prompts.neutral_system``) if one is set; by default there is none.
        :func:`preview_input` with ``generation=True`` shows the exact input.

        alphas/alpha_unit: the offset ``alpha * v_l`` is added to the residual stream at every
        token position of each steered layer (divided across layers when steer_distribute=True,
        and compounding through subsequent layers). ``"raw"`` uses alpha as is. ``"sigma"``
        multiplies it by sigma_best_layer, the pooled within-pole SD of the per-item eval scores
        at the best layer, so it is a scale knob, not literally "n standard deviations of
        behavior". ``"resid_norm"`` multiplies it, per layer, by the mean residual norm of the
        training tokens at that layer: alpha=0.1 adds 10% of a typical token's norm.

        ``score_mean`` averages the completion's tokens, up to and not including the
        end-of-sequence token; ``score_mean_z`` standardizes that mean with the per-item references
        (0 is the midpoint between the eval poles, 1 their pooled within-pole SD). The npz and the
        heatmaps carry single-token z scores, standardized with the per-token references.

        seed: seeds python/numpy/torch RNGs before generation for reproducible
        sampling. Note: GPU kernels are not bit-deterministic across
        hardware/library versions, so reproducibility is best-effort.
        """
        self._require_trained()
        alpha_unit = _validate_alpha_unit(alpha_unit)
        if seed is not None:
            set_seed(int(seed))
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = self.config["prompts"].get("neutral_system") if system_prompt is None else system_prompt
        save_html = self.config["plots"].get("heatmap_html", True) if save_html is None else save_html
        save_segments = False if save_segments is None else save_segments
        do_steering = bool(self.config["steering"].get("do_steering", True))
        save_token_scores_npz = self.config["output"].get("save_token_scores_npz", True)
        if save_generation_logits is None:
            save_generation_logits = bool(self.config["output"].get("save_generation_logits", False))
        else:
            save_generation_logits = bool(save_generation_logits)
        if generation_logits_top_k is None:
            generation_logits_top_k = self.config["output"].get("generation_logits_top_k", None)
        if generation_logits_dtype is None:
            generation_logits_dtype = self.config["output"].get("generation_logits_dtype", "float16")
        if save_generation_logits and not save_token_scores_npz:
            self._warn("save_generation_logits requested but output.save_token_scores_npz=false; logits will not be saved.")
        save_generation_logits = bool(save_generation_logits and save_token_scores_npz)
        scoring_seed: Optional[int] = None
        steering_seed: Optional[int] = None
        scoring_vectors = self.concept_vectors
        steering_vectors = self.concept_vectors
        if random_scoring_vector or random_steering_vector:
            control_vectors, control_seed = self._random_control_vectors(random_vector_seed)
        if random_scoring_vector:
            scoring_seed, scoring_vectors = control_seed, control_vectors
            self._warn(
                f"score_prompts using random scoring vector control (same-norm per layer, seed={scoring_seed})."
            )
        if random_steering_vector:
            steering_seed, steering_vectors = control_seed, control_vectors
            self._warn(
                f"score_prompts using random steering vector control (same-norm per layer, seed={steering_seed})."
            )

        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config["steering"]["steer_max_new_tokens"]
        greedy = greedy if greedy is not None else (not self.config["steering"]["steer_sampling"])
        temperature = temperature if temperature is not None else self.config["steering"]["steer_temperature"]
        top_p = top_p if top_p is not None else self.config["steering"]["steer_top_p"]

        if alphas is None:
            alphas = [0.0]
        nonzero_alpha_requested = any(abs(float(a)) > 1e-12 for a in alphas)
        if nonzero_alpha_requested and not do_steering:
            self._warn(
                "Non-zero alphas provided; overriding steering.do_steering=false and enabling steering."
            )
            do_steering = True
        if random_steering_vector and not nonzero_alpha_requested:
            self._warn("random_steering_vector=true but all alphas are zero; no steering is applied.")
        if nonzero_alpha_requested:
            self._check_alpha_unit(alpha_unit)

        if steer_layers is None:
            steer_layers = self.config["steering"]["steer_layers"]
        if steer_window_radius is None:
            steer_window_radius = self.config["steering"]["steer_window_radius"]
        if steer_distribute is None:
            steer_distribute = self.config["steering"]["steer_distribute"]

        steer_layer_list = _resolve_steer_layers(
            steer_layers, int(self.best_layer), steer_window_radius, self.model_bundle.num_layers
        )

        base_dir = os.path.join(self.run_dir, output_subdir)
        ensure_dir(base_dir)
        if batch_subdir is None:
            batch_subdir = f"batch_{now_tag()}"
        out_dir = base_dir if batch_subdir == "" else os.path.join(base_dir, batch_subdir)
        ensure_dir(out_dir)
        results = []
        batch_entries = []
        total_prompts = len(prompts)
        total_alphas = len(alphas)
        total_runs = total_prompts * total_alphas
        self._status(
            f"Scoring {total_prompts} prompts x {total_alphas} alphas ({total_runs} generations) -> {out_dir}"
        )
        show_progress = self.console is not None and self.console.enabled
        if show_progress and total_runs:
            _progress_print(0, total_runs, label="Generating", enabled=True)

        eos_ids = _eos_token_ids(model, tokenizer)
        run_idx = 0
        progress_every = 1
        for i, prompt in enumerate(prompts):
            prompt_slug = _prompt_slug_any(prompt)
            for j, alpha in enumerate(alphas):
                alpha_val, layer_scales = self._steering_alpha(alpha, alpha_unit, steer_layer_list)

                messages, used_prompt_system = _normalize_messages(
                    prompt,
                    sys_prompt,
                    warn=self._warn,
                    warn_prefix=f"score_prompts.prompts[{i}]",
                )
                input_ids = apply_chat(tokenizer, messages, add_generation_prompt=True).to(model.device)
                attn = attention_mask_from_ids(input_ids).to(model.device)
                prompt_len = int(input_ids.shape[-1])

                should_steer = do_steering and abs(float(alpha_val)) > 1e-12
                if should_steer:
                    context = MultiLayerSteererLayerwise(
                        model, steer_layer_list, steering_vectors, alpha_val, distribute=steer_distribute,
                        layer_scales=layer_scales,
                    )
                else:
                    context = None

                if context is not None:
                    with context:
                        gen_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attn,
                            max_new_tokens=max_new_tokens,
                            do_sample=not greedy,
                            temperature=temperature if not greedy else None,
                            top_p=top_p if not greedy else None,
                            pad_token_id=tokenizer.eos_token_id,
                        )[0].detach().cpu()
                        if save_generation_logits:
                            # Generation logits must reflect the steered model that produced the text.
                            _, generation_step_logits = forward_hidden_states_all_layers(
                                model,
                                gen_ids.unsqueeze(0),
                                return_generation_logits=True,
                                generation_prompt_len=prompt_len,
                            )
                        else:
                            generation_step_logits = None
                    # Score with the hooks removed: with them active, the injected vector reads
                    # itself back (score inflation ~ alpha * cos(steer_dir, score_dir) at steered
                    # layers), contaminating the probe readout of the generated text.
                    hs_layers = forward_hidden_states_all_layers(model, gen_ids.unsqueeze(0))
                else:
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_new_tokens,
                        do_sample=not greedy,
                        temperature=temperature if not greedy else None,
                        top_p=top_p if not greedy else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0].detach().cpu()
                    if save_generation_logits:
                        hs_layers, generation_step_logits = forward_hidden_states_all_layers(
                            model,
                            gen_ids.unsqueeze(0),
                            return_generation_logits=True,
                            generation_prompt_len=prompt_len,
                        )
                    else:
                        hs_layers = forward_hidden_states_all_layers(model, gen_ids.unsqueeze(0))
                        generation_step_logits = None
                scores_per_layer, scores_agg = token_scores_from_hs_layers(
                    hs_layers, scoring_vectors, layers, aggregate=aggregate
                )
                generation_logits_payload: Dict[str, np.ndarray] = {}
                if save_generation_logits and generation_step_logits is not None:
                    generation_logits_payload = _pack_generation_logits_arrays(
                        generation_step_logits,
                        logits_top_k=generation_logits_top_k,
                        logits_dtype_name=generation_logits_dtype,
                        logits_source="steered_model" if should_steer else "raw_model",
                    )

                token_ids = gen_ids.numpy().astype(np.int32)
                tokens = _decode_tokens(tokenizer, token_ids)
                completion_end = _completion_end(token_ids, prompt_len, eos_ids)
                completion_text = tokenizer.decode(token_ids[prompt_len:completion_end], skip_special_tokens=True)
                completion_scores = scores_agg[prompt_len:completion_end]
                score_mean = float(np.mean(completion_scores)) if completion_scores.size else float("nan")
                standardized = None if random_scoring_vector else self._token_z(scores_per_layer, layers, aggregate)
                score_mean_z = (
                    None if random_scoring_vector
                    else self._item_z(scores_per_layer[prompt_len:completion_end], layers, aggregate)
                )

                alpha_label = _alpha_label(float(alpha), alpha_unit)
                base = f"prompt_{i:03d}_{prompt_slug}_{alpha_label}"
                if save_token_scores_npz:
                    npz_path = os.path.join(out_dir, f"{base}.npz")
                    npz_payload: Dict[str, Any] = dict(
                        token_ids=token_ids,
                        prompt_len=np.array([prompt_len], dtype=np.int32),
                        completion_end=np.array([completion_end], dtype=np.int32),
                        scores_per_layer=scores_per_layer,
                        scores_agg=scores_agg,
                        layer_indices=np.array(layers, dtype=np.int32),
                    )
                    if standardized is not None:
                        npz_payload.update(scores_z_per_layer=standardized[0], scores_agg_z=standardized[1])
                    npz_payload.update(generation_logits_payload)
                    np.savez_compressed(npz_path, **npz_payload)
                else:
                    npz_path = None

                if save_html:
                    shown = scores_agg if standardized is None else standardized[1]
                    unit = "raw projection" if standardized is None else "token z: 0 = midpoint of the eval poles"
                    html_path = os.path.join(out_dir, f"{base}.html")
                    title = f"prompt {i:03d} | {prompt_slug} | {alpha_label} | {unit}"
                    render_token_heatmap(tokens, shown.tolist(), html_path, title=title)
                    batch_entries.append((title, tokens, shown.tolist()))
                else:
                    html_path = None

                segments_path = None
                if save_segments:
                    segments_path = os.path.join(out_dir, f"{base}_segments.json")
                    segments = segment_token_scores(
                        tokens,
                        scores_agg.tolist(),
                        prompt_len=prompt_len,
                    )
                    with open(segments_path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {"prompt_len": prompt_len, "segments": segments},
                            handle,
                            ensure_ascii=False,
                            indent=2,
                        )

                rec = {
                    "prompt": prompt,
                    "system_prompt": sys_prompt,
                    "system_prompt_overridden": bool(used_prompt_system),
                    "prompt_text": tokenizer.decode(token_ids[:prompt_len].tolist(), skip_special_tokens=False),
                    "alpha": float(alpha),
                    "alpha_unit": alpha_unit,
                    "alpha_value": float(alpha_val),
                    "steer_layer_scales": layer_scales,
                    "sampling_seed": seed,
                    "steer_layers": steer_layer_list,
                    "completion": completion_text,
                    "prompt_len": prompt_len,
                    "completion_end": completion_end,
                    "layers": layers,
                    "npz_path": npz_path,
                    "html_path": html_path,
                    "segments_path": segments_path,
                    "score_mean": score_mean,
                    "score_mean_z": score_mean_z,
                    "do_steering": bool(should_steer),
                    "random_scoring_vector": bool(random_scoring_vector),
                    "random_scoring_vector_seed": scoring_seed,
                    "random_steering_vector": bool(random_steering_vector),
                    "random_steering_vector_seed": steering_seed,
                    "generation_logits_saved": bool(generation_logits_payload),
                    "generation_logits_top_k": (
                        int(generation_logits_payload["generation_logits_topk_k"][0])
                        if "generation_logits_topk_k" in generation_logits_payload
                        else None
                    ),
                    "generation_logits_steps": (
                        int(generation_logits_payload["generation_logits_steps"][0])
                        if "generation_logits_steps" in generation_logits_payload
                        else 0
                    ),
                    "generation_logits_source": (
                        str(generation_logits_payload["generation_logits_source"][0])
                        if "generation_logits_source" in generation_logits_payload
                        else None
                    ),
                    "batch_dir": out_dir,
                }
                results.append(rec)
                self.logger.log("score_prompt", rec)
                run_idx += 1
                if total_runs and (
                    run_idx % progress_every == 0 or run_idx == total_runs
                ):
                    _progress_print(
                        run_idx,
                        total_runs,
                        label="Generating",
                        enabled=show_progress,
                        last=run_idx == total_runs,
                    )
            # Progress bar handles per-run updates; avoid extra log spam here.

        if save_html and batch_entries:
            batch_path = os.path.join(out_dir, "batch_prompts.html")
            render_batch_heatmap(batch_entries, batch_path, title="Batch prompt scores")
            self.logger.log("score_prompt_batch", {"html_path": batch_path, "count": len(batch_entries)})

        return results

    @classmethod
    def load(cls, run_dir: str, model_bundle: ModelBundle) -> "ConceptProbe":
        config_path = os.path.join(run_dir, "config.json")
        concept_path = os.path.join(run_dir, "concept.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.loads(f.read())
        with open(concept_path, "r", encoding="utf-8") as f:
            c = ConceptSpec.from_config({"concept": json.loads(f.read())})

        logger = JsonlLogger(
            os.path.join(run_dir, "log.jsonl"),
            enabled=bool(cfg.get("output", {}).get("log_jsonl", True)),
        )
        console = ConsoleLogger(bool(cfg.get("output", {}).get("console", True)))
        obj = cls(
            model_bundle=model_bundle,
            config=cfg,
            concept=c,
            run_dir=run_dir,
            logger=logger,
            console=console,
        )

        tensors = np.load(os.path.join(run_dir, "tensors.npz"))
        obj.concept_vectors = tensors["concept_vectors"]
        obj.best_layer = int(tensors["best_layer"][0])
        obj.sweep_d = tensors["sweep_d"]
        obj.sweep_p = tensors["sweep_p"]
        obj.score_center = tensors["score_center"]
        obj.score_scale = tensors["score_scale"]
        obj.token_center = tensors["token_center"]
        obj.token_scale = tensors["token_scale"]
        obj.resid_norm = tensors["resid_norm"]
        sigma = float(obj.score_scale[obj.best_layer])
        obj.sigma_best_layer = sigma if np.isfinite(sigma) else None
        return obj


def _dotted_differences(a: Dict[str, Any], b: Dict[str, Any], prefix: str = "") -> List[str]:
    """Dotted keys whose values differ between two nested config dicts."""
    out: List[str] = []
    for key in sorted(set(a) | set(b), key=str):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_dotted_differences(va, vb, prefix=f"{prefix}{key}."))
        elif va != vb:
            out.append(f"{prefix}{key}")
    return out


# Model settings that change the activations: a probe must be used with the ones it was trained with.
_MODEL_IDENTITY_KEYS = ("model_id", "dtype", "use_4bit", "chat_template", "chat_template_date")


class ProbeWorkspace:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        defaults_path: Optional[str] = None,
        model_id: Optional[str] = None,
        project_directory: Optional[str] = None,
    ) -> None:
        self.project_directory = project_directory

        if project_directory:
            run_dir = os.path.abspath(project_directory)
            cfg_path = os.path.join(run_dir, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"project_directory must point to a trained run dir containing config.json: {run_dir}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.loads(f.read())
            if config_overrides:
                check_config_keys(config_overrides, deep_merge(load_defaults(), cfg))
                cfg = deep_merge(cfg, config_overrides)
            if model_id:
                cfg["model"]["model_id"] = model_id
            # root_dir isn't used for loading, but keep it consistent for callers.
            if root_dir:
                cfg["output"]["root_dir"] = root_dir

            self.config = cfg
            self.root_dir = cfg["output"].get("root_dir", "outputs")
            self.console = ConsoleLogger(cfg["output"].get("console", False))
            self.console.info(f"Loading model {cfg['model']['model_id']}...")
            self.model_bundle = ModelBundle.load(cfg["model"])
            self.console.info(f"Model loaded ({self.model_bundle.num_layers} layers).")
            return

        cfg = resolve_config(config_overrides, defaults_path=defaults_path)
        if model_id:
            cfg["model"]["model_id"] = model_id
        if root_dir:
            cfg["output"]["root_dir"] = root_dir
        self.config = cfg
        self.root_dir = cfg["output"]["root_dir"]
        ensure_dir(self.root_dir)
        self.console = ConsoleLogger(cfg["output"].get("console", False))
        self.console.info(f"Loading model {cfg['model']['model_id']}...")
        self.model_bundle = ModelBundle.load(cfg["model"])
        self.console.info(f"Model loaded ({self.model_bundle.num_layers} layers).")

    def get_probe(
        self,
        name: Optional[str] = None,
        project_directory: Optional[str] = None,
        allow_model_mismatch: bool = False,
    ) -> "ConceptProbe":
        """Load a previously trained probe without retraining.

        If this workspace was created with project_directory=..., that directory is used by default.
        The workspace's model settings (model_id, dtype, use_4bit, chat template and its date)
        must match the run's, since the probe's directions only apply to the activations they
        were trained on; ``allow_model_mismatch=True`` loads it anyway. Scoring then uses the
        workspace config, and a warning lists the settings that differ from the run's.
        """
        if not (project_directory or self.project_directory):
            raise ValueError("No project_directory provided. Create ProbeWorkspace(project_directory=...) or pass it to get_probe().")
        run_dir = os.path.abspath(project_directory or self.project_directory)

        probe = ConceptProbe.load(run_dir, model_bundle=self.model_bundle)
        saved_model = probe.config.get("model", {})
        mine = self.config.get("model", {})
        mismatched = [f"{k}: run={saved_model.get(k)!r} workspace={mine.get(k)!r}" for k in _MODEL_IDENTITY_KEYS
                      if saved_model.get(k) != mine.get(k)]
        if mismatched and not allow_model_mismatch:
            raise ValueError(
                "The workspace model differs from the one this probe was trained on ("
                + "; ".join(mismatched)
                + "). Load it with ProbeWorkspace(project_directory=run_dir) to use the run's own model "
                "settings, or pass allow_model_mismatch=True."
            )
        differences = [d for d in _dotted_differences(probe.config, self.config) if not d.startswith("output.")]
        if differences:
            self.console.warn(
                "Scoring uses the workspace config, which differs from the saved run config in: "
                + ", ".join(differences[:12]) + (" ..." if len(differences) > 12 else "") + "."
            )
        probe.config = self.config
        probe.console = self.console

        if name is not None:
            if safe_slug(str(probe.concept.name)) != safe_slug(str(name)):
                raise ValueError(
                    f"Loaded probe concept name '{probe.concept.name}' does not match requested name '{name}'."
                )
        return probe

    def train_concept(
        self,
        concept: Optional[ConceptSpec] = None,
        concept_overrides: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> ConceptProbe:
        """Train ``concept`` in ``<root_dir>/<concept name>/<run_name or timestamp>``.

        A run directory that already holds files raises, so two runs never mix their logs and
        artifacts; ``overwrite=True`` deletes it first.
        """
        if config_overrides:
            check_config_keys(config_overrides, deep_merge(load_defaults(), self.config))
        cfg = deep_merge(self.config, config_overrides or {})
        if concept is None:
            concept = ConceptSpec.from_config(cfg, concept_overrides)
        elif concept_overrides:
            concept = ConceptSpec.from_config({"concept": concept.to_dict()}, concept_overrides)

        run_tag = run_name or now_tag()
        concept_dir = os.path.join(self.root_dir, safe_slug(concept.name), run_tag)
        if os.path.isdir(concept_dir) and os.listdir(concept_dir):
            if not overwrite:
                raise FileExistsError(
                    f"{concept_dir} already holds a run; pass another run_name, or overwrite=True to replace it."
                )
            shutil.rmtree(concept_dir)
        ensure_dir(concept_dir)
        log_path = os.path.join(concept_dir, "log.jsonl")
        logger = JsonlLogger(log_path, enabled=bool(cfg.get("output", {}).get("log_jsonl", True)))

        probe = ConceptProbe(
            model_bundle=self.model_bundle,
            config=cfg,
            concept=concept,
            run_dir=concept_dir,
            logger=logger,
            console=self.console,
        )
        return probe.train()
