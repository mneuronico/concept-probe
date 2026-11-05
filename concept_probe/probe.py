import json
import os
import shutil
import sys
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from .config import ConceptSpec, check_config_keys, load_defaults, resolve_config
from .logger import JsonlLogger, NullLogger
from .modeling import (
    ModelBundle,
    _chat_template_kwargs,
    _resolve_attr_path,
    apply_chat,
    attention_mask_from_ids,
    identity_from_config,
    identity_mismatches,
    identity_of,
)
from .record import Choice, announce
from .settings import settings
from .directions import estimator_spec, fit_directions, make_estimator
from .linear_probe import Calibration, _token_references, calibrations_from_tensors, sigmoid, steer_directions_for
from .directions import (  # re-exported: they lived here before
    DIRECTION_NORMALIZATIONS,
    concept_direction,
    normed_np,
)
from .stats import (  # re-exported: they lived here before
    SWEEP_SELECTS,
    _bootstrap_d_ci,
    _cohen_d_columns,
    _max_d_permutation_p,
    _pooled_sd_columns,
    _select_best_layer,
    _split_indices,
    _welch_p_columns,
    cohen_d_np,
    ttest_ind,
    ttest_rel,
)
from .reading import (  # re-exported: they lived here before
    EmptySpanError,
    NonFiniteActivationsError,
    READ_MODES,
    Readout,
    Reps,
    _FINAL_NORM_PATHS,
    _get_final_norm_module,
    _model_dtype,
    _read,
    _slice_generation_step_logits,
    _span_bounds,
    check_finite_hidden_states,
    extract_completion_reps,
    extract_text_reps,
    forward_hidden_states_all_layers,
    read_encoded,
    read_texts,
    reps_from_hs_layers,
)
from .inputs import (  # re-exported: they lived here before
    Encoded,
    ReadContext,
    encode_text,
    PromptLike,
    PromptMessage,
    TEXT_ROLES,
    _TEXT_SENTINEL,
    _completion_input_ids,
    _decode_tokens,
    _encode_for_read,
    _encode_text_for_read,
    _has_chat_template,
    _no_bos_sink_start,
    _normalize_messages,
    _prompt_to_text,
    _token_count,
    _user_text_span,
    preview_input,
)
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
    """A random-direction control for ``base_vectors``: one random direction, used at every layer
    with that layer's norm.

    One shared direction keeps the control comparable to the concept vectors, which point nearly
    the same way at adjacent layers: independent directions per layer would partly cancel when
    steering several layers, so the control would push the residual stream about half as far.
    """
    src = np.asarray(base_vectors, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"base_vectors must be 2D [num_layers, hidden_dim], got shape={src.shape}")
    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(src.shape[1])
    direction = (direction / np.linalg.norm(direction)).astype(np.float32)
    norms = np.linalg.norm(src, axis=1)
    norms = np.where(np.isfinite(norms), norms, 0.0)
    return (norms[:, None] * direction[None, :]).astype(np.float32)


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


def _training_estimator(train_cfg: Dict[str, Any]) -> Tuple[Any, List[Choice]]:
    """The estimator of ``training.estimator``, or of the deprecated ``training.direction_normalization``
    (the normalization of the difference of means), which must then leave the estimator alone."""
    estimator = train_cfg.get("estimator", "mean_diff")
    normalization = train_cfg.get("direction_normalization")
    if normalization is None:
        return estimator, []
    if normalization not in DIRECTION_NORMALIZATIONS:
        raise ValueError(f"training.direction_normalization must be one of {DIRECTION_NORMALIZATIONS} or null; "
                         f"got {normalization!r}")
    if estimator not in ("mean_diff", {"kind": "mean_diff"}):
        raise ValueError("training.direction_normalization and training.estimator disagree; set only "
                         "training.estimator (for example 'mean_diff_var').")
    warnings.warn("training.direction_normalization is deprecated; set training.estimator, e.g. 'mean_diff_var'.",
                  DeprecationWarning, stacklevel=3)
    spec = {"kind": "mean_diff", "normalization": normalization}
    return spec, [Choice("training.estimator", f"direction_normalization={normalization!r}", spec,
                         "direction_normalization is the old name of the difference of means' normalization")]


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


def _has_system_message(prompt: Any) -> bool:
    return isinstance(prompt, list) and any(isinstance(m, dict) and m.get("role") == "system" for m in prompt)


def _text_context(role: str, system: Optional[str], prompt: Optional[str], mode: str) -> Dict[str, Any]:
    """How texts were read: the context a set of z references applies to (see :class:`ReadContext`)."""
    return ReadContext(kind="text", role=role, system=system or None, prompt=prompt, readout=mode).to_dict()


GENERATED_CONTEXT = {"kind": "generated"}


def _same_context(reference: Optional[Dict[str, Any]], context: Dict[str, Any]) -> bool:
    if not reference:
        return False
    return ReadContext.from_dict(reference).matches(ReadContext.from_dict(context))


def _prepare_batch_dir(out_dir: str, overwrite: bool) -> None:
    """Create ``out_dir`` for a scoring batch. A directory that already holds files raises unless
    ``overwrite`` (which deletes those files and the analysis folder): an earlier batch's files,
    such as its coherence ratings, would otherwise be read together with the new ones."""
    if os.path.isdir(out_dir):
        existing = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
        if existing and not overwrite:
            raise FileExistsError(
                f"{out_dir} already holds a scoring batch ({len(existing)} files); pass another batch_subdir, or "
                "overwrite=True to replace it."
            )
        if existing:
            for f in existing:
                os.remove(os.path.join(out_dir, f))
            shutil.rmtree(os.path.join(out_dir, "analysis"), ignore_errors=True)
    ensure_dir(out_dir)


def _prompt_slug(text: str, max_words: int = 8, max_len: int = 48) -> str:
    words = text.strip().split()
    snippet = " ".join(words[:max_words])
    slug = safe_slug(snippet)[:max_len]
    return slug or "prompt"


def _prompt_slug_any(prompt: PromptLike, max_words: int = 8, max_len: int = 48) -> str:
    return _prompt_slug(_prompt_to_text(prompt), max_words=max_words, max_len=max_len)


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


def _default_concept() -> ConceptSpec:
    return ConceptSpec(name="concept", pos_label="positive", neg_label="negative", pos_system="", neg_system="")


def _as_reps(reps: Any) -> np.ndarray:
    data = np.asarray(getattr(reps, "data", reps))  # a reading.Reps, or an array
    if data.ndim != 3:
        raise ValueError(f"representations must be (items, layers, dim); got shape {data.shape}")
    return data


@dataclass
class ConceptProbe:
    """A concept probe: one direction per layer of a model to read the concept with, the layer it
    reads at, calibrations that turn its projections into z scores, and the directions it steers with.

    A probe always belongs to a model (``identity``: the model, dtype, quantization and chat
    template its directions were computed on), but needs that model loaded (``model_bundle``) only to
    read texts, generate and steer. With representations already read (:func:`read_texts`), it
    trains, reads, calibrates and classifies without it. Build one by training
    (:meth:`ProbeWorkspace.train_concept`), from representations (:meth:`fit`), from directions
    computed elsewhere (:meth:`from_directions`), or from a saved run (:meth:`load`); a loaded model
    is added with :meth:`attach`. Probes save to their ``run_dir`` by default.
    """

    #: (layers, dim) unit directions to read with, pointing toward the positive pole
    directions: Optional[np.ndarray] = None
    #: the layer the probe reads at
    layer: Optional[int] = None
    #: (layers, dim) directions steering adds (decision D1: the raw difference of means by default,
    #: which may differ from the reading directions)
    steer_directions: Optional[np.ndarray] = None
    #: one calibration per reading context: the zero and unit of z scores, a threshold, neutral texts
    calibrations: List[Calibration] = field(default_factory=list)
    #: the unit of alpha_unit="sigma": the eval items' pooled within-pole SD on the steering direction
    steer_sigma: Optional[float] = None
    #: the unit of alpha_unit="resid_norm": the mean residual norm per layer of the training tokens
    resid_norm: Optional[np.ndarray] = None
    #: the model the directions belong to (see :func:`concept_probe.modeling.identity_of`)
    identity: Optional[Dict[str, Any]] = None
    #: the estimators the reading and steering directions came from
    estimator: Optional[Dict[str, Any]] = None
    steer_estimator: Optional[Dict[str, Any]] = None
    concept: ConceptSpec = field(default_factory=_default_concept)
    config: Dict[str, Any] = field(default_factory=resolve_config)
    model_bundle: Optional[ModelBundle] = None
    run_dir: Optional[str] = None
    #: where a probe without a run folder creates one when it first saves (None: the session's
    #: output_root; see :func:`concept_probe.configure`)
    save_root: Optional[str] = None
    logger: Any = field(default_factory=NullLogger)
    console: Optional[ConsoleLogger] = None
    #: training diagnostics: Cohen's d and its p per layer on the eval items
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
        estimator_cfg, early_choices = _training_estimator(train_cfg)
        read_estimator = make_estimator(estimator_cfg)
        read_spec = estimator_spec(read_estimator)
        direction_normalization = read_spec.get("normalization") if read_spec.get("kind") == "mean_diff" else None
        steer_direction_cfg = self.config.get("steering", {}).get("steer_direction", "mean_diff")
        if steer_direction_cfg != "read":
            make_estimator(steer_direction_cfg)  # an unknown estimator fails here, before anything runs
        sweep_select = train_cfg.get("sweep_select", "effect_size")
        if sweep_select not in SWEEP_SELECTS:
            raise ValueError(f"training.sweep_select must be one of {SWEEP_SELECTS}; got {sweep_select!r}")
        eval_mode = eval_cfg.get("eval_mode", "auto")
        if eval_mode not in ("auto", "read", "generate", "skip"):
            raise ValueError(f"Unknown evaluation.eval_mode: {eval_mode!r} (use 'auto', 'read', 'generate' or 'skip').")
        selection_split = eval_cfg.get("selection_split", 0.5)
        if selection_split is not None and not 0.0 < float(selection_split) < 1.0:
            raise ValueError(f"evaluation.selection_split must be in (0, 1) or null; got {selection_split!r}")
        generated_questions = {
            "shared": {"prompts.train_questions": train_prompts},
            "opposed": {"prompts.train_questions_pos": train_prompts_pos, "prompts.train_questions_neg": train_prompts_neg},
            "texts": {},
        }[train_prompt_mode]
        if eval_cfg.get("do_eval", True) and (eval_mode == "generate" or (eval_mode == "auto" and train_prompt_mode != "texts")):
            generated_questions["prompts.eval_questions"] = list(prompts_cfg.get("eval_questions", []))
        for key, questions in generated_questions.items():
            with_system = [i for i, q in enumerate(questions) if _has_system_message(q)]
            if with_system:
                raise ValueError(
                    f"{key}{with_system} carry their own system message, which would replace pos_system/neg_system "
                    "and make the two conditions identical. Pass the questions without a system message."
                )
        if self.concept.pos_system == self.concept.neg_system and (
            train_prompt_mode in ("shared", "opposed") or (eval_cfg.get("do_eval", True) and eval_mode == "generate")
        ):
            raise ValueError(
                "concept.pos_system == concept.neg_system: contrasting generations under identical system "
                "prompts learns noise. Set the two system prompts that define the concept."
            )

        # Labelled texts (the training texts in texts mode, the eval texts in every mode) are read
        # with training.train_text_role: by default the raw text, with no chat template and no
        # system prompt. Every evaluation (texts or generated completions) is pooled like the
        # vector's training data: with read_mode in texts mode, and with train_mode (the readout
        # of the model's own completions) otherwise.
        if train_prompt_mode == "texts":
            train_readout_mode, train_readout_last_k = readout["read_mode"], readout["read_last_k"]
        else:
            train_readout_mode, train_readout_last_k = readout["train_mode"], readout["train_last_k"]
        for key, mode in (("read_mode", readout["read_mode"]), ("train_mode", readout["train_mode"])):
            if mode not in READ_MODES:
                raise ValueError(f"readout.{key} must be one of {READ_MODES}; got {mode!r}")
        eval_readout_mode, eval_readout_last_k = train_readout_mode, train_readout_last_k
        if str(train_readout_mode).endswith("_tokens"):
            if eval_cfg.get("score_aggregate", "mean") != "mean":
                raise ValueError("Training on token samples scores eval items by their mean token score; "
                                 "set evaluation.score_aggregate to 'mean'.")
            eval_readout_mode = "assistant_all_mean" if train_readout_mode.startswith("assistant") else "sequence_all_mean"
            early_choices.append(Choice(
                "evaluation readout", train_readout_mode, eval_readout_mode,
                "every token is a training sample; eval items are scored by the mean over their tokens, so the "
                "statistics compare items, not tokens of the same text",
            ))
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
        if (
            train_prompt_mode == "texts" and text_role == "assistant" and not train_text_system
            and text_prompt is None
        ):
            raise ValueError(
                "training.train_text_role='assistant' reads each training text as the model's reply, and a chat "
                "template needs a turn before a reply. Set training.train_text_prompt (a user turn before every "
                "text, e.g. a neutral request) or a system prompt (training.train_text_system or "
                "prompts.neutral_system). preview_input shows the resulting input."
            )
        if reads_eval_texts and text_role == "assistant" and not eval_system and text_prompt is None:
            raise ValueError(
                "Eval texts are read as the assistant's reply, which needs a turn before them: set "
                "training.train_text_prompt (a user turn), or a system prompt (evaluation.eval_system or "
                "prompts.neutral_system)."
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
        # The saved config holds the concept actually trained, not the defaults' concept section.
        self.config = {**self.config, "concept": self.concept.to_dict()}
        json_dump(os.path.join(self.run_dir, "config.json"), self.config)
        json_dump(os.path.join(self.run_dir, "concept.json"), self.concept.to_dict())
        self.logger.log("config", {"config": self.config, "concept": self.concept.to_dict()})
        self.identity = model_identity_of(self.model_bundle, self.config)
        self.logger.log("model_identity", self.identity)
        # Every choice made on the caller's behalf goes to the run log and to metrics.json; the ones
        # made when the model was loaded were printed then.
        run_choices: List[Choice] = list(getattr(self.model_bundle, "choices", []) or [])
        announce(run_choices, console=False, logger=self.logger)
        run_choices.extend(early_choices)
        announce(early_choices, console=self.console is not None and self.console.enabled, logger=self.logger)

        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        num_layers = self.model_bundle.num_layers
        eos_ids = _eos_token_ids(model, tokenizer)
        gen_bs = int(train_cfg.get("gen_batch_size", 16))

        dropped_items: Dict[str, List[int]] = {}

        read_batch_size = int(readout.get("batch_size", 16))
        rep_cache = self._rep_cache_dir()
        rep_cache_dtype = str(out_cfg.get("rep_cache_dtype", "float32"))

        def note(choices: Sequence[Choice]) -> None:
            """Announce and log reading's choices once per run (they repeat for every set read)."""
            fresh = [c for c in choices if all(c != seen for seen in run_choices)]
            run_choices.extend(fresh)
            announce(fresh, console=self.console is not None and self.console.enabled, logger=self.logger)

        def pooled_reps(ids_list, starts, ends, mode, last_k, label, vectors=None):
            """Pooled reps ``(n, layers, dim)`` and mean token norms ``(n, layers)`` over each span, and,
            given ``vectors``, each item's per-token projections ``(n_tokens, layers)``. The items go
            through the model ``readout.batch_size`` at a time, and through the representations cache
            when it is on.

            An item with no tokens of its own (an empty completion or text) is left out and listed
            under ``dropped_items[label]``; a side left with no items raises.
            """
            encodings = [Encoded(ids.reshape(-1), start, end) for ids, start, end in zip(ids_list, starts, ends)]
            result = _read(model, tokenizer, encodings, [Readout.parse(mode, last_k)], batch_size=read_batch_size,
                           norms=True, project_on=vectors, cache=rep_cache if rep_cache else False,
                           cache_dtype=rep_cache_dtype, identity=self.identity, skip_empty=True,
                           label=label)[0]
            note(result.choices)
            if result.skipped:
                dropped_items[label] = sorted(result.skipped)
                self._warn(
                    f"{label}: items {dropped_items[label]} have no tokens to pool (an empty completion or text) "
                    "and are left out."
                )
            if result.data.shape[0] == 0:
                raise ValueError(f"{label}: no item has tokens to pool (every completion or text is empty).")
            return result.data, result.norms, result.projections or [], result.read_items

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
        reps_pos, norms_pos, _, train_pos_kept = pooled_reps(train_pos_ids, train_pos_starts, train_pos_ends,
                                                             train_readout_mode, train_readout_last_k, "train_pos")
        reps_neg, norms_neg, _, train_neg_kept = pooled_reps(train_neg_ids, train_neg_starts, train_neg_ends,
                                                             train_readout_mode, train_readout_last_k, "train_neg")
        if reps_pos.shape[1] != num_layers:
            raise ValueError("Layer count mismatch between model and reps.")
        # Mean residual norm of the training tokens per layer: the unit of alpha_unit="resid_norm".
        resid_norm = np.concatenate([norms_pos, norms_neg], axis=0).mean(axis=0)

        concept_vectors = fit_directions(read_estimator, reps_pos, reps_neg)
        # Decision D1: steering has its own direction (the raw difference of means by default).
        steer_vectors, steer_spec, steer_choice = steer_directions_for(
            read_estimator, steer_direction_cfg, reps_pos, reps_neg, concept_vectors
        )
        steer_differs = steer_vectors is not concept_vectors
        if steer_choice is not None:
            note([steer_choice])
        steer_scores: Dict[str, np.ndarray] = {}

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
        transfer_pos_tokens: List[np.ndarray] = []
        transfer_neg_tokens: List[np.ndarray] = []
        eval_prompts = list(prompts_cfg["eval_questions"])
        # Generated eval items are pairs: row i of both poles answers eval question eval_question_idx[i].
        eval_paired = False
        eval_question_idx: List[int] = []
        did_eval = False
        eval_readout: Optional[Dict[str, Any]] = None
        transfer_readout: Optional[Dict[str, Any]] = None
        text_readout = {
            "role": text_role,
            "prompt": text_prompt,
            "system": eval_system,
            "mode": eval_readout_mode,
            "last_k": eval_readout_last_k,
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
                reps, _, tokens, _ = pooled_reps(ids_list, starts, ends, eval_readout_mode, eval_readout_last_k,
                                                 f"{event.split('_')[0]}_{side}", vectors=concept_vectors)
                out[side] = (np.einsum("nld,ld->nl", reps, concept_vectors).astype(np.float32), tokens)
                if steer_differs:
                    steer_scores[f"{event}_{side}"] = np.einsum("nld,ld->nl", reps, steer_vectors)
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
                    if steer_differs:
                        steer_scores["eval_pos"] = steer_scores["eval_read_pos"]
                        steer_scores["eval_neg"] = steer_scores["eval_read_neg"]
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
                        reps, _, tokens, kept = pooled_reps(
                            [ids for ids, _ in results],
                            [plen for _, plen in results],
                            [_completion_end(ids, plen, eos_ids) for ids, plen in results],
                            eval_readout_mode,
                            eval_readout_last_k,
                            f"eval_{side}",
                            vectors=concept_vectors,
                        )
                        side_scores[side] = (np.einsum("nld,ld->nl", reps, concept_vectors).astype(np.float32), tokens, kept)
                        if steer_differs:
                            steer_scores[f"eval_gen_{side}"] = np.einsum("nld,ld->nl", reps, steer_vectors)
                    # Keep the questions answered on both poles, so the two poles stay paired.
                    eval_question_idx = sorted(set(side_scores["pos"][2]) & set(side_scores["neg"][2]))
                    for side in ("pos", "neg"):
                        unpaired = sorted(set(side_scores[side][2]) - set(eval_question_idx))
                        if unpaired:
                            dropped_items.setdefault(f"eval_{side}", []).extend(unpaired)
                            self._warn(
                                f"eval_{side}: questions {unpaired} are left out because the other pole's "
                                "completion is empty; the poles are compared question by question."
                            )
                    rows = {side: [side_scores[side][2].index(q) for q in eval_question_idx] for side in ("pos", "neg")}
                    eval_pos_scores_mat = side_scores["pos"][0][rows["pos"]]
                    eval_neg_scores_mat = side_scores["neg"][0][rows["neg"]]
                    eval_pos_tokens = [side_scores["pos"][1][r] for r in rows["pos"]]
                    eval_neg_tokens = [side_scores["neg"][1][r] for r in rows["neg"]]
                    if steer_differs:
                        steer_scores["eval_pos"] = steer_scores["eval_gen_pos"][rows["pos"]]
                        steer_scores["eval_neg"] = steer_scores["eval_gen_neg"][rows["neg"]]
                    eval_paired = True
                    did_eval = len(eval_question_idx) > 0
                    eval_readout = {
                        "role": "assistant",
                        "source": "completions generated for eval_questions under pos_system/neg_system",
                        "mode": eval_readout_mode,
                        "last_k": eval_readout_last_k,
                    }
                if eval_texts_given and train_prompt_mode != "texts":
                    self._status("Reading eval texts (transfer check)")
                    read = read_eval_texts("transfer_read")
                    (transfer_pos_scores_mat, transfer_pos_tokens), (transfer_neg_scores_mat, transfer_neg_tokens) = (
                        read["pos"], read["neg"]
                    )
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
            sweep_p = _welch_p_columns(P, N, paired=eval_paired)
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
        selection_d: Optional[np.ndarray] = None
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
                if eval_paired:
                    # Both poles of a question go to the same side of the split; split independently,
                    # a question can help pick the layer on one pole and then be measured on the other.
                    neg_sel, neg_est = pos_sel, pos_est
                else:
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
                selection_d = sel_d
                sel_p = _welch_p_columns(P[pos_sel], N[neg_sel], paired=eval_paired)
                best_layer = _select_best_layer(sel_d, sel_p, search_layers, sweep_select)
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
                    P[:, search_layers], N[:, search_layers], n_permutations, np.random.default_rng([seed, 2]),
                    paired=eval_paired,
                )
        else:
            best_layer = pick_layer(num_layers, train_cfg.get("layer_idx"), train_cfg.get("layer_frac", 0.75))
            if did_eval:
                selection = "fixed_layer"
        if did_eval:
            best_d = float(cohen_d_np(est_pos[:, best_layer], est_neg[:, best_layer]))
            best_p_arr = _welch_p_columns(est_pos[:, [best_layer]], est_neg[:, [best_layer]], paired=eval_paired)
            best_p = None if np.isnan(best_p_arr[0]) else float(best_p_arr[0])
            best_d_all_eval = float(sweep_d[best_layer])
            best_d_ci95 = _bootstrap_d_ci(
                est_pos[:, best_layer], est_neg[:, best_layer],
                int(eval_cfg.get("n_bootstrap") or 0), np.random.default_rng([seed, 3]), paired=eval_paired,
            )
        notes = {
            "split": "best_layer was chosen on the selection part of the eval items; best_d, best_p and "
                     "best_d_ci95 are measured on the held-out rest (split by question when the poles are "
                     "paired by question).",
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

        # z references only apply to inputs read in the context they were measured in: completions
        # generated in a chat, or texts read with a given role and system prompt.
        reference_context: Optional[Dict[str, Any]] = None
        if did_eval:
            reference_context = (
                {"kind": "generated", "mode": eval_readout_mode} if eval_mode == "generate"
                else _text_context(text_role, eval_system, text_prompt, eval_readout_mode)
            )
        transfer_refs: Optional[Dict[str, Any]] = None
        if transfer_eval is not None:
            t_token_center, t_token_scale = _token_references(transfer_pos_tokens, transfer_neg_tokens)
            transfer_refs = {
                "context": _text_context(text_role, eval_system, text_prompt, eval_readout_mode),
                "score_center": 0.5 * (TP.mean(axis=0) + TN.mean(axis=0)),
                "score_scale": _pooled_sd_columns(TP, TN),
                "token_center": t_token_center,
                "token_scale": t_token_scale,
            }
            transfer_eval["reference_context"] = transfer_refs["context"]

        sigma = float(score_scale[best_layer])
        # The unit of alpha_unit="sigma" is measured on the direction that steers (D1).
        steer_sigma = sigma
        if steer_differs and did_eval and "eval_pos" in steer_scores:
            steer_sigma = float(_pooled_sd_columns(steer_scores["eval_pos"][:, [best_layer]],
                                                   steer_scores["eval_neg"][:, [best_layer]])[0])

        # The references above as calibrations (layer 3): the evaluation's, with a classifier's
        # threshold (entry 26), and the transfer check's.
        calibrations: List[Calibration] = []
        threshold: Optional[Dict[str, Any]] = None
        if did_eval:
            main_cal = Calibration(context=ReadContext.from_dict(reference_context), center=score_center,
                                   scale=score_scale, token_center=token_center, token_scale=token_scale,
                                   n_pos=n_eval_pos, n_neg=n_eval_neg, source="evaluation")
            if selection == "split":
                threshold = main_cal.add_threshold(P[pos_sel, best_layer], N[neg_sel, best_layer], best_layer,
                                                   "the selection part of the eval items")
            else:
                threshold = main_cal.add_threshold(P[:, best_layer], N[:, best_layer], best_layer, "all eval items")
            calibrations.append(main_cal)
        if transfer_refs is not None:
            calibrations.append(Calibration(
                context=ReadContext.from_dict(transfer_refs["context"]), center=transfer_refs["score_center"],
                scale=transfer_refs["score_scale"], token_center=transfer_refs["token_center"],
                token_scale=transfer_refs["token_scale"], n_pos=int(TP.shape[0]), n_neg=int(TN.shape[0]),
                source="transfer check",
            ))

        # Where texts without the concept fall (entry 19): read like the eval texts, reported with the
        # calibration of that reading, not used as the zero.
        neutral_report: Optional[Dict[str, Any]] = None
        neutral_texts = list(getattr(self.concept, "neutral_texts", []) or [])
        if neutral_texts:
            self._status(f"Reading {len(neutral_texts)} neutral texts")
            encs = [_encode_text_for_read(tokenizer, t, eval_system, role=text_role, prompt=text_prompt,
                                          warn=self._warn, warn_prefix=f"neutral_texts[{i}]")
                    for i, t in enumerate(neutral_texts)]
            reps_neutral, _, _, _ = pooled_reps([e[0][0] for e in encs], [e[1] for e in encs], [e[2] for e in encs],
                                                eval_readout_mode, eval_readout_last_k, "neutral")
            neutral_scores = np.einsum("nld,ld->nl", reps_neutral, concept_vectors)
            text_ctx = ReadContext.from_dict(_text_context(text_role, eval_system, text_prompt, eval_readout_mode))
            cal = next((c for c in calibrations if c.context.matches(text_ctx)), None)
            mean_score = float(neutral_scores[:, best_layer].mean())
            neutral_report = {"n": int(neutral_scores.shape[0]), "mean_score_at_best_layer": mean_score,
                              "z_at_best_layer": None, "z_sd_at_best_layer": None, "calibration": None}
            if cal is not None and cal.usable([best_layer]):
                cal.add_neutral(neutral_scores)
                z = cal.items_z(neutral_scores[:, [best_layer]], [best_layer])
                neutral_report.update(z_at_best_layer=float(z.mean()), z_sd_at_best_layer=float(z.std(ddof=1)) if len(z) > 1 else None,
                                      calibration=cal.source)
            else:
                neutral_report["note"] = ("no calibration was measured on texts read like these, so only the raw "
                                          "score is reported")

        self.directions = concept_vectors
        self.layer = int(best_layer)
        self.steer_directions = steer_vectors
        self.calibrations = calibrations
        self.steer_sigma = steer_sigma if np.isfinite(steer_sigma) else None
        self.resid_norm = resid_norm
        self.estimator, self.steer_estimator = read_spec, steer_spec
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
                    selection_d=selection_d,
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
            "reference_context": np.array([json.dumps(reference_context)]),
            "transfer_references": np.array([json.dumps(
                None if transfer_refs is None
                else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in transfer_refs.items()}
            )]),
            "eval_paired": np.array([eval_paired]),
            "eval_question_idx": np.asarray(eval_question_idx, dtype=np.int32),
            "resid_norm": resid_norm.astype(np.float64),
            "transfer_pos_scores_mat": transfer_pos_scores_mat.astype(np.float32),
            "transfer_neg_scores_mat": transfer_neg_scores_mat.astype(np.float32),
            "transfer_sweep_d": transfer_d.astype(np.float32),
            "transfer_sweep_p": transfer_p.astype(np.float64),
            "steer_vectors": np.asarray(steer_vectors, dtype=np.float32),
            "steer_sigma_best_layer": np.array([steer_sigma], dtype=np.float64),
            "calibrations": np.array([json.dumps([c.to_dict() for c in calibrations])]),
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

        n_train_pos = len(train_pos_kept)
        n_train_neg = len(train_neg_kept)
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
            "estimator": read_spec,
            "steer_direction": steer_spec,
            "steer_sigma_best_layer": _opt(steer_sigma),
            "threshold": threshold,
            "neutral_texts": neutral_report,
            "train_samples": ({"pos": int(reps_pos.shape[0]), "neg": int(reps_neg.shape[0])}
                              if str(train_readout_mode).endswith("_tokens") else None),
            "train_text_role": text_role,
            "gen_batch_size": gen_bs if train_prompt_mode == "shared" else None,
            "n_eval": n_eval_pos + n_eval_neg,
            "n_eval_pos": n_eval_pos,
            "n_eval_neg": n_eval_neg,
            "eval_mode_used": eval_mode if did_eval else None,
            "eval_paired": eval_paired,
            "reference_context": reference_context,
            "eval_readout": eval_readout,
            "transfer_eval": transfer_eval,
            "dropped_items": dropped_items or None,
            "model_identity": self.identity,
            "read_batch_size": read_batch_size,
            "rep_cache": rep_cache,
            "choices": [c.to_dict() for c in run_choices],
        }
        json_dump(os.path.join(self.run_dir, "metrics.json"), metrics)
        self.logger.log("artifacts_saved", {"npz_path": npz_path})
        if self.config["output"].get("pretty_json_log", False) and self.config["output"].get("log_jsonl", True):
            pretty_path = os.path.join(self.run_dir, "log.pretty.json")
            jsonl_to_pretty(os.path.join(self.run_dir, "log.jsonl"), pretty_path)
        self._status(f"Saved artifacts to {self.run_dir}")
        return self

    def _rep_cache_dir(self) -> Optional[str]:
        """Where this probe caches representations: ``output.rep_cache_dir``, or else
        ``<output.root_dir>/rep_cache``; None when ``output.rep_cache`` is false or the session does
        not save (``concept_probe.configure(save=False)`` turns off every implicit write)."""
        out_cfg = self.config.get("output", {})
        if not out_cfg.get("rep_cache", True) or not settings().save:
            return None
        return out_cfg.get("rep_cache_dir") or os.path.join(out_cfg.get("root_dir", "outputs"), "rep_cache")

    def _require_trained(self) -> None:
        if self.directions is None or self.layer is None:
            raise ValueError("The probe has no directions yet: train, fit, or load it first.")

    def _ensure_run_dir(self) -> str:
        """The probe's run folder, created under ``output.root_dir`` the first time it saves."""
        if not self.run_dir:
            root = self.save_root or settings().output_root
            self.run_dir = os.path.join(root, safe_slug(self.concept.name), now_tag())
            ensure_dir(self.run_dir)
        return self.run_dir

    def _require_model(self, what: str) -> None:
        self._require_trained()
        if self.model_bundle is None:
            raise ValueError(f"{what} needs the model loaded: attach it with probe.attach(model, tokenizer) "
                             "(the probe's identity says which model it belongs to).")

    def _status(self, message: str) -> None:
        if self.console is not None:
            self.console.info(message)

    def _warn(self, message: str) -> None:
        if self.console is not None:
            self.console.warn(message)
        self.logger.log("warning", {"message": message})

    @property
    def num_layers(self) -> int:
        self._require_trained()
        return int(np.asarray(self.directions).shape[0])

    def _layer_selection(self, override: Optional[Dict[str, Any]] = None) -> List[int]:
        self._require_trained()
        selection = dict(self.config["evaluation"]["layer_selection"])
        if override:
            selection = deep_merge(selection, override)
        return select_layers(selection, int(self.layer), self.num_layers)

    def _random_control_vectors(self, random_vector_seed: Optional[int]) -> Tuple[np.ndarray, int]:
        """Same-norm random vectors per layer: this probe's random-direction control.

        The seed combines ``random_vector_seed`` (or ``random.seed``) with the probe's identity
        (concept name and run directory), so score_texts, score_prompts (scoring and steering)
        and multi_probe_score_prompts all use the same control vectors for the same probe.
        """
        base = _resolve_random_control_seed(random_vector_seed, self.config.get("random", {}).get("seed", 0))
        run = os.path.basename(os.path.normpath(self.run_dir)) if self.run_dir else ""
        seed = _mix_random_control_seed(base, zlib.crc32(f"{self.concept.name}|{run}".encode("utf-8")))
        return _random_vectors_same_norm(self.directions, seed=seed), seed

    # ------------------------------------------------------------------ calibrations and z
    def calibration_for(self, context: Optional[Union[ReadContext, Dict[str, Any]]] = None) -> Optional[Calibration]:
        """The calibration measured on items read like ``context`` (a :class:`ReadContext` or its
        dict); with no context, the only calibration if there is one."""
        if context is None:
            return self.calibrations[0] if len(self.calibrations) == 1 else None
        target = context if isinstance(context, ReadContext) else ReadContext.from_dict(context)
        return next((c for c in self.calibrations if c.context.matches(target)), None)

    def _missing_z_reason(self, context: Optional[Union[ReadContext, Dict[str, Any]]]) -> str:
        """Why items read in ``context`` get no z score."""
        have = "; ".join(c.context.describe() for c in self.calibrations) or "in no context"
        if context is None:
            return (f"the probe has {len(self.calibrations)} calibrations (items read {have}); say which reading "
                    "context the items come from (context=...).")
        target = context if isinstance(context, ReadContext) else ReadContext.from_dict(context)
        return (f"the probe has no calibration for items read {target.describe()} (it has one for items read "
                f"{have}); z scores from another reading context are not comparable. Add one with "
                "probe.calibrate(pos, neg, ...).")

    def _token_z(
        self, scores_per_layer: np.ndarray, layers: List[int], aggregate: str, context: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Single-token z scores per selected layer and their aggregate, with the calibration of
        ``context``; None when there is none (see :meth:`Calibration.tokens_z`)."""
        cal = self.calibration_for(context)
        return None if cal is None else cal.tokens_z(scores_per_layer, layers, aggregate)

    def _item_z(
        self, span_scores_per_layer: np.ndarray, layers: List[int], aggregate: str, context: Dict[str, Any]
    ) -> Optional[float]:
        """The z score of an item, from its tokens' scores at the selected layers: their mean,
        standardized with the calibration of ``context``. None without such a calibration, or when it
        was measured with a pooling other than the mean (its zero would not apply to a mean)."""
        cal = self.calibration_for(context)
        span = np.asarray(span_scores_per_layer, dtype=np.float64)
        if cal is None or (cal.context.readout or "assistant_all_mean") != "assistant_all_mean" or span.shape[0] == 0:
            return None
        z = cal.items_z(span.mean(axis=0)[None, :], layers, aggregate)
        return None if z is None else float(z[0])

    # ------------------------------------------------------------------ reading arrays
    def project(self, reps: Any) -> np.ndarray:
        """Each item's projection on each layer's reading direction: ``(items, layers)``, from
        representations ``(items, layers, dim)`` (an array, or what :func:`read_texts` returns)."""
        self._require_trained()
        R = _as_reps(reps)
        if R.shape[1:] != np.asarray(self.directions).shape:
            raise ValueError(f"representations of shape {R.shape[1:]} (layers, dim) do not fit the probe's directions "
                             f"{np.asarray(self.directions).shape}")
        return np.einsum("nld,ld->nl", R.astype(np.float64), np.asarray(self.directions, dtype=np.float64))

    def scores(self, reps: Any) -> np.ndarray:
        """Each item's raw score: its projection at the probe's layers (``evaluation.layer_selection``,
        the probe's layer by default), aggregated as ``evaluation.score_aggregate``. The offset is
        arbitrary; see :meth:`z`."""
        layers = self._layer_selection()
        proj = self.project(reps)[:, layers]
        return proj.mean(axis=1) if self.config["evaluation"]["score_aggregate"] == "mean" else proj.sum(axis=1)

    def z(self, reps: Any, context: Optional[Union[ReadContext, Dict[str, Any]]] = None) -> np.ndarray:
        """Each item's z score, with the calibration of ``context`` (how the items were read; with one
        calibration it may be left out). 0 is the calibration's zero (the midpoint between the poles,
        not "without the concept"), 1 its within-pole SD. Raises, saying which contexts the probe
        has, when there is no calibration for ``context``."""
        cal = self.calibration_for(context)
        if cal is None:
            raise ValueError(self._missing_z_reason(context))
        layers = self._layer_selection()
        z = cal.items_z(self.project(reps)[:, layers], layers, self.config["evaluation"]["score_aggregate"])
        if z is None:
            raise ValueError(f"the calibration for items read {cal.context.describe()} has no usable references "
                             f"at layers {layers}.")
        return z

    def classify(self, reps: Any, context: Optional[Union[ReadContext, Dict[str, Any]]] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Whether each item belongs to the positive pole, and the probability, from the calibrated
        threshold of ``context`` (not from z = 0, which is the midpoint, not the best cut)."""
        cal = self.calibration_for(context)
        if cal is None:
            raise ValueError(self._missing_z_reason(context))
        if not cal.threshold:
            raise ValueError("That calibration has no threshold; calibrate again to fit one.")
        t = cal.threshold
        prob = sigmoid(t["slope"] * self.project(reps)[:, int(t["layer"])] + t["intercept"])
        return prob >= 0.5, prob

    # ------------------------------------------------------------------ building
    @classmethod
    def fit(
        cls,
        pos: Any,
        neg: Any,
        *,
        estimator: Any = "mean_diff",
        steer: Any = "mean_diff",
        layer: Optional[int] = None,
        select: Optional[Tuple[Any, Any]] = None,
        search_layers: Optional[Sequence[int]] = None,
        sweep_select: str = "effect_size",
        calibrate: Optional[Tuple[Any, Any]] = None,
        context: Optional[Union[ReadContext, Dict[str, Any]]] = None,
        baseline: bool = False,
        concept: Optional[ConceptSpec] = None,
        model: Any = None,
        tokenizer: Any = None,
        identity: Optional[Union[Dict[str, Any], str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        root_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        save: Optional[bool] = None,
        overwrite: bool = False,
    ) -> "ConceptProbe":
        """Train a probe from representations already read: ``(items, layers, dim)`` of each pole
        (arrays, or what :func:`read_texts` returns, which says which model they come from).

        ``estimator`` and ``steer`` set the reading and steering directions (see
        :mod:`concept_probe.directions`). The layer is ``layer``, or the one whose direction separates
        ``select = (pos, neg)`` the most (``sweep_select``), within ``search_layers``; choose it on
        items that did not train the directions. ``calibrate = (pos, neg)`` adds the calibration of
        ``context`` (how those items were read; default raw text), with a threshold; use items that
        did not train or choose. ``baseline=True`` reads ``neg`` as a baseline (text without the
        concept) rather than an opposite pole, so the zero is the baseline's mean.

        The probe belongs to the model the representations came from: the one ``read_texts`` recorded,
        ``model`` (loaded, then attached), or ``identity`` (a dict, or a model id). One is required.
        It saves to ``<root>/<concept name>/<run_name or timestamp>`` unless ``save=False`` (or the
        session does not save), where ``root`` is ``root_dir``, else ``output.root_dir`` if
        ``config_overrides`` sets it, else the session's ``output_root``.
        """
        P, N = _as_reps(pos), _as_reps(neg)
        probe = cls._new(concept, model, tokenizer, identity, getattr(pos, "identity", None), config_overrides, root_dir)
        read = fit_directions(estimator, P, N)
        steer_dirs, steer_spec, choice = steer_directions_for(estimator, steer, P, N, read)
        probe.directions, probe.steer_directions = read, steer_dirs
        probe.estimator, probe.steer_estimator = estimator_spec(estimator), steer_spec
        if layer is None:
            if select is None:
                raise ValueError("Pass layer=, or select=(pos, neg) to choose the layer on items that did not train.")
            sp = np.einsum("nld,ld->nl", _as_reps(select[0]).astype(np.float64), read.astype(np.float64))
            sn = np.einsum("nld,ld->nl", _as_reps(select[1]).astype(np.float64), read.astype(np.float64))
            layers = list(range(read.shape[0])) if search_layers is None else list(search_layers)
            layer = _select_best_layer(_cohen_d_columns(sp, sn), _welch_p_columns(sp, sn), layers, sweep_select)
            probe.sweep_d, probe.sweep_p = _cohen_d_columns(sp, sn), _welch_p_columns(sp, sn)
        probe.layer = resolve_layer_index(layer, read.shape[0], "layer")
        choices = [choice] if choice is not None else []
        if calibrate is not None:
            cal = probe.calibrate(calibrate[0], calibrate[1], context=context, baseline=baseline)
            if steer_dirs is not read:
                probe.steer_sigma = float(_pooled_sd_columns(
                    np.einsum("nld,ld->nl", _as_reps(calibrate[0]).astype(np.float64), steer_dirs.astype(np.float64)),
                    np.einsum("nld,ld->nl", _as_reps(calibrate[1]).astype(np.float64), steer_dirs.astype(np.float64)),
                )[probe.layer])
            else:
                probe.steer_sigma = float(cal.scale[probe.layer])
        announce(choices, console=probe.console is not None and probe.console.enabled)
        probe._save_new(root_dir, run_name, save, overwrite, "fit", choices)
        return probe

    @classmethod
    def from_directions(
        cls,
        directions: np.ndarray,
        layer: int,
        *,
        steer_directions: Optional[np.ndarray] = None,
        concept: Optional[ConceptSpec] = None,
        model: Any = None,
        tokenizer: Any = None,
        identity: Optional[Union[Dict[str, Any], str]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        root_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        save: Optional[bool] = None,
        overwrite: bool = False,
    ) -> "ConceptProbe":
        """A probe from directions computed elsewhere: ``(layers, dim)``, scaled to unit length per
        layer (steering uses them too unless ``steer_directions`` is given). Calibrate it with
        :meth:`calibrate` to get z scores. The model and saving work as in :meth:`fit`."""
        probe = cls._new(concept, model, tokenizer, identity, None, config_overrides, root_dir)
        v = np.asarray(directions, dtype=np.float64)
        if v.ndim != 2:
            raise ValueError(f"directions must be (layers, dim); got shape {v.shape}")
        probe.directions = (v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)
        probe.steer_directions = probe.directions if steer_directions is None else np.asarray(steer_directions, np.float32)
        probe.layer = resolve_layer_index(layer, v.shape[0], "layer")
        probe.estimator = {"kind": "given"}
        probe.steer_estimator = probe.estimator if steer_directions is None else {"kind": "given"}
        probe._save_new(root_dir, run_name, save, overwrite, "from_directions", [])
        return probe

    @classmethod
    def _new(cls, concept, model, tokenizer, identity, reps_identity, config_overrides, root_dir) -> "ConceptProbe":
        cfg = resolve_config(config_overrides)
        # where it saves: root_dir, else a root_dir given in the config overrides, else the session's
        save_root = root_dir or ((config_overrides or {}).get("output") or {}).get("root_dir")
        probe = cls(concept=concept or _default_concept(), config=cfg, save_root=save_root,
                    console=ConsoleLogger(cfg["output"].get("console", True)))
        if isinstance(identity, str):
            identity = {"model_id": identity}
        if model is not None:
            probe.attach(model, tokenizer)
            if reps_identity and identity_mismatches(reps_identity, probe.identity):
                raise ValueError("The representations were read with another model than the one passed: "
                                 + "; ".join(identity_mismatches(reps_identity, probe.identity)))
        else:
            probe.identity = reps_identity or identity
        if not probe.identity:
            raise ValueError("Which model are these representations from? Pass what read_texts returns (it records "
                             "the model), model= (the loaded model), or identity= (a dict, or the model id).")
        return probe

    def attach(self, model: Any, tokenizer: Any = None, *, allow_model_mismatch: bool = False) -> "ConceptProbe":
        """Give the probe a loaded model (a :class:`ModelBundle`, or a model and its tokenizer), so it
        can read texts, generate and steer. A model of another identity than the probe's raises
        unless ``allow_model_mismatch``."""
        bundle = model if isinstance(model, ModelBundle) else None
        if bundle is None:
            if tokenizer is None:
                raise ValueError("attach a ModelBundle, or a model together with its tokenizer.")
            bundle = ModelBundle.from_loaded(model, tokenizer, model_cfg=self.config.get("model"))
        mine = bundle.identity or identity_of(bundle.model, bundle.tokenizer, bundle.model_id)
        if self.identity:
            mismatched = identity_mismatches(self.identity, mine)
            if mismatched and not allow_model_mismatch:
                raise ValueError("The model differs from the one this probe belongs to (probe vs model: "
                                 + "; ".join(mismatched) + "); pass allow_model_mismatch=True to use it anyway.")
        else:
            self.identity = mine
        self.model_bundle = bundle
        return self

    def _save_new(self, root_dir, run_name, save, overwrite, how: str, choices: List[Choice]) -> None:
        if not (settings().save if save is None else save):
            return
        root = self.save_root or settings().output_root
        run_dir = os.path.join(root, safe_slug(self.concept.name), run_name or now_tag())
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            if not overwrite:
                raise FileExistsError(f"{run_dir} already holds a run; pass another run_name, or overwrite=True.")
            shutil.rmtree(run_dir)
        self.save(run_dir, how=how, choices=choices)

    def save(self, run_dir: Optional[str] = None, *, how: str = "save", choices: Sequence[Choice] = ()) -> str:
        """Write the probe to ``run_dir`` (default: its own) in the format of a training run, so
        :meth:`load` reads it back. Into a training run's folder it updates the probe's own entries
        (directions, layer, calibrations, steering units) and keeps everything else."""
        self._require_trained()
        run_dir = run_dir or self.run_dir
        if not run_dir:
            raise ValueError("The probe has no run folder; pass run_dir.")
        ensure_dir(run_dir)
        self.run_dir = run_dir
        cal_eval = next((c for c in self.calibrations if c.source == "evaluation"), None)
        state = {
            "concept_vectors": np.asarray(self.directions, dtype=np.float32),
            "best_layer": np.array([int(self.layer)], dtype=np.int32),
            "steer_vectors": np.asarray(self.steer_directions if self.steer_directions is not None else self.directions,
                                        dtype=np.float32),
            "steer_sigma_best_layer": np.array([np.nan if self.steer_sigma is None else self.steer_sigma]),
            "calibrations": np.array([json.dumps([c.to_dict() for c in self.calibrations])]),
        }
        if self.resid_norm is not None:
            state["resid_norm"] = np.asarray(self.resid_norm, dtype=np.float64)
        tensors_path = os.path.join(run_dir, "tensors.npz")
        if os.path.exists(tensors_path):
            with np.load(tensors_path) as old:
                state = {**{k: old[k] for k in old.files}, **state}
        np.savez_compressed(tensors_path, **state)
        metrics_path = os.path.join(run_dir, "metrics.json")
        metrics: Dict[str, Any] = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        metrics.update({
            "best_layer": int(self.layer), "num_layers": self.num_layers, "estimator": self.estimator,
            "steer_direction": self.steer_estimator, "steer_sigma_best_layer": self.steer_sigma,
            "threshold": None if cal_eval is None else cal_eval.threshold, "model_identity": self.identity,
            "calibrations": [{"context": c.context.to_dict(), "source": c.source, "n_pos": c.n_pos, "n_neg": c.n_neg}
                             for c in self.calibrations],
        })
        metrics.setdefault("built_by", how)
        if choices:
            metrics["choices"] = list(metrics.get("choices", [])) + [c.to_dict() for c in choices]
        json_dump(metrics_path, metrics)
        if not os.path.exists(os.path.join(run_dir, "config.json")):
            json_dump(os.path.join(run_dir, "config.json"), {**self.config, "concept": self.concept.to_dict()})
        json_dump(os.path.join(run_dir, "concept.json"), self.concept.to_dict())
        return run_dir

    # ------------------------------------------------------------------ calibrating
    def calibrate(
        self,
        pos: Any,
        neg: Any,
        *,
        context: Optional[Union[ReadContext, Dict[str, Any]]] = None,
        as_reply_to: Optional[Union[str, List[str]]] = None,
        role: str = "raw",
        system_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        neutral: Any = None,
        baseline: bool = False,
        batch_size: Optional[int] = None,
    ) -> Calibration:
        """Measure a calibration on items of each pole, so items read the same way get z scores and
        a calibrated threshold. It replaces any calibration for the same reading context.

        ``pos`` / ``neg`` (and ``neutral``, texts without the concept, whose place is recorded) are
        representations already read, with ``context`` saying how (default: raw text); or texts,
        which the probe reads with its model: as replies to ``as_reply_to`` (one user turn, or one
        per text), which is how generated completions are read, or with ``role`` /
        ``system_prompt`` / ``prompt``, as :meth:`score_texts` reads them. ``baseline=True`` sets the
        zero at ``neg``'s mean instead of the midpoint. Call :meth:`save` to keep it.
        """
        self._require_trained()
        if isinstance(pos, np.ndarray) or hasattr(pos, "data"):
            ctx = ReadContext() if context is None else (
                context if isinstance(context, ReadContext) else ReadContext.from_dict(context))
            scores = {"pos": self.project(pos), "neg": self.project(neg)}
            if neutral is not None:
                scores["neutral"] = self.project(neutral)
            tokens: Dict[str, Optional[List[np.ndarray]]] = {"pos": None, "neg": None}
        else:
            ctx, scores, tokens = self._read_for_calibration(pos, neg, neutral, as_reply_to, role, system_prompt,
                                                             prompt, batch_size)
        cal = Calibration.from_scores(scores["pos"], scores["neg"], ctx, pos_tokens=tokens["pos"],
                                      neg_tokens=tokens["neg"], zero="baseline" if baseline else "midpoint",
                                      source="calibrate()")
        layer = int(self.layer)
        cal.add_threshold(scores["pos"][:, layer], scores["neg"][:, layer], layer, "the calibration items")
        if "neutral" in scores:
            cal.add_neutral(scores["neutral"])
        self.calibrations = [c for c in self.calibrations if not c.context.matches(ctx)] + [cal]
        self.logger.log("calibrate", {"context": ctx.to_dict(), "n_pos": cal.n_pos, "n_neg": cal.n_neg})
        return cal

    def _read_for_calibration(self, pos_texts, neg_texts, neutral_texts, as_reply_to, role, system_prompt, prompt,
                              batch_size):
        self._require_model("calibrating on texts")
        tokenizer, model = self.model_bundle.tokenizer, self.model_bundle.model
        sys_prompt = self.config["prompts"].get("neutral_system") if system_prompt is None else system_prompt

        def encode(texts: List[PromptLike], label: str) -> List[Encoded]:
            if as_reply_to is None:
                sp = self._system_for(role, system_prompt, prompt)
                return [encode_text(tokenizer, t, sp, role=role, prompt=prompt, warn=self._warn,
                                    warn_prefix=f"calibrate.{label}[{i}]") for i, t in enumerate(texts)]
            turns = [as_reply_to] * len(texts) if isinstance(as_reply_to, str) else list(as_reply_to)
            if len(turns) != len(texts):
                raise ValueError(f"as_reply_to has {len(turns)} turns for {len(texts)} {label} texts.")
            out = []
            for turn, text in zip(turns, texts):
                messages, _ = _normalize_messages(turn, sys_prompt)
                ids, plen = _completion_input_ids(tokenizer, messages, str(text))
                out.append(Encoded(ids[0], plen, int(ids.shape[-1])))
            return out

        ctx = (ReadContext.generated() if as_reply_to is not None else
               ReadContext(kind="text", role=role, system=self._system_for(role, system_prompt, prompt),
                           prompt=prompt, readout="assistant_all_mean"))
        bs = int(batch_size or self.config.get("readout", {}).get("batch_size", 16))
        scores, tokens = {}, {}
        for label, texts in (("pos", pos_texts), ("neg", neg_texts), ("neutral", neutral_texts or [])):
            if not texts:
                continue
            reps = _read(model, tokenizer, encode(list(texts), label), [Readout.parse("assistant_all_mean")],
                         batch_size=bs, project_on=self.directions, cache=self._rep_cache_dir() or False,
                         identity=self.identity, label=f"calibrate.{label}")[0]
            announce(reps.choices, console=self.console is not None and self.console.enabled, logger=self.logger)
            scores[label] = np.einsum("nld,ld->nl", reps.data, self.directions)
            tokens[label] = reps.projections
        return ctx, scores, tokens

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
        overwrite: bool = False,
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
        tokens), which the heatmaps then show. The midpoint is not "without the concept": text that
        belongs to neither pole can land well away from 0. Each record has the text's mean score over its own
        tokens, ``score_mean``, and ``score_mean_z``, standardized with the per-item references.
        z scores need references measured on texts read the same way (role, system prompt, prompt):
        the evaluation's in texts mode, the transfer check's (eval texts) in shared/opposed mode.
        Otherwise only the raw projections are saved, with a warning.
        """
        self._require_model("score_texts")
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = self._system_for(role, system_prompt, prompt)
        read_context = _text_context(role, sys_prompt, prompt, "assistant_all_mean")
        save_html = self.config["plots"].get("heatmap_html", True) if save_html is None else save_html
        save_token_scores_npz = self.config["output"].get("save_token_scores_npz", True)
        scoring_seed: Optional[int] = None
        scoring_vectors = self.directions
        if random_scoring_vector:
            scoring_vectors, scoring_seed = self._random_control_vectors(random_vector_seed)
            self._warn(
                f"score_texts using random scoring vector control (same-norm per layer, seed={scoring_seed})."
            )
        elif reference_layer is None and self.calibration_for(read_context) is None:
            self._warn(f"score_texts: only raw projections are saved: {self._missing_z_reason(read_context)}")

        base_dir = os.path.join(self._ensure_run_dir(), output_subdir)
        ensure_dir(base_dir)
        if batch_subdir is None:
            batch_subdir = f"batch_{now_tag()}"
        out_dir = base_dir if batch_subdir == "" else os.path.join(base_dir, batch_subdir)
        _prepare_batch_dir(out_dir, overwrite)
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
            standardized = self._token_z(scores_per_layer, layers, aggregate, read_context) if calibrated else None
            span = slice(text_start, text_end) if text_start is not None else slice(0, 0)
            score_mean = float(np.mean(scores_agg[span])) if scores_agg[span].size else None
            score_mean_z = self._item_z(scores_per_layer[span], layers, aggregate, read_context) if calibrated else None
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

    def _steer_sigma(self) -> Optional[float]:
        """The unit of alpha_unit="sigma": the eval items' pooled within-pole SD on the steering
        direction at the probe's layer (decision D1)."""
        sigma = self.steer_sigma
        return None if sigma is None or not np.isfinite(sigma) else float(sigma)

    def _check_alpha_unit(self, alpha_unit: str) -> None:
        """Raise if the probe lacks the reference that ``alpha_unit`` scales by."""
        if alpha_unit == "sigma" and self._steer_sigma() is None:
            raise ValueError("alpha_unit='sigma' needs the probe's steering unit (steer_sigma), which comes from "
                             "evaluation or calibration items; this probe has none.")
        if alpha_unit == "resid_norm" and self.resid_norm is None:
            raise ValueError("resid_norm is not available for alpha_unit='resid_norm'; re-train the probe.")

    def _steering_alpha(
        self, alpha: float, alpha_unit: str, steer_layers: List[int]
    ) -> Tuple[float, Optional[Dict[int, float]]]:
        """The alpha to inject, and per-layer scales of it, for ``alpha`` in ``alpha_unit``."""
        value = float(alpha)
        if alpha_unit == "sigma" and abs(value) > 1e-12:
            return value * float(self._steer_sigma()), None
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
        overwrite: bool = False,
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
        (0 is the midpoint between the eval poles, not "without the concept"; 1 their pooled
        within-pole SD). The npz and the heatmaps carry single-token z scores, standardized with
        the per-token references.

        Steering: by default only the probe's layer is steered (``steering.steer_layers =
        "probe"``). With several layers (``"window"`` or a list), each layer adds its own probe
        vector ``v_l``, and with ``steer_distribute`` alpha is divided by the number of layers
        actually steered (fewer at the model's edges). Each record has the alpha applied at each
        layer, ``steer_alpha_per_layer``.

        Generation is greedy by default (``steering.steer_sampling``). When sampling, the RNGs
        are seeded with ``seed``, or else ``random.seed``, so completions are reproducible
        (best-effort: GPU kernels are not bit-deterministic across hardware/library versions);
        each record has the generation settings under ``generation``.
        """
        self._require_model("score_prompts")
        alpha_unit = _validate_alpha_unit(alpha_unit)
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
        scoring_vectors = self.directions
        steering_vectors = self.steer_directions
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
        if not greedy:
            # Sampled completions are reproducible: without an explicit seed, random.seed is used.
            seed = int(self.config.get("random", {}).get("seed", 0)) if seed is None else int(seed)
            set_seed(seed)
        elif seed is not None:
            set_seed(int(seed))
        generation = {"greedy": bool(greedy), "max_new_tokens": int(max_new_tokens),
                      "temperature": None if greedy else temperature, "top_p": None if greedy else top_p,
                      "seed": None if greedy else seed}

        if alphas is None:
            alphas = [0.0]
        if len(alphas) == 0:
            raise ValueError("alphas is empty; pass at least one alpha (0.0 for no steering).")
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
            steer_layers, int(self.layer), steer_window_radius, self.num_layers
        )

        base_dir = os.path.join(self._ensure_run_dir(), output_subdir)
        ensure_dir(base_dir)
        if batch_subdir is None:
            batch_subdir = f"batch_{now_tag()}"
        out_dir = base_dir if batch_subdir == "" else os.path.join(base_dir, batch_subdir)
        _prepare_batch_dir(out_dir, overwrite)
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
        z_missing = None
        if not random_scoring_vector and self.calibration_for(GENERATED_CONTEXT) is None:
            z_missing = self._missing_z_reason(GENERATED_CONTEXT)
            self._warn(f"score_prompts: score_mean_z is None: {z_missing}")
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
                    steer_alpha_per_layer = context.per_layer_alpha()
                else:
                    context = None
                    steer_alpha_per_layer = {}

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
                score_mean = float(np.mean(completion_scores)) if completion_scores.size else None
                standardized = None if random_scoring_vector else self._token_z(
                    scores_per_layer, layers, aggregate, GENERATED_CONTEXT
                )
                score_mean_z = (
                    None if random_scoring_vector
                    else self._item_z(scores_per_layer[prompt_len:completion_end], layers, aggregate, GENERATED_CONTEXT)
                )

                alpha_label = _alpha_label(float(alpha), alpha_unit)
                # j keeps repeated alphas (several samples of one prompt) in separate files
                base = f"prompt_{i:03d}_{prompt_slug}_a{j:02d}_{alpha_label}"
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
                    segment_scores = scores_agg if standardized is None else standardized[1]
                    segments = segment_token_scores(
                        tokens,
                        segment_scores.tolist(),
                        prompt_len=prompt_len,
                    )
                    with open(segments_path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {"prompt_len": prompt_len, "score_kind": "raw" if standardized is None else "z",
                             "segments": segments},
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
                    "steer_alpha_per_layer": steer_alpha_per_layer,
                    "steer_vectors": "each steered layer's own probe vector",
                    "generation": generation,
                    "sampling_seed": generation["seed"],
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
                    "z_missing": z_missing,
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
    def load(cls, run_dir: str, model: Any = None, tokenizer: Any = None, *,
             allow_model_mismatch: bool = False) -> "ConceptProbe":
        """A probe saved in ``run_dir`` (by training, :meth:`fit`, :meth:`from_directions` or
        :meth:`save`), with ``model`` attached if given (see :meth:`attach`). Runs saved by earlier
        versions load too."""
        with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.loads(f.read())
        with open(os.path.join(run_dir, "concept.json"), "r", encoding="utf-8") as f:
            concept = ConceptSpec.from_config({"concept": json.loads(f.read())})
        metrics: Dict[str, Any] = {}
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.loads(f.read())
        with np.load(os.path.join(run_dir, "tensors.npz")) as t:
            tensors = {k: t[k] for k in t.files}
        out_cfg = cfg.get("output", {})
        obj = cls(config=cfg, concept=concept, run_dir=run_dir,
                  logger=JsonlLogger(os.path.join(run_dir, "log.jsonl"), enabled=bool(out_cfg.get("log_jsonl", True))),
                  console=ConsoleLogger(bool(out_cfg.get("console", True))))
        obj.directions = tensors["concept_vectors"]
        obj.layer = int(tensors["best_layer"][0])
        obj.sweep_d, obj.sweep_p = tensors.get("sweep_d"), tensors.get("sweep_p")
        obj.resid_norm = tensors.get("resid_norm")
        obj.calibrations = calibrations_from_tensors(tensors)
        obj.identity = metrics.get("model_identity") or identity_from_config(cfg.get("model", {}))
        estimator, _ = _training_estimator(cfg.get("training", {}))
        obj.estimator = metrics.get("estimator") or estimator_spec(estimator)
        if "steer_vectors" in tensors:
            obj.steer_directions = tensors["steer_vectors"]
            if np.array_equal(obj.steer_directions, obj.directions):
                obj.steer_directions = obj.directions
            sigma = float(tensors["steer_sigma_best_layer"][0])
            obj.steer_estimator = metrics.get("steer_direction") or obj.estimator
        else:
            # Saved before steering had its own direction: it steers with the one it reads with.
            obj.steer_directions, obj.steer_estimator = obj.directions, obj.estimator
            sigma = float(tensors["score_scale"][obj.layer]) if "score_scale" in tensors else float("nan")
            if obj.estimator != {"kind": "mean_diff", "normalization": "none"}:
                announce([Choice("steering.steer_direction", "mean_diff", "the run's reading direction",
                                 f"this run ({obj.estimator}) was saved before steering had its own direction, so it "
                                 "steers with the one it reads with; retrain it to steer with the raw difference of means")],
                         console=obj.console.enabled, logger=obj.logger)
        obj.steer_sigma = sigma if np.isfinite(sigma) else None
        if model is not None:
            obj.attach(model, tokenizer, allow_model_mismatch=allow_model_mismatch)
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


def model_identity_of(obj: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The model identity of a bundle or probe: the one recorded when it was loaded or trained, or
    else the one its config describes (runs saved before identities were recorded)."""
    identity = getattr(obj, "identity", None)
    if identity:
        return identity
    cfg = config if config is not None else getattr(obj, "config", None) or {}
    return identity_from_config(cfg.get("model", {}))


class ProbeWorkspace:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        defaults_path: Optional[str] = None,
        model_id: Optional[str] = None,
        project_directory: Optional[str] = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """A model and a config to train and load probes with.

        The model is loaded from the config's ``model`` section, or passed in already loaded:
        ``model`` is a :class:`ModelBundle`, or a Hugging Face model together with its
        ``tokenizer``. A passed model's own settings (name, dtype, quantization) replace the
        config's, and that replacement is printed and logged like every automatic choice.
        """
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
        else:
            cfg = resolve_config(config_overrides, defaults_path=defaults_path)
            if model_id:
                cfg["model"]["model_id"] = model_id
            if root_dir:
                cfg["output"]["root_dir"] = root_dir
            self.config = cfg
            self.root_dir = cfg["output"]["root_dir"]
            ensure_dir(self.root_dir)
            self.console = ConsoleLogger(cfg["output"].get("console", False))
        self.model_bundle = self._model_bundle(model, tokenizer, model_id)
        announce(self.model_bundle.choices, console=self.console.enabled)
        self.console.info(f"Model ready ({self.model_bundle.num_layers} layers).")

    def _model_bundle(self, model: Any, tokenizer: Any, model_id: Optional[str]) -> ModelBundle:
        cfg = self.config
        if model is None:
            if tokenizer is not None:
                raise ValueError("tokenizer was passed without a model.")
            self.console.info(f"Loading model {cfg['model']['model_id']}...")
            return ModelBundle.load(cfg["model"])
        if isinstance(model, ModelBundle):
            if tokenizer is not None:
                raise ValueError("Pass either a ModelBundle or a model and its tokenizer, not both.")
            bundle = model
            if not bundle.identity:
                bundle.identity = identity_of(bundle.model, bundle.tokenizer, bundle.model_id)
        else:
            if tokenizer is None:
                raise ValueError("A model passed to ProbeWorkspace needs its tokenizer (tokenizer=...).")
            bundle = ModelBundle.from_loaded(model, tokenizer, model_id=model_id, model_cfg=cfg["model"])
        identity = bundle.identity
        described = identity_from_config(cfg["model"])
        replaced = {k: identity.get(k) for k in ("model_id", "dtype", "quantization")
                    if described.get(k) != identity.get(k)}
        if replaced:
            bundle.choices = list(bundle.choices) + [Choice(
                "model", {k: described.get(k) for k in replaced}, replaced,
                "a loaded model was passed in; its own settings replace the config's model section",
            )]
        cfg["model"].update(model_id=identity.get("model_id") or cfg["model"]["model_id"],
                            dtype=identity.get("dtype") or cfg["model"].get("dtype"),
                            quantization=identity.get("quantization"), use_4bit=False)
        return bundle

    def get_probe(
        self,
        name: Optional[str] = None,
        project_directory: Optional[str] = None,
        allow_model_mismatch: bool = False,
    ) -> "ConceptProbe":
        """Load a previously trained probe without retraining.

        If this workspace was created with project_directory=..., that directory is used by default.
        The workspace's model identity (model, dtype actually used, quantization, chat template and
        its date) must match the run's, since the probe's directions only apply to the activations
        they were trained on; ``allow_model_mismatch=True`` loads it anyway. Scoring then uses the
        workspace config, and a warning lists the settings that differ from the run's.
        """
        if not (project_directory or self.project_directory):
            raise ValueError("No project_directory provided. Create ProbeWorkspace(project_directory=...) or pass it to get_probe().")
        run_dir = os.path.abspath(project_directory or self.project_directory)

        probe = ConceptProbe.load(run_dir)
        mismatched = identity_mismatches(model_identity_of(probe), model_identity_of(self.model_bundle, self.config))
        if mismatched and not allow_model_mismatch:
            raise ValueError(
                "The workspace model differs from the one this probe was trained on (run vs workspace: "
                + "; ".join(mismatched)
                + "). Load it with ProbeWorkspace(project_directory=run_dir) to use the run's own model "
                "settings, or pass allow_model_mismatch=True."
            )
        differences = [
            d for d in _dotted_differences(probe.config, self.config) if not d.startswith(("output.", "concept."))
        ]
        if differences:
            self.console.warn(
                "Scoring uses the workspace config, which differs from the saved run config in: "
                + ", ".join(differences[:12]) + (" ..." if len(differences) > 12 else "") + "."
            )
        probe.config = self.config
        probe.console = self.console
        probe.model_bundle = self.model_bundle

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
