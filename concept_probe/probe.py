import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .config import ConceptSpec, resolve_config
from .logger import JsonlLogger
from .modeling import ModelBundle, _resolve_attr_path, apply_chat, attention_mask_from_ids
from .steering import MultiLayerSteererLayerwise
from .utils import deep_merge, ensure_dir, json_dump, jsonl_to_pretty, now_tag, safe_slug, set_seed
from .visuals import plot_score_hist, plot_sweep, render_batch_heatmap, render_token_heatmap, segment_token_scores
from .console import ConsoleLogger

try:
    from scipy.stats import ttest_ind
except Exception:
    ttest_ind = None


def normed_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


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


def _validate_layer_index(value: Any, num_layers: int, what: str) -> int:
    idx = int(value)
    if not (0 <= idx < num_layers):
        raise ValueError(f"{what} must be in [0, {num_layers - 1}]; got {idx}.")
    return idx


def pick_layer(num_layers: int, layer_idx: Optional[int], layer_frac: float) -> int:
    if layer_idx is not None:
        return _validate_layer_index(layer_idx, num_layers, "training.layer_idx")
    return int(layer_frac * (num_layers - 1))


def _layer_index_from_value(value: Any, num_layers: int) -> int:
    if isinstance(value, bool):
        raise TypeError("Layer values must be int or float, not bool.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if 0.0 <= float(value) <= 1.0:
            return int(float(value) * (num_layers - 1))
        raise ValueError(
            f"Float layer values are interpreted as fractions of depth in [0, 1]; got {value}. "
            "Use an int for an absolute layer index."
        )
    raise TypeError(f"Layer values must be int or float; got {type(value)}")


def _normalize_best_layer_search(
    search_spec: Optional[Any], num_layers: int
) -> Dict[str, Any]:
    if search_spec is None:
        return {
            "intervals": [(0, num_layers - 1)],
            "layers": list(range(num_layers)),
            "is_default": True,
        }

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
        lo = _layer_index_from_value(raw[0], num_layers)
        hi = _layer_index_from_value(raw[1], num_layers)
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0, min(num_layers - 1, lo))
        hi = max(0, min(num_layers - 1, hi))
        intervals.append((lo, hi))

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

    if not layers:
        raise ValueError("training.best_layer_search did not resolve to any valid layers.")

    return {"intervals": merged_intervals, "layers": layers, "is_default": False}


def select_layers(selection: Dict[str, Any], best_layer: int, num_layers: int) -> List[int]:
    mode = selection.get("mode", "best")
    if mode == "best":
        return [best_layer]
    if mode == "window":
        radius = int(selection.get("window_radius", 0))
        lo = max(0, best_layer - radius)
        hi = min(num_layers - 1, best_layer + radius)
        return list(range(lo, hi + 1))
    if mode == "explicit":
        return [
            _validate_layer_index(l, num_layers, "layer_selection.layers entries")
            for l in selection.get("layers", [])
        ]
    if mode == "all":
        return list(range(num_layers))
    raise ValueError(f"Unknown layer selection mode: {mode}")


def _decode_tokens(tokenizer, token_ids: Iterable[int]) -> List[str]:
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids]


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


_CHAT_TEMPLATE_FALLBACK_WARNED: set = set()


def _encode_for_read(
    tokenizer,
    prompt: "PromptLike",
    default_system: str,
    *,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "read",
) -> torch.Tensor:
    """Encode a 'read' input (no generation). Uses the chat template when the
    tokenizer has one; otherwise falls back to raw tokenization (system text
    is dropped — base models have no chat structure to wrap it in).
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
            f"{warn_prefix}: tokenizer has no chat_template; "
            f"falling back to raw tokenization (system text ignored)."
        )
        _CHAT_TEMPLATE_FALLBACK_WARNED.add(fallback_key)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]


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
    default_system: str,
    *,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "prompt",
) -> Tuple[List[PromptMessage], bool]:
    """Return (messages, used_prompt_system).

    - If prompt is a string: wrap as system+user.
    - If prompt is a message list:
        - If it contains any system message, do not inject default_system; warn if default_system is non-empty.
        - If it contains no system message, prepend default_system as a system message.
    """
    if isinstance(prompt, str):
        return (
            [
                {"role": "system", "content": str(default_system)},
                {"role": "user", "content": prompt},
            ],
            False,
        )

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

    return ([{"role": "system", "content": str(default_system)}] + messages, False)


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


def reps_from_hs_layers(
    hs_layers: List[torch.Tensor],
    prompt_len: Optional[int],
    mode: str,
    last_k: int,
    warn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    num_layers = len(hs_layers)
    dim = int(hs_layers[0].shape[1])
    reps = np.zeros((num_layers, dim), dtype=np.float32)

    if (
        warn is not None
        and prompt_len is not None
        and mode in ("assistant_last_k_mean", "assistant_all_mean")
        and prompt_len >= int(hs_layers[0].shape[0])
    ):
        warn(
            f"mode={mode}: no completion tokens (prompt_len={prompt_len} >= "
            f"seq_len={int(hs_layers[0].shape[0])}); falling back to full-sequence pooling, "
            "which includes prompt/system tokens and can leak the system-prompt difference "
            "into the concept vector."
        )

    for l in range(num_layers):
        hs = hs_layers[l]
        total = hs.shape[0]

        if mode == "sequence_last":
            reps[l] = hs[-1].numpy()
        elif mode == "sequence_last_k_mean":
            k = min(last_k, total)
            reps[l] = hs[-k:].mean(dim=0).numpy()
        elif mode == "sequence_all_mean":
            reps[l] = hs.mean(dim=0).numpy()
        elif mode == "assistant_last":
            reps[l] = hs[-1].numpy()
        elif mode == "assistant_last_k_mean":
            if prompt_len is None or prompt_len >= total:
                k = min(last_k, total)
                reps[l] = hs[-k:].mean(dim=0).numpy()
            else:
                span = hs[prompt_len:]
                if span.shape[0] == 0:
                    reps[l] = hs.mean(dim=0).numpy()
                else:
                    k = min(last_k, span.shape[0])
                    reps[l] = span[-k:].mean(dim=0).numpy()
        elif mode == "assistant_all_mean":
            if prompt_len is None or prompt_len >= total:
                reps[l] = hs.mean(dim=0).numpy()
            else:
                span = hs[prompt_len:]
                reps[l] = (span.mean(dim=0) if span.shape[0] > 0 else hs.mean(dim=0)).numpy()
        else:
            raise ValueError(f"Unknown mode: {mode}")

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
        prompt_ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
        plen = int(prompt_ids.shape[-1])
        comp_ids = tokenizer(str(completion), add_special_tokens=False, return_tensors="pt")["input_ids"]
        if comp_ids.dim() == 1:
            comp_ids = comp_ids.unsqueeze(0)
        full_ids = torch.cat([prompt_ids, comp_ids.to(prompt_ids.dtype)], dim=1)
        hs_layers = forward_hidden_states_all_layers(model, full_ids)
        out_reps.append(reps_from_hs_layers(hs_layers, prompt_len=plen, mode=pool, last_k=last_k))
    return np.stack(out_reps, axis=0).astype(np.float32)


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
        ref = int(reference_layer)
        if ref < 0:
            ref += num_vectors
        if not (0 <= ref < num_vectors):
            raise ValueError(
                f"reference_layer={reference_layer} out of range for {num_vectors} concept vectors."
            )
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
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    """Score every token of a RAW text against the per-layer concept directions, in memory.

    Low-level read primitive for callers that need per-token, per-layer scores together with
    character offsets (e.g. to target a specific word/subtoken), without the artifact-writing or
    chat-template wrapping of ``ConceptProbe.score_texts``. The text is tokenized as-is (no chat
    template); the forward pass uses the same all-layers capture as training (with the final-norm
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
        add_special_tokens: passed to the tokenizer (default True; BOS gets offset (0, 0)).

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
        else [_validate_layer_index(i, num_layers, "layer_indices entries") for i in layer_indices]
    )

    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=add_special_tokens)
    om = enc["offset_mapping"]
    if hasattr(om, "tolist"):
        om = om.tolist()
    offsets = [list(span) for span in om[0]]
    input_ids = enc["input_ids"]

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
    ``[prompt_tokens + completion_tokens]`` with all padding stripped -- the same
    contract as ``generate_once``, so callers can use the results interchangeably.
    Prompts are left-padded within each batch and an attention mask is passed so
    greedy decoding matches the unbatched path; left padding and any trailing right
    padding are stripped per item before returning.
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
            row = gen[j][maxlen - plen :]  # drop left padding -> [prompt + completion (+ right pad)]
            keep = int(row.shape[-1])
            while keep > plen and int(row[keep - 1]) == pad_id:
                keep -= 1
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
    proj_std_best_layer: Optional[float] = None
    sweep_d: Optional[np.ndarray] = None
    sweep_p: Optional[np.ndarray] = None

    def train(self) -> "ConceptProbe":
        if ttest_ind is None:
            self._warn("scipy not available; sweep_p will be NaN.")
        set_seed(int(self.config["random"]["seed"]))
        ensure_dir(self.run_dir)

        config_path = os.path.join(self.run_dir, "config.json")
        concept_path = os.path.join(self.run_dir, "concept.json")
        json_dump(config_path, self.config)
        json_dump(concept_path, self.concept.to_dict())

        self.logger.log("config", {"config": self.config, "concept": self.concept.to_dict()})

        readout = self.config["readout"]
        train_cfg = self.config["training"]
        eval_cfg = self.config["evaluation"]
        out_cfg = self.config["output"]
        plot_cfg = self.config["plots"]
        prompts_cfg = self.config["prompts"]
        train_prompt_mode = train_cfg.get("train_prompt_mode", "shared")
        train_prompts = list(prompts_cfg.get("train_questions", []))
        train_prompts_pos = list(prompts_cfg.get("train_questions_pos", []))
        train_prompts_neg = list(prompts_cfg.get("train_questions_neg", []))

        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        num_layers = self.model_bundle.num_layers

        self._status(f"Training concept '{self.concept.name}' with model {self.model_bundle.model_id}")
        if train_prompt_mode in ("shared", "opposed") and self.concept.pos_system == self.concept.neg_system:
            self._warn(
                "pos_system == neg_system; contrastive training will compare identical conditions "
                "and the learned direction will be noise."
            )
        phase1_action = "Reading training texts" if train_prompt_mode == "texts" else "Generating training completions"
        self._status(f"[Phase 1/4] {phase1_action} ({train_prompt_mode})")
        train_pos_ids, train_neg_ids = [], []
        train_pos_plens, train_neg_plens = [], []

        if train_prompt_mode == "shared":
            total_train = len(train_prompts)
            progress_every = max(1, total_train // 5) if total_train else 1
            if not train_prompts:
                raise ValueError("prompts.train_questions is empty; provide training prompts.")
            if train_prompts_pos or train_prompts_neg:
                self._warn(
                    "train_prompt_mode=shared; ignoring prompts.train_questions_pos/train_questions_neg."
                )
            gen_bs = int(train_cfg.get("gen_batch_size", 16))
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
                train_pos_plens.append(plen_pos)
                train_neg_plens.append(plen_neg)

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
            total_pos = len(train_prompts_pos)
            progress_every_pos = max(1, total_pos // 5) if total_pos else 1
            for i, q in enumerate(train_prompts_pos):
                ids_pos, plen_pos = generate_once(
                    model,
                    tokenizer,
                    self.concept.pos_system,
                    q,
                    train_cfg["train_max_new_tokens"],
                    train_cfg["train_greedy"],
                    temperature=train_cfg.get("train_temperature", 0.7),
                    top_p=train_cfg.get("train_top_p", 0.9),
                    warn=self._warn,
                    warn_prefix=f"train_questions_pos[{i}] (pos_system)",
                )
                train_pos_ids.append(ids_pos)
                train_pos_plens.append(plen_pos)
                pos_completion = tokenizer.decode(ids_pos[plen_pos:], skip_special_tokens=True)
                self.logger.log(
                    "train_gen_pos",
                    {
                        "i": i,
                        "question": q,
                        "label": self.concept.pos_label,
                        "system": self.concept.pos_system,
                        "text": tokenizer.decode(ids_pos, skip_special_tokens=False),
                        "completion": pos_completion,
                    },
                )
                if total_pos and ((i + 1) % progress_every_pos == 0 or (i + 1) == total_pos):
                    self._status(f"Generated pos training completions {i + 1}/{total_pos}")

            total_neg = len(train_prompts_neg)
            progress_every_neg = max(1, total_neg // 5) if total_neg else 1
            for i, q in enumerate(train_prompts_neg):
                ids_neg, plen_neg = generate_once(
                    model,
                    tokenizer,
                    self.concept.neg_system,
                    q,
                    train_cfg["train_max_new_tokens"],
                    train_cfg["train_greedy"],
                    temperature=train_cfg.get("train_temperature", 0.7),
                    top_p=train_cfg.get("train_top_p", 0.9),
                    warn=self._warn,
                    warn_prefix=f"train_questions_neg[{i}] (neg_system)",
                )
                train_neg_ids.append(ids_neg)
                train_neg_plens.append(plen_neg)
                neg_completion = tokenizer.decode(ids_neg[plen_neg:], skip_special_tokens=True)
                self.logger.log(
                    "train_gen_neg",
                    {
                        "i": i,
                        "question": q,
                        "label": self.concept.neg_label,
                        "system": self.concept.neg_system,
                        "text": tokenizer.decode(ids_neg, skip_special_tokens=False),
                        "completion": neg_completion,
                    },
                )
                if total_neg and ((i + 1) % progress_every_neg == 0 or (i + 1) == total_neg):
                    self._status(f"Generated neg training completions {i + 1}/{total_neg}")
        elif train_prompt_mode == "texts":
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
            train_text_system = train_cfg.get("train_text_system")
            if train_text_system is None:
                train_text_system = prompts_cfg.get("neutral_system", "")

            total_pos = len(train_pos_texts)
            progress_every_pos = max(1, total_pos // 5) if total_pos else 1
            for i, s in enumerate(train_pos_texts):
                input_ids = _encode_for_read(
                    tokenizer,
                    s,
                    train_text_system,
                    warn=self._warn,
                    warn_prefix=f"train_pos_texts[{i}]",
                )
                ids_1d = input_ids[0].detach().cpu()
                train_pos_ids.append(ids_1d)
                train_pos_plens.append(int(ids_1d.shape[-1]))
                self.logger.log(
                    "train_read_pos",
                    {
                        "i": i,
                        "label": self.concept.pos_label,
                        "system": train_text_system,
                        "text": tokenizer.decode(ids_1d, skip_special_tokens=False),
                    },
                )
                if total_pos and ((i + 1) % progress_every_pos == 0 or (i + 1) == total_pos):
                    self._status(f"Read pos training texts {i + 1}/{total_pos}")

            total_neg = len(train_neg_texts)
            progress_every_neg = max(1, total_neg // 5) if total_neg else 1
            for i, s in enumerate(train_neg_texts):
                input_ids = _encode_for_read(
                    tokenizer,
                    s,
                    train_text_system,
                    warn=self._warn,
                    warn_prefix=f"train_neg_texts[{i}]",
                )
                ids_1d = input_ids[0].detach().cpu()
                train_neg_ids.append(ids_1d)
                train_neg_plens.append(int(ids_1d.shape[-1]))
                self.logger.log(
                    "train_read_neg",
                    {
                        "i": i,
                        "label": self.concept.neg_label,
                        "system": train_text_system,
                        "text": tokenizer.decode(ids_1d, skip_special_tokens=False),
                    },
                )
                if total_neg and ((i + 1) % progress_every_neg == 0 or (i + 1) == total_neg):
                    self._status(f"Read neg training texts {i + 1}/{total_neg}")
        else:
            raise ValueError(f"Unknown train_prompt_mode: {train_prompt_mode}")

        train_readout_mode = readout["read_mode"] if train_prompt_mode == "texts" else readout["train_mode"]
        train_readout_last_k = readout["read_last_k"] if train_prompt_mode == "texts" else readout["train_last_k"]

        self._status("[Phase 2/4] Forward passes for training reps")
        reps_pos = np.stack(
            [
                reps_from_hs_layers(
                    forward_hidden_states_all_layers(model, train_pos_ids[i].unsqueeze(0)),
                    prompt_len=train_pos_plens[i],
                    mode=train_readout_mode,
                    last_k=train_readout_last_k,
                    warn=None if train_prompt_mode == "texts" else self._warn,
                )
                for i in range(len(train_pos_ids))
            ],
            axis=0,
        ).astype(np.float32)

        reps_neg = np.stack(
            [
                reps_from_hs_layers(
                    forward_hidden_states_all_layers(model, train_neg_ids[i].unsqueeze(0)),
                    prompt_len=train_neg_plens[i],
                    mode=train_readout_mode,
                    last_k=train_readout_last_k,
                    warn=None if train_prompt_mode == "texts" else self._warn,
                )
                for i in range(len(train_neg_ids))
            ],
            axis=0,
        ).astype(np.float32)

        num_layers_check = reps_pos.shape[1]
        if num_layers_check != num_layers:
            raise ValueError("Layer count mismatch between model and reps.")

        concept_vectors = np.zeros((num_layers, reps_pos.shape[2]), dtype=np.float32)
        for l in range(num_layers):
            mu_pos = reps_pos[:, l, :].mean(axis=0)
            mu_neg = reps_neg[:, l, :].mean(axis=0)
            concept_vectors[l] = normed_np(mu_pos - mu_neg)

        eval_pos_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        eval_neg_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        sweep_d = np.zeros((num_layers,), dtype=np.float32)
        sweep_p = np.full((num_layers,), np.nan, dtype=np.float32)
        eval_pos_mean = np.zeros((num_layers,), dtype=np.float32)
        eval_neg_mean = np.zeros((num_layers,), dtype=np.float32)
        eval_pos_std = np.zeros((num_layers,), dtype=np.float32)
        eval_neg_std = np.zeros((num_layers,), dtype=np.float32)

        eval_prompts = list(self.config["prompts"]["eval_questions"])
        did_eval = False
        n_eval_pos = 0
        n_eval_neg = 0
        eval_mode = eval_cfg.get("eval_mode", "auto")
        if eval_cfg.get("do_eval", True):
            if eval_mode == "auto":
                if len(self.concept.eval_pos_texts) > 0 and len(self.concept.eval_neg_texts) > 0:
                    eval_mode = "read"
                elif train_prompt_mode == "texts":
                    self._warn(
                        "eval_mode=auto with train_prompt_mode=texts but no eval_pos_texts/"
                        "eval_neg_texts; skipping evaluation (generate-mode eval would contrast "
                        "pos_system vs neg_system, which texts-mode concepts typically leave identical)."
                    )
                    eval_mode = "skip"
                else:
                    eval_mode = "generate"

            if eval_mode == "generate" and self.concept.pos_system == self.concept.neg_system:
                self._warn(
                    "pos_system == neg_system; generate-mode eval contrasts identical conditions, "
                    "so the layer sweep cannot separate the concept."
                )

            if eval_mode == "read":
                eval_pos_texts = list(self.concept.eval_pos_texts)
                eval_neg_texts = list(self.concept.eval_neg_texts)
                eval_system = eval_cfg.get("eval_system", self.config["prompts"]["neutral_system"])
                if len(eval_pos_texts) > 0 and len(eval_neg_texts) > 0:
                    self._status("[Phase 3/4] Reading eval texts")
                    eval_pos_reps_list: List[np.ndarray] = []
                    total_eval_pos = len(eval_pos_texts)
                    progress_every_pos = max(1, total_eval_pos // 5) if total_eval_pos else 1
                    for i, s in enumerate(eval_pos_texts):
                        input_ids = _encode_for_read(
                            tokenizer,
                            s,
                            eval_system,
                            warn=self._warn,
                            warn_prefix=f"eval_pos_texts[{i}]",
                        )
                        eval_pos_reps_list.append(
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(model, input_ids),
                                prompt_len=None,
                                mode=readout["read_mode"],
                                last_k=readout["read_last_k"],
                            )
                        )
                        if total_eval_pos and ((i + 1) % progress_every_pos == 0 or (i + 1) == total_eval_pos):
                            self._status(f"Read eval pos text {i + 1}/{total_eval_pos}")
                    eval_pos_reps = np.stack(eval_pos_reps_list, axis=0).astype(np.float32)

                    eval_neg_reps_list: List[np.ndarray] = []
                    total_eval_neg = len(eval_neg_texts)
                    progress_every_neg = max(1, total_eval_neg // 5) if total_eval_neg else 1
                    for i, s in enumerate(eval_neg_texts):
                        input_ids = _encode_for_read(
                            tokenizer,
                            s,
                            eval_system,
                            warn=self._warn,
                            warn_prefix=f"eval_neg_texts[{i}]",
                        )
                        eval_neg_reps_list.append(
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(model, input_ids),
                                prompt_len=None,
                                mode=readout["read_mode"],
                                last_k=readout["read_last_k"],
                            )
                        )
                        if total_eval_neg and ((i + 1) % progress_every_neg == 0 or (i + 1) == total_eval_neg):
                            self._status(f"Read eval neg text {i + 1}/{total_eval_neg}")
                    eval_neg_reps = np.stack(eval_neg_reps_list, axis=0).astype(np.float32)

                    eval_pos_scores_mat = np.zeros((eval_pos_reps.shape[0], num_layers), dtype=np.float32)
                    eval_neg_scores_mat = np.zeros((eval_neg_reps.shape[0], num_layers), dtype=np.float32)

                    for l in range(num_layers):
                        v = concept_vectors[l]
                        pos_scores = eval_pos_reps[:, l, :] @ v
                        neg_scores = eval_neg_reps[:, l, :] @ v
                        eval_pos_scores_mat[:, l] = pos_scores
                        eval_neg_scores_mat[:, l] = neg_scores
                        eval_pos_mean[l] = pos_scores.mean()
                        eval_neg_mean[l] = neg_scores.mean()
                        eval_pos_std[l] = pos_scores.std(ddof=1)
                        eval_neg_std[l] = neg_scores.std(ddof=1)
                        sweep_d[l] = cohen_d_np(pos_scores, neg_scores)
                        if ttest_ind is not None:
                            _, p = ttest_ind(pos_scores, neg_scores, equal_var=False)
                            sweep_p[l] = float(p)
                    did_eval = True
                    n_eval_pos = len(eval_pos_texts)
                    n_eval_neg = len(eval_neg_texts)
                else:
                    self._warn("eval_mode=read but eval_pos_texts/eval_neg_texts are empty.")
            elif eval_mode == "generate":
                if len(eval_prompts) > 0:
                    self._status("[Phase 3/4] Generating eval completions")
                    eval_pos_ids, eval_neg_ids = [], []
                    eval_pos_plens, eval_neg_plens = [], []
                    total_eval = len(eval_prompts)
                    progress_every = max(1, total_eval // 5) if total_eval else 1

                    gen_bs = int(eval_cfg.get("gen_batch_size", train_cfg.get("gen_batch_size", 16)))
                    pos_results = generate_batch(
                        model,
                        tokenizer,
                        self.concept.pos_system,
                        eval_prompts,
                        eval_cfg["eval_max_new_tokens"],
                        eval_cfg["eval_greedy"],
                        temperature=eval_cfg.get("eval_temperature", 0.7),
                        top_p=eval_cfg.get("eval_top_p", 0.9),
                        batch_size=gen_bs,
                        warn=self._warn,
                        warn_prefix="eval_questions (pos_system)",
                    )
                    neg_results = generate_batch(
                        model,
                        tokenizer,
                        self.concept.neg_system,
                        eval_prompts,
                        eval_cfg["eval_max_new_tokens"],
                        eval_cfg["eval_greedy"],
                        temperature=eval_cfg.get("eval_temperature", 0.7),
                        top_p=eval_cfg.get("eval_top_p", 0.9),
                        batch_size=gen_bs,
                        warn=self._warn,
                        warn_prefix="eval_questions (neg_system)",
                    )
                    for i, q in enumerate(eval_prompts):
                        ids_pos, plen_pos = pos_results[i]
                        ids_neg, plen_neg = neg_results[i]
                        eval_pos_ids.append(ids_pos)
                        eval_neg_ids.append(ids_neg)
                        eval_pos_plens.append(plen_pos)
                        eval_neg_plens.append(plen_neg)

                        self.logger.log(
                            "eval_gen",
                            {
                                "i": i,
                                "question": q,
                                "pos_text": tokenizer.decode(ids_pos, skip_special_tokens=False),
                                "neg_text": tokenizer.decode(ids_neg, skip_special_tokens=False),
                                "pos_completion": tokenizer.decode(ids_pos[plen_pos:], skip_special_tokens=True),
                                "neg_completion": tokenizer.decode(ids_neg[plen_neg:], skip_special_tokens=True),
                            },
                        )
                        if total_eval and ((i + 1) % progress_every == 0 or (i + 1) == total_eval):
                            self._status(f"Generated eval completions {i + 1}/{total_eval}")

                    eval_pos_reps = np.stack(
                        [
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(model, eval_pos_ids[i].unsqueeze(0)),
                                prompt_len=eval_pos_plens[i],
                                mode=readout["train_mode"],
                                last_k=readout["train_last_k"],
                                warn=self._warn,
                            )
                            for i in range(len(eval_prompts))
                        ],
                        axis=0,
                    ).astype(np.float32)

                    eval_neg_reps = np.stack(
                        [
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(model, eval_neg_ids[i].unsqueeze(0)),
                                prompt_len=eval_neg_plens[i],
                                mode=readout["train_mode"],
                                last_k=readout["train_last_k"],
                                warn=self._warn,
                            )
                            for i in range(len(eval_prompts))
                        ],
                        axis=0,
                    ).astype(np.float32)

                    eval_pos_scores_mat = np.zeros((eval_pos_reps.shape[0], num_layers), dtype=np.float32)
                    eval_neg_scores_mat = np.zeros((eval_neg_reps.shape[0], num_layers), dtype=np.float32)

                    for l in range(num_layers):
                        v = concept_vectors[l]
                        pos_scores = eval_pos_reps[:, l, :] @ v
                        neg_scores = eval_neg_reps[:, l, :] @ v
                        eval_pos_scores_mat[:, l] = pos_scores
                        eval_neg_scores_mat[:, l] = neg_scores
                        eval_pos_mean[l] = pos_scores.mean()
                        eval_neg_mean[l] = neg_scores.mean()
                        eval_pos_std[l] = pos_scores.std(ddof=1)
                        eval_neg_std[l] = neg_scores.std(ddof=1)
                        sweep_d[l] = cohen_d_np(pos_scores, neg_scores)
                        if ttest_ind is not None:
                            _, p = ttest_ind(pos_scores, neg_scores, equal_var=False)
                            sweep_p[l] = float(p)
                    did_eval = True
                    n_eval_pos = len(eval_prompts)
                    n_eval_neg = len(eval_prompts)
            elif eval_mode != "skip":
                raise ValueError(f"Unknown evaluation.eval_mode: {eval_mode!r} (use 'auto', 'read', or 'generate').")

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

        search_intervals = None
        search_layers = None
        search_default = None
        search_intervals_plot = None
        if train_cfg["do_layer_sweep"] and did_eval:
            search_info = _normalize_best_layer_search(train_cfg.get("best_layer_search"), num_layers)
            search_intervals = search_info["intervals"]
            search_layers = search_info["layers"]
            search_default = search_info["is_default"]
            if not search_default:
                search_intervals_plot = search_intervals
            allowed_mask = np.zeros((num_layers,), dtype=bool)
            allowed_mask[search_layers] = True
            if train_cfg["sweep_select"] == "effect_size":
                masked = np.where(allowed_mask & np.isfinite(sweep_d), sweep_d, -np.inf)
                best_layer = int(np.argmax(masked))
                if not np.isfinite(masked[best_layer]):
                    raise ValueError(
                        "No finite Cohen's d in the best_layer_search range. This typically means "
                        "fewer than 2 eval samples per side; add eval data or disable the layer sweep."
                    )
            else:
                masked = np.where(allowed_mask, sweep_p, np.nan)
                if np.all(np.isnan(masked)):
                    raise ValueError(
                        "All sweep p-values are NaN for the selected best_layer_search. "
                        "Install scipy or use sweep_select='effect_size'."
                    )
                best_layer = int(np.nanargmin(masked))
            self._warn(
                f"best_layer was selected by optimizing over {len(search_layers)} layers on the eval "
                "set; best_d/best_p at that layer are selection diagnostics (no multiple-comparison "
                "correction), not confirmatory statistics. Confirm on held-out data before reporting."
            )
        else:
            best_layer = pick_layer(num_layers, train_cfg.get("layer_idx"), train_cfg.get("layer_frac", 0.75))

        if eval_pos_scores_mat.size > 0:
            combined = np.concatenate(
                [eval_pos_scores_mat[:, best_layer], eval_neg_scores_mat[:, best_layer]], axis=0
            )
            proj_std = float(np.std(combined, ddof=1) + 1e-12)
        else:
            proj_std = float("nan")

        self.concept_vectors = concept_vectors
        self.best_layer = best_layer
        self.proj_std_best_layer = proj_std
        self.sweep_d = sweep_d
        self.sweep_p = sweep_p

        self._status("[Phase 4/4] Sweep + artifact saving")
        self.logger.log(
            "sweep_full",
            {
                "best_layer": best_layer,
                "best_d": float(sweep_d[best_layer]),
                "best_p": None if np.isnan(sweep_p[best_layer]) else float(sweep_p[best_layer]),
                "best_layer_search": train_cfg.get("best_layer_search"),
                "best_layer_search_intervals": search_intervals,
                "best_layer_search_layers": search_layers,
                "best_layer_search_default": search_default,
                "sweep_d": sweep_d.tolist(),
                "sweep_p": [None if np.isnan(x) else float(x) for x in sweep_p],
                "eval_pos_mean": eval_pos_mean.tolist(),
                "eval_neg_mean": eval_neg_mean.tolist(),
                "eval_pos_std": eval_pos_std.tolist(),
                "eval_neg_std": eval_neg_std.tolist(),
                "proj_std_best_layer": proj_std,
            },
        )

        plots_dir = os.path.join(self.run_dir, "plots")
        ensure_dir(plots_dir)
        if plot_cfg.get("enabled", True) and eval_pos_scores_mat.size > 0:
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

        tensors = {
            "best_layer": np.array([best_layer], dtype=np.int32),
            "best_v": concept_vectors[best_layer].astype(np.float32),
            "concept_vectors": concept_vectors.astype(np.float32),
            "sweep_d": sweep_d.astype(np.float32),
            "sweep_p": sweep_p.astype(np.float32),
            "eval_pos_mean": eval_pos_mean.astype(np.float32),
            "eval_neg_mean": eval_neg_mean.astype(np.float32),
            "eval_pos_std": eval_pos_std.astype(np.float32),
            "eval_neg_std": eval_neg_std.astype(np.float32),
            "eval_pos_scores_mat": eval_pos_scores_mat.astype(np.float32),
            "eval_neg_scores_mat": eval_neg_scores_mat.astype(np.float32),
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
            "best_d": float(sweep_d[best_layer]),
            "best_p": None if np.isnan(sweep_p[best_layer]) else float(sweep_p[best_layer]),
            "best_layer_selection_note": (
                "best_d/best_p are post-selection values (best layer chosen by optimizing over the "
                "layer sweep on the same eval data); confirm on held-out data before reporting."
                if (train_cfg["do_layer_sweep"] and did_eval)
                else None
            ),
            "best_layer_search": train_cfg.get("best_layer_search"),
            "best_layer_search_intervals": search_intervals,
            "best_layer_search_layers": search_layers,
            "best_layer_search_default": search_default,
            "proj_std_best_layer": proj_std,
            "num_layers": num_layers,
            "n_train": n_train,
            "n_train_pos": n_train_pos,
            "n_train_neg": n_train_neg,
            "train_prompt_mode": train_prompt_mode,
            "n_eval": n_eval_pos + n_eval_neg,
            "n_eval_pos": n_eval_pos,
            "n_eval_neg": n_eval_neg,
            "eval_mode_used": eval_mode if did_eval else None,
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
    ) -> List[Dict[str, Any]]:
        self._require_trained()
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = self.config["prompts"]["neutral_system"] if system_prompt is None else system_prompt
        save_html = self.config["plots"].get("heatmap_html", True) if save_html is None else save_html
        save_token_scores_npz = self.config["output"].get("save_token_scores_npz", True)
        base_random_seed = _resolve_random_control_seed(
            random_vector_seed,
            self.config.get("random", {}).get("seed", 0),
        )
        scoring_seed: Optional[int] = None
        scoring_vectors = self.concept_vectors
        if random_scoring_vector:
            scoring_seed = _mix_random_control_seed(base_random_seed, 101)
            scoring_vectors = _random_vectors_same_norm(self.concept_vectors, seed=scoring_seed)
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
            if _has_chat_template(tokenizer):
                messages, used_prompt_system = _normalize_messages(
                    text,
                    sys_prompt,
                    warn=self._warn,
                    warn_prefix=f"score_texts.texts[{i}]",
                )
                input_ids = apply_chat(tokenizer, messages, add_generation_prompt=False)
            else:
                input_ids = _encode_for_read(
                    tokenizer,
                    text,
                    sys_prompt,
                    warn=self._warn,
                    warn_prefix=f"score_texts.texts[{i}]",
                )
                used_prompt_system = False
            hs_layers = forward_hidden_states_all_layers(model, input_ids)
            scores_per_layer, scores_agg = token_scores_from_hs_layers(
                hs_layers, scoring_vectors, layers, aggregate=aggregate,
                reference_layer=reference_layer,
            )
            token_ids = input_ids[0].detach().cpu().numpy().astype(np.int32)
            tokens = _decode_tokens(tokenizer, token_ids)

            slug = _prompt_slug_any(text)
            base = f"text_{i:03d}_{slug}"
            if save_token_scores_npz:
                npz_path = os.path.join(out_dir, f"{base}.npz")
                np.savez_compressed(
                    npz_path,
                    token_ids=token_ids,
                    scores_per_layer=scores_per_layer,
                    scores_agg=scores_agg,
                    layer_indices=np.array(layers, dtype=np.int32),
                )
            else:
                npz_path = None

            if save_html:
                html_path = os.path.join(out_dir, f"{base}.html")
                render_token_heatmap(tokens, scores_agg.tolist(), html_path, title=slug)
                batch_entries.append((f"text {i:03d}: {slug}", tokens, scores_agg.tolist()))
            else:
                html_path = None

            rec = {
                "text": text,
                "system_prompt": sys_prompt,
                "system_prompt_overridden": bool(used_prompt_system),
                "layers": layers,
                "npz_path": npz_path,
                "html_path": html_path,
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

        alphas/alpha_unit: with alpha_unit="sigma", alpha is multiplied by
        proj_std_best_layer — the std of the *sequence-pooled* eval projections at the
        best layer — and the resulting raw offset is added to the residual stream at
        every token position of each steered layer (divided across layers when
        steer_distribute=True, and compounding through subsequent layers). It is a
        convenient scale knob, not literally "n standard deviations of behavior".

        seed: seeds python/numpy/torch RNGs before generation for reproducible
        sampling. Note: GPU kernels are not bit-deterministic across
        hardware/library versions, so reproducibility is best-effort.
        """
        self._require_trained()
        if seed is not None:
            set_seed(int(seed))
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = self.config["prompts"]["neutral_system"] if system_prompt is None else system_prompt
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
        base_random_seed = _resolve_random_control_seed(
            random_vector_seed,
            self.config.get("random", {}).get("seed", 0),
        )
        scoring_seed: Optional[int] = None
        steering_seed: Optional[int] = None
        scoring_vectors = self.concept_vectors
        steering_vectors = self.concept_vectors
        if random_scoring_vector:
            scoring_seed = _mix_random_control_seed(base_random_seed, 201)
            scoring_vectors = _random_vectors_same_norm(self.concept_vectors, seed=scoring_seed)
            self._warn(
                f"score_prompts using random scoring vector control (same-norm per layer, seed={scoring_seed})."
            )
        if random_steering_vector:
            steering_seed = _mix_random_control_seed(base_random_seed, 202)
            steering_vectors = _random_vectors_same_norm(self.concept_vectors, seed=steering_seed)
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

        if steer_layers is None:
            steer_layers = self.config["steering"]["steer_layers"]
        if steer_window_radius is None:
            steer_window_radius = self.config["steering"]["steer_window_radius"]
        if steer_distribute is None:
            steer_distribute = self.config["steering"]["steer_distribute"]

        if steer_layers == "probe":
            steer_layer_list = [int(self.best_layer)]
        elif steer_layers == "window":
            lo = max(0, int(self.best_layer) - int(steer_window_radius))
            hi = min(self.model_bundle.num_layers - 1, int(self.best_layer) + int(steer_window_radius))
            steer_layer_list = list(range(lo, hi + 1))
        elif isinstance(steer_layers, list):
            steer_layer_list = list(map(int, steer_layers))
        else:
            steer_layer_list = list(map(int, steer_layers))

        run_idx = 0
        progress_every = 1
        for i, prompt in enumerate(prompts):
            prompt_slug = _prompt_slug_any(prompt)
            for j, alpha in enumerate(alphas):
                alpha_val = float(alpha)
                if alpha_unit == "sigma":
                    if self.proj_std_best_layer is None or np.isnan(self.proj_std_best_layer):
                        raise ValueError("proj_std_best_layer is not available for sigma scaling.")
                    alpha_val = alpha_val * float(self.proj_std_best_layer)

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
                        model, steer_layer_list, steering_vectors, alpha_val, distribute=steer_distribute
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
                completion_ids = token_ids[prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completion_scores = scores_agg[prompt_len:]
                score_mean = float(np.mean(completion_scores)) if completion_scores.size else float("nan")

                alpha_label = _alpha_label(float(alpha), alpha_unit)
                base = f"prompt_{i:03d}_{prompt_slug}_{alpha_label}"
                if save_token_scores_npz:
                    npz_path = os.path.join(out_dir, f"{base}.npz")
                    npz_payload: Dict[str, Any] = dict(
                        token_ids=token_ids,
                        prompt_len=np.array([prompt_len], dtype=np.int32),
                        scores_per_layer=scores_per_layer,
                        scores_agg=scores_agg,
                        layer_indices=np.array(layers, dtype=np.int32),
                    )
                    npz_payload.update(generation_logits_payload)
                    np.savez_compressed(npz_path, **npz_payload)
                else:
                    npz_path = None

                if save_html:
                    html_path = os.path.join(out_dir, f"{base}.html")
                    title = f"prompt {i:03d} | {prompt_slug} | {alpha_label}"
                    render_token_heatmap(tokens, scores_agg.tolist(), html_path, title=title)
                    batch_entries.append((title, tokens, scores_agg.tolist()))
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
                    "alpha": float(alpha),
                    "alpha_unit": alpha_unit,
                    "alpha_value": float(alpha_val),
                    "sampling_seed": seed,
                    "steer_layers": steer_layer_list,
                    "completion": completion_text,
                    "prompt_len": prompt_len,
                    "layers": layers,
                    "npz_path": npz_path,
                    "html_path": html_path,
                    "segments_path": segments_path,
                    "score_mean": score_mean,
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
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.loads(f.read())
                obj.proj_std_best_layer = metrics.get("proj_std_best_layer")
        return obj


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

    def get_probe(self, name: Optional[str] = None, project_directory: Optional[str] = None) -> "ConceptProbe":
        """Load a previously trained probe without retraining.

        If this workspace was created with project_directory=..., that directory is used by default.
        """
        run_dir = os.path.abspath(project_directory or self.project_directory or "")
        if not run_dir:
            raise ValueError("No project_directory provided. Create ProbeWorkspace(project_directory=...) or pass it to get_probe().")

        probe = ConceptProbe.load(run_dir, model_bundle=self.model_bundle)
        # Use the workspace's config + console so scoring behavior matches workspace.config.
        if probe.config != self.config:
            self.console.warn(
                "Workspace config differs from the saved run config; scoring will use the "
                "workspace config (settings like layer_selection or score_aggregate may not "
                "match what was used at training time)."
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
    ) -> ConceptProbe:
        cfg = deep_merge(self.config, config_overrides or {})
        if concept is None:
            concept = ConceptSpec.from_config(cfg, concept_overrides)
        elif concept_overrides:
            concept = ConceptSpec.from_config({"concept": concept.to_dict()}, concept_overrides)

        run_tag = run_name or now_tag()
        concept_dir = os.path.join(self.root_dir, safe_slug(concept.name), run_tag)
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
