import json
import math
import os
import random
import re
import time
from typing import Any, Dict

import numpy as np
import torch


_LAST_TAG_US = [-1]


def now_tag() -> str:
    # Microsecond suffix: second-resolution tags collide when runs/batches are created
    # in a loop, silently mixing artifacts in the same directory. The clock can be
    # coarser than 1us (notably on Windows), so bump past the last tag to stay unique
    # and sortable within a process.
    us = int(time.time() * 1e6)
    if us <= _LAST_TAG_US[0]:
        us = _LAST_TAG_US[0] + 1
    _LAST_TAG_US[0] = us
    base = time.strftime("%Y%m%d_%H%M%S", time.localtime(us / 1e6))
    return f"{base}_{us % 1_000_000:06d}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_sanitize(obj: Any) -> Any:
    """Map NaN/Inf to None, and numpy scalars/arrays and sets to Python values and lists, so emitted
    files are valid strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_json_sanitize(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def json_dump(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)


def resolve_layer_index(value: Any, num_layers: int, what: str = "layer") -> int:
    """One layer index in ``[0, num_layers)``.

    An int is a layer index; negative ints count from the end, as in Python. ``{"frac": x}``
    with ``x`` in [0, 1] is a fraction of depth, ``int(x * (num_layers - 1))``. Anything out of
    range raises; bare floats raise too, since 1 and 1.0 would otherwise mean different layers.
    """
    if isinstance(value, dict):
        if set(value) != {"frac"}:
            raise ValueError(f"{what}: a fraction of depth is written {{'frac': x}}; got {value!r}")
        frac = value["frac"]
        if isinstance(frac, bool) or not isinstance(frac, (int, float, np.integer, np.floating)) or not 0.0 <= float(frac) <= 1.0:
            raise ValueError(f"{what}: 'frac' must be a number in [0, 1]; got {frac!r}")
        return int(float(frac) * (num_layers - 1))
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{what} must be an int layer index (negative counts from the end) or {{'frac': x}}; got {value!r}"
        )
    idx = int(value)
    resolved = idx + num_layers if idx < 0 else idx
    if not 0 <= resolved < num_layers:
        raise ValueError(f"{what} {idx} out of range for {num_layers} layers")
    return resolved


def safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return slug.strip("._-") or "concept"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype_from_str(value: str) -> torch.dtype:
    v = (value or "").lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {value}")
