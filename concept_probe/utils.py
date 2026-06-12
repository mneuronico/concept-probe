import json
import math
import os
import random
import re
import time
from typing import Any, Dict

import numpy as np
import torch


def now_tag() -> str:
    # Microsecond suffix: second-resolution tags collide when runs/batches are created
    # in a loop, silently mixing artifacts in the same directory.
    t = time.time()
    base = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
    return f"{base}_{int((t % 1) * 1e6):06d}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_sanitize(obj: Any) -> Any:
    """Map NaN/Inf to None (and numpy scalars to Python) so emitted files are valid strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
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


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_sanitize(obj), ensure_ascii=True, allow_nan=False) + "\n")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def safe_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return slug.strip("._-") or "concept"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonl_to_pretty(jsonl_path: str, out_path: str) -> None:
    events = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    json_dump(out_path, {"events": events})


def torch_dtype_from_str(value: str) -> torch.dtype:
    v = (value or "").lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {value}")
