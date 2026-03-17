import json
import os
import random
import re
import time
from typing import Any, Dict

import numpy as np
import torch


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, sort_keys=True)


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


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
