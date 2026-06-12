import os
from typing import Any, Dict, Optional

from .utils import ensure_dir, jsonl_append


class JsonlLogger:
    def __init__(self, path: str, *, enabled: bool = True) -> None:
        parent = os.path.dirname(path) or "."
        ensure_dir(parent)
        self.path = path
        self.enabled = bool(enabled)

    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        rec = dict(payload) if payload else {}
        rec["event"] = event  # the event name must not be clobbered by a payload key
        jsonl_append(self.path, rec)
