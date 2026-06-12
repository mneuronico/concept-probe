"""Tests for strict-JSON artifacts, now_tag uniqueness, and final-norm resolution."""
import json

import numpy as np
import pytest
import torch

from concept_probe.probe import _get_final_norm_module
from concept_probe.utils import deep_merge, json_dump, jsonl_append, now_tag


def _strict_loads(text):
    def _reject(name):
        raise ValueError(f"non-strict JSON constant: {name}")

    return json.loads(text, parse_constant=_reject)


def test_json_dump_is_strict_json(tmp_path):
    path = tmp_path / "m.json"
    json_dump(str(path), {"nan": float("nan"), "inf": float("inf"), "np": np.float32(1.5), "i": np.int64(3)})
    data = _strict_loads(path.read_text(encoding="utf-8"))
    assert data["nan"] is None
    assert data["inf"] is None
    assert data["np"] == 1.5
    assert data["i"] == 3


def test_jsonl_append_is_strict_json(tmp_path):
    path = tmp_path / "log.jsonl"
    jsonl_append(str(path), {"score": float("nan"), "vals": [1.0, float("-inf")]})
    line = path.read_text(encoding="utf-8").strip()
    data = _strict_loads(line)
    assert data["score"] is None
    assert data["vals"] == [1.0, None]


def test_now_tag_unique_and_sortable():
    tags = {now_tag() for _ in range(5)}
    assert len(tags) == 5
    assert all(len(t) == len("YYYYmmdd_HHMMSS_uuuuuu") for t in tags)


def test_deep_merge_nested():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"c": 9}})
    assert merged == {"a": {"b": 1, "c": 9}, "d": 3}
    assert base["a"]["c"] == 2  # no mutation


def _wrapper_with(path: str) -> torch.nn.Module:
    root = torch.nn.Module()
    node = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = torch.nn.Module()
        setattr(node, part, child)
        node = child
    setattr(node, parts[-1], torch.nn.LayerNorm(4))
    return root


@pytest.mark.parametrize(
    "path",
    ["model.norm", "transformer.ln_f", "model.language_model.norm", "ln_f"],
)
def test_final_norm_resolution_across_wrappers(path):
    model = _wrapper_with(path)
    assert isinstance(_get_final_norm_module(model), torch.nn.LayerNorm)


def test_final_norm_missing_raises():
    with pytest.raises(AttributeError):
        _get_final_norm_module(torch.nn.Module())
