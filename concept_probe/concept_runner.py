import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import ConceptSpec
from .probe import ConceptProbe, ProbeWorkspace
from .utils import deep_merge


def _load_concept_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Concept JSON must be an object: {path}")
    if "concept" not in data:
        raise ValueError(f"Concept JSON missing 'concept' section: {path}")
    if "prompts" not in data:
        raise ValueError(f"Concept JSON missing 'prompts' section: {path}")
    return data


def train_concept_from_json(
    json_path: str,
    alphas: Optional[Sequence[float]] = None,
    *,
    model_id: Optional[str] = None,
    root_dir: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ConceptProbe:
    """Train a concept probe from a JSON spec and score eval prompts."""
    data = _load_concept_json(json_path)
    concept = ConceptSpec.from_config(data)

    prompt_overrides = {"prompts": dict(data.get("prompts", {}))}
    json_cfg_overrides = data.get("config_overrides", {})
    if json_cfg_overrides is None:
        json_cfg_overrides = {}
    if not isinstance(json_cfg_overrides, dict):
        raise ValueError(f"'config_overrides' must be an object in concept JSON: {json_path}")
    overrides = deep_merge(prompt_overrides, json_cfg_overrides)
    overrides = deep_merge(overrides, config_overrides or {})

    json_model_id = data.get("model_id")
    resolved_model_id = model_id
    if resolved_model_id is None and isinstance(json_model_id, str) and json_model_id.strip():
        resolved_model_id = json_model_id.strip()

    workspace = ProbeWorkspace(
        model_id=resolved_model_id,
        root_dir=root_dir,
        config_overrides=overrides,
    )
    probe = workspace.train_concept(concept)

    eval_prompts = workspace.config["prompts"]["eval_questions"]
    alpha_values = list(alphas) if alphas is not None else [0.0]
    probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=alpha_values,
        alpha_unit="sigma",
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
    )
    return probe
