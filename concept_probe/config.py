import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PromptMessage = Dict[str, str]
PromptLike = Union[str, List[PromptMessage]]

from .utils import deep_merge


DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.json"


def load_defaults(path: Optional[str] = None) -> Dict[str, Any]:
    defaults_path = Path(path) if path else DEFAULTS_PATH
    with open(defaults_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config(
    overrides: Optional[Dict[str, Any]] = None,
    defaults_path: Optional[str] = None,
) -> Dict[str, Any]:
    config = load_defaults(defaults_path)
    if overrides:
        config = deep_merge(config, overrides)
    return config


@dataclass
class ConceptSpec:
    name: str
    pos_label: str
    neg_label: str
    pos_system: str
    neg_system: str
    eval_pos_texts: List[PromptLike] = field(default_factory=list)
    eval_neg_texts: List[PromptLike] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> "ConceptSpec":
        base = dict(config.get("concept", {}))
        if overrides:
            base = deep_merge(base, overrides)
        return cls(
            name=str(base.get("name", "concept")),
            pos_label=str(base.get("pos_label", "positive")),
            neg_label=str(base.get("neg_label", "negative")),
            pos_system=str(base.get("pos_system", "")),
            neg_system=str(base.get("neg_system", "")),
            eval_pos_texts=list(base.get("eval_pos_texts", [])),
            eval_neg_texts=list(base.get("eval_neg_texts", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pos_label": self.pos_label,
            "neg_label": self.neg_label,
            "pos_system": self.pos_system,
            "neg_system": self.neg_system,
            "eval_pos_texts": list(self.eval_pos_texts),
            "eval_neg_texts": list(self.eval_neg_texts),
        }
