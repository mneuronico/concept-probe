import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PromptMessage = Dict[str, str]
PromptLike = Union[str, List[PromptMessage]]

from .utils import deep_merge, unknown_config_keys


DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.json"


def check_config_keys(override: Dict[str, Any], reference: Dict[str, Any], what: str = "config override") -> None:
    """Raise if ``override`` has keys that ``reference`` does not (a misspelled key is otherwise ignored)."""
    unknown = unknown_config_keys(override, reference)
    if unknown:
        raise ValueError(f"Unknown keys in {what}: {', '.join(unknown)}.")


def load_defaults(path: Optional[str] = None) -> Dict[str, Any]:
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
        packaged = json.load(f)
    if not path:
        return packaged
    with open(path, "r", encoding="utf-8") as f:
        custom = json.load(f)
    check_config_keys(custom, packaged, what=f"defaults file {path}")
    return custom


def resolve_config(
    overrides: Optional[Dict[str, Any]] = None,
    defaults_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Defaults merged with ``overrides``. Keys the defaults do not have raise."""
    config = load_defaults(defaults_path)
    if overrides:
        check_config_keys(overrides, load_defaults())
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
    train_pos_texts: List[PromptLike] = field(default_factory=list)
    train_neg_texts: List[PromptLike] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> "ConceptSpec":
        base = dict(config.get("concept", {}))
        if overrides:
            base = deep_merge(base, overrides)
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(base) - fields)
        if unknown:
            raise ValueError(f"Unknown concept keys: {', '.join(unknown)}; expected some of {sorted(fields)}.")
        return cls(
            name=str(base.get("name", "concept")),
            pos_label=str(base.get("pos_label", "positive")),
            neg_label=str(base.get("neg_label", "negative")),
            pos_system=str(base.get("pos_system", "")),
            neg_system=str(base.get("neg_system", "")),
            eval_pos_texts=list(base.get("eval_pos_texts", [])),
            eval_neg_texts=list(base.get("eval_neg_texts", [])),
            train_pos_texts=list(base.get("train_pos_texts", [])),
            train_neg_texts=list(base.get("train_neg_texts", [])),
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
            "train_pos_texts": list(self.train_pos_texts),
            "train_neg_texts": list(self.train_neg_texts),
        }
