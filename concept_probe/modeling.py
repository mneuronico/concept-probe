import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import torch_dtype_from_str


def attention_mask_from_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


def apply_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )


@dataclass
class ModelBundle:
    model_id: str
    tokenizer: Any
    model: Any

    @classmethod
    def load(cls, model_cfg: Dict[str, Any]) -> "ModelBundle":
        model_id = model_cfg["model_id"]
        hf_token = os.environ.get(model_cfg.get("hf_token_env", "HF_TOKEN"), None)
        dtype = torch_dtype_from_str(model_cfg.get("dtype", "bfloat16"))
        requested_4bit = bool(model_cfg.get("use_4bit", False))
        use_4bit = requested_4bit

        if requested_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception:
                warnings.warn(
                    "model.use_4bit is true but bitsandbytes is unavailable; falling back to non-4bit load."
                )
                use_4bit = False

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant = BitsAndBytesConfig(load_in_4bit=True) if use_4bit else None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                device_map=model_cfg.get("device_map", "auto"),
                torch_dtype=dtype,
                quantization_config=quant if use_4bit else None,
            )
        except Exception as exc:
            if requested_4bit and use_4bit:
                warnings.warn(f"4-bit model load failed ({exc}); retrying without 4-bit quantization.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=hf_token,
                    device_map=model_cfg.get("device_map", "auto"),
                    torch_dtype=dtype,
                    quantization_config=None,
                )
            else:
                raise
        model.eval()
        return cls(model_id=model_id, tokenizer=tokenizer, model=model)

    @property
    def num_layers(self) -> int:
        return len(get_transformer_layers(self.model))


def _resolve_attr_path(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def get_transformer_layers(model: Any) -> Iterable[Any]:
    """
    Resolve the transformer block list across model families/wrappers.
    """
    candidate_paths = [
        # Llama/Mistral/Gemma text-only
        "model.layers",
        # Gemma 3 multimodal wrappers
        "model.language_model.layers",
        # Alternative wrappers seen in HF stacks
        "language_model.layers",
        "model.model.layers",
        "model.model.language_model.layers",
        "base_model.model.layers",
        "base_model.model.language_model.layers",
        # GPT-like fallbacks
        "transformer.h",
    ]
    for path in candidate_paths:
        layers = _resolve_attr_path(model, path)
        if layers is None:
            continue
        try:
            len(layers)
            return layers
        except TypeError:
            continue
    tried = ", ".join(candidate_paths)
    raise AttributeError(f"Could not resolve transformer layers for {type(model).__name__}; tried: {tried}")
