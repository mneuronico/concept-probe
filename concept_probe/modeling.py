import datetime
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import torch_dtype_from_str


def attention_mask_from_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


# The date Llama 3.x chat templates write when no current date is available.
DEFAULT_CHAT_TEMPLATE_DATE = "2024-07-26"
_CHAT_TEMPLATE_DATE_ATTR = "concept_probe_chat_template_date"


def _chat_template_kwargs(tokenizer) -> Dict[str, Any]:
    """Extra variables for rendering ``tokenizer``'s chat template.

    Some templates (Llama 3.2 and 3.3, among others) write today's date into every prompt
    through transformers' ``strftime_now``, so the same input tokenizes differently on different
    days. We render them with a fixed date instead: ``model.chat_template_date`` (ISO
    ``YYYY-MM-DD``, default 2024-07-26), or the real date when that setting is null.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or "strftime_now" not in str(template):
        return {}
    date = getattr(tokenizer, _CHAT_TEMPLATE_DATE_ATTR, DEFAULT_CHAT_TEMPLATE_DATE)
    if date is None:
        return {}
    return {"strftime_now": datetime.date.fromisoformat(str(date)).strftime}


def apply_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> torch.Tensor:
    """Tokenize chat messages into a ``(1, seq_len)`` input_ids tensor.

    Depending on the transformers version, ``apply_chat_template`` returns either a
    bare tensor of token ids or a ``BatchEncoding``/dict (when it tokenizes into a
    full encoding). Normalize both cases to a 2D LongTensor of input ids. Templates
    that write the current date get a fixed one (see :func:`_chat_template_kwargs`).
    """
    out = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_tensors="pt",
        **_chat_template_kwargs(tokenizer),
    )
    if not isinstance(out, torch.Tensor):
        # BatchEncoding / dict-like result from newer transformers.
        out = out["input_ids"]
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out


def apply_tokenizer_overrides(tokenizer, model_cfg: Dict[str, Any]) -> None:
    """Apply the tokenizer settings of ``model_cfg``.

    ``model.chat_template`` (a Jinja string, default null) replaces the tokenizer's own chat
    template for every use of this bundle: training, eval, scoring, generation. Use it to
    drop parts a template injects unasked, such as the dated system block of Llama 3.x.

    ``model.chat_template_date`` (ISO ``YYYY-MM-DD``, default 2024-07-26) is the date written by
    templates that would otherwise write today's date; null writes the real current date.
    """
    chat_template = model_cfg.get("chat_template")
    if chat_template:
        tokenizer.chat_template = str(chat_template)
    date = model_cfg.get("chat_template_date", DEFAULT_CHAT_TEMPLATE_DATE)
    if date is not None:
        try:
            datetime.date.fromisoformat(str(date))
        except ValueError as exc:
            raise ValueError(f"model.chat_template_date must be an ISO date (YYYY-MM-DD) or null; got {date!r}") from exc
    setattr(tokenizer, _CHAT_TEMPLATE_DATE_ATTR, date)


def _transformers_version() -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])


def _dtype_kwargs(dtype: torch.dtype) -> Dict[str, Any]:
    """The dtype argument of ``from_pretrained``: ``dtype`` from transformers 4.56, which deprecates
    ``torch_dtype``; ``torch_dtype`` before."""
    return {"dtype": dtype} if _transformers_version() >= (4, 56) else {"torch_dtype": dtype}


_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _hub_offline() -> bool:
    """Whether ``HF_HUB_OFFLINE`` is set, now or when huggingface_hub was imported."""
    if os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in _TRUE_VALUES:
        return True
    try:
        from huggingface_hub import constants
    except Exception:
        return False
    return bool(getattr(constants, "HF_HUB_OFFLINE", False))


def resolve_model_source(model_id: str, token: Optional[str] = None) -> str:
    """Where to load ``model_id`` from.

    A local directory, or a repo id while the Hub is reachable, is returned as is. With
    ``HF_HUB_OFFLINE`` set, a repo id resolves to its snapshot in the local Hugging Face cache:
    loading by repo id fails offline in some transformers versions (4.57.3 asks the Hub whether
    the tokenizer comes from Mistral), even when every file is cached.
    """
    if os.path.isdir(model_id) or not _hub_offline():
        return model_id
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_id, local_files_only=True, token=token)
    except Exception as exc:
        raise OSError(
            f"HF_HUB_OFFLINE is set and {model_id!r} is not in the local Hugging Face cache; download it "
            "first, or unset HF_HUB_OFFLINE."
        ) from exc


def _warn_if_gemma3_float16(model: Any, dtype: torch.dtype, model_id: str) -> None:
    """Gemma 3 overflows in float16: its hidden states become inf/NaN (seen with 270M unquantized,
    and 4B and 12B in NF4 with float16 compute)."""
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "")
    if dtype == torch.float16 and model_type.startswith("gemma3"):
        warnings.warn(
            f"{model_id} is a Gemma 3 model loaded with float16 (compute) dtype, in which its activations "
            "overflow to inf/NaN. Use bfloat16 if the GPU supports it, or float32.",
            RuntimeWarning,
            stacklevel=3,
        )


@dataclass
class ModelBundle:
    model_id: str
    tokenizer: Any
    model: Any

    @classmethod
    def load(cls, model_cfg: Dict[str, Any]) -> "ModelBundle":
        """Load the tokenizer and model of ``model_cfg``.

        ``use_4bit`` loads NF4 4-bit weights with ``dtype`` as the compute dtype. It needs
        bitsandbytes, and it fails loudly rather than loading the full-precision model instead,
        since the two give different activations. With ``HF_HUB_OFFLINE`` set, a repo id is loaded
        from its cached snapshot (see :func:`resolve_model_source`).
        """
        model_id = model_cfg["model_id"]
        hf_token = os.environ.get(model_cfg.get("hf_token_env", "HF_TOKEN"), None)
        dtype = torch_dtype_from_str(model_cfg.get("dtype", "bfloat16"))
        use_4bit = bool(model_cfg.get("use_4bit", False))

        quant = None
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:
                raise ImportError(
                    "model.use_4bit is true but bitsandbytes is not available; install it or set use_4bit=false."
                ) from exc
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

        source = resolve_model_source(model_id, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(source, token=hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        apply_tokenizer_overrides(tokenizer, model_cfg)

        model = AutoModelForCausalLM.from_pretrained(
            source,
            token=hf_token,
            device_map=model_cfg.get("device_map", "auto"),
            quantization_config=quant,
            **_dtype_kwargs(dtype),
        )
        model.eval()
        _warn_if_gemma3_float16(model, dtype, model_id)
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
