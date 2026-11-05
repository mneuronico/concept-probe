import datetime
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .record import Choice
from .utils import torch_dtype_from_str


def attention_mask_from_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


# The date Llama 3.x chat templates write when no current date is available.
DEFAULT_CHAT_TEMPLATE_DATE = "2024-07-26"
_CHAT_TEMPLATE_DATE_ATTR = "concept_probe_chat_template_date"
_CHAT_TEMPLATE_OVERRIDE_ATTR = "concept_probe_chat_template_override"


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
        setattr(tokenizer, _CHAT_TEMPLATE_OVERRIDE_ATTR, str(chat_template))
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


# ------------------------------------------------------------------------------------------- dtype
def _dtype_name(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def _on_cpu(device_map: Any) -> bool:
    """Whether ``device_map`` keeps the whole model on the CPU (None does: no device map)."""
    if device_map is None or (isinstance(device_map, str) and device_map == "cpu"):
        return True
    if isinstance(device_map, dict):
        return bool(device_map) and all(str(v) in ("cpu", "disk") for v in device_map.values())
    return False


def resolve_dtype(requested: Any, model_type: str = "", device_map: Any = "auto") -> Tuple[torch.dtype, Optional[Choice]]:
    """The dtype to load a model in, and the choice made when ``requested`` is ``"auto"``.

    ``"auto"``: bfloat16 on GPUs that compute in it (compute capability 8.0 and up); otherwise
    float16, except float32 for Gemma 3, which overflows in float16; float32 without a CUDA GPU.
    The compute capability decides, not ``torch.cuda.is_bf16_supported()``, which also says yes on
    GPUs that only emulate bfloat16 (slowly), such as the Turing T4 and RTX 20xx.
    """
    if requested is None or str(requested) != "auto":
        return torch_dtype_from_str(requested if requested is not None else "bfloat16"), None
    gemma3 = str(model_type or "").startswith("gemma3")
    if not torch.cuda.is_available() or _on_cpu(device_map):
        return torch.float32, Choice("model.dtype", "auto", "float32", "no CUDA GPU in use: float32 on the CPU")
    caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    major, minor = min(caps)
    gpu = f"{torch.cuda.get_device_name(0)} (compute capability {major}.{minor})"
    if major >= 8:
        return torch.bfloat16, Choice("model.dtype", "auto", "bfloat16", f"{gpu} computes in bfloat16")
    if gemma3:
        return torch.float32, Choice("model.dtype", "auto", "float32",
                                     f"{gpu} has no native bfloat16, and Gemma 3 overflows in float16")
    return torch.float16, Choice("model.dtype", "auto", "float16", f"{gpu} has no native bfloat16")


# ------------------------------------------------------------------------------------ quantization
#: Quantization levels with a name. Any other transformers quantization config can be passed as an
#: object or as a dict with its ``quant_method``.
QUANTIZATION_LEVELS = ("nf4", "int8", "hqq4", "hqq3", "hqq2")


def _require(package: str, what: str) -> None:
    try:
        __import__(package)
    except Exception as exc:
        raise ImportError(f"{what} needs {package}; install it, or set model.quantization to null.") from exc


def _patch_hqq_compute_dtype() -> bool:
    """Make transformers' HQQ quantizer compute in the model's dtype. Returns True if it patched now.

    transformers 4.57.3 sets the HQQ compute dtype when the quantizer is built, before it knows the
    model's dtype, so it is always float32 and a float16 model fails on its first matmul. The dtype
    does reach ``validate_environment``; the patch takes it from there. Where the quantizer already
    gets it right, this sets the same value.
    """
    from transformers.quantizers.quantizer_hqq import HqqHfQuantizer

    if getattr(HqqHfQuantizer, "_concept_probe_dtype_patch", False):
        return False
    base = HqqHfQuantizer.validate_environment

    def validate_environment(self, *args, **kwargs):
        base(self, *args, **kwargs)
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        if dtype is not None:
            self.dtype = dtype

    HqqHfQuantizer.validate_environment = validate_environment
    HqqHfQuantizer._concept_probe_dtype_patch = True
    return True


def quantization_identity(config: Any) -> Any:
    """A comparable description of a quantization config: the level name when it is one of
    :data:`QUANTIZATION_LEVELS`, otherwise its settings as a dict; None for no quantization."""
    if config is None:
        return None
    if isinstance(config, str):
        return config
    d = config if isinstance(config, dict) else (config.to_dict() if hasattr(config, "to_dict") else vars(config))
    method = d.get("quant_method", "")
    method = str(getattr(method, "value", method)).lower()
    if method == "bitsandbytes":
        if d.get("load_in_8bit"):
            return "int8"
        if d.get("load_in_4bit") and d.get("bnb_4bit_quant_type") == "nf4" and not d.get("bnb_4bit_use_double_quant"):
            return "nf4"
    if method == "hqq":
        nbits = d.get("nbits", (d.get("quant_config") or {}).get("weight_quant_params", {}).get("nbits"))
        group = d.get("group_size", (d.get("quant_config") or {}).get("weight_quant_params", {}).get("group_size"))
        if nbits in (2, 3, 4) and group == (16 if nbits == 2 else 64):
            return f"hqq{nbits}"
    keep = {k: v for k, v in d.items() if not k.startswith("_")}
    return json.loads(json.dumps(keep, sort_keys=True, default=str))


def resolve_quantization(model_cfg: Dict[str, Any], dtype: torch.dtype) -> Tuple[Any, List[Choice]]:
    """The transformers quantization config for ``model.quantization``.

    ``model.quantization`` is null, a level of :data:`QUANTIZATION_LEVELS`, a transformers
    quantization config object, or a dict with its ``quant_method`` (e.g. bitsandbytes with double
    quantization). NF4 computes in the model's ``dtype``. A quantization that cannot load fails
    instead of loading the full-precision model, since the two give different activations.
    """
    choices: List[Choice] = []
    spec = model_cfg.get("quantization")
    if spec is None:
        return None, choices
    if isinstance(spec, str):
        if spec not in QUANTIZATION_LEVELS:
            raise ValueError(f"model.quantization must be null, one of {QUANTIZATION_LEVELS}, a dict with "
                             f"'quant_method', or a transformers quantization config; got {spec!r}")
        if spec in ("nf4", "int8"):
            _require("bitsandbytes", f"model.quantization={spec!r}")
            if spec == "int8":
                return BitsAndBytesConfig(load_in_8bit=True), choices
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=dtype), choices
        _require("hqq", f"model.quantization={spec!r}")
        from transformers import HqqConfig

        if _patch_hqq_compute_dtype():
            choices.append(Choice("transformers HQQ quantizer", "its own compute dtype", "the model's dtype",
                                  "transformers 4.57 computes HQQ layers in float32 whatever the model's dtype"))
        nbits = int(spec[3:])
        return HqqConfig(nbits=nbits, group_size=16 if nbits == 2 else 64), choices
    if isinstance(spec, dict):
        if "quant_method" not in spec:
            raise ValueError("model.quantization as a dict needs its 'quant_method' (as in transformers' configs).")
        from transformers.quantizers.auto import AutoQuantizationConfig

        return AutoQuantizationConfig.from_dict(dict(spec)), choices
    return spec, choices


# ------------------------------------------------------------------------------------------ identity
#: The identity settings two uses of a probe must share: they change the activations.
IDENTITY_KEYS = ("model_id", "dtype", "quantization", "chat_template", "chat_template_date")


def _repo_id_from_path(path: str) -> str:
    match = re.search(r"models--([^/\\]+)--([^/\\]+)", str(path))
    return f"{match.group(1)}/{match.group(2)}" if match else str(path)


def identity_of(model: Any, tokenizer: Any, model_id: Optional[str] = None) -> Dict[str, Any]:
    """What makes this model's activations what they are: the model (and its revision when known),
    the dtype, the quantization, the chat template override and its date."""
    config = getattr(model, "config", None)
    name = model_id or getattr(model, "_concept_probe_model_id", None) or getattr(config, "_name_or_path", None)
    dtype = getattr(model, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        try:
            dtype = next(model.parameters()).dtype
        except Exception:
            dtype = None
    return {
        "model_id": None if name is None else _repo_id_from_path(name),
        "revision": getattr(config, "_commit_hash", None),
        "dtype": _dtype_name(dtype),
        "quantization": quantization_identity(getattr(config, "quantization_config", None)),
        "chat_template": getattr(tokenizer, _CHAT_TEMPLATE_OVERRIDE_ATTR, None),
        "chat_template_date": getattr(tokenizer, _CHAT_TEMPLATE_DATE_ATTR, DEFAULT_CHAT_TEMPLATE_DATE),
    }


def _config_dtype_name(value: Any) -> Optional[str]:
    """A config's dtype as a dtype name; None for ``"auto"`` (unknown until a model is loaded)."""
    if value in (None, "auto"):
        return None
    try:
        return _dtype_name(torch_dtype_from_str(value))
    except ValueError:
        return str(value)


def upgrade_saved_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A run's saved config in today's terms: runs saved by earlier versions wrote
    ``model.use_4bit`` (now ``model.quantization = "nf4"``) and ``training.direction_normalization``
    (now ``training.estimator``). Returns a new dict; today's configs come back unchanged."""
    cfg = json.loads(json.dumps(cfg))
    model = cfg.get("model", {})
    if "use_4bit" in model:
        if model.pop("use_4bit") and model.get("quantization") is None:
            model["quantization"] = "nf4"
        model.setdefault("quantization", None)
    training = cfg.get("training", {})
    if "direction_normalization" in training:
        normalization = training.pop("direction_normalization") or "none"
        training.setdefault("estimator", {"kind": "mean_diff", "normalization": normalization})
    return cfg


def identity_from_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The identity a saved config describes (for runs saved before identities were recorded)."""
    quant = upgrade_saved_config({"model": dict(model_cfg)})["model"].get("quantization")
    return {
        "model_id": model_cfg.get("model_id"),
        "dtype": _config_dtype_name(model_cfg.get("dtype")),
        "quantization": quantization_identity(quant),
        "chat_template": model_cfg.get("chat_template") or None,
        "chat_template_date": model_cfg.get("chat_template_date", DEFAULT_CHAT_TEMPLATE_DATE),
    }


def identity_mismatches(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """The identity settings (:data:`IDENTITY_KEYS`) on which ``a`` and ``b`` differ, as text.
    A dtype that one side does not know (an old run saved with ``"auto"``) is not compared."""
    out = []
    for key in IDENTITY_KEYS:
        va, vb = a.get(key), b.get(key)
        if key == "dtype" and (va is None or vb is None):
            continue
        if va != vb:
            out.append(f"{key}: {va!r} vs {vb!r}")
    return out


# ------------------------------------------------------------------------------------------- bundle
@dataclass
class ModelBundle:
    model_id: str
    tokenizer: Any
    model: Any
    #: what changes the activations (see :func:`identity_of`); filled by ``load`` and ``from_loaded``
    identity: Dict[str, Any] = field(default_factory=dict)
    #: the automatic choices made while loading (see :mod:`concept_probe.record`)
    choices: List[Choice] = field(default_factory=list)

    @classmethod
    def load(cls, model_cfg: Dict[str, Any]) -> "ModelBundle":
        """Load the tokenizer and model of ``model_cfg`` (the config's ``model`` section).

        ``dtype`` may be ``"auto"`` (see :func:`resolve_dtype`); ``quantization`` is null, a level of
        :data:`QUANTIZATION_LEVELS`, a dict or a transformers config (see
        :func:`resolve_quantization`). With ``HF_HUB_OFFLINE`` set, a repo id is loaded from its
        cached snapshot (see :func:`resolve_model_source`). The choices made on the caller's behalf
        are in ``choices``; they are not printed here.
        """
        model_id = model_cfg["model_id"]
        hf_token = os.environ.get(model_cfg.get("hf_token_env", "HF_TOKEN"), None)
        device_map = model_cfg.get("device_map", "auto")
        choices: List[Choice] = []
        requested_dtype = model_cfg.get("dtype", "auto")
        source = resolve_model_source(model_id, token=hf_token)
        if source != model_id:
            choices.append(Choice("model source", model_id, source, "HF_HUB_OFFLINE is set: the cached snapshot"))
        model_type = ""
        if requested_dtype == "auto":
            from transformers import AutoConfig

            model_type = str(getattr(AutoConfig.from_pretrained(source, token=hf_token), "model_type", "") or "")
        dtype, dtype_choice = resolve_dtype(requested_dtype, model_type, device_map)
        if dtype_choice is not None:
            choices.append(dtype_choice)
        quant, quant_choices = resolve_quantization(model_cfg, dtype)
        choices.extend(quant_choices)

        tokenizer = AutoTokenizer.from_pretrained(source, token=hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        apply_tokenizer_overrides(tokenizer, model_cfg)

        model = AutoModelForCausalLM.from_pretrained(
            source,
            token=hf_token,
            device_map=device_map,
            quantization_config=quant,
            **_dtype_kwargs(dtype),
        )
        model.eval()
        _warn_if_gemma3_float16(model, dtype, model_id)
        try:
            setattr(model, "_concept_probe_model_id", model_id)
        except Exception:
            pass
        identity = identity_of(model, tokenizer, model_id)
        identity["dtype"] = _dtype_name(dtype)  # the compute dtype, also for quantized models
        return cls(model_id=model_id, tokenizer=tokenizer, model=model, identity=identity, choices=choices)

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, model_id: Optional[str] = None,
                    model_cfg: Optional[Dict[str, Any]] = None) -> "ModelBundle":
        """Wrap a model and tokenizer that are already loaded (entry 4 of LIBRARY_ISSUES).

        ``model_cfg`` applies its tokenizer settings (``chat_template``, ``chat_template_date``) to
        ``tokenizer`` in place, as :meth:`load` does. The identity is read from the model itself.
        """
        if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if model_cfg is not None:
            apply_tokenizer_overrides(tokenizer, model_cfg)
        elif getattr(tokenizer, _CHAT_TEMPLATE_DATE_ATTR, "missing") == "missing":
            apply_tokenizer_overrides(tokenizer, {})
        identity = identity_of(model, tokenizer, model_id)
        return cls(model_id=identity["model_id"] or "unknown", tokenizer=tokenizer, model=model, identity=identity)

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
