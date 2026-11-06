"""Loading: the dtype chosen for "auto", quantization settings, the model identity, and
workspaces built around a model that is already loaded."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers
from transformers import BitsAndBytesConfig

from concept_probe import ConceptProbe, ModelBundle, modeling, read_texts
from concept_probe.modeling import (
    identity_mismatches,
    quantization_identity,
    resolve_dtype,
    resolve_quantization,
)
from fakes import CharChatTok
from tiny_model import TinyCausalLM


# ----------------------------------------------------------------------------------------- dtype
def _gpu(monkeypatch, capability, name="Fake GPU"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: name)
    # what torch says on Turing GPUs, which only emulate bfloat16: it must not decide
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)


@pytest.mark.parametrize("capability, model_type, expected", [
    ((8, 6), "llama", torch.bfloat16),
    ((8, 0), "gemma3_text", torch.bfloat16),
    ((7, 5), "llama", torch.float16),
    ((7, 5), "qwen2", torch.float16),
    ((7, 5), "gemma3_text", torch.float32),
    ((7, 5), "gemma3", torch.float32),
])
def test_auto_dtype_follows_the_gpu_and_the_model_family(monkeypatch, capability, model_type, expected):
    _gpu(monkeypatch, capability)
    dtype, choice = resolve_dtype("auto", model_type, "auto")
    assert dtype is expected
    assert choice.setting == "dtype" and choice.requested == "auto"
    assert f"{capability[0]}.{capability[1]}" in choice.reason


def test_auto_dtype_without_a_gpu_is_float32(monkeypatch):
    _gpu(monkeypatch, None)
    assert resolve_dtype("auto", "llama")[0] is torch.float32
    _gpu(monkeypatch, (8, 6))
    assert resolve_dtype("auto", "llama", device_map="cpu")[0] is torch.float32


def test_an_explicit_dtype_is_not_a_choice():
    assert resolve_dtype("float16") == (torch.float16, None)


def test_load_resolves_auto_from_the_model_config_and_records_it(monkeypatch):
    _gpu(monkeypatch, (7, 5))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    import huggingface_hub.constants
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(transformers, "AutoConfig",
                        SimpleNamespace(from_pretrained=lambda src, **kw: SimpleNamespace(model_type="gemma3_text")))
    loaded = {}
    monkeypatch.setattr(modeling, "AutoTokenizer", SimpleNamespace(
        from_pretrained=lambda src, **kw: SimpleNamespace(pad_token_id=0, eos_token="e", chat_template=None)))

    def model_from_pretrained(src, **kw):
        loaded.update(kw)
        return SimpleNamespace(config=SimpleNamespace(model_type="gemma3_text"), eval=lambda: None,
                               dtype=kw.get("dtype"))

    monkeypatch.setattr(modeling, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=model_from_pretrained))
    bundle = ModelBundle.load("google/gemma-3-270m-it", console=False)
    assert loaded["dtype"] is torch.float32
    assert bundle.identity["dtype"] == "float32"
    assert [c.chosen for c in bundle.choices if c.setting == "dtype"] == ["float32"]


def test_the_default_dtype_is_auto_and_none_means_auto():
    import inspect

    assert inspect.signature(ModelBundle.load).parameters["dtype"].default == "auto"
    assert resolve_dtype(None)[1] is not None and resolve_dtype(None)[1].requested == "auto"


# ---------------------------------------------------------------------------------- quantization
def test_named_levels_build_their_configs():
    pytest.importorskip("bitsandbytes")  # an optional extra: concept-probe[quantization]
    cfg, _ = resolve_quantization("nf4", torch.float16)
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4" and cfg.bnb_4bit_compute_dtype == torch.float16
    cfg, _ = resolve_quantization("int8", torch.float16)
    assert cfg.load_in_8bit
    assert resolve_quantization(None, torch.float16) == (None, [])


def test_a_level_whose_package_is_missing_says_what_to_install():
    with pytest.raises(ImportError, match="needs no_such_package; install it"):
        modeling._require("no_such_package", "quantization='x'")


def test_hqq_levels_patch_the_quantizer_once_and_say_so():
    pytest.importorskip("hqq")
    from transformers.quantizers.quantizer_hqq import HqqHfQuantizer

    cfg, choices = resolve_quantization("hqq2", torch.float16)
    assert quantization_identity(cfg) == "hqq2"
    assert getattr(HqqHfQuantizer, "_concept_probe_dtype_patch", False)
    _, again = resolve_quantization("hqq4", torch.float16)
    assert len(choices) <= 1 and again == []


def test_a_dict_or_object_config_is_passed_through():
    cfg, _ = resolve_quantization({"quant_method": "bitsandbytes", "load_in_4bit": True,
                                                    "bnb_4bit_quant_type": "nf4",
                                                    "bnb_4bit_use_double_quant": True}, torch.float16)
    assert cfg.bnb_4bit_use_double_quant
    assert isinstance(quantization_identity(cfg), dict)  # no name for double quantization
    own = BitsAndBytesConfig(load_in_8bit=True)
    assert resolve_quantization(own, torch.float16)[0] is own


def test_bad_quantization_settings_raise():
    with pytest.raises(ValueError, match="one of"):
        resolve_quantization("q4", torch.float16)
    with pytest.raises(ValueError, match="quant_method"):
        resolve_quantization({"bits": 4}, torch.float16)


def test_quantization_identities_name_the_levels():
    assert quantization_identity(BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")) == "nf4"
    assert quantization_identity(BitsAndBytesConfig(load_in_8bit=True)) == "int8"
    assert quantization_identity(None) is None


# -------------------------------------------------------------------------------------- identity
def test_identity_mismatches_compare_what_both_sides_know():
    old = {"model_id": "m", "dtype": "bfloat16", "quantization": "nf4"}
    new = dict(old, dtype="float16")
    assert identity_mismatches(old, new) == ["dtype: 'bfloat16' vs 'float16'"]
    assert identity_mismatches(dict(old, dtype=None), new) == []  # a dtype one side does not know
    assert identity_mismatches({"model_id": "m"}, new) == []  # only the model id given


def test_identity_reads_the_repo_id_out_of_a_cached_snapshot_path():
    model = TinyCausalLM()
    model.config = SimpleNamespace(_name_or_path="/h/hub/models--org--name/snapshots/abc123", _commit_hash="abc123")
    ident = modeling.identity_of(model, CharChatTok())
    assert ident["model_id"] == "org/name" and ident["revision"] == "abc123" and ident["dtype"] == "float64"
    # A model given by its snapshot's path is the same model as the repo.
    path = "C:\\Users\\u\\.cache\\huggingface\\hub\\models--org--name\\snapshots\\abc123"
    assert identity_mismatches(dict(ident, model_id=path), ident) == []
    assert identity_mismatches(dict(ident, model_id="org/other"), ident) == ["model_id: 'org/other' vs 'org/name'"]


# -------------------------------------------------------------------------------- loaded models
def test_a_loaded_model_is_wrapped_and_its_tokenizer_set_up_and_announced():
    tok = CharChatTok()
    tok.pad_token_id = None
    tok.eos_token = "|"
    bundle = ModelBundle.from_loaded(TinyCausalLM(), tok, model_id="tiny-test", console=False)
    assert bundle.model_id == "tiny-test" and bundle.num_layers == 4 and bundle.identity["dtype"] == "float64"
    assert "tokenizer pad token" in [c.setting for c in bundle.choices]


def test_a_fitted_probe_records_its_model_identity_and_refuses_another(tmp_path):
    model, tok = TinyCausalLM(), CharChatTok()
    P, N = read_texts(model, tok, ["pp x", "p y"], cache=False), read_texts(model, tok, ["nn x", "n y"], cache=False)
    probe = ConceptProbe.fit(P, N, layer=1, save=True, output_root=str(tmp_path), run_name="r")
    assert probe.identity["dtype"] == "float64"
    assert ConceptProbe.load(probe.out_dir).identity == probe.identity
    with pytest.raises(ValueError, match="dtype: 'float64' vs 'float32'"):
        ConceptProbe.load(probe.out_dir, TinyCausalLM().float(), tok)
    other = read_texts(TinyCausalLM().float(), tok, ["pp x", "p y"], cache=False)
    with pytest.raises(ValueError, match="another model"):
        ConceptProbe.fit(other, other, layer=1, model=model, tokenizer=tok, save=False)
    assert ConceptProbe.fit(other, other, layer=1, model=model, tokenizer=tok, allow_model_mismatch=True,
                            save=False).identity["dtype"] == "float64"


def test_a_probe_given_only_its_model_id_attaches_to_that_model():
    pos, neg = np.random.default_rng(0).normal(size=(2, 6, 3, 8))
    probe = ConceptProbe.fit(pos + 1.0, neg, layer=1, identity="tiny-test", save=False)
    probe.attach(ModelBundle.from_loaded(TinyCausalLM(), CharChatTok(), model_id="tiny-test"))  # only its id was known
    assert probe.identity["model_id"] == "tiny-test" and probe.identity["dtype"] == "float64"
    other = ConceptProbe.fit(pos + 1.0, neg, layer=1, identity="another-model", save=False)
    with pytest.raises(ValueError, match="model_id: 'another-model' vs 'tiny-test'"):
        other.attach(ModelBundle.from_loaded(TinyCausalLM(), CharChatTok(), model_id="tiny-test"))


def test_float32_on_a_gpu_leaves_out_the_memory_efficient_attention_kernel(monkeypatch):
    from torch.nn import attention

    gpu32 = SimpleNamespace(dtype=torch.float32, device=torch.device("cuda", 0),
                            config=SimpleNamespace(_attn_implementation="sdpa"))
    used = []
    monkeypatch.setattr(attention, "sdpa_kernel", lambda backends: (used.append(list(backends)), __import__("contextlib").nullcontext())[1])
    with modeling.safe_attention(gpu32):
        pass
    assert used and attention.SDPBackend.EFFICIENT_ATTENTION not in used[0]
    assert modeling.attention_choice(gpu32).setting == "attention kernel"
    for other in (SimpleNamespace(**{**vars(gpu32), "dtype": torch.float16}),
                  SimpleNamespace(**{**vars(gpu32), "device": torch.device("cpu")}), TinyCausalLM()):
        assert modeling.attention_choice(other) is None
    used.clear()
    with modeling.safe_attention(TinyCausalLM()):
        pass
    assert not used  # nothing changes elsewhere
