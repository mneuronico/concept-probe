"""Loading (layer 0): the dtype chosen for "auto", quantization settings, the model identity, and
workspaces built around a model that is already loaded."""
import json
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import BitsAndBytesConfig

from concept_probe import ConceptSpec, ModelBundle, ProbeWorkspace, modeling
from concept_probe.modeling import (
    identity_from_config,
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
    assert choice.setting == "model.dtype" and choice.requested == "auto"
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
    bundle = ModelBundle.load({"model_id": "google/gemma-3-270m-it", "dtype": "auto"})
    assert loaded["dtype"] is torch.float32
    assert bundle.identity["dtype"] == "float32"
    assert [c.chosen for c in bundle.choices if c.setting == "model.dtype"] == ["float32"]


def test_the_default_dtype_is_auto():
    assert concept_probe_defaults()["model"]["dtype"] == "auto"


def concept_probe_defaults():
    from concept_probe import load_defaults

    return load_defaults()


# ---------------------------------------------------------------------------------- quantization
def test_named_levels_build_their_configs():
    cfg, _ = resolve_quantization({"quantization": "nf4"}, torch.float16)
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4" and cfg.bnb_4bit_compute_dtype == torch.float16
    cfg, _ = resolve_quantization({"quantization": "int8"}, torch.float16)
    assert cfg.load_in_8bit
    assert resolve_quantization({"quantization": None}, torch.float16) == (None, [])


def test_hqq_levels_patch_the_quantizer_once_and_say_so():
    pytest.importorskip("hqq")
    from transformers.quantizers.quantizer_hqq import HqqHfQuantizer

    cfg, choices = resolve_quantization({"quantization": "hqq2"}, torch.float16)
    assert quantization_identity(cfg) == "hqq2"
    assert getattr(HqqHfQuantizer, "_concept_probe_dtype_patch", False)
    _, again = resolve_quantization({"quantization": "hqq4"}, torch.float16)
    assert len(choices) <= 1 and again == []


def test_a_dict_or_object_config_is_passed_through():
    cfg, _ = resolve_quantization({"quantization": {"quant_method": "bitsandbytes", "load_in_4bit": True,
                                                    "bnb_4bit_quant_type": "nf4",
                                                    "bnb_4bit_use_double_quant": True}}, torch.float16)
    assert cfg.bnb_4bit_use_double_quant
    assert isinstance(quantization_identity(cfg), dict)  # no name for double quantization
    own = BitsAndBytesConfig(load_in_8bit=True)
    assert resolve_quantization({"quantization": own}, torch.float16)[0] is own


def test_use_4bit_is_an_alias_of_nf4_with_a_notice():
    with pytest.warns(DeprecationWarning, match="quantization = 'nf4'"):
        cfg, choices = resolve_quantization({"use_4bit": True}, torch.float16)
    assert quantization_identity(cfg) == "nf4" and choices[0].chosen == "nf4"
    with pytest.raises(ValueError, match="disagree"):
        resolve_quantization({"use_4bit": True, "quantization": "int8"}, torch.float16)
    with pytest.raises(ValueError, match="one of"):
        resolve_quantization({"quantization": "q4"}, torch.float16)
    with pytest.raises(ValueError, match="quant_method"):
        resolve_quantization({"quantization": {"bits": 4}}, torch.float16)


def test_quantization_identities_name_the_levels():
    assert quantization_identity(BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")) == "nf4"
    assert quantization_identity(BitsAndBytesConfig(load_in_8bit=True)) == "int8"
    assert quantization_identity(None) is None


# -------------------------------------------------------------------------------------- identity
def test_identity_from_an_old_config_and_mismatches():
    old = identity_from_config({"model_id": "m", "dtype": "bfloat16", "use_4bit": True})
    assert old["dtype"] == "bfloat16" and old["quantization"] == "nf4"
    new = dict(old, dtype="float16")
    assert identity_mismatches(old, new) == ["dtype: 'bfloat16' vs 'float16'"]
    unknown = identity_from_config({"model_id": "m", "dtype": "auto", "use_4bit": True})
    assert identity_mismatches(unknown, new) == []  # an "auto" config does not say which dtype was used


def test_identity_reads_the_repo_id_out_of_a_cached_snapshot_path():
    model = TinyCausalLM()
    model.config = SimpleNamespace(_name_or_path="/h/hub/models--org--name/snapshots/abc123", _commit_hash="abc123")
    ident = modeling.identity_of(model, CharChatTok())
    assert ident["model_id"] == "org/name" and ident["revision"] == "abc123" and ident["dtype"] == "float64"


# -------------------------------------------------------------------------------- loaded models
def _ws(tmp_path, **kw):
    return ProbeWorkspace(config_overrides={"output": {"root_dir": str(tmp_path), "console": False}}, **kw)


def test_a_workspace_takes_a_loaded_model_and_its_settings_replace_the_config(tmp_path):
    ws = _ws(tmp_path, model=TinyCausalLM(), tokenizer=CharChatTok(), model_id="tiny-test")
    assert ws.model_bundle.model_id == "tiny-test" and ws.model_bundle.num_layers == 4
    assert ws.config["model"]["model_id"] == "tiny-test" and ws.config["model"]["dtype"] == "float64"
    replaced = [c for c in ws.model_bundle.choices if c.setting == "model"]
    assert replaced and replaced[0].chosen == {"dtype": "float64"}  # model_id was passed, so the config had it


def test_a_workspace_takes_a_model_bundle(tmp_path):
    bundle = ModelBundle.from_loaded(TinyCausalLM(), CharChatTok(), model_id="tiny-test")
    assert _ws(tmp_path, model=bundle).model_bundle is bundle
    with pytest.raises(ValueError, match="not both"):
        _ws(tmp_path, model=bundle, tokenizer=CharChatTok())
    with pytest.raises(ValueError, match="tokenizer"):
        _ws(tmp_path, model=TinyCausalLM())


def test_a_trained_probe_records_its_model_identity_and_refuses_another(tmp_path):
    ws = _ws(tmp_path, model=TinyCausalLM(), tokenizer=CharChatTok(), model_id="tiny-test")
    concept = ConceptSpec(name="c", pos_label="p", neg_label="n", pos_system="", neg_system="",
                          train_pos_texts=["pp x", "p y"], train_neg_texts=["nn x", "n y"],
                          eval_pos_texts=["pxp", "ppz"], eval_neg_texts=["nxn", "nnz"])
    probe = ws.train_concept(concept, run_name="r",
                             config_overrides={"training": {"train_prompt_mode": "texts"},
                                               "evaluation": {"selection_split": None}})
    metrics = json.loads(open(f"{probe.run_dir}/metrics.json").read())
    assert metrics["model_identity"]["model_id"] == "tiny-test" and metrics["model_identity"]["dtype"] == "float64"
    assert metrics["read_batch_size"] == 16
    assert ws.get_probe(project_directory=probe.run_dir).identity["model_id"] == "tiny-test"
    other = _ws(tmp_path, model=TinyCausalLM().float(), tokenizer=CharChatTok(), model_id="tiny-test")
    with pytest.raises(ValueError, match="dtype: 'float64' vs 'float32'"):
        other.get_probe(project_directory=probe.run_dir)
