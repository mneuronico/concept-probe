"""Model-free tests of config validation, run directories, loading old runs, and model loading."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concept_probe import coherence, modeling
from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.console import ConsoleLogger
from concept_probe.modeling import ModelBundle
from concept_probe.probe import ConceptProbe, ProbeWorkspace


# ------------------------------------------------------------- config keys
@pytest.mark.parametrize("override", [
    {"trainig": {"layer_idx": 3}},
    {"training": {"sweep_selct": "p_value"}},
    {"evaluation": {"layer_selection": {"modes": "all"}}},
])
def test_misspelled_config_keys_raise(override):
    with pytest.raises(ValueError, match="Unknown keys"):
        resolve_config(override)


def test_a_removed_key_is_an_unknown_key():
    with pytest.raises(ValueError, match="steering.steer_system"):
        resolve_config({"steering": {"steer_system": "x"}})


def test_misspelled_concept_keys_raise():
    with pytest.raises(ValueError, match="Unknown concept keys"):
        ConceptSpec.from_config({"concept": {"name": "c", "eval_pos_text": ["x"]}})


def test_known_keys_are_accepted():
    cfg = resolve_config({"evaluation": {"gen_batch_size": 4, "selection_split": None}})
    assert cfg["evaluation"]["gen_batch_size"] == 4


# ------------------------------------------------------------- model loading
def test_requested_4bit_without_bitsandbytes_fails_instead_of_loading_full_precision(monkeypatch):
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # makes `import bitsandbytes` fail
    with pytest.raises(ImportError, match="bitsandbytes"):
        ModelBundle.load({"model_id": "unused", "quantization": "nf4", "dtype": "float32"})


def test_quantization_is_off_by_default_and_the_old_switch_is_gone():
    assert resolve_config()["model"]["quantization"] is None
    with pytest.raises(ValueError, match="model.use_4bit"):
        resolve_config({"model": {"use_4bit": True}})


def _fake_from_pretrained(monkeypatch, model_type="llama"):
    """Replace transformers' loaders with stand-ins that record where and how they were called."""
    calls = {}

    def tokenizer_from_pretrained(source, **kwargs):
        calls["tokenizer"] = (source, kwargs)
        return SimpleNamespace(pad_token_id=0, eos_token="</s>", chat_template=None)

    def model_from_pretrained(source, **kwargs):
        calls["model"] = (source, kwargs)
        return SimpleNamespace(config=SimpleNamespace(model_type=model_type), eval=lambda: None)

    monkeypatch.setattr(modeling, "AutoTokenizer", SimpleNamespace(from_pretrained=tokenizer_from_pretrained))
    monkeypatch.setattr(modeling, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=model_from_pretrained))
    return calls


def _online(monkeypatch):
    import huggingface_hub.constants

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


@pytest.mark.parametrize("version, key", [("4.57.3", "dtype"), ("4.56.0", "dtype"), ("4.55.4", "torch_dtype"),
                                          ("4.45.0", "torch_dtype")])
def test_dtype_argument_follows_the_transformers_version(monkeypatch, version, key):
    # transformers 4.56 renamed torch_dtype to dtype and deprecated the old name.
    _online(monkeypatch)
    calls = _fake_from_pretrained(monkeypatch)
    monkeypatch.setattr(modeling.transformers, "__version__", version)
    ModelBundle.load({"model_id": "org/model", "dtype": "float32"})
    kwargs = calls["model"][1]
    assert kwargs[key] is torch.float32
    assert ({"dtype", "torch_dtype"} - {key}).pop() not in kwargs


def test_online_loading_uses_the_repo_id(monkeypatch):
    _online(monkeypatch)
    calls = _fake_from_pretrained(monkeypatch)
    ModelBundle.load({"model_id": "org/model", "dtype": "float32"})
    assert calls["tokenizer"][0] == calls["model"][0] == "org/model"


def test_offline_loading_uses_the_cached_snapshot(monkeypatch):
    # Loading by repo id fails offline in transformers 4.57.3 even when everything is cached.
    import huggingface_hub

    calls = _fake_from_pretrained(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, **kw: asked.append((repo_id, kw)) or "/cache/org--model/snap")
    bundle = ModelBundle.load({"model_id": "org/model", "dtype": "float32"})
    assert asked[0][0] == "org/model" and asked[0][1]["local_files_only"] is True
    assert calls["tokenizer"][0] == calls["model"][0] == "/cache/org--model/snap"
    assert bundle.model_id == "org/model"  # the identity is still the repo id


def test_offline_loading_of_an_uncached_model_says_so(monkeypatch):
    import huggingface_hub

    _fake_from_pretrained(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def not_cached(repo_id, **kwargs):
        raise FileNotFoundError(repo_id)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    with pytest.raises(OSError, match="not in the local Hugging Face cache"):
        ModelBundle.load({"model_id": "org/model", "dtype": "float32"})


def test_a_local_directory_is_loaded_as_is_offline(monkeypatch, tmp_path):
    calls = _fake_from_pretrained(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ModelBundle.load({"model_id": str(tmp_path), "dtype": "float32"})
    assert calls["model"][0] == str(tmp_path)


@pytest.mark.parametrize("model_type, dtype, warns", [
    ("gemma3_text", "float16", True), ("gemma3", "float16", True),
    ("gemma3_text", "bfloat16", False), ("llama", "float16", False),
])
def test_gemma3_in_float16_warns(monkeypatch, recwarn, model_type, dtype, warns):
    _online(monkeypatch)
    _fake_from_pretrained(monkeypatch, model_type=model_type)
    ModelBundle.load({"model_id": "org/model", "dtype": dtype})
    overflow = [w for w in recwarn if "overflow" in str(w.message)]
    assert bool(overflow) == warns


# ------------------------------------------------------------- workspaces
def _workspace(tmp_path, model_cfg=None):
    ws = ProbeWorkspace.__new__(ProbeWorkspace)  # no model load
    ws.project_directory = None
    ws.config = resolve_config({"output": {"root_dir": str(tmp_path / "out")}, "model": model_cfg or {}})
    ws.root_dir = ws.config["output"]["root_dir"]
    ws.console = ConsoleLogger(False)
    ws.model_bundle = SimpleNamespace(model_id=ws.config["model"]["model_id"], num_layers=2)
    return ws


def _saved_run(tmp_path, model_cfg):
    run = tmp_path / "run"
    run.mkdir()
    cfg = resolve_config({"model": model_cfg})
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "concept.json").write_text(json.dumps({"name": "c", "pos_label": "p", "neg_label": "n",
                                                  "pos_system": "a", "neg_system": "b"}))
    tensors = dict(
        concept_vectors=np.eye(2, dtype=np.float32), best_layer=np.array([1]),
        sweep_d=np.zeros(2), sweep_p=np.zeros(2),
        eval_pos_mean=np.array([1.0, 2.0]), eval_neg_mean=np.array([-1.0, 0.0]),
        eval_pos_std=np.array([1.0, 2.0]), eval_neg_std=np.array([1.0, 2.0]),
        score_center=np.array([0.0, 1.0]), score_scale=np.array([1.0, 2.0]),
        token_center=np.array([0.0, 1.0]), token_scale=np.array([3.0, 4.0]), resid_norm=np.ones(2),
        reference_context=np.array([json.dumps({"kind": "generated", "mode": "assistant_all_mean"})]),
        transfer_references=np.array([json.dumps(None)]),
    )
    np.savez(run / "tensors.npz", **tensors)
    (run / "metrics.json").write_text(json.dumps({"eval_mode_used": "read", "n_eval_pos": 4, "n_eval_neg": 4}))
    return str(run)


def test_get_probe_refuses_a_different_model(tmp_path):
    run = _saved_run(tmp_path, {"model_id": "model-a"})
    ws = _workspace(tmp_path, {"model_id": "model-b"})
    with pytest.raises(ValueError, match="model_id"):
        ws.get_probe(project_directory=run)
    assert ws.get_probe(project_directory=run, allow_model_mismatch=True).layer == 1


def test_get_probe_loads_with_the_same_model(tmp_path):
    run = _saved_run(tmp_path, {"model_id": "model-a"})
    probe = _workspace(tmp_path, {"model_id": "model-a"}).get_probe(project_directory=run)
    assert probe.steer_sigma == 2.0  # a run saved before steering had its own unit: the eval SD at its layer


def test_load_restores_every_reference(tmp_path):
    run = _saved_run(tmp_path, {"model_id": "model-a"})
    probe = ConceptProbe.load(run)
    cal = probe.calibration_for({"kind": "generated"})
    np.testing.assert_allclose(cal.scale, [1.0, 2.0])
    np.testing.assert_allclose(cal.token_scale, [3.0, 4.0])
    assert probe.steer_sigma == 2.0 and probe.resid_norm is not None


def test_a_repeated_run_name_raises_unless_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(ConceptProbe, "train", lambda self: self)
    ws = _workspace(tmp_path)
    concept = ConceptSpec(name="c", pos_label="p", neg_label="n", pos_system="a", neg_system="b")
    probe = ws.train_concept(concept, run_name="same")
    open(os.path.join(probe.run_dir, "log.jsonl"), "w").write("old\n")
    with pytest.raises(FileExistsError):
        ws.train_concept(concept, run_name="same")
    probe = ws.train_concept(concept, run_name="same", overwrite=True)
    assert not os.path.exists(os.path.join(probe.run_dir, "log.jsonl"))


def test_train_concept_validates_its_overrides(tmp_path):
    concept = ConceptSpec(name="c", pos_label="p", neg_label="n", pos_system="a", neg_system="b")
    with pytest.raises(ValueError, match="Unknown keys"):
        _workspace(tmp_path).train_concept(concept, config_overrides={"training": {"layer_frc": 0.5}})


# ------------------------------------------------------------- coherence env
def test_no_env_file_is_read_without_env_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-dotenv\n")
    assert coherence._read_env_file(None) == {}


def test_an_env_file_is_read_without_touching_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CP_TEST_KEY", raising=False)
    path = tmp_path / "keys.env"
    path.write_text("# comment\nCP_TEST_KEY='secret'\n")
    assert coherence._read_env_file(str(path)) == {"CP_TEST_KEY": "secret"}
    assert "CP_TEST_KEY" not in os.environ
