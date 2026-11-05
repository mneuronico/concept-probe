"""Model-free tests for multi_probe_score_prompts: each probe is scored with its own vectors."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concept_probe import multi_probe as multi_mod
from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.logger import JsonlLogger
from concept_probe.multi_probe import multi_probe_score_prompts
from concept_probe.probe import ConceptProbe


class _Tok:
    chat_template = "fake"
    eos_token_id = pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None, **kwargs):
        return torch.tensor([[5, 6, 7]], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(list(ids))


class _Model:
    device = torch.device("cpu")

    def generate(self, input_ids, max_new_tokens=8, **kwargs):
        return torch.cat([input_ids, torch.tensor([[8, 9]], dtype=torch.long)], dim=1)


def _fake_forward(model, input_ids, **kwargs):
    n = int(input_ids.shape[-1])
    return [torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(1).repeat(1, 2)]  # one layer, dim 2


def _probe(tmp_path, name, vectors, bundle):
    probe = ConceptProbe(
        model_bundle=bundle,
        config=resolve_config({"output": {"console": False}, "plots": {"heatmap_html": False}}),
        concept=ConceptSpec(name=name, pos_label="p", neg_label="n", pos_system="a", neg_system="b"),
        run_dir=str(tmp_path / name),
        logger=JsonlLogger(str(tmp_path / name / "log.jsonl")),
    )
    probe.directions = np.asarray(vectors, dtype=np.float32)
    probe.steer_directions = probe.directions
    probe.layer = 0
    probe.steer_sigma = 1.0
    return probe


@pytest.fixture
def bundle():
    return SimpleNamespace(model_id="fake", tokenizer=_Tok(), model=_Model(), num_layers=1)


def test_each_probe_is_scored_with_its_own_vectors(tmp_path, monkeypatch, bundle):
    monkeypatch.setattr(multi_mod, "forward_hidden_states_all_layers", _fake_forward)
    a = _probe(tmp_path, "a", [[1.0, 0.0]], bundle)
    b = _probe(tmp_path, "b", [[-1.0, 0.0]], bundle)
    out = multi_probe_score_prompts([a, b], ["hi"], output_root=str(tmp_path / "multi"), save_html=False)
    rows = np.load(out["results"][0]["npz_path"])["scores_agg"]
    np.testing.assert_allclose(rows[0], [1, 2, 3, 4, 5])
    np.testing.assert_allclose(rows[1], -rows[0])
    assert out["results"][0]["score_mean_by_probe"] == {"a": 4.5, "b": -4.5}


@pytest.mark.parametrize("names", [("a", "a"), ("my concept", "my_concept")])
def test_probes_with_the_same_name_are_rejected(tmp_path, bundle, names):
    probes = [_probe(tmp_path, n, [[1.0, 0.0]], bundle) for n in names]
    with pytest.raises(ValueError, match="distinct concept names"):
        multi_probe_score_prompts(probes, ["hi"], output_root=str(tmp_path / "multi"))
    assert not (tmp_path / "multi").exists()


@pytest.mark.parametrize("kwargs", [{"alpha_unit": "Sigma"}, {"steer_layers": "12"}])
def test_bad_steering_options_are_rejected_before_writing(tmp_path, bundle, kwargs):
    probes = [_probe(tmp_path, "a", [[1.0, 0.0]], bundle)]
    with pytest.raises(ValueError):
        multi_probe_score_prompts(probes, ["hi"], alphas=[1.0], steer_probe=0,
                                  output_root=str(tmp_path / "multi"), **kwargs)
    assert not (tmp_path / "multi").exists()
