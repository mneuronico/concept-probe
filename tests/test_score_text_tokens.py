"""Model-free tests for score_text_tokens (additive low-level read primitive).

No real model/tokenizer: a tiny fake tokenizer provides input_ids + offset_mapping, and
forward_hidden_states_all_layers is monkeypatched to return controlled hidden states.
"""
import numpy as np
import torch

import concept_probe
from concept_probe import probe as probe_mod


class _FakeTokenizer:
    """Returns 2 tokens with known offsets; decode -> 'tok<id>'."""

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, add_special_tokens=True):
        out = {"input_ids": torch.tensor([[10, 11]], dtype=torch.long)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([[[0, 3], [3, 7]]], dtype=torch.long)
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "tok" + "_".join(str(int(i)) for i in ids)


# hidden states: 3 layers, 2 tokens, dim 4; token0=[1,2,3,4], token1=[5,6,7,8] (igual por capa)
def _fake_forward(model, input_ids, **kwargs):
    base = torch.tensor([[1.0, 2, 3, 4], [5, 6, 7, 8]])
    return [base.clone(), base.clone(), base.clone()]


CV = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)  # filas unitarias


def test_exposed_in_public_api():
    assert "score_text_tokens" in concept_probe.__all__
    assert concept_probe.score_text_tokens is probe_mod.score_text_tokens


def test_per_layer_scores_and_offsets(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    r = concept_probe.score_text_tokens(object(), _FakeTokenizer(), CV, "siete car")
    # layer l proyecta sobre el eje l: token0 -> [1,2,3], token1 -> [5,6,7]
    assert r["scores_per_layer"].shape == (2, 3)
    np.testing.assert_allclose(r["scores_per_layer"], [[1, 2, 3], [5, 6, 7]], atol=1e-6)
    assert r["offset_mapping"] == [[0, 3], [3, 7]]
    assert r["token_ids"].tolist() == [10, 11]
    assert r["tokens"] == ["tok10", "tok11"]
    assert r["layer_indices"] == [0, 1, 2]


def test_reference_layer_logit_lens(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    r = concept_probe.score_text_tokens(object(), _FakeTokenizer(), CV, "x", reference_layer=0)
    # todas las capas usan el eje 0 -> token0 = 1 en las 3 capas, token1 = 5
    np.testing.assert_allclose(r["scores_per_layer"], [[1, 1, 1], [5, 5, 5]], atol=1e-6)


def test_unit_norm_divides_by_vector_norm(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    cv = CV.copy(); cv[0] = [2, 0, 0, 0]  # norma 2 en la capa 0
    no_norm = concept_probe.score_text_tokens(object(), _FakeTokenizer(), cv, "x")
    with_norm = concept_probe.score_text_tokens(object(), _FakeTokenizer(), cv, "x", unit_norm=True)
    # sin normalizar: capa0 token0 = 1*2 = 2 ; con unit_norm: vuelve a 1
    assert no_norm["scores_per_layer"][0, 0] == 2.0
    np.testing.assert_allclose(with_norm["scores_per_layer"][0, 0], 1.0, atol=1e-6)


def test_layer_indices_subset(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    r = concept_probe.score_text_tokens(object(), _FakeTokenizer(), CV, "x", layer_indices=[2])
    assert r["scores_per_layer"].shape == (2, 1)
    np.testing.assert_allclose(r["scores_per_layer"][:, 0], [3, 7], atol=1e-6)
    assert r["layer_indices"] == [2]


def test_layer_mismatch_raises(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    bad = np.zeros((5, 4), dtype=np.float32)  # 5 capas vs 3 que devuelve el forward
    try:
        concept_probe.score_text_tokens(object(), _FakeTokenizer(), bad, "x")
    except ValueError as e:
        assert "Layer mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on layer mismatch")
