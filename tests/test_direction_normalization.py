"""Model-free tests for concept_direction and for reading labelled texts as assistant replies."""
import numpy as np
import pytest
import torch

import concept_probe
from concept_probe.probe import _encode_text_for_read, concept_direction, normed_np

RNG = np.random.default_rng(0)
POS = RNG.normal(0.5, 1.0, size=(8, 5)).astype(np.float32)
NEG = RNG.normal(-0.5, 2.0, size=(8, 5)).astype(np.float32)


def test_none_is_the_plain_difference_of_means():
    np.testing.assert_array_equal(concept_direction(POS, NEG), normed_np(POS.mean(0) - NEG.mean(0)))
    np.testing.assert_array_equal(concept_direction(POS, NEG, "none"), concept_direction(POS, NEG))


def test_var_matches_the_estimator_of_the_paper():
    # build_var_probes.py: (mu_pos - mu_neg) / (0.5 * (var_pos + var_neg) + 1e-12), unit length
    p, n = POS.astype(np.float64), NEG.astype(np.float64)
    v = (p.mean(0) - n.mean(0)) / (0.5 * (p.var(0) + n.var(0)) + 1e-12)
    np.testing.assert_allclose(concept_direction(POS, NEG, "var"), v / np.linalg.norm(v), atol=1e-6)


def test_sd_divides_by_the_pooled_standard_deviation():
    p, n = POS.astype(np.float64), NEG.astype(np.float64)
    v = (p.mean(0) - n.mean(0)) / np.sqrt(0.5 * (p.var(0) + n.var(0)) + 1e-12)
    np.testing.assert_allclose(concept_direction(POS, NEG, "sd"), v / np.linalg.norm(v), atol=1e-6)


@pytest.mark.parametrize("norm", ["none", "var", "sd"])
def test_directions_are_unit_float32(norm):
    v = concept_direction(POS, NEG, norm)
    assert v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_var_keeps_a_massive_coordinate_from_dominating():
    # Coordinate 0: huge variance, gap of 10. Coordinate 1: tiny variance, gap of 1 (the concept).
    pos = np.array([[110, 1.1], [-90, 0.9], [110, 1.1], [-90, 0.9]], dtype=np.float32)
    neg = np.array([[100, 0.1], [-100, -0.1], [100, 0.1], [-100, -0.1]], dtype=np.float32)
    assert abs(concept_direction(pos, neg, "none")[0]) > 0.99
    assert abs(concept_direction(pos, neg, "var")[1]) > 0.99


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError):
        concept_direction(POS, NEG, "whiten")


def test_exposed_in_public_api():
    assert "concept_direction" in concept_probe.__all__
    assert concept_probe.concept_direction is concept_direction


def test_defaults_keep_the_historical_behaviour():
    training = concept_probe.load_defaults()["training"]
    assert training["direction_normalization"] == "none"
    assert training["train_text_role"] == "user"
    assert training["train_text_prompt"] is None


class _ChatTok:
    chat_template = "fake"

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        self.messages = messages
        assert add_generation_prompt
        return torch.tensor([[1] * (2 * len(messages) + 1)], dtype=torch.long)

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        assert not add_special_tokens
        return {"input_ids": torch.tensor([[7] * len(text.split())], dtype=torch.long)}


def test_assistant_role_puts_the_text_after_the_assistant_header():
    tok = _ChatTok()
    ids, plen = _encode_text_for_read(tok, "one two three", "sys", role="assistant", prompt="question?")
    assert tok.messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "question?"}]
    assert plen == 5 and ids.shape == (1, 8)
    assert ids[0, plen:].tolist() == [7, 7, 7]


def test_assistant_role_without_system_or_prompt():
    tok = _ChatTok()
    ids, plen = _encode_text_for_read(tok, "one two", "", role="assistant", prompt=None)
    assert tok.messages == []
    assert ids[0, plen:].tolist() == [7, 7]


def test_assistant_role_rejects_message_lists_and_unknown_roles():
    with pytest.raises(TypeError):
        _encode_text_for_read(_ChatTok(), [{"role": "user", "content": "x"}], "", role="assistant")
    with pytest.raises(ValueError):
        _encode_text_for_read(_ChatTok(), "x", "", role="narrator")
