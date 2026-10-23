"""Model-free tests for reading texts raw, the chat-template override and extract_text_reps."""
import numpy as np
import pytest
import torch

import concept_probe
from concept_probe import probe as probe_mod
from concept_probe.modeling import apply_tokenizer_overrides
from concept_probe.probe import _encode_text_for_read


class _Tok:
    chat_template = "fake"

    def __init__(self, bos=1):
        self.bos_token_id = bos

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        return torch.tensor([[5] * (2 * len(messages) + 1)], dtype=torch.long)

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        assert not add_special_tokens
        return {"input_ids": torch.tensor([[7] * len(text.split())], dtype=torch.long)}


def test_raw_is_bos_then_the_text():
    ids, plen = _encode_text_for_read(_Tok(), "one two three", "sys", role="raw")
    assert ids.tolist() == [[1, 7, 7, 7]] and plen == 1


def test_raw_without_bos():
    ids, plen = _encode_text_for_read(_Tok(bos=None), "one two", "", role="raw")
    assert ids.tolist() == [[7, 7]] and plen == 0


def test_raw_warns_that_system_and_prompt_are_ignored():
    seen = []
    _encode_text_for_read(_Tok(), "x", "sys", role="raw", prompt="q?", warn=seen.append)
    assert seen and "ignored" in seen[0]


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        _encode_text_for_read(_Tok(), "x", "", role="narrator")


def test_chat_template_override_replaces_the_tokenizer_template():
    tok = _Tok()
    apply_tokenizer_overrides(tok, {"chat_template": "{{ messages }}"})
    assert tok.chat_template == "{{ messages }}"
    apply_tokenizer_overrides(tok, {"chat_template": None})
    assert tok.chat_template == "{{ messages }}"  # null leaves it alone


def test_defaults_leave_the_template_alone():
    assert concept_probe.load_defaults()["model"]["chat_template"] is None


def _fake_forward(model, input_ids, **kwargs):
    # one layer, dim 2; token t has value t, so a mean over a span is easy to check
    n = int(input_ids.shape[-1])
    return [torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 2)]


def test_extract_text_reps_pools_over_the_text_only(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    # raw: [BOS, t1, t2, t3] -> text rows 1..3, mean 2
    raw = concept_probe.extract_text_reps(object(), _Tok(), ["a b c"], role="raw")
    np.testing.assert_allclose(raw[0, 0], [2, 2])
    # assistant with system + prompt: 5 prompt tokens, then 3 text tokens (rows 5..7), mean 6
    asst = concept_probe.extract_text_reps(object(), _Tok(), ["a b c"], role="assistant", prompt="q", system="s")
    np.testing.assert_allclose(asst[0, 0], [6, 6])


def test_extract_text_reps_several_pools_in_one_pass(monkeypatch):
    calls = []

    def counting(model, input_ids, **kwargs):
        calls.append(1)
        return _fake_forward(model, input_ids)

    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", counting)
    out = concept_probe.extract_text_reps(object(), _Tok(), ["a b c d"], role="raw",
                                          pool=["assistant_all_mean", "assistant_last", "assistant_last_k_mean"],
                                          last_k=2)
    assert len(calls) == 1
    np.testing.assert_allclose(out["assistant_all_mean"][0, 0], [2.5, 2.5])  # rows 1..4
    np.testing.assert_allclose(out["assistant_last"][0, 0], [4, 4])
    np.testing.assert_allclose(out["assistant_last_k_mean"][0, 0], [3.5, 3.5])


def test_exposed_in_public_api():
    assert "extract_text_reps" in concept_probe.__all__
