"""Model-free test for extract_completion_reps.

A fake tokenizer provides a 2-token prompt and a 3-token completion;
forward_hidden_states_all_layers is monkeypatched to return controlled hidden
states. The rep must be the mean over the completion tokens only (prompt excluded).
"""
import numpy as np
import torch

import concept_probe
from concept_probe import probe as probe_mod


class _FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        return torch.tensor([[5, 5]], dtype=torch.long)  # prompt_len = 2

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return {"input_ids": torch.tensor([[7, 7, 7]], dtype=torch.long)}  # completion_len = 3


def _fake_forward(model, input_ids, **kwargs):
    # one layer, seq=5 (2 prompt + 3 completion), dim=4; completion rows have values 1,2,3
    return [torch.tensor([[0.0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])]


def test_pools_over_completion_tokens_only(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    reps = concept_probe.extract_completion_reps(object(), _FakeTok(), [("hi", "abc")])
    assert reps.shape == (1, 1, 4)
    # assistant_all_mean over completion rows [1..],[2..],[3..] -> [2,2,2,2] (prompt rows excluded)
    np.testing.assert_allclose(reps[0, 0], [2, 2, 2, 2], atol=1e-6)


def test_exposed_in_public_api():
    assert "extract_completion_reps" in concept_probe.__all__
    assert concept_probe.extract_completion_reps is probe_mod.extract_completion_reps
