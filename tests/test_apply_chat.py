"""Model-free tests for apply_chat.

``tokenizer.apply_chat_template`` returns different shapes across transformers
versions: a bare 2D tensor, a 1D tensor, or a BatchEncoding/dict. apply_chat must
normalize all of them to a (1, seq_len) LongTensor. A fake tokenizer stands in for
each case (no real model/tokenizer needed).
"""
import torch

from concept_probe import modeling


class _TokRetTensor2D:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        assert tokenize is True
        assert return_tensors == "pt"
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


class _TokRetTensor1D:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        return torch.tensor([1, 2, 3], dtype=torch.long)


class _TokRetBatchEncoding:
    """Newer transformers can return a dict/BatchEncoding instead of a bare tensor."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        return {
            "input_ids": torch.tensor([[4, 5, 6, 7]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


MSGS = [{"role": "user", "content": "hi"}]


def test_tensor_2d_passthrough():
    out = modeling.apply_chat(_TokRetTensor2D(), MSGS, add_generation_prompt=True)
    assert isinstance(out, torch.Tensor) and out.dim() == 2
    assert out.tolist() == [[1, 2, 3]]


def test_tensor_1d_promoted_to_2d():
    out = modeling.apply_chat(_TokRetTensor1D(), MSGS, add_generation_prompt=True)
    assert isinstance(out, torch.Tensor) and out.dim() == 2
    assert out.tolist() == [[1, 2, 3]]


def test_batch_encoding_input_ids_extracted():
    out = modeling.apply_chat(_TokRetBatchEncoding(), MSGS, add_generation_prompt=False)
    assert isinstance(out, torch.Tensor) and out.dim() == 2
    assert out.tolist() == [[4, 5, 6, 7]]
