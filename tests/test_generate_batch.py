"""Model-free tests for generate_batch.

Batched generation must match the generate_once ``(ids, prompt_len)`` contract:
left-pad inputs, run generate, then strip left padding and any trailing right
padding per item -- and per-item results must not depend on how prompts are
chunked into batches. A fake tokenizer/model stand in (no real model needed).
"""
import torch

from concept_probe import probe as probe_mod


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None):
        user = [m for m in messages if m.get("role") == "user"][-1]["content"]
        # prompt length encodes content length; ids are all 5s (distinct from pad 0)
        return torch.tensor([[5] * len(user)], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=False):
        return "x"


class _FakeModel:
    """generate appends a content-dependent completion (length = nonpad-1, value =
    40 + nonpad), right-padded with pad_id within the batch. Output depends only on
    each row's prompt content, not its position in the batch -- so chunking must not
    change per-item results."""

    device = torch.device("cpu")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, pad_token_id=0, **kw):
        gens = []
        for r in range(input_ids.shape[0]):
            nonpad = int((input_ids[r] != pad_token_id).sum())
            gens.append([40 + nonpad] * max(1, nonpad - 1))
        maxk = max(len(g) for g in gens)
        rows = []
        for r in range(input_ids.shape[0]):
            padded = gens[r] + [pad_token_id] * (maxk - len(gens[r]))
            rows.append(torch.cat([input_ids[r], torch.tensor(padded, dtype=torch.long)]))
        return torch.stack(rows)


def _run(batch_size):
    return probe_mod.generate_batch(
        _FakeModel(), _FakeTok(), "sys", ["ab", "abcd"],
        max_new_tokens=8, greedy=True, temperature=None, top_p=None,
        batch_size=batch_size, warn=None,
    )


def test_left_pad_and_right_pad_strip():
    res = _run(batch_size=2)
    ids0, plen0 = res[0]
    ids1, plen1 = res[1]
    # prompt "ab" (len 2): nonpad=2 -> completion [42]; trailing right-pad (to maxk=3) stripped
    assert plen0 == 2 and ids0.tolist() == [5, 5, 42]
    # prompt "abcd" (len 4): nonpad=4 -> completion [44, 44, 44]
    assert plen1 == 4 and ids1.tolist() == [5, 5, 5, 5, 44, 44, 44]


def test_chunking_invariance():
    # batch_size 1 vs 2 must yield identical per-item results
    one = _run(batch_size=1)
    two = _run(batch_size=2)
    for (a_ids, a_p), (b_ids, b_p) in zip(one, two):
        assert a_p == b_p and a_ids.tolist() == b_ids.tolist()


def test_empty_prompts_returns_empty():
    assert probe_mod.generate_batch(_FakeModel(), _FakeTok(), "sys", [], 8, True, None, None) == []
