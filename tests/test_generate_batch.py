"""Model-free tests for generate_batch.

Batched generation must match the generate_once ``(ids, prompt_len)`` contract:
left-pad inputs, run generate, then strip left padding and the padding after the
first end-of-sequence token per item (the EOS itself is kept, as generate_once keeps
it) -- and per-item results must not depend on how prompts are chunked into batches.
A fake tokenizer/model stand in (no real model needed).
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
    40 + nonpad) and the end-of-sequence token 0, right-padded with pad_id within the
    batch, as HF generate does. Output depends only on each row's prompt content, not
    its position in the batch -- so chunking must not change per-item results."""

    device = torch.device("cpu")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, pad_token_id=0, **kw):
        gens = []
        for r in range(input_ids.shape[0]):
            nonpad = int((input_ids[r] != pad_token_id).sum())
            gens.append([40 + nonpad] * max(1, nonpad - 1) + [0])
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
    # prompt "ab" (len 2): completion [42] + EOS 0; the padding after the EOS is stripped
    assert plen0 == 2 and ids0.tolist() == [5, 5, 42, 0]
    # prompt "abcd" (len 4): completion [44, 44, 44] + EOS 0
    assert plen1 == 4 and ids1.tolist() == [5, 5, 5, 5, 44, 44, 44, 0]
    # the completion ends before the EOS
    assert probe_mod._completion_end(ids0, plen0, {0}) == 3


def test_a_pad_token_distinct_from_eos_is_stripped_after_the_eos():
    class _Tok(_FakeTok):
        pad_token_id = 9

    class _Model:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask=None, max_new_tokens=8, pad_token_id=9, **kw):
            # row 0 finishes early: completion [41], EOS 0, then padding 9
            rows = [torch.cat([input_ids[0], torch.tensor([41, 0, 9])]),
                    torch.cat([input_ids[1], torch.tensor([43, 43, 0])])]
            return torch.stack(rows)

    res = probe_mod.generate_batch(_Model(), _Tok(), "sys", ["ab", "ab"], 8, True, None, None, batch_size=2)
    assert res[0][0].tolist() == [5, 5, 41, 0]
    assert res[1][0].tolist() == [5, 5, 43, 43, 0]


def test_chunking_invariance():
    # batch_size 1 vs 2 must yield identical per-item results
    one = _run(batch_size=1)
    two = _run(batch_size=2)
    for (a_ids, a_p), (b_ids, b_p) in zip(one, two):
        assert a_p == b_p and a_ids.tolist() == b_ids.tolist()


def test_empty_prompts_returns_empty():
    assert probe_mod.generate_batch(_FakeModel(), _FakeTok(), "sys", [], 8, True, None, None) == []
