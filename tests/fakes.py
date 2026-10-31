"""Model-free stand-ins shared by the tests.

A character-level chat tokenizer and a forward pass whose hidden states are known in closed
form: a token is worth +1 for 'p', -1 for 'n' and 0 otherwise (dimension 0; dimension 1 is
constant), so any pooled span has an exact expected value.
"""
from types import SimpleNamespace

import torch


class CharChatTok:
    """One token per character (id = ord). Template: ``^<s>system|<u>text|``, plus ``<a>`` to prompt a reply."""

    chat_template = "fake"
    bos_token_id = ord("^")
    eos_token_id = pad_token_id = ord("$")

    def __init__(self, strip_content=False, offsets=True):
        self.strip_content = strip_content
        self.offsets = offsets
        self.rendered = []

    def _render(self, messages, add_generation_prompt):
        body = "".join(
            f"<{m['role'][0]}>{m['content'].strip() if self.strip_content else m['content']}|" for m in messages
        )
        return "^" + body + ("<a>" if add_generation_prompt else "")

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_tensors=None, **kwargs):
        text = self._render(messages, add_generation_prompt)
        self.rendered.append(text)
        if not tokenize:
            return text
        return torch.tensor([[ord(c) for c in text]], dtype=torch.long)

    def __call__(self, text, add_special_tokens=True, return_tensors=None, return_offsets_mapping=False):
        if return_offsets_mapping and not self.offsets:
            raise NotImplementedError("slow tokenizer")
        ids = [ord(c) for c in text]
        out = {"input_ids": torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


def fake_forward(model, input_ids, **kwargs):
    """Two layers of (seq_len, 2) hidden states; layer 1 is layer 0 doubled."""
    v = torch.tensor([{"p": 1.0, "n": -1.0}.get(chr(int(t)), 0.0) for t in input_ids.reshape(-1)])
    layer = torch.stack([v, torch.ones_like(v)], dim=1)
    return [layer, 2 * layer]


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # steering hooks read the device from a parameter

    def forward(self, x):
        return x


class CharModel(torch.nn.Module):
    """generate appends ``completion`` and the end-of-sequence token '$', as HF generate does.

    It has two (identity) blocks, so steering hooks can be registered on it.
    """

    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=ord("$"))

    def __init__(self, completion="pp"):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block(), _Block()])
        self.completion = completion
        self.calls = []

    def generate(self, input_ids, max_new_tokens=8, **kwargs):
        self.calls.append(input_ids)
        tail = torch.tensor([[ord(c) for c in self.completion + "$"]], dtype=torch.long)
        return torch.cat([input_ids, tail], dim=1)
