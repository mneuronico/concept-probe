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
            offsets = [(i, i + 1) for i in range(len(text))]
            # batched like input_ids when tensors are asked for, as Hugging Face tokenizers do
            out["offset_mapping"] = torch.tensor([offsets], dtype=torch.long) if return_tensors == "pt" else offsets
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


def fake_forward(model, input_ids, **kwargs):
    """Two layers of (seq_len, 2) hidden states; layer 1 is layer 0 doubled."""
    v = torch.tensor([{"p": 1.0, "n": -1.0}.get(chr(int(t)), 0.0) for t in input_ids.reshape(-1)])
    layer = torch.stack([v, torch.ones_like(v)], dim=1)
    return [layer, 2 * layer]


class _Block(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # a parameter, as real blocks have
        self.scale = scale

    def forward(self, x, *args, **kwargs):
        return x * self.scale


class FunctionModel(torch.nn.Module):
    """A model whose hidden states come from ``fn(model, input_ids)``, a function of the kind tests
    used to put in place of ``forward_hidden_states_all_layers``: given one input ``(1, seq_len)``,
    it returns one ``(seq_len, dim)`` tensor per layer. Batches are computed row by row; ``calls``
    counts forward passes."""

    device = torch.device("cpu")

    def __init__(self, fn, num_layers=None):
        super().__init__()
        self.fn = fn
        probe = fn(None, torch.zeros(1, 1, dtype=torch.long))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block() for _ in range(num_layers or len(probe))])
        self.model.norm = torch.nn.Identity()
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        self.calls += 1
        rows = [self.fn(None, row.unsqueeze(0)) for row in input_ids]
        layers = [torch.stack([r[l] for r in rows]).float() for l in range(len(rows[0]))]
        states = [torch.zeros_like(layers[0])] + layers[:-1] + [self.model.norm(layers[-1])]
        return SimpleNamespace(hidden_states=tuple(states), logits=None)


class _Norm(torch.nn.Module):
    def forward(self, x):
        return x / (x.abs().amax(dim=-1, keepdim=True) + 1.0)


class CharModel(torch.nn.Module):
    """generate appends ``completion`` and the end-of-sequence token '$', as HF generate does.

    Its forward pass gives the hidden states of :func:`fake_forward` (block 0 passes the
    embeddings ``[value, 1]`` through, block 1 doubles them), through two real blocks that steering
    hooks can be registered on, and a final norm, as the library's readers expect.
    """

    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=ord("$"))

    def __init__(self, completion="pp"):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Block(1.0), _Block(2.0)])
        self.model.norm = _Norm()
        self.completion = completion
        self.calls = []
        self.forward_calls = []

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        self.forward_calls.append(input_ids.shape)
        v = torch.tensor([[{"p": 1.0, "n": -1.0}.get(chr(int(t)), 0.0) for t in row] for row in input_ids])
        x = torch.stack([v, torch.ones_like(v)], dim=-1)
        states = [x]
        h = x
        for block in self.model.layers:
            h = block(h)
            states.append(h)
        states[-1] = self.model.norm(h)
        logits = torch.zeros(input_ids.shape + (128,))
        return SimpleNamespace(hidden_states=tuple(states) if output_hidden_states else None, logits=logits)

    def generate(self, input_ids, max_new_tokens=8, **kwargs):
        self.calls.append(input_ids)
        tail = torch.tensor([[ord(c) for c in self.completion + "$"]], dtype=torch.long)
        return torch.cat([input_ids, tail.expand(input_ids.shape[0], -1)], dim=1)


@torch.no_grad()
def hidden_states_alone(model, input_ids):
    """Reference reading of one input, alone and unpadded: one ``(seq_len, dim)`` float32 tensor per
    block output, the last one taken before the final norm (as the library reads it). Tests compare
    the batched readers with it."""
    ids = input_ids.reshape(1, -1)
    captured = {}
    norm = model.model.norm
    handle = norm.register_forward_pre_hook(lambda module, args: captured.__setitem__("pre", args[0]))
    try:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    finally:
        handle.remove()
    hs = list(out.hidden_states)
    hs[-1] = captured["pre"]
    return [h[0].float() for h in hs[1:]]
