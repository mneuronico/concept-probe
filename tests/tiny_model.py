"""A tiny deterministic causal LM for end-to-end tests.

Unlike the stand-ins in tests/fakes.py, it is a real (if minuscule) transformer-shaped model:
token embeddings, blocks under ``model.layers`` that forward hooks can edit, a final norm under
``model.norm``, logits tied to the embeddings, an attention mask, and greedy or sampled ``generate``
over a batch. Its hidden states depend on the context and on every intervention, so training,
scoring, steering and batched generation run through the same code paths as on a real model.

The weights come from a fixed seed and the arithmetic is float64, so the outputs are reproducible
across machines. Pair it with ``fakes.CharChatTok`` (one token per character, id = ``ord``).

Built in: lowercase ``p`` and ``n`` move the embedding in opposite directions along one "concept"
axis, and coordinate 0 is a large, high-variance coordinate that the concept does not move (like
the massive activations of some models), so ``direction_normalization`` changes the direction.
"""
from types import SimpleNamespace

import torch
from torch import nn

VOCAB = 128
EOS = ord("$")
GEN_CHARS = "pnxyz $"  # what generate may produce; '$' ends a completion


class _Block(nn.Module):
    """``h + 0.5 tanh(W h) + 0.3 * (mean of the visible positions up to t)``."""

    def __init__(self, dim: int, gen: torch.Generator):
        super().__init__()
        self.mix = nn.Linear(dim, dim, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.mix.weight.copy_(torch.randn(dim, dim, generator=gen, dtype=torch.float64) / dim ** 0.5)

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.to(h.dtype).unsqueeze(-1)
        context = torch.cumsum(h * m, dim=1) / torch.cumsum(m, dim=1).clamp(min=1.0)
        return h + 0.5 * torch.tanh(self.mix(h)) + 0.3 * context


class _RMSNorm(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h / torch.sqrt((h * h).mean(dim=-1, keepdim=True) + 1e-6)


class TinyCausalLM(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 16, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(VOCAB, dim, generator=gen, dtype=torch.float64)
        concept = torch.randn(dim, generator=gen, dtype=torch.float64)
        concept[0] = 0.0
        concept = concept / concept.norm()
        emb[ord("p")] += 3.0 * concept
        emb[ord("n")] -= 3.0 * concept
        emb[:, 0] += 20.0 + 10.0 * torch.rand(VOCAB, generator=gen, dtype=torch.float64)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding.from_pretrained(emb, freeze=True)
        self.model.layers = nn.ModuleList([_Block(dim, gen) for _ in range(num_layers)])
        self.model.norm = _RMSNorm()
        head = emb.clone()
        head[:, 0] = 0.0  # the large coordinate would otherwise pick the next token on its own
        self.register_buffer("head", head, persistent=False)
        self.register_buffer("allowed", torch.tensor([ord(c) for c in GEN_CHARS]), persistent=False)
        self.generation_config = SimpleNamespace(eos_token_id=EOS)
        self.eval()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False,
                return_dict=True, **kwargs):
        input_ids = input_ids.to("cpu")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        h = self.model.embed_tokens(input_ids)
        states = [h]
        for block in self.model.layers:
            h = block(h, attention_mask.to("cpu"))
            states.append(h)
        normed = self.model.norm(h)
        states[-1] = normed  # like Hugging Face decoders: the last hidden state is after the final norm
        logits = normed @ self.head.T
        return SimpleNamespace(hidden_states=tuple(states) if output_hidden_states else None, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, do_sample=False, temperature=None,
                 top_p=None, pad_token_id=None, **kwargs):
        """Greedy (or sampled) continuation of every row, as Hugging Face's generate returns it: the
        input, the new tokens, and padding after a row's end-of-sequence token until all rows end."""
        ids = input_ids.to("cpu").clone()
        mask = torch.ones_like(ids) if attention_mask is None else attention_mask.to("cpu").clone()
        pad = EOS if pad_token_id is None else int(pad_token_id)
        done = torch.zeros(ids.shape[0], dtype=torch.bool)
        for _ in range(int(max_new_tokens)):
            logits = self.forward(ids, attention_mask=mask).logits[:, -1, :]
            masked = torch.full_like(logits, float("-inf"))
            masked[:, self.allowed] = logits[:, self.allowed]
            if do_sample:
                probs = torch.softmax(masked / float(temperature or 1.0), dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            else:
                nxt = masked.argmax(dim=-1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            mask = torch.cat([mask, torch.ones_like(nxt[:, None])], dim=1)
            done |= nxt == EOS
            if bool(done.all()):
                break
        return ids
