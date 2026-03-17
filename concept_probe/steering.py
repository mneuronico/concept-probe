from dataclasses import dataclass
from typing import List, Optional

import torch

from .modeling import get_transformer_layers


@dataclass
class LayerSteerer:
    model: torch.nn.Module
    layer_idx: int
    direction_cpu: torch.Tensor
    alpha: float
    handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _hook(self, module, inputs, output):
        d = self.direction_cpu.to(next(module.parameters()).device)
        if isinstance(output, tuple):
            hs = output[0] + (self.alpha * d).view(1, 1, -1).to(output[0].dtype)
            return (hs,) + output[1:]
        return output + (self.alpha * d).view(1, 1, -1).to(output.dtype)

    def __enter__(self):
        layer = get_transformer_layers(self.model)[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MultiLayerSteererLayerwise:
    def __init__(self, model, layer_indices: List[int], concept_vectors, alpha: float, distribute: bool):
        self.model = model
        self.layer_indices = list(map(int, layer_indices))
        self.alpha = float(alpha)
        self.distribute = bool(distribute)
        self.concept_vectors = concept_vectors
        self.steerers: List[LayerSteerer] = []

    def __enter__(self):
        if self.alpha == 0.0 or len(self.layer_indices) == 0:
            return self
        per = self.alpha / len(self.layer_indices) if self.distribute else self.alpha
        for li in self.layer_indices:
            v_li = self.concept_vectors[li]
            s = LayerSteerer(
                model=self.model,
                layer_idx=li,
                direction_cpu=torch.tensor(v_li, dtype=torch.float32),
                alpha=float(per),
            )
            s.__enter__()
            self.steerers.append(s)
        return self

    def __exit__(self, exc_type, exc, tb):
        for s in self.steerers:
            s.__exit__(exc_type, exc, tb)
        self.steerers = []
