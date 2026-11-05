from dataclasses import dataclass
from typing import Dict, List, Optional

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
        hidden = output[0] if isinstance(output, tuple) else output
        # The device of the block's output, not of its parameters: with CPU offload (accelerate),
        # a block's parameters sit on the "meta" device outside its forward pass.
        offset = (self.alpha * self.direction_cpu).view(1, 1, -1).to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (hidden + offset,) + output[1:]
        return hidden + offset

    def __enter__(self):
        layer = get_transformer_layers(self.model)[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MultiLayerSteererLayerwise:
    """Add ``alpha_l * v_l`` to the output of each block ``l`` in ``layer_indices``.

    Each layer adds its own concept vector ``concept_vectors[l]`` (the probe trained at that
    layer), not one layer's vector copied to the others. ``alpha_l`` is ``alpha``, divided by
    the number of steered layers when ``distribute``, and multiplied by ``layer_scales[l]`` when
    given (e.g. that layer's mean residual norm for ``alpha_unit="resid_norm"``);
    :meth:`per_layer_alpha` returns them.
    """

    def __init__(
        self,
        model,
        layer_indices: List[int],
        concept_vectors,
        alpha: float,
        distribute: bool,
        layer_scales: Optional[Dict[int, float]] = None,
    ):
        self.model = model
        self.layer_indices = list(map(int, layer_indices))
        self.alpha = float(alpha)
        self.distribute = bool(distribute)
        self.concept_vectors = concept_vectors
        self.layer_scales = layer_scales
        self.steerers: List[LayerSteerer] = []

    def per_layer_alpha(self) -> Dict[int, float]:
        """The alpha each steered layer adds its vector with."""
        if not self.layer_indices:
            return {}
        per = self.alpha / len(self.layer_indices) if self.distribute else self.alpha
        return {
            li: float(per * (1.0 if self.layer_scales is None else float(self.layer_scales[li])))
            for li in self.layer_indices
        }

    def __enter__(self):
        if self.alpha == 0.0 or len(self.layer_indices) == 0:
            return self
        for li, alpha_li in self.per_layer_alpha().items():
            s = LayerSteerer(
                model=self.model,
                layer_idx=li,
                direction_cpu=torch.tensor(self.concept_vectors[li], dtype=torch.float32),
                alpha=alpha_li,
            )
            s.__enter__()
            self.steerers.append(s)
        return self

    def __exit__(self, exc_type, exc, tb):
        for s in self.steerers:
            s.__exit__(exc_type, exc, tb)
        self.steerers = []
