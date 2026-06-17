"""Residual-stream editing: activation patching, ablation, projection removal.

This module provides the generic *mechanism* for intervening on the residual
stream (the unopinionated counterpart of :mod:`concept_probe.steering`). The
library supplies the hook plumbing; the caller decides what edit to apply.

Layer convention (shared with steering and ``forward_hidden_states_all_layers``):
a hook on transformer block ``l`` edits that block's *output* residual, i.e.
``hidden_states[l+1]`` -- the same tensor that ``hs_layers[l]`` holds and that
``concept_vectors[l]`` is meant to score. So "patch layer K" == "replace the
residual stream after block K" (``resid_post[K]``).

Core pieces:
  * :class:`ResidualStreamEditor` -- context manager that registers forward
    hooks on chosen blocks and routes each block's output through a user
    ``edit_fn(layer_idx, hidden) -> hidden``. Steering, patching, ablation and
    projection-removal are all instances of this.
  * :class:`ActivationPatch` -- a declarative "at block ``layer``, overwrite the
    residual at ``positions`` with ``values``".
  * :func:`patched_forward_hidden_states` -- run one forward with patches applied
    and return per-layer hidden states exactly like
    ``forward_hidden_states_all_layers`` (so the patched run is scored with the
    same downstream code as the clean run).
  * :func:`cache_residual_stream` -- donor activations (thin alias over the clean
    forward), to be read at ``(layer, position)`` and fed into a patch.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .modeling import attention_mask_from_ids, get_transformer_layers

# edit_fn(layer_idx, hidden[batch, seq, hidden_dim]) -> hidden[batch, seq, hidden_dim]
EditFn = Callable[[int, torch.Tensor], torch.Tensor]


class ResidualStreamEditor:
    """Edit the residual stream of selected transformer blocks via forward hooks.

    ``edit_fn`` is called as ``edit_fn(layer_idx, hidden)`` for every hooked
    block, where ``hidden`` is that block's output residual of shape
    ``(batch, seq, hidden_dim)``; it must return a tensor of the same shape (the
    same tensor unchanged, or an edited copy). Use ``layers`` to restrict which
    blocks are hooked (default: all). Returning the input unchanged for layers
    you do not care about is also fine.

    This is intentionally unopinionated: it is the residual-stream intervention
    primitive, not a particular experiment. ``LayerSteerer`` (additive steering)
    is the same idea specialized to ``hidden += alpha * direction``.
    """

    def __init__(self, model, edit_fn: EditFn, layers: Optional[Sequence[int]] = None):
        self.model = model
        self.edit_fn = edit_fn
        n = len(get_transformer_layers(model))
        if layers is None:
            self.layers = list(range(n))
        else:
            self.layers = [int(l) + n if int(l) < 0 else int(l) for l in layers]
            for l in self.layers:
                if not (0 <= l < n):
                    raise ValueError(f"layer index {l} out of range for {n} blocks")
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                new = self.edit_fn(layer_idx, output[0])
                return (new,) + tuple(output[1:])
            return self.edit_fn(layer_idx, output)

        return hook

    def __enter__(self) -> "ResidualStreamEditor":
        blocks = get_transformer_layers(self.model)
        for li in self.layers:
            self.handles.append(blocks[li].register_forward_hook(self._make_hook(li)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []


@dataclass
class ActivationPatch:
    """Overwrite the residual at ``positions`` of block ``layer`` with ``values``.

    ``positions`` are token indices into the sequence being run. ``values`` is a
    ``(len(positions), hidden_dim)`` tensor (or ``(hidden_dim,)`` broadcast to a
    single position). Donor activations come from :func:`cache_residual_stream`
    on the donor input, read at ``hs_layers[layer][donor_positions]``; the donor
    and receiver positions need not match.
    """

    layer: int
    positions: Sequence[int]
    values: torch.Tensor

    def __post_init__(self):
        self.layer = int(self.layer)
        if isinstance(self.positions, int):
            self.positions = [self.positions]
        self.positions = [int(p) for p in self.positions]
        if self.values.dim() == 1:
            self.values = self.values.unsqueeze(0)
        if self.values.shape[0] != len(self.positions):
            raise ValueError(
                f"values has {self.values.shape[0]} rows but {len(self.positions)} positions given"
            )


def build_patch_edit_fn(patches: Sequence[ActivationPatch]) -> EditFn:
    """Compile a list of :class:`ActivationPatch` into an ``edit_fn``.

    The edit is out-of-place (the block output is cloned before writing) so the
    model's own forward bookkeeping is never mutated underneath it.
    """
    by_layer: Dict[int, List[ActivationPatch]] = {}
    for p in patches:
        by_layer.setdefault(int(p.layer), []).append(p)

    def edit(layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        ps = by_layer.get(layer_idx)
        if not ps:
            return hidden
        seq_len = hidden.shape[1]
        out = hidden.clone()
        for p in ps:
            vals = p.values.to(device=hidden.device, dtype=hidden.dtype)
            for j, pos in enumerate(p.positions):
                if not (0 <= pos < seq_len):
                    raise IndexError(
                        f"patch position {pos} out of range for seq_len={seq_len} (layer {layer_idx})"
                    )
                out[:, pos, :] = vals[j]
        return out

    return edit


@torch.no_grad()
def cache_residual_stream(model, input_ids: torch.Tensor) -> List[torch.Tensor]:
    """Donor activations: per-layer residual stream (block outputs) for ``input_ids``.

    Returns a list of length ``num_layers`` of ``(seq_len, hidden_dim)`` CPU
    float32 tensors, where entry ``l`` is ``resid_post[l]`` (== ``hidden_states[l+1]``).
    Thin alias over ``forward_hidden_states_all_layers`` so donor and clean reads
    use identical capture semantics (including the final-norm scale fix).
    """
    from .probe import forward_hidden_states_all_layers

    return forward_hidden_states_all_layers(model, input_ids)


@torch.no_grad()
def patched_forward_hidden_states(
    model,
    input_ids: torch.Tensor,
    patches: Sequence[ActivationPatch],
) -> List[torch.Tensor]:
    """Run one forward with ``patches`` applied; return per-layer hidden states.

    Output matches ``forward_hidden_states_all_layers`` (list of
    ``(seq_len, hidden_dim)`` CPU float32 tensors, ``hs_layers[l] == resid_post[l]``),
    so a patched run is scored with the same downstream code as a clean run. The
    patch is reflected at the patched layer AND every layer after it; layers
    before it are unchanged.

    Residuals are captured from block-output hooks *after* the edit is applied,
    rather than from ``output_hidden_states``. This matters precisely at the
    patched layer: under ``output_hidden_states`` a forward hook's replacement of
    block ``K``'s output propagates downstream but is *not* reflected in that
    block's own recorded entry (HF records it pre-hook), so ``hs_layers[K]`` would
    misleadingly show the unpatched value. Capturing post-edit makes the whole
    returned stream self-consistent. (The downstream readout, e.g. the final
    layer, is identical either way -- it is always post-edit.)
    """
    edit_fn = build_patch_edit_fn(patches)
    blocks = get_transformer_layers(model)
    n = len(blocks)
    captured: List[torch.Tensor] = [None] * n  # type: ignore[list-item]
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_idx: int):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output
            hs = edit_fn(layer_idx, hs)
            captured[layer_idx] = hs[0].detach().float().cpu()
            if is_tuple:
                return (hs,) + tuple(output[1:])
            return hs

        return hook

    for li in range(n):
        handles.append(blocks[li].register_forward_hook(_make_hook(li)))
    try:
        attn = attention_mask_from_ids(input_ids).to(model.device)
        model(
            input_ids=input_ids.to(model.device),
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    if any(c is None for c in captured):
        raise RuntimeError("Block-output capture hook did not fire for every layer; unexpected model layout.")
    return captured
