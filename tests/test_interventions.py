"""Interventions on the residual stream: what each one does to the stream, at which layers
and positions, in batches padded on either side, and how reading reports the edited stream."""
import numpy as np
import pytest
import torch

from concept_probe import (Ablate, Add, Compose, Edit, Encoded, Patch, Rotate, generate, intervene, read_encoded,
                           read_texts)
from concept_probe.modeling import get_transformer_layers
from fakes import CharChatTok
from tiny_model import TinyCausalLM

TEXTS = ["pp x", "nn yzx", "p", "xyzzy pp n"]
RNG = np.random.default_rng(0)
V = RNG.normal(size=16)
W = RNG.normal(size=16)


def _ids(text):
    return torch.tensor([ord(c) for c in "^" + text])


def _stream(model, texts=TEXTS, batch_size=4):
    """Every token's residual stream at every layer, per text: a list of (tokens, layers, dim)."""
    reps = read_encoded(model, CharChatTok(), [Encoded(_ids(t)) for t in texts], "through_text_tokens",
                        batch_size=batch_size, cache=False)
    return [reps.data[reps.items == i] for i in range(len(texts))]


def _unit(v):
    return np.asarray(v) / np.linalg.norm(v)


# ------------------------------------------------------------------------------------------- add
def test_add_puts_alpha_times_the_vector_on_the_chosen_layer_and_nothing_before_it():
    model = TinyCausalLM()
    clean = _stream(model)
    with intervene(model, Add({1: V}, alpha=2.0)):
        steered = _stream(model)
    for c, s in zip(clean, steered):
        np.testing.assert_allclose(s[:, 0], c[:, 0])  # the layer before is untouched
        np.testing.assert_allclose(s[:, 1] - c[:, 1], np.tile(2.0 * V, (len(c), 1)), atol=1e-5)
        assert not np.allclose(s[:, 2], c[:, 2])  # and later layers see it


def test_add_takes_an_array_of_vectors_per_layer_negative_layers_and_alpha_per_layer():
    model = TinyCausalLM()
    vectors = RNG.normal(size=(4, 16))
    clean = _stream(model, TEXTS[:1])[0]
    with intervene(model, Add(vectors, alpha={0: 1.0, -1: 0.5}, layers=[0, -1])):
        steered = _stream(model, TEXTS[:1])[0]
    np.testing.assert_allclose(steered[:, 0] - clean[:, 0], np.tile(vectors[0], (len(clean), 1)), atol=1e-5)
    edit = Add(vectors, alpha={0: 1.0, -1: 0.5}, layers=[0, -1])
    edit.prepare(4)
    assert edit.alpha_per_layer == {0: 1.0, 3: 0.5} and edit.to_dict()["layers"] == [0, 3]


def test_alpha_zero_changes_nothing_and_hooks_go_away():
    model = TinyCausalLM()
    clean = _stream(model)
    with intervene(model, Add({1: V}, alpha=0.0)):
        same = _stream(model)
    for c, s in zip(clean, same):
        np.testing.assert_array_equal(c, s)
    with pytest.raises(RuntimeError):
        with intervene(model, Add({1: V}, alpha=5.0)):
            raise RuntimeError("inside")
    assert all(not block._forward_hooks for block in get_transformer_layers(model))
    assert not model._forward_pre_hooks


# ------------------------------------------------------------------------------------- positions
@pytest.mark.parametrize("positions", [[-1], [0, 1], slice(1, 3), slice(-2, None), "prompt"])
def test_positions_mean_the_same_tokens_in_a_padded_batch_and_alone(positions):
    model = TinyCausalLM()
    edit = Add({1: V}, alpha=3.0, positions=positions)
    with intervene(model, edit):
        batched = _stream(model, batch_size=4)
        alone = _stream(model, batch_size=1)
    for b, a in zip(batched, alone):
        np.testing.assert_allclose(b, a, atol=1e-5)


def test_negative_positions_count_back_from_the_end_of_each_input():
    model = TinyCausalLM()
    clean = _stream(model)
    with intervene(model, Add({0: V}, alpha=1.0, positions=[-1])):
        steered = _stream(model)
    for c, s in zip(clean, steered):
        changed = np.where(np.abs(s[:, 0] - c[:, 0]).max(axis=1) > 1e-6)[0]
        assert changed.tolist() == [len(c) - 1]


def test_a_position_past_an_input_raises():
    model = TinyCausalLM()
    with intervene(model, Add({0: V}, positions=[40])):
        with pytest.raises(IndexError, match="past the end"):
            _stream(model)


# ----------------------------------------------------------------------------------------- ablate
def test_ablate_removes_the_direction_or_sets_its_coordinate():
    model = TinyCausalLM()
    u = _unit(V)
    with intervene(model, Ablate(V, layers=[2])):
        removed = _stream(model)
    with intervene(model, Ablate({2: V}, value=4.0, positions=[-1])):
        set_last = _stream(model)
    clean = _stream(model)
    for r, s, c in zip(removed, set_last, clean):
        np.testing.assert_allclose(r[:, 2] @ u, 0.0, atol=1e-5)
        np.testing.assert_allclose(s[-1, 2] @ u, 4.0, atol=1e-5)
        np.testing.assert_allclose(s[:-1, 2], c[:-1, 2])  # other positions keep their coordinate
        # what is orthogonal to the direction is left as it was
        np.testing.assert_allclose(r[:, 2] - np.outer(r[:, 2] @ u, u), c[:, 2] - np.outer(c[:, 2] @ u, u), atol=1e-5)


def test_ablating_a_subspace_removes_every_direction_in_it():
    model = TinyCausalLM()
    with intervene(model, Ablate({1: np.stack([V, W])})):
        removed = _stream(model)
    for r in removed:
        np.testing.assert_allclose(r[:, 1] @ np.stack([_unit(V), _unit(W)]).T, 0.0, atol=1e-4)
    with pytest.raises(ValueError, match="linearly dependent"):
        with intervene(model, Ablate({1: np.stack([V, 2 * V])})):
            pass


# ------------------------------------------------------------------------------------------ patch
def test_patch_writes_donor_activations_at_the_positions_and_later_layers_follow():
    model = TinyCausalLM()
    donor = _stream(model, ["zz pp"])[0]
    receiver_clean = _stream(model, ["nn x"])[0]
    with intervene(model, Patch(1, [-1], donor[-1, 1])):
        patched = _stream(model, ["nn x"])[0]
    np.testing.assert_allclose(patched[-1, 1], donor[-1, 1], atol=1e-6)
    np.testing.assert_allclose(patched[:-1, 1], receiver_clean[:-1, 1])
    assert not np.allclose(patched[-1, 2], receiver_clean[-1, 2])
    with pytest.raises(ValueError, match="rows"):
        Patch(1, [0, 1], donor[0, 1])


def test_a_patch_with_one_donor_per_prompt_equals_patching_each_prompt_alone():
    from concept_probe import answer_logprobs

    model, tok = TinyCausalLM(), CharChatTok()
    prompts, answers = ["nn x", "pp z", "xz n"], ["p", "n"]
    rng = np.random.default_rng(0)
    last = rng.normal(size=(3, 1, 16)).astype(np.float32)  # each prompt's own value at its last position
    together = answer_logprobs(model, tok, prompts, answers, intervention=Patch(1, [-1], last), save=False)
    for i, p in enumerate(prompts):
        alone = answer_logprobs(model, tok, [p], answers, intervention=Patch(1, [-1], last[i]), save=False)
        np.testing.assert_allclose(together.logprobs[i], alone.logprobs[0], atol=1e-9)
    # each prompt its own positions too
    own = [[-1], [-2, -1], [-3]]
    values = [rng.normal(size=(len(p), 16)).astype(np.float32) for p in own]
    together = answer_logprobs(model, tok, prompts, answers, intervention=Patch(1, own, values), save=False)
    for i, p in enumerate(prompts):
        alone = answer_logprobs(model, tok, [p], answers, intervention=Patch(1, own[i], values[i]), save=False)
        np.testing.assert_allclose(together.logprobs[i], alone.logprobs[0], atol=1e-9)
    gens = generate(model, tok, prompts, max_new_tokens=2, intervention=Patch(1, [-1], last), save=False)
    assert len(gens) == 3
    with pytest.raises(RuntimeError, match="which input"):  # a direct call does not say which input a row is
        with intervene(model, Patch(1, [-1], last)):
            _stream(model, ["nn x"])
    with intervene(model, Patch(1, [-1], last), items=[2]):
        _stream(model, ["nn x"])


# ----------------------------------------------------------------------------------------- rotate
def test_rotate_turns_the_stream_in_the_plane_and_keeps_its_norm():
    model = TinyCausalLM()
    clean = _stream(model)
    with intervene(model, Rotate(V, W, np.pi / 2, layers=[0])):
        turned = _stream(model)
    e1 = _unit(V)
    e2 = _unit(W - (W @ e1) * e1)
    for t, c in zip(turned, clean):
        np.testing.assert_allclose(np.linalg.norm(t[:, 0], axis=1), np.linalg.norm(c[:, 0], axis=1), rtol=1e-5)
        np.testing.assert_allclose(t[:, 0] @ e2, c[:, 0] @ e1, atol=1e-4)  # the u part went to v
    with intervene(model, Rotate(V, W, 2 * np.pi, layers=[0])):
        full_turn = _stream(model)
    for f, c in zip(full_turn, clean):
        np.testing.assert_allclose(f, c, atol=1e-4)


# ------------------------------------------------------------------------------------ edit, compose
def test_edit_gets_positions_and_compose_applies_in_order():
    model = TinyCausalLM()
    seen = []

    def spy(layer, hidden, where):
        seen.append((layer, where.positions.tolist(), where.prompt_lens.tolist()))
        return hidden

    with intervene(model, Edit(spy, [3])):
        _stream(model, ["ab", "abcd"])
    assert seen == [(3, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], [3, 5])]  # right padding: positions from 0

    u = _unit(V)
    with intervene(model, Add({1: V}, alpha=5.0) + Ablate({1: V})):
        added_then_removed = _stream(model)
    with intervene(model, Compose(Ablate({1: V}), Add({1: V}, alpha=5.0))):
        removed_then_added = _stream(model)
    for a, r in zip(added_then_removed, removed_then_added):
        np.testing.assert_allclose(a[:, 1] @ u, 0.0, atol=1e-4)
        np.testing.assert_allclose(r[:, 1] @ u, 5.0 * np.linalg.norm(V), atol=1e-4)


# ------------------------------------------------------------------------------ reading and caching
def test_reading_under_an_intervention_does_not_use_the_cache(tmp_path):
    model = TinyCausalLM()
    model.name_or_path = "tiny-test"
    tok = CharChatTok()
    clean = read_texts(model, tok, TEXTS, cache=str(tmp_path))
    with intervene(model, Add({0: V}, alpha=3.0)):
        steered = read_texts(model, tok, TEXTS, cache=str(tmp_path))
    again = read_texts(model, tok, TEXTS, cache=str(tmp_path))
    assert not np.allclose(steered.data, clean.data)
    np.testing.assert_array_equal(again.data, clean.data)
    assert any(c.chosen is None and "intervention" in c.reason for c in steered.choices)
