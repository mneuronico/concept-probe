"""Reading: batches, readouts, the embedding layer and the representations cache.

The tiny model (tests/tiny_model.py) is context-dependent and float64, so a batched read must equal
reading one input at a time, and every readout can be checked against the hidden states directly.
"""
import os

import numpy as np
import pytest
import torch

import concept_probe
from concept_probe import EmptySpanError, NonFiniteActivationsError, Readout, read_texts
from concept_probe.inputs import encode_text
from concept_probe.modeling import get_transformer_layers
from concept_probe.reading import READ_MODES, _RepCache, read_encoded, reps_from_hs_layers
from fakes import hidden_states_alone as forward_hidden_states_alone
from fakes import CharChatTok
from tiny_model import TinyCausalLM

TEXTS = ["pp x", "nn yzx", "p", "xyzzy pp n", "z"]


def _named(seed=0, name="tiny-test"):
    model = TinyCausalLM(seed=seed)
    model._concept_probe_model_id = name  # gives the cache a model identity
    return model


def _hidden(model, tok, text, **kw):
    enc = encode_text(tok, text, **kw)
    hs = forward_hidden_states_alone(model, enc.input_ids.unsqueeze(0))
    return enc, torch.stack(hs).numpy()  # (layers, seq, dim)


# ------------------------------------------------------------------------------------- batching
@pytest.mark.parametrize("role, kw", [("raw", {}), ("user", {"system": "x y"}), ("assistant", {"prompt": "say"})])
def test_a_batch_reads_like_one_input_at_a_time(role, kw):
    model, tok = TinyCausalLM(), CharChatTok()
    one = read_texts(model, tok, TEXTS, role=role, batch_size=1, cache=False, **kw).data
    many = read_texts(model, tok, TEXTS, role=role, batch_size=4, cache=False, **kw).data
    np.testing.assert_allclose(many, one, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", [m for m in READ_MODES if not m.endswith("_tokens")])
def test_every_readout_mode_matches_the_one_input_reader(mode):
    model, tok = TinyCausalLM(), CharChatTok()
    batched = read_texts(model, tok, TEXTS, mode, role="user", system="x y", last_k=2, batch_size=3, cache=False).data
    for i, text in enumerate(TEXTS):
        enc = encode_text(tok, text, "x y", role="user")
        hs = forward_hidden_states_alone(model, enc.input_ids.unsqueeze(0))
        expected = reps_from_hs_layers(hs, enc.start, mode, 2, span_end=enc.end)
        np.testing.assert_allclose(batched[i], expected, rtol=1e-6, atol=1e-6)


def test_reading_asks_the_model_for_one_position_of_logits_only():
    seen = {}

    class Spy(TinyCausalLM):
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, logits_to_keep=0, **kw):
            seen["logits_to_keep"] = logits_to_keep
            return super().forward(input_ids, attention_mask, output_hidden_states, **kw)

    read_texts(Spy(), CharChatTok(), TEXTS[:2], cache=False)
    assert seen["logits_to_keep"] == 1


def test_non_finite_activations_stop_a_batched_read():
    model = TinyCausalLM()
    handle = get_transformer_layers(model)[1].register_forward_hook(lambda m, i, o: o * float("inf"))
    try:
        with pytest.raises(NonFiniteActivationsError, match=r"layers \[1, 2, 3\]"):
            read_texts(model, CharChatTok(), TEXTS, cache=False)
    finally:
        handle.remove()


# ------------------------------------------------------------------------------------- readouts
def test_max_first_and_tokens_readouts():
    model, tok = TinyCausalLM(), CharChatTok()
    enc, hs = _hidden(model, tok, "pp xn")
    own = hs[:, enc.start:enc.end]
    np.testing.assert_allclose(read_texts(model, tok, ["pp xn"], "text_max", cache=False).data[0], own.max(axis=1), rtol=1e-6)
    np.testing.assert_allclose(read_texts(model, tok, ["pp xn"], "text_first", cache=False).data[0], own[:, 0], rtol=1e-6)
    tokens = read_texts(model, tok, ["pp xn", "z"], "text_tokens", cache=False)
    assert tokens.data.shape == (5 + 1, 4, 16)
    assert tokens.items.tolist() == [0] * 5 + [1]
    assert tokens.positions.tolist() == [1, 2, 3, 4, 5, 1]  # after the BOS
    np.testing.assert_allclose(tokens.data[:5], own.transpose(1, 0, 2), rtol=1e-6)


def test_several_readouts_come_from_one_read():
    model, tok = TinyCausalLM(), CharChatTok()
    texts = ["pp xn", "z", "nn px"]
    both = read_texts(model, tok, texts, ["text_mean", Readout("text", "last")], cache=False)
    assert list(both) == ["text_mean", "text_last"]
    for name, reps in both.items():
        np.testing.assert_allclose(reps.data, read_texts(model, tok, texts, name, cache=False).data, rtol=1e-6)
    with pytest.raises(ValueError, match="distinct"):
        read_texts(model, tok, texts, ["text_last", Readout("text", "last")], cache=False)


def test_a_prompt_is_read_as_generation_feeds_it():
    from concept_probe import generate

    model, tok = TinyCausalLM(), CharChatTok()
    enc = encode_text(tok, "ppx", add_generation_prompt=True)
    assert tok.decode(enc.input_ids.tolist()) == "^<u>ppx|<a>"
    assert tok.decode(enc.input_ids[enc.start:enc.end].tolist()) == "ppx"
    whole = read_texts(model, tok, ["ppx"], Readout("input", "tokens"), add_generation_prompt=True, cache=False)
    assert whole.positions.tolist() == list(range(len(enc.input_ids)))  # up to the token that predicts the reply
    assert Readout("input", "last").name == "input:last"
    gens = generate(model, tok, ["ppx"], max_new_tokens=1, save=False)
    assert gens[0].token_ids[:gens[0].prompt_len].tolist() == enc.input_ids.tolist()
    with pytest.raises(ValueError, match="do not apply"):
        encode_text(tok, "ppx", role="assistant", prompt="q", add_generation_prompt=True)


def test_position_readouts_count_within_the_text():
    model, tok = TinyCausalLM(), CharChatTok()
    enc, hs = _hidden(model, tok, "ab pn", role="user", system="s")
    first = read_texts(model, tok, ["ab pn"], Readout(position=0), role="user", system="s", cache=False).data[0]
    last = read_texts(model, tok, ["ab pn"], Readout(position=-1), role="user", system="s", cache=False).data[0]
    np.testing.assert_allclose(first, hs[:, enc.start], rtol=1e-6)
    np.testing.assert_allclose(last, hs[:, enc.end - 1], rtol=1e-6)
    with pytest.raises(EmptySpanError, match="no position 9"):
        read_texts(model, tok, ["ab pn"], Readout(position=9), cache=False)


@pytest.mark.parametrize("role, kw", [("raw", {}), ("user", {"system": "s"}), ("assistant", {"prompt": "q"})])
def test_substring_readouts_find_the_word_in_every_role(role, kw):
    model, tok = TinyCausalLM(), CharChatTok()
    text = "the pp and the nn"
    enc, hs = _hidden(model, tok, text, role=role, **kw)
    at = enc.start + text.index("nn")
    got = read_texts(model, tok, [text], Readout(substring="nn"), role=role, cache=False, **kw).data[0]
    np.testing.assert_allclose(got, hs[:, at:at + 2].mean(axis=1), rtol=1e-6)
    second_the = enc.start + text.index("the", 1)
    got = read_texts(model, tok, [text], Readout(substring="the", occurrence=1, agg="first"), role=role,
                     cache=False, **kw).data[0]
    np.testing.assert_allclose(got, hs[:, second_the], rtol=1e-6)


def test_one_substring_per_text_and_missing_ones_are_skipped_or_raise():
    model, tok = TinyCausalLM(), CharChatTok()
    readout = Readout(substring=["pp", "zz", "nn"])
    with pytest.raises(EmptySpanError, match=r"texts\[1\].*'zz'"):
        read_texts(model, tok, ["a pp", "b", "nn c"], readout, cache=False)
    reps = read_texts(model, tok, ["a pp", "b", "nn c"], readout, cache=False, skip_empty=True)
    assert reps.items.tolist() == [0, 2] and list(reps.skipped) == [1]


def test_the_embedding_layer_is_asked_for_by_name():
    model, tok = TinyCausalLM(), CharChatTok()
    reps = read_texts(model, tok, ["pp x"], layers=["embeddings", 0, -1], cache=False)
    assert reps.layers == ["embeddings", 0, 3]
    enc = encode_text(tok, "pp x")
    emb = model.model.embed_tokens(enc.input_ids)[enc.start:enc.end].mean(dim=0).float().numpy()
    np.testing.assert_allclose(reps.data[0, 0], emb, rtol=1e-6)
    every = read_texts(model, tok, ["pp x"], cache=False)
    np.testing.assert_allclose(reps.data[0, 1:], every.data[0, [0, 3]], rtol=1e-6)


def test_readout_names_round_trip_and_bad_combinations_raise():
    for mode in READ_MODES:
        assert Readout._parse(mode, 3).name == mode
    with pytest.raises(ValueError, match="at most one"):
        Readout(last_k=2, position=1)
    with pytest.raises(ValueError, match="span='text'"):
        Readout(span="through_text", substring="x")
    with pytest.raises(ValueError, match="Unknown readout"):
        Readout._parse("text_median")
    with pytest.raises(ValueError, match="Unknown readout"):
        Readout._parse("assistant_all_mean")


# ----------------------------------------------------------------------------------------- cache
@pytest.fixture
def saving():
    previous = concept_probe.configure(save=True)
    yield
    concept_probe.configure(save=previous.save)


def _entries(cache_dir):
    import sqlite3

    with sqlite3.connect(os.path.join(cache_dir, "reps.sqlite")) as db:
        return db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


def test_the_cache_returns_what_a_fresh_read_returns(tmp_path):
    model, tok = _named(), CharChatTok()
    first = read_texts(model, tok, TEXTS, cache=str(tmp_path))
    assert first.n_from_cache == 0 and _entries(first.cache_dir) == len(TEXTS)
    again = read_texts(model, tok, TEXTS, cache=str(tmp_path))
    assert again.n_from_cache == len(TEXTS)
    np.testing.assert_array_equal(again.data, first.data)
    assert any(c.setting == "representations cache" for c in first.choices)


def test_cache_entries_are_per_model_identity_and_readout(tmp_path):
    tok = CharChatTok()
    read_texts(_named(), tok, TEXTS, cache=str(tmp_path))
    assert read_texts(_named(), tok, TEXTS, "text_last", cache=str(tmp_path)).n_from_cache == 0
    other = read_texts(_named(seed=1, name="tiny-other"), tok, TEXTS, cache=str(tmp_path))
    assert other.n_from_cache == 0 and other.cache_dir != read_texts(_named(), tok, TEXTS, cache=str(tmp_path)).cache_dir
    assert read_texts(_named(), tok, TEXTS, role="user", cache=str(tmp_path)).n_from_cache == 0  # other token ids


def test_float16_cache_rounds_consistently_and_falls_back_when_values_do_not_fit(tmp_path):
    model, tok = _named(), CharChatTok()
    exact = read_texts(model, tok, TEXTS[:2], cache=False).data
    first = read_texts(model, tok, TEXTS[:2], cache=str(tmp_path), cache_dtype="float16")
    again = read_texts(model, tok, TEXTS[:2], cache=str(tmp_path), cache_dtype="float16")
    np.testing.assert_array_equal(first.data, again.data)  # a fresh read returns what is stored
    np.testing.assert_allclose(first.data, exact, rtol=2e-3)
    store = _RepCache(str(tmp_path / "big"), {"model_id": "x"}, dtype="float16")
    kept = store.put("k", {"reps": np.array([[7e4]], dtype=np.float32)})
    assert kept["reps"][0, 0] == np.float32(7e4) and store.fallbacks == ["k"]
    store.close()


def test_the_session_default_saves_into_its_cache(tmp_path, saving):
    previous = concept_probe.configure(output_root=str(tmp_path))
    try:
        reps = read_texts(_named(), CharChatTok(), TEXTS[:2])
    finally:
        concept_probe.configure(output_root=previous.output_root)
    assert reps.cache_dir.startswith(str(tmp_path / "rep_cache"))


def test_nothing_is_cached_without_saving_or_without_a_model_name(tmp_path, saving):
    assert read_texts(_named(), CharChatTok(), TEXTS[:2], cache=False).cache_dir is None
    unnamed = read_texts(TinyCausalLM(), CharChatTok(), TEXTS[:2], cache=str(tmp_path))
    assert unnamed.cache_dir is None and any("no name" in c.reason for c in unnamed.choices)
    concept_probe.configure(save=False)
    assert read_texts(_named(), CharChatTok(), TEXTS[:2]).cache_dir is None


def test_read_encoded_reads_generated_ids_without_retokenizing():
    from concept_probe.inputs import Encoded

    model, tok = TinyCausalLM(), CharChatTok()
    ids = torch.tensor([ord(c) for c in "^<u>hi|<a>ppx$"])
    reps = read_encoded(model, tok, [Encoded(ids, 11, 14)], cache=False)
    hs = forward_hidden_states_alone(model, ids.unsqueeze(0))
    np.testing.assert_allclose(reps.data[0], reps_from_hs_layers(hs, 11, "text_mean", 12, 14), rtol=1e-6)
