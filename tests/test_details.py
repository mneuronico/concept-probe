"""Details that make every reading, choice and number what it says: the span of a text, which tokens a
readout pools, how the layer is chosen, which layers steer and with what dose, the files and plots."""
import json

import numpy as np
import pytest
import torch

from concept_probe import ConceptProbe, EmptySpanError, Encoded, generate, read_encoded, read_texts
from concept_probe.inputs import _encode_text_for_read
from concept_probe.reading import reps_from_hs_layers
from concept_probe.stats import _select_best_layer
from concept_probe.utils import json_dump
from concept_probe.visuals import _token_for_display, plot_sweep
from fakes import CharChatTok, CharModel, fake_forward


# ----------------------------------------------------------------- text spans
def test_user_role_span_is_the_text_itself():
    tok = CharChatTok()
    ids, start, end = _encode_text_for_read(tok, "ppx", "sys", role="user")
    assert tok.decode(ids[0]) == "^<s>sys|<u>ppx|"
    assert tok.decode(ids[0, start:end]) == "ppx"


def test_user_role_span_when_the_template_strips_the_text():
    tok = CharChatTok(strip_content=True)
    ids, start, end = _encode_text_for_read(tok, "  ppx ", "sys", role="user")
    assert tok.decode(ids[0, start:end]) == "ppx"


def test_user_role_span_of_a_message_list_is_its_last_message():
    tok = CharChatTok()
    msgs = [{"role": "user", "content": "nn"}, {"role": "assistant", "content": "pp"}, {"role": "user", "content": "px"}]
    ids, start, end = _encode_text_for_read(tok, msgs, "sys", role="user")
    assert tok.decode(ids[0, start:end]) == "px"


def test_a_text_that_also_occurs_in_the_template_is_found_in_its_own_turn():
    tok = CharChatTok()
    ids, start, end = _encode_text_for_read(tok, "s", "s", role="user")  # "s" is also the system text
    assert (start, end) == (len("^<s>s|<u>"), len("^<s>s|<u>s"))


def test_unknown_span_makes_text_readouts_refuse_instead_of_reading_the_template():
    ids, start, end = _encode_text_for_read(CharChatTok(offsets=False), "ppx", "sys", role="user")
    assert start is None and end is None
    hs = fake_forward(None, ids)
    with pytest.raises(ValueError, match="through_text_"):
        reps_from_hs_layers(hs, start, "text_mean", 4, span_end=end)
    reps_from_hs_layers(hs, start, "through_text_mean", 4)  # whole-sequence readouts still work


def test_text_readouts_pool_over_the_span_only():
    ids, start, end = _encode_text_for_read(CharChatTok(), "ppx", "sys", role="user")
    hs = fake_forward(None, ids)
    np.testing.assert_allclose(reps_from_hs_layers(hs, start, "text_mean", 4, span_end=end)[0], [2 / 3, 1])
    # the last text token is 'x', not the end-of-turn '|'
    np.testing.assert_allclose(reps_from_hs_layers(hs, start, "text_last", 4, span_end=end)[0], [0, 1])
    np.testing.assert_allclose(reps_from_hs_layers(hs, start, "text_last_k", 2, span_end=end)[0], [0.5, 1])


def test_through_text_pooling_depends_on_length_and_text_pooling_does_not():
    # The template prefix is identical in every input, so its weight in a sequence mean is 1/length.
    tok = CharChatTok()
    seq, own = [], []
    for text in ("pp", "pppppp"):
        ids, start, end = _encode_text_for_read(tok, text, "sys", role="user")
        hs = fake_forward(None, ids)
        seq.append(reps_from_hs_layers(hs, None, "through_text_mean", 4)[0, 0])
        own.append(reps_from_hs_layers(hs, start, "text_mean", 4, span_end=end)[0, 0])
    assert seq[0] < seq[1]
    np.testing.assert_allclose(own, [1.0, 1.0])


def test_through_text_modes_stop_where_the_item_ends():
    hs = [torch.tensor([[1.0], [2.0], [3.0], [9.0]])]  # the last position is a trailing EOS
    assert reps_from_hs_layers(hs, 1, "through_text_last", 1, span_end=3)[0, 0] == 3.0
    assert reps_from_hs_layers(hs, 1, "through_text_mean", 1, span_end=3)[0, 0] == pytest.approx(2.0)
    assert reps_from_hs_layers(hs, 1, "through_text_last_k", 2, span_end=3)[0, 0] == pytest.approx(2.5)


# ----------------------------------------------------------------- generated completions
def test_completions_are_read_without_their_end_of_sequence_token_and_with_or_without_their_system():
    model, tok = CharModel(), CharChatTok()
    gens = generate(model, tok, ["hi", "yo"], system="S", max_new_tokens=4, save=False)
    as_generated = read_encoded(model, tok, gens.encodings(), cache=False)
    np.testing.assert_allclose(as_generated.data[:, 0, 0], [1.0, 1.0])  # 'pp' only; '$' would make it 2/3
    alone = gens.encodings(system=None)
    assert tok.decode(alone[0].input_ids.tolist()) == "^<u>hi|<a>pp"
    assert tok.decode(gens.encodings()[0].input_ids.tolist()).startswith("^<s>S|<u>hi|<a>")
    assert gens.texts == ["pp", "pp"] and gens.prompts == ["hi", "yo"]


def test_an_empty_item_is_left_out_when_asked_and_named_otherwise():
    model, tok = CharModel(), CharChatTok()
    empty = Encoded(torch.tensor([ord(c) for c in "^<u>q|<a>"]), 9, 9)
    full = Encoded(torch.tensor([ord(c) for c in "^<u>q|<a>pp"]), 9, 11)
    out = read_encoded(model, tok, [full, empty], skip_empty=True, cache=False)
    assert out.read_items == [0] and 1 in out.skipped
    with pytest.raises(EmptySpanError, match=r"items\[1\]"):
        read_encoded(model, tok, [full, empty], cache=False)
    with pytest.raises(EmptySpanError, match=r"texts\[1\]"):
        read_texts(model, tok, ["pp", ""], cache=False)


# ----------------------------------------------------------------- choosing the layer
def test_p_value_selection_skips_layers_with_reversed_separation():
    # layer 0 has the smallest p but separates the eval sets the wrong way (d < 0)
    assert _select_best_layer([-3.0, 0.5, 1.0], [1e-9, 0.2, 0.01], [0, 1, 2], "p_value") == 2


def test_p_value_ties_go_to_the_larger_effect():
    assert _select_best_layer([2.0, 5.0, 3.0], [0.0, 0.0, 0.0], [0, 1, 2], "p_value") == 1


def test_selection_respects_the_search_range():
    assert _select_best_layer([5.0, 1.0, 2.0], [0.1, 0.1, 0.1], [1, 2], "effect_size") == 2


def test_p_value_selection_without_a_positive_layer_raises():
    with pytest.raises(ValueError, match="d > 0"):
        _select_best_layer([-1.0, -2.0], [0.01, 0.01], [0, 1], "p_value")


@pytest.mark.parametrize("bad", ["pvalue", "p", "Effect_size"])
def test_an_unknown_layer_selection_raises(bad):
    with pytest.raises(ValueError, match="select_layer_by"):
        _select_best_layer([1.0], [0.1], [0], bad)


# ----------------------------------------------------------------- steering layers and units
def _probe(num_layers=6, layer=3):
    v = np.zeros((num_layers, 4), dtype=np.float32)
    v[:, 0] = 1.0
    probe = ConceptProbe.from_directions(v, layer, identity="m", save=False)
    probe.steer_sd, probe.resid_norm = 0.5, np.arange(1, num_layers + 1, dtype=float)
    return probe


def _alphas(add, num_layers=6):
    add.prepare(num_layers)
    return add.alpha_per_layer


def test_by_default_only_the_probe_layer_is_steered_with_the_whole_dose():
    probe = _probe()
    assert _alphas(probe.steering(2.0)) == {3: pytest.approx(8.0)}  # in residual norms by default (layer 3's is 4)
    assert _alphas(probe.steering(2.0, "raw", layers=[1, -1])) == {1: 2.0, 5: 2.0}  # each listed layer, the whole dose
    assert _alphas(probe.steering(2.0, "raw", layers=[1, 5], distribute=True)) == {1: 1.0, 5: 1.0}  # only when asked


def test_a_window_is_clipped_at_the_edges_and_says_so(capsys):
    import concept_probe

    probe = _probe(layer=1)
    previous = concept_probe.configure(console=True)
    try:
        layers = probe.steer_layer_list(window=2)
    finally:
        concept_probe.configure(console=previous.console)
    assert layers == [0, 1, 2, 3] and "reaches past the model" in capsys.readouterr().out
    with pytest.raises(ValueError, match="not both"):
        probe.steer_layer_list(layers=[1], window=1)
    with pytest.raises(ValueError, match="radius"):
        probe.steer_layer_list(window=-1)


def test_units_scale_the_dose_and_need_their_reference():
    probe = _probe()
    assert _alphas(probe.steering(2.0, "sd")) == {3: pytest.approx(1.0)}
    assert _alphas(probe.steering(0.5, "resid_norm", layers=[0, 3])) == {0: pytest.approx(0.5), 3: pytest.approx(2.0)}
    with pytest.raises(ValueError, match="unit must be one of"):
        probe.steering(1.0, "Sigma")
    probe.steer_sd = None
    with pytest.raises(ValueError, match="steer_sd"):
        probe.steering(1.0, "sd")
    assert _alphas(probe.steering(0.0, "sd")) == {3: 0.0}  # dose 0 needs no unit
    probe.resid_norm = None
    with pytest.raises(ValueError, match="resid_norm"):
        probe.steering(1.0, "resid_norm")
    with pytest.raises(ValueError):
        probe.steering(1.0, layers=[6])


def test_each_steered_layer_adds_its_own_vector():
    probe = _probe()
    probe.steer_directions = np.eye(6, 4, dtype=np.float32)
    add = probe.steering(1.0, layers=[0, 2])
    add.prepare(6)
    np.testing.assert_allclose(add._by_layer[0][1].numpy(), probe.steer_directions[0])
    np.testing.assert_allclose(add._by_layer[2][1].numpy(), probe.steer_directions[2])


# ----------------------------------------------------------------- generation settings
def test_generation_is_greedy_by_default_and_sampling_settings_are_recorded_or_warned():
    model, tok = CharModel(), CharChatTok()
    gens = generate(model, tok, ["hi"], max_new_tokens=2, save=False)
    assert gens.settings["greedy"] is True and gens.settings["temperature"] is None
    with pytest.warns(UserWarning, match="temperature, seed do not apply"):
        generate(model, tok, ["hi"], max_new_tokens=2, temperature=0.5, seed=1, save=False)


# ----------------------------------------------------------------- files and plots
def test_json_takes_sets_numpy_bools_and_arrays(tmp_path):
    json_dump(str(tmp_path / "x.json"), {"expected": {"b", "a"}, "correct": np.bool_(True), "v": np.arange(2)})
    assert json.loads((tmp_path / "x.json").read_text()) == {"correct": True, "expected": ["a", "b"], "v": [0, 1]}


@pytest.mark.parametrize("token", ["<bos>", "<eos>", "<pad>", "<|endoftext|>"])
def test_special_tokens_are_shown_as_they_are(token):
    assert _token_for_display(token) == token


def test_the_sweep_plot_can_show_the_selection_half(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "sweep.png"
    assert plot_sweep([0.1, 0.5], [0.5, 0.01], str(out), best_layer=1, selection_d=[0.2, 0.9])
    assert out.exists()


# ------------------------------------------------------------------------------------- progress
def test_progress_rewrites_one_line_or_prints_three(capsys, monkeypatch):
    from concept_probe import record

    monkeypatch.setattr(record, "_rewritable_output", lambda: False)
    report = record.progress_line("Generated condition", True)
    for done in range(1, 8):
        report(done, 7)
    assert [line.split()[-1] for line in capsys.readouterr().out.splitlines()] == ["1/7", "4/7", "7/7"]
    monkeypatch.setattr(record, "_rewritable_output", lambda: True)
    report = record.progress_line("Generated condition", True)
    for done in range(1, 4):
        report(done, 3)
    out = capsys.readouterr().out
    assert out.count("\r") == 3 and out.count("\n") == 1 and out.endswith("3/3\n")
    record.progress_line("x", False)(1, 1)
    assert capsys.readouterr().out == ""
