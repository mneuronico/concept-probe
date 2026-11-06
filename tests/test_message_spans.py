"""Encoding (layer 1): the token span of every message (entry 28) and the character offsets of a
text's tokens, which readouts at a word use."""
import pytest

from concept_probe import encode_text, preview_input
from concept_probe.inputs import message_spans
from concept_probe.record import Choice, announce
from concept_probe import configure
from fakes import CharChatTok


def _span_text(tok, ids, span):
    return tok.decode(ids[span[0]:span[1]])


def test_every_message_has_its_span():
    tok = CharChatTok()
    msgs = [{"role": "system", "content": "be calm"}, {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)[0].tolist()
    spans = message_spans(tok, msgs, add_generation_prompt=True, input_ids=ids)
    assert [_span_text(tok, ids, sp) for sp in spans] == ["be calm", "hi there", "ok", "more"]


def test_spans_survive_a_template_that_strips_and_repeated_contents():
    tok = CharChatTok(strip_content=True)
    msgs = [{"role": "user", "content": "  x "}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "x"}]
    ids = tok.apply_chat_template(msgs)[0].tolist()
    spans = message_spans(tok, msgs, input_ids=ids)
    assert [_span_text(tok, ids, sp) for sp in spans] == ["x", "x", "x"]
    assert len(set(spans)) == 3  # each in its own turn


def test_spans_are_none_when_they_cannot_be_located():
    msgs = [{"role": "user", "content": "hi"}]
    assert message_spans(CharChatTok(offsets=False), msgs) == [None]
    assert message_spans(CharChatTok(), msgs, input_ids=[1, 2, 3]) == [None]  # not this encoding


def test_preview_input_lists_the_messages():
    tok = CharChatTok()
    gen = preview_input(tok, "hello", system="S", generation=True)
    assert [(m["role"], m["text"]) for m in gen["message_spans"]] == [("system", "S"), ("user", "hello")]
    user = preview_input(tok, "hello", role="user", system="S")
    assert [m["text"] for m in user["message_spans"]] == ["S", "hello"]
    reply = preview_input(tok, "pp x", role="assistant", system="S", prompt="q")
    assert [(m["role"], m["text"]) for m in reply["message_spans"]] == [("system", "S"), ("user", "q"), ("assistant", "pp x")]
    assert preview_input(tok, "pp x")["message_spans"] == []


@pytest.mark.parametrize("role, kw", [("raw", {}), ("user", {"system": "S"}), ("assistant", {"prompt": "q"})])
def test_encoded_texts_carry_each_tokens_characters(role, kw):
    tok = CharChatTok()
    enc = encode_text(tok, "ab cd", role=role, **kw)
    assert enc.text == "ab cd"
    assert enc.offsets == [(i, i + 1) for i in range(5)]
    assert tok.decode(enc.input_ids[enc.start:enc.end]) == "ab cd"


def test_a_stripping_template_reports_the_text_it_kept():
    enc = encode_text(CharChatTok(strip_content=True), " ab ", role="user")
    assert enc.text == "ab" and enc.offsets == [(0, 1), (1, 2)]


def test_choices_print_once_and_always_reach_the_log(capsys):
    logged = []

    class Log:
        def log(self, event, payload):
            logged.append((event, payload))

    choice = Choice("dtype", "auto", "float16", "a test GPU without bfloat16")
    announce([choice], console=True, logger=Log())
    announce([choice], console=True, logger=Log())
    printed = capsys.readouterr().out
    assert printed.count("dtype: 'auto' -> 'float16'") == 1
    assert [e for e, _ in logged] == ["choice", "choice"] and logged[0][1]["chosen"] == "float16"


def test_configure_rejects_unknown_settings():
    with pytest.raises(ValueError, match="Unknown settings"):
        configure(saev=False)
