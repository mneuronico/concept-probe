"""Model-free tests that templates writing today's date get a fixed one."""
import pytest
import torch

import concept_probe
from concept_probe.modeling import apply_chat, apply_tokenizer_overrides

MESSAGES = [{"role": "user", "content": "x"}]


class _Tok:
    def __init__(self, template):
        self.chat_template = template
        self.kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=True, return_tensors=None, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2]])


DATED = "Today Date: {{ strftime_now('%d %b %Y') }}"


def test_a_dated_template_gets_the_default_fixed_date():
    tok = _Tok(DATED)
    apply_chat(tok, MESSAGES, add_generation_prompt=True)
    assert tok.kwargs["strftime_now"]("%d %b %Y") == "26 Jul 2024"


def test_the_date_follows_the_config_and_null_restores_the_real_date():
    tok = _Tok(DATED)
    apply_tokenizer_overrides(tok, {"chat_template_date": "2025-01-31"})
    apply_chat(tok, MESSAGES, add_generation_prompt=True)
    assert tok.kwargs["strftime_now"]("%d %b %Y") == "31 Jan 2025"
    apply_tokenizer_overrides(tok, {"chat_template_date": None})
    apply_chat(tok, MESSAGES, add_generation_prompt=True)
    assert tok.kwargs == {}


def test_templates_without_a_date_get_no_extra_variables():
    tok = _Tok("{{ messages }}")
    apply_chat(tok, MESSAGES, add_generation_prompt=True)
    assert tok.kwargs == {}


def test_a_malformed_date_is_rejected():
    with pytest.raises(ValueError, match="chat_template_date"):
        apply_tokenizer_overrides(_Tok(DATED), {"chat_template_date": "26/07/2024"})


def test_the_default_config_pins_the_date():
    assert concept_probe.load_defaults()["model"]["chat_template_date"] == "2024-07-26"
