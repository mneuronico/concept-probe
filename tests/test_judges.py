"""Judges: a rubric on any dimension, answered by a model behind an API (any of several providers,
no network here: the requests are checked and answered by a fake) or by a model loaded here."""
import io
import json
import urllib.error

import numpy as np
import pytest

from concept_probe import COHERENCE, APIJudge, LocalJudge, Rubric, answer_logprobs, rate_saved
from concept_probe import judges
from fakes import CharChatTok
from tiny_model import TinyCausalLM

TONE = Rubric("tone", "Is this text friendly?", {"yes": 1.0, "no": 0.0}, guide="yes: friendly. no: not friendly.")


def _fake_service(calls, answer=lambda item: "YES", style="openai"):
    def post(url, headers, payload, timeout):
        calls.append((url, headers, payload))
        user = payload["messages"][-1]["content"]
        content = json.dumps([{"id": x["id"], "answer": answer(x)} for x in json.loads(user)])
        if style == "anthropic":
            return {"content": [{"type": "text", "text": content}]}
        return {"choices": [{"message": {"content": content}}]}

    return post


@pytest.mark.parametrize("provider, url", [
    ("openai", "https://api.openai.com/v1/chat/completions"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
    ("groq", "https://api.groq.com/openai/v1/chat/completions"),
])
def test_openai_style_providers_get_the_rubric_and_the_texts(provider, url, monkeypatch):
    calls = []
    monkeypatch.setattr(judges, "_post_json", _fake_service(calls))
    judge = APIJudge(provider, "m", api_key="secret", texts_per_request=2)
    assert judge.rate(["a", "b", "c"], TONE) == ["yes", "yes", "yes"]  # answers are matched ignoring case
    assert len(calls) == 2 and all(c[0] == url for c in calls)
    headers, payload = calls[0][1], calls[0][2]
    assert headers == {"Authorization": "Bearer secret"} and payload["model"] == "m" and payload["temperature"] == 0.0
    system = payload["messages"][0]["content"]
    assert "Is this text friendly?" in system and "yes, no" in system and "friendly. no" in system
    assert json.loads(payload["messages"][1]["content"]) == [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}]
    assert "secret" not in json.dumps(judge.describe())


def test_anthropic_uses_its_own_api(monkeypatch):
    calls = []
    monkeypatch.setattr(judges, "_post_json", _fake_service(calls, lambda x: "no", style="anthropic"))
    judge = APIJudge("anthropic", "claude-model", api_key="k")
    np.testing.assert_array_equal(judge.score(["x", "y"], TONE), [0.0, 0.0])
    url, headers, payload = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers == {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    assert "Is this text friendly?" in payload["system"] and payload["messages"][0]["role"] == "user"


def test_local_servers_and_other_services(monkeypatch):
    calls = []
    monkeypatch.setattr(judges, "_post_json", _fake_service(calls))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    APIJudge("ollama", "llama3").rate(["a"], TONE)  # no key needed
    APIJudge("openai-compatible", "m", base_url="http://localhost:8000/v1/").rate(["a"], TONE)
    assert [c[0] for c in calls] == ["http://localhost:11434/v1/chat/completions",
                                     "http://localhost:8000/v1/chat/completions"]
    assert calls[0][1] == {} and calls[1][1] == {}
    with pytest.raises(ValueError, match="base_url"):
        APIJudge("openai-compatible", "m")
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        APIJudge("openai", "m")
    with pytest.raises(ValueError, match="model is required"):
        APIJudge("openai", "", api_key="k")
    with pytest.raises(ValueError, match="provider"):
        APIJudge("somewhere", "m", api_key="k")


def test_an_explicit_env_file_wins_over_the_environment(tmp_path, monkeypatch):
    env = tmp_path / "keys.env"
    env.write_text("GROQ_API_KEY=from-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "from-environment")
    calls = []
    monkeypatch.setattr(judges, "_post_json", _fake_service(calls))
    APIJudge("groq", "m", env_path=str(env)).rate(["a"], TONE)
    APIJudge("groq", "m").rate(["a"], TONE)
    APIJudge("groq", "m", api_key="explicit", env_path=str(env)).rate(["a"], TONE)
    assert [c[1]["Authorization"] for c in calls] == ["Bearer from-file", "Bearer from-environment", "Bearer explicit"]


def test_answers_outside_the_rubric_or_missing_raise(monkeypatch):
    monkeypatch.setattr(judges, "_post_json", _fake_service([], lambda x: "maybe"))
    with pytest.raises(ValueError, match="'maybe'"):
        APIJudge("openai", "m", api_key="k").rate(["a"], TONE)

    def drops_one(url, headers, payload, timeout):
        return {"choices": [{"message": {"content": '[{"id": 0, "answer": "yes"}]'}}]}

    monkeypatch.setattr(judges, "_post_json", drops_one)
    with pytest.raises(RuntimeError, match="no answer for 1 texts"):
        APIJudge("openai", "m", api_key="k").rate(["a", "b"], TONE)


def test_rate_limits_are_retried(monkeypatch):
    attempts = []

    def flaky(url, headers, payload, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(url, 429, "slow down", {}, io.BytesIO(b"rate limited"))
        return _fake_service([])(url, headers, payload, timeout)

    monkeypatch.setattr(judges, "_post_json", flaky)
    monkeypatch.setattr(judges.time, "sleep", lambda s: None)
    assert APIJudge("openai", "m", api_key="k").rate(["a"], TONE) == ["yes"] and len(attempts) == 2

    def denied(url, headers, payload, timeout):
        raise urllib.error.HTTPError(url, 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(judges, "_post_json", denied)
    with pytest.raises(RuntimeError, match="401.*API key"):
        APIJudge("openai", "m", api_key="k").rate(["a"], TONE)


def test_a_local_judge_reads_the_probability_of_each_answer():
    model, tok = TinyCausalLM(), CharChatTok()
    rubric = Rubric("has_p", "Does the text say p?", {"p": 1.0, "n": 0.0})
    judge = LocalJudge(model, tok, name="tiny")
    texts = ["pp", "nn x"]
    probs = judge.probabilities(texts, rubric)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    direct = answer_logprobs(model, tok, [rubric.prompt(t) for t in texts], ["p", "n"], save=False).probs()
    np.testing.assert_allclose(probs, direct)
    np.testing.assert_allclose(judge.score(texts, rubric), probs[:, 0])  # the expected value of 1 and 0
    assert judge.rate(texts, rubric) == [["p", "n"][k] for k in probs.argmax(axis=1)]
    assert judge.describe() == {"judge": "local", "model": "tiny", "system": None}


def test_a_saved_batch_is_rated_by_index(tmp_path, monkeypatch):
    (tmp_path / "scores.jsonl").write_text("\n".join(json.dumps({"text": t}) for t in ["ok", "bad"]) + "\n",
                                           encoding="utf-8")
    monkeypatch.setattr(judges, "_post_json", _fake_service([], lambda x: "NONSENSE" if x["text"] == "bad" else "COHERENT"))
    out = rate_saved(str(tmp_path), APIJudge("openai", "m", api_key="k"))
    saved = json.loads((tmp_path / "coherence_rating.json").read_text(encoding="utf-8"))
    assert saved == out and [(r["index"], r["rating"], r["value"]) for r in saved] == [(0, "COHERENT", 1.0),
                                                                                       (1, "NONSENSE", 0.0)]
    assert COHERENCE.match(" partially_coherent. ") == "PARTIALLY_COHERENT"


# ---------------------------------------------------------------------------- agents and critiques
def test_an_agent_judge_writes_a_task_and_checks_the_answers(tmp_path):
    from concept_probe import AgentJudge, AnswersPending

    judge = AgentJudge(str(tmp_path))
    texts = ["hello friend", "go away", "fine"]
    with pytest.raises(AnswersPending) as pending:
        judge.rate(texts, TONE)
    path = pending.value.path
    assert json.loads(open(f"{path}/items.jsonl", encoding="utf-8").readline()) == {"id": 0, "text": "hello friend"}
    task = open(f"{path}/TASK.md", encoding="utf-8").read()
    assert "Is this text friendly?" in task and '"yes", "no"' in task and "ids 0 to 2" in task
    with open(f"{path}/answers.jsonl", "w", encoding="utf-8") as f:
        f.write('{"id": 0, "answer": "Yes"}\n{"id": 1, "answer": "maybe"}\n')
    with pytest.raises(ValueError, match="'maybe' is not an allowed answer.*2 ids have no answer"):
        judge.rate(texts, TONE)
    with open(f"{path}/answers.jsonl", "w", encoding="utf-8") as f:
        f.write('{"id": 0, "answer": "Yes"}\n{"id": 1, "answer": "no"}\n{"id": 2, "answer": "yes"}\n')
    np.testing.assert_array_equal(judge.score(texts, TONE), [1.0, 0.0, 1.0])
    assert judge.task_dir(texts[:2], TONE) != judge.task_dir(texts, TONE)  # other texts, another task


POS = ["I love this day!", "What a great gift!", "We won!", "Such a lovely evening!"]
NEG = ["I hate this day.", "What a bad gift.", "We lost.", "Such a dreadful evening."]


def test_a_critique_task_round_trip_gives_checks_for_the_audit(tmp_path):
    from concept_probe import compare_poles, critique_task, read_critique

    folder = critique_task(POS, NEG, str(tmp_path), concept="positive vs negative feeling", paired=True, n=4)
    sample = json.load(open(f"{folder}/sample.json", encoding="utf-8"))
    assert len(sample["pairs"]) == 4
    assert "positive vs negative feeling" in open(f"{folder}/TASK.md", encoding="utf-8").read()
    with pytest.raises(FileNotFoundError, match="TASK.md"):
        read_critique(folder)
    answer = {"differences": [{"name": "exclamation", "description": "positives end in !", "more_in": "pos",
                               "examples": ["pos:0", "neg:0"], "regex": "!"}],
              "formulaic": [{"description": "What a ...", "examples": ["pos:1"]}],
              "mislabeled": [], "suggestions": ["vary the punctuation"]}
    json.dump(answer, open(f"{folder}/critique.json", "w", encoding="utf-8"))
    critique = read_critique(folder)
    assert critique.checks()["exclamation"]("Wow!!") == 2.0 and critique.source["by"] == "agent"
    rows = compare_poles(POS, NEG, paired=True, checks=critique.checks())
    assert rows[0]["feature"] == "exclamation"
    answer["differences"][0]["examples"] = ["pos:99"]
    answer["differences"][0]["regex"] = "("
    json.dump(answer, open(f"{folder}/critique.json", "w", encoding="utf-8"))
    with pytest.raises(ValueError, match="unknown ids.*does not compile"):
        read_critique(folder)


def test_a_judge_can_write_the_critique_directly(monkeypatch):
    from concept_probe import critique_dataset

    reply = {"differences": [], "formulaic": [], "mislabeled": [{"id": "neg:2", "reason": "not negative"}],
             "suggestions": []}
    calls = []

    def fake_post(url, headers, payload, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "Here it is: " + json.dumps(reply)}}]}

    monkeypatch.setattr(judges, "_post_json", fake_post)
    critique = critique_dataset(POS, NEG, APIJudge("openai", "m", api_key="k"), concept="valence", n=4)
    assert critique.mislabeled == reply["mislabeled"] and "valence" in calls[0]["messages"][0]["content"]
