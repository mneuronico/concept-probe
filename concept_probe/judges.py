"""Judges: another model rates texts on any dimension you describe.

A :class:`Rubric` says what to ask about each text and which answers count, each with a number:
``Rubric("positivity", "How positive is the feeling this text expresses?", {"1": 1, "2": 2, "3": 3,
"4": 4, "5": 5}, guide="1 very negative, 3 neutral, 5 very positive")``. :data:`COHERENCE` is one,
ready made. A judge answers a rubric for many texts:

- :class:`APIJudge`: a model behind an API. ``provider`` is ``"openai"``, ``"anthropic"``,
  ``"gemini"``, ``"openrouter"``, ``"groq"``, ``"ollama"`` (a local Ollama server) or
  ``"openai-compatible"`` with a ``base_url`` (vLLM, LM Studio, llama.cpp's server, and most other
  services). It sends the texts to that service (nothing is sent unless you build one), several
  texts per request, and reads its answers. The library calls the APIs with Python's standard
  library, so no client package is needed. ``model`` is always yours to choose.
- :class:`LocalJudge`: a model loaded here (transformers). It reads the probability of each answer
  after the question, with no sampling, and nothing leaves the machine; its score is the expected
  value of the answers' numbers.
- :class:`AgentJudge`: whoever answers a task written to a folder, for example an agent (or a
  subagent it delegates to) working with this library: no API and no model here. The task file
  says what to do, how to read the input and exactly how to write the answer.

Every judge has ``rate(texts, rubric)`` (the answers), ``score(texts, rubric)`` (their numbers) and
``measure(rubric)``, a function ``texts -> numbers`` to pass to
:func:`concept_probe.evaluate_steering` as a measure. A judge's answers are another model's
labels, not ground truth: check a sample before relying on them.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------------------- rubrics
@dataclass(frozen=True)
class Rubric:
    """What to ask about each text: ``question``, the ``answers`` allowed (each with the number it
    counts as), and optionally a ``guide`` saying what the answers mean. ``name`` names the measure
    and the files it is saved in."""

    name: str
    question: str
    answers: Dict[str, float]
    guide: str = ""

    def __post_init__(self):
        if not self.answers:
            raise ValueError("a rubric needs at least one answer")
        if len({a.casefold() for a in self.answers}) != len(self.answers):
            raise ValueError("a rubric's answers must differ in more than their case")

    def prompt(self, text: str) -> str:
        """The question about one text, as a :class:`LocalJudge` asks it."""
        guide = f"\n{self.guide}" if self.guide else ""
        return (f"{self.question}{guide}\n\nText:\n\"\"\"\n{text}\n\"\"\"\n\n"
                f"Answer with exactly one of: {', '.join(self.answers)}.")

    def match(self, answer: Any) -> Optional[str]:
        """The allowed answer ``answer`` stands for (case and surrounding space ignored), or None."""
        a = str(answer).strip().strip(".").casefold()
        return next((k for k in self.answers if k.casefold() == a), None)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "question": self.question, "answers": dict(self.answers), "guide": self.guide}


COHERENCE = Rubric(
    "coherence",
    "How coherent is this text?",
    {"COHERENT": 1.0, "PARTIALLY_COHERENT": 0.5, "NONSENSE": 0.0},
    guide=("COHERENT: syntactically well formed and understandable. PARTIALLY_COHERENT: some grammatical "
           "or syntactic errors, but still understandable. NONSENSE: not understandable as meaningful text."),
)


class Judge:
    """What every judge offers; subclasses implement :meth:`rate` (and :class:`LocalJudge` its own
    :meth:`score`)."""

    def rate(self, texts: Sequence[str], rubric: Rubric) -> List[str]:
        raise NotImplementedError

    def score(self, texts: Sequence[str], rubric: Rubric) -> np.ndarray:
        """Each text's answer as its number."""
        return np.array([np.nan if a is None else rubric.answers[a] for a in self.rate(texts, rubric)],
                        dtype=np.float64)

    def as_measure(self, rubric: Rubric) -> Callable[[List[str]], np.ndarray]:
        """A measure for :func:`concept_probe.evaluate_steering`: ``texts -> numbers``."""

        def judged(texts: List[str]) -> np.ndarray:
            return self.score(texts, rubric)

        judged.__name__ = f"judge_{rubric.name}"
        return judged

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> str:
        """A free-text reply to ``user`` after ``system`` (used by :mod:`concept_probe.critique`)."""
        raise NotImplementedError(f"{type(self).__name__} does not write free text")


# ------------------------------------------------------------------------------------- API judges
#: provider -> (base URL, the environment variable with its key, the API it speaks)
PROVIDERS: Dict[str, tuple] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "openai"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "anthropic"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "openai"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai"),
    "ollama": ("http://localhost:11434/v1", None, "openai"),
    "openai-compatible": (None, None, "openai"),
}


def _read_env_file(env_path: Optional[str]) -> Dict[str, str]:
    """KEY=VALUE pairs of ``env_path`` (none without a path). The process environment is not modified."""
    if not env_path:
        return {}
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_json_list(content: str) -> List[Any]:
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end <= start:
        raise ValueError(f"the judge did not answer with a JSON list: {content[:200]!r}")
    data = json.loads(content[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("the judge did not answer with a JSON list")
    return data


class APIJudge(Judge):
    """A judge behind an API (see the module's docstring for the providers).

    The key is ``api_key``, else the provider's variable (``api_key_env`` overrides its name) in the
    file ``env_path`` if one is given, else in the environment; no ``.env`` file is read unless
    passed, and the key is never saved or shown. Up to ``texts_per_request`` texts go in one
    request, as a JSON list, and the judge returns one answer per text; ``temperature`` 0 makes the
    answers as repeatable as the service allows. Failed requests (rate limits, server errors) are
    retried ``retries`` times.
    """

    def __init__(self, provider: str, model: str, *, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 api_key_env: Optional[str] = None, env_path: Optional[str] = None, texts_per_request: int = 8,
                 temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 120.0, retries: int = 2):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}; got {provider!r}")
        if not model:
            raise ValueError("model is required: the judge's model id at that provider")
        default_url, default_env, self.api = PROVIDERS[provider]
        self.provider, self.model = provider, str(model)
        self.base_url = (base_url or default_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("provider 'openai-compatible' needs base_url (e.g. http://localhost:8000/v1)")
        env_name = api_key_env or default_env
        self._key = api_key or _read_env_file(env_path).get(env_name or "") or (os.environ.get(env_name) if env_name
                                                                              else None)
        if env_name and not self._key and provider != "ollama":
            raise EnvironmentError(f"no API key for {provider}: pass api_key, or set {env_name} in the environment "
                                   "or in a file given as env_path")
        if int(texts_per_request) < 1:
            raise ValueError("texts_per_request must be at least 1")
        self.texts_per_request, self.temperature = int(texts_per_request), float(temperature)
        self.max_tokens, self.timeout, self.retries = int(max_tokens), float(timeout), int(retries)

    def to_dict(self) -> Dict[str, Any]:
        return {"judge": "api", "provider": self.provider, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature}

    def complete(self, system: str, user: str) -> str:
        """One request: the model's reply to ``user`` after ``system``."""
        if self.api == "anthropic":
            url = f"{self.base_url}/messages"
            headers = {"x-api-key": self._key, "anthropic-version": "2023-06-01"}
            payload = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                       "system": system, "messages": [{"role": "user", "content": user}]}
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
            payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        for attempt in range(self.retries + 1):
            try:
                data = _post_json(url, headers, payload, self.timeout)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:300]
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                hint = " (check the API key and the model's name)" if exc.code in (401, 403, 404) else ""
                raise RuntimeError(f"{self.provider} API error {exc.code}{hint}: {body}") from None
            except urllib.error.URLError as exc:
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"could not reach {url}: {exc.reason}") from None
        if self.api == "anthropic":
            return "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        return data["choices"][0]["message"]["content"]

    def _system(self, rubric: Rubric) -> str:
        guide = f"\n{rubric.guide}" if rubric.guide else ""
        return (f"You rate texts. For each text you receive, answer this question: {rubric.question}{guide}\n"
                f"The only allowed answers are: {', '.join(rubric.answers)}.\n"
                "You will receive a JSON list of objects with an id (an integer) and a text. Return only a JSON "
                "list with one object per text, each exactly {\"id\": <the text's id>, \"answer\": <an allowed "
                "answer>}, with no other text and no code fences.")

    def rate(self, texts: Sequence[str], rubric: Rubric) -> List[Optional[str]]:
        """Each text's answer. A text the judge answered with something not allowed, or not at all,
        gets None (and a NaN score); how many and why is warned, and the rest are kept."""
        texts = [str(t) for t in texts]
        out: Dict[int, str] = {}
        problems: List[str] = []
        system = self._system(rubric)
        for start in range(0, len(texts), self.texts_per_request):
            chunk = [{"id": i, "text": texts[i]} for i in range(start, min(len(texts), start + self.texts_per_request))]
            try:
                answered = _parse_json_list(self.complete(system, json.dumps(chunk, ensure_ascii=False)))
            except (ValueError, json.JSONDecodeError) as exc:
                problems.append(f"texts {chunk[0]['id']}-{chunk[-1]['id']}: {exc}")
                continue
            for item in answered:
                if not isinstance(item, dict) or "id" not in item:
                    problems.append(f"an answer {item!r} instead of {{'id': ..., 'answer': ...}}")
                    continue
                answer = rubric.match(item.get("answer"))
                if answer is None:
                    problems.append(f"text {item['id']}: {item.get('answer')!r}, not one of {list(rubric.answers)}")
                    continue
                out[int(item["id"])] = answer
        missing = [i for i in range(len(texts)) if i not in out]
        if missing:
            import warnings

            warnings.warn(f"the judge gave no usable answer for {len(missing)} of {len(texts)} texts (they score NaN); "
                          + "; ".join(problems[:5]), stacklevel=2)
        return [out.get(i) for i in range(len(texts))]


# ----------------------------------------------------------------------------------- local judges
class LocalJudge(Judge):
    """A judge loaded here: a model and its tokenizer (transformers). For each text it asks
    :meth:`Rubric.prompt` as a user turn (after ``system``, if given) and reads the probability of
    each answer as the start of its reply (:func:`concept_probe.answer_logprobs`), with no sampling;
    :meth:`rate` takes the most probable answer and :meth:`score` the expected value of the answers'
    numbers. Use a model other than the one you study, so the judge does not share its
    representations; write the answers as the judge would start its reply."""

    def __init__(self, model: Any, tokenizer: Any, *, system: Optional[str] = None, batch_size: int = 16,
                 name: Optional[str] = None):
        self.model, self.tokenizer, self.system, self.batch_size = model, tokenizer, system, int(batch_size)
        self.name = name or getattr(getattr(model, "config", None), "_name_or_path", None) or type(model).__name__

    def to_dict(self) -> Dict[str, Any]:
        return {"judge": "local", "model": self.name, "system": self.system}

    def probabilities(self, texts: Sequence[str], rubric: Rubric) -> np.ndarray:
        """``(texts, answers)``: each answer's probability among the allowed ones."""
        from .generation import answer_logprobs

        out = answer_logprobs(self.model, self.tokenizer, [rubric.prompt(str(t)) for t in texts], list(rubric.answers),
                              system=self.system, batch_size=self.batch_size, save=False)
        return out.probs()

    def rate(self, texts: Sequence[str], rubric: Rubric) -> List[str]:
        names = list(rubric.answers)
        return [names[k] for k in self.probabilities(texts, rubric).argmax(axis=1)]

    def score(self, texts: Sequence[str], rubric: Rubric) -> np.ndarray:
        values = np.array(list(rubric.answers.values()), dtype=np.float64)
        return self.probabilities(texts, rubric) @ values

    def complete(self, system: str, user: str, max_new_tokens: int = 1024) -> str:
        """The model's greedy reply to ``user`` after ``system``."""
        from .generation import generate

        return generate(self.model, self.tokenizer, [user], system=system, max_new_tokens=max_new_tokens,
                        save=False)[0].text


# ----------------------------------------------------------------------------------- agent judges
class AnswersPending(RuntimeError):
    """The task was written and nobody has answered it yet: ``task_dir`` is its folder."""

    def __init__(self, task_dir: str, message: str):
        super().__init__(message)
        self.task_dir = task_dir


def _task_id(*parts: Any) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]


class AgentJudge(Judge):
    """A judge that is whoever answers a task written to a folder: an agent such as Claude Code
    (or a subagent it delegates to), another program, or a person. No API, no model loaded here.

    :meth:`rate` writes ``<out_dir>/<rubric name>-<id>/`` with ``TASK.md`` (what to do, how to read
    the input, the exact format of the answer and where to save it) and ``items.jsonl`` (one text
    per line, with its id), and raises :class:`AnswersPending` with that folder. Once
    ``answers.jsonl`` is written there, calling :meth:`rate` again with the same texts reads it,
    checks that every text has an allowed answer, and returns them. The id depends on the texts and
    the rubric, so a different request never reads another's answers. :meth:`request` only writes.
    """

    def __init__(self, out_dir: str):
        self.out_dir = str(out_dir)

    def to_dict(self) -> Dict[str, Any]:
        return {"judge": "agent", "out_dir": self.out_dir}

    def task_dir(self, texts: Sequence[str], rubric: Rubric) -> Path:
        return Path(self.out_dir) / f"{rubric.name}-{_task_id(rubric.to_dict(), [str(t) for t in texts])}"

    def request(self, texts: Sequence[str], rubric: Rubric) -> str:
        """Write the task (if it is not there yet) and return its folder."""
        path = self.task_dir(texts, rubric)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "items.jsonl", "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                f.write(json.dumps({"id": i, "text": str(t)}, ensure_ascii=False) + "\n")
        answers = ", ".join(f'"{a}"' for a in rubric.answers)
        guide = f"\n\n{rubric.guide}" if rubric.guide else ""
        (path / "TASK.md").write_text(f"""# Rating task: {rubric.name}

Rate each text of `items.jsonl` (in this folder) by answering this question about it:

> {rubric.question}{guide}

The only allowed answers are: {answers}.

## Input

`items.jsonl` has one JSON object per line: `{{"id": <integer>, "text": "<the text>"}}`. There are
{len(texts)} texts, with ids 0 to {len(texts) - 1}.

## What to write

Write `answers.jsonl` in this folder: one JSON object per line, one line per text, exactly

    {{"id": <the text's id>, "answer": <one of the allowed answers, as a string>}}

Answer every id once. Judge each text on its own, from the text alone: do not compare texts, do not
skip any, and do not add other fields or lines. If the work is split (for example among subagents),
merge every part into this one file before you finish.

When the file is written, the code that asked for the ratings reads it (calling it again with the
same texts and rubric) and checks that every id has an allowed answer; anything missing or not
allowed is reported, so it can be fixed in the file.
""", encoding="utf-8")
        return str(path)

    def rate(self, texts: Sequence[str], rubric: Rubric) -> List[str]:
        path = self.task_dir(texts, rubric)
        answers_path = path / "answers.jsonl"
        if not answers_path.exists():
            self.request(texts, rubric)
            raise AnswersPending(str(path), f"The ratings are a task for an agent (or anyone): follow {path / 'TASK.md'} "
                                            f"and write {answers_path}; then call this again with the same texts.")
        out: Dict[int, str] = {}
        problems: List[str] = []
        for n, line in enumerate(answers_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                i, answer = int(item["id"]), rubric.match(item["answer"])
            except Exception:
                problems.append(f"line {n} is not {{\"id\": <integer>, \"answer\": <answer>}}: {line[:80]!r}")
                continue
            if answer is None:
                problems.append(f"id {i}: {item['answer']!r} is not an allowed answer")
            elif not 0 <= i < len(texts):
                problems.append(f"id {i} is not an id of the texts (0 to {len(texts) - 1})")
            else:
                out[i] = answer
        missing = [i for i in range(len(texts)) if i not in out]
        if missing:
            problems.append(f"{len(missing)} ids have no answer (the first is {missing[0]})")
        if problems:
            raise ValueError(f"{answers_path} cannot be used: " + "; ".join(problems[:5]))
        return [out[i] for i in range(len(texts))]


# --------------------------------------------------------------------------------- saved batches
def rate_saved(out_dir: str, judge: Judge, rubric: Rubric = COHERENCE) -> List[Dict[str, Any]]:
    """Rate the texts of a saved batch of scores (its ``scores.jsonl``) and save the answers to
    ``<out_dir>/<rubric name>_rating.json``: one entry per text with its ``index`` in the scores,
    the answer, its number, the judge and when it rated. The eval analysis reads the coherence
    ratings (``coherence_rating.json``) from there."""
    path = Path(out_dir).resolve()
    lines_path = path / "scores.jsonl"
    if not lines_path.exists():
        raise FileNotFoundError(f"No scores.jsonl in {out_dir}")
    texts = [json.loads(l)["text"] for l in lines_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    answers = judge.rate(texts, rubric)
    rated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = [{"index": i, "rating": a, "value": None if a is None else rubric.answers[a], "judge": judge.to_dict(), "rated_at": rated_at}
           for i, a in enumerate(answers)]
    (path / f"{rubric.name}_rating.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


__all__ = ["Rubric", "COHERENCE", "Judge", "APIJudge", "LocalJudge", "AgentJudge", "AnswersPending", "PROVIDERS",
           "rate_saved"]
