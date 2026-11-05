import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from groq import Groq
except Exception as exc:
    Groq = None
    _GROQ_IMPORT_ERROR = exc
else:
    _GROQ_IMPORT_ERROR = None

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_PER_REQUEST = 8
ALLOWED_RATINGS = {"COHERENT", "PARTIALLY_COHERENT", "NONSENSE"}

SYSTEM_PROMPT = (
    "You are a strict rater of text coherence. You will receive a JSON list of objects, "
    "each with fields: id (an integer) and completion (a string). "
    "Your job is to return a JSON list with the same ids and a rating for each completion.\n\n"
    "Ratings:\n"
    "- COHERENT: The text is syntactically well-formed and understandable.\n"
    "- PARTIALLY_COHERENT: Some grammatical/syntactic errors, but still understandable.\n"
    "- NONSENSE: Not understandable as meaningful text.\n\n"
    "Return only valid JSON (no code fences, no extra text). "
    "Each output object must have exactly: {\"id\": <int>, \"rating\": <one of the three labels>}."
)


def _read_env_file(env_path: Optional[str]) -> Dict[str, str]:
    """KEY=VALUE pairs of ``env_path`` (none without a path). The process environment is not modified."""
    if not env_path:
        return {}
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _chunked(items: List[Dict[str, object]], size: int) -> Iterable[List[Dict[str, object]]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _call_groq(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    if Groq is None:
        raise RuntimeError(
            "groq package is not installed. Install it to use coherence rating "
            "(e.g. `pip install groq`)."
        ) from _GROQ_IMPORT_ERROR
    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg:
            raise RuntimeError(f"Groq API error: {msg} (check GROQ_API_KEY and model access)") from exc
        raise RuntimeError(f"Groq API error: {msg}") from exc
    return resp.choices[0].message.content


def _parse_json_list(content: str) -> List[Dict[str, object]]:
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response is not a JSON list.")
    snippet = content[start : end + 1]
    data = json.loads(snippet)
    if not isinstance(data, list):
        raise ValueError("Model response is not a JSON list.")
    return data


def _validate_ratings(items: List[Dict[str, object]]) -> Dict[int, str]:
    ratings: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Model response items must be objects.")
        raw_id = item.get("id")
        rating = item.get("rating")
        if isinstance(raw_id, int):
            idx = raw_id
        elif isinstance(raw_id, str) and raw_id.isdigit():
            idx = int(raw_id)
        else:
            raise ValueError(f"Invalid id in model response: {raw_id}")
        if rating not in ALLOWED_RATINGS:
            raise ValueError(f"Invalid rating for id {idx}: {rating}")
        ratings[idx] = rating
    return ratings


def _credentials(model: Optional[str], api_key: Optional[str], env_path: Optional[str]) -> Tuple[str, str]:
    env_file = _read_env_file(env_path)
    api_key = api_key or env_file.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY (set it in the environment, or pass api_key or env_path).")
    if model is None:
        model = env_file.get("GROQ_MODEL") or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    return model, api_key


def rate_coherence(
    texts: Sequence[str],
    max_elements_per_request: int = DEFAULT_MAX_PER_REQUEST,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
) -> List[str]:
    """Rate each text as COHERENT, PARTIALLY_COHERENT or NONSENSE with a judge model; returns the
    ratings in order.

    NOTE: this sends the texts to the external Groq API. Do not use it for data that must not leave
    the machine. The ratings are labels from another model, not ground truth.

    api_key: explicit key; when None, GROQ_API_KEY from the file ``env_path`` if one is given,
    then from the environment (no .env file is read unless passed). model: explicit model id; when
    None, GROQ_MODEL from ``env_path``, then from the environment, then the package default.
    """
    if max_elements_per_request <= 0:
        raise ValueError("max_elements_per_request must be positive.")
    model, api_key = _credentials(model, api_key, env_path)
    items = [{"id": i, "completion": str(t)} for i, t in enumerate(texts)]
    print(f"Sending {len(items)} completions to the Groq API (model={model}) for coherence rating.")
    ratings_by_id: Dict[int, str] = {}
    for chunk in _chunked(items, max_elements_per_request):
        content = _call_groq(api_key, model, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)},
        ])
        ratings_by_id.update(_validate_ratings(_parse_json_list(content)))
        time.sleep(0.1)
    missing = [i for i in range(len(items)) if i not in ratings_by_id]
    if missing:
        raise RuntimeError(f"Missing rating for id {missing[0]} ({len(missing)} missing).")
    return [ratings_by_id[i] for i in range(len(items))]


def rate_batch_coherence(
    batch_dir: str,
    max_elements_per_request: int = DEFAULT_MAX_PER_REQUEST,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rate the completions of a saved batch of scores (its ``scores.jsonl``) with
    :func:`rate_coherence`, and save them to ``<batch_dir>/coherence_rating.json``: one entry per
    generation, with its ``index`` in the scores, the rating, the judge model and when it rated.
    """
    batch_path = Path(batch_dir).resolve()
    lines_path = batch_path / "scores.jsonl"
    if not lines_path.exists():
        raise FileNotFoundError(f"No scores.jsonl in {batch_dir}")
    lines = [json.loads(l) for l in lines_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    model, api_key = _credentials(model, api_key, env_path)
    ratings = rate_coherence([l["text"] for l in lines], max_elements_per_request, model=model, api_key=api_key)
    rated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = [{"index": i, "rating": r, "judge_model": model, "rated_at": rated_at} for i, r in enumerate(ratings)]
    (batch_path / "coherence_rating.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    return results


__all__ = ["rate_coherence", "rate_batch_coherence"]
