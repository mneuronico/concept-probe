import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _load_env(env_path: Optional[Path] = None) -> None:
    path = env_path or Path(".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_log_jsonl(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for parent in [cur] + list(cur.parents):
        candidate = parent / "log.jsonl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find log.jsonl above {start_dir}")


def _normalize_npz_path(path_str: str, repo_root: Path, log_dir: Path) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw.resolve()
    candidate = (repo_root / raw).resolve()
    if candidate.exists():
        return candidate
    return (log_dir / raw).resolve()


def _iter_log_entries(log_path: Path) -> Iterable[Dict[str, object]]:
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _collect_completion_map(log_path: Path, repo_root: Path) -> Dict[str, str]:
    completion_map: Dict[str, str] = {}
    log_dir = log_path.parent
    for entry in _iter_log_entries(log_path):
        event = entry.get("event")
        payload = entry.get("data", entry)
        if event not in {"score_prompt", "multi_score_prompt"}:
            continue
        npz_path = payload.get("npz_path")
        completion = payload.get("completion")
        if not isinstance(npz_path, str) or not isinstance(completion, str):
            continue
        normalized = _normalize_npz_path(npz_path, repo_root, log_dir)
        completion_map[str(normalized)] = completion
    return completion_map


def _list_batch_npz(batch_dir: Path) -> List[Path]:
    npz_files = sorted(batch_dir.glob("prompt_*.npz"))
    if npz_files:
        return npz_files
    return sorted(p for p in batch_dir.glob("*.npz") if p.name != "tensors.npz")


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


def rate_batch_coherence(
    batch_dir: str,
    max_elements_per_request: int = DEFAULT_MAX_PER_REQUEST,
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Rate completion coherence for a score batch directory.

    Saves output to <batch_dir>/coherence_rating.json and returns the list.
    """
    if max_elements_per_request <= 0:
        raise ValueError("max_elements_per_request must be positive.")

    _load_env(Path(env_path) if env_path else None)
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY (set it in .env or environment).")
    if model == DEFAULT_MODEL:
        model = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL

    batch_path = Path(batch_dir).resolve()
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    repo_root = Path.cwd().resolve()
    log_path = _find_log_jsonl(batch_path)
    completion_map = _collect_completion_map(log_path, repo_root)
    npz_files = _list_batch_npz(batch_path)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {batch_dir}")

    records: List[Tuple[int, Path, str]] = []
    missing: List[str] = []
    for idx, npz_path in enumerate(npz_files):
        key = str(npz_path.resolve())
        completion = completion_map.get(key)
        if completion is None:
            missing.append(npz_path.name)
            continue
        records.append((idx, npz_path, completion))

    if missing:
        raise RuntimeError(
            "Missing completions for some .npz files. "
            f"First missing: {missing[0]} (total missing: {len(missing)})"
        )

    items = [{"id": idx, "completion": completion} for idx, _, completion in records]
    ratings_by_id: Dict[int, str] = {}
    for chunk in _chunked(items, max_elements_per_request):
        user_payload = json.dumps(chunk, ensure_ascii=False)
        content = _call_groq(
            api_key,
            model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
        )
        parsed = _parse_json_list(content)
        ratings_by_id.update(_validate_ratings(parsed))
        time.sleep(0.1)

    results: List[Dict[str, str]] = []
    for idx, npz_path, _ in records:
        rating = ratings_by_id.get(idx)
        if rating is None:
            raise RuntimeError(f"Missing rating for id {idx}.")
        results.append({"example": npz_path.name, "rating": rating})

    out_path = batch_path / "coherence_rating.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


__all__ = ["rate_batch_coherence"]
