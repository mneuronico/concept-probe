"""Automatic choices: what the library decided that the caller did not ask for, and why.

Every such choice (the dtype resolved from ``"auto"``, the source a model was loaded from, how a
cache was used, ...) is recorded with its reason, travels with the result it shaped, is printed to
the console when that is on, and goes to the run log and the saved metrics. Each one has an option
to choose something else (decisions D1 and D5 in docs/DESIGN.md).
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from .settings import settings


@dataclass(frozen=True)
class Choice:
    #: the option the choice is about, as it is set (e.g. ``"model.dtype"``)
    setting: str
    #: what the caller asked for (often ``"auto"`` or a default left untouched)
    requested: Any
    #: what the library used
    chosen: Any
    #: why, in a sentence
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"setting": self.setting, "requested": _plain(self.requested), "chosen": _plain(self.chosen),
                "reason": self.reason}

    def __str__(self) -> str:
        return f"{self.setting}: {self.requested!r} -> {self.chosen!r} ({self.reason})"


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


_ANNOUNCED: Set[Tuple[str, str, str]] = set()


def announce(choices: Iterable[Choice], *, console: Optional[bool] = None, logger: Any = None,
             once: bool = True) -> None:
    """Print ``choices`` (when the console is on) and log them (when a run logger is given).

    ``once`` prints a choice only the first time it is made in the session, so a loop does not
    repeat it; the run log always gets it.
    """
    show = settings().console if console is None else bool(console)
    for choice in choices:
        if logger is not None:
            logger.log("choice", choice.to_dict())
        key = (choice.setting, repr(choice.chosen), choice.reason)
        if show and not (once and key in _ANNOUNCED):
            print(f"[concept-probe] {choice}", flush=True)
        _ANNOUNCED.add(key)
