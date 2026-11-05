"""Session-wide defaults for saving and for the console.

Functions that produce results save them by default (decision D4 in docs/DESIGN.md: better to save
too much than too little) and announce the choices they make on their own (D1, D5). Each such
function takes ``save=`` / ``console=`` arguments; when they are None, these defaults apply.
``configure(...)`` changes them for the rest of the session::

    import concept_probe
    concept_probe.configure(save=False)            # nothing is written unless asked
    concept_probe.configure(output_root="runs")    # where results without a run directory go

The high-level API (``ProbeWorkspace``) follows its own config (``output.*``) instead.
"""
import os
from dataclasses import asdict, dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class Settings:
    #: write results to disk, including the representations cache of reading
    save: bool = True
    #: where results go when there is no run directory
    output_root: str = "outputs"
    #: the representations cache; None means ``<output_root>/rep_cache``
    rep_cache_dir: Optional[str] = None
    #: print automatic choices (and progress) to the console
    console: bool = True


_SETTINGS = Settings()


def settings() -> Settings:
    """The current session defaults."""
    return _SETTINGS


def configure(**changes) -> Settings:
    """Change session defaults (``save``, ``output_root``, ``rep_cache_dir``, ``console``).

    Returns the previous settings, so ``configure(**asdict(previous))`` restores them.
    """
    global _SETTINGS
    unknown = sorted(set(changes) - set(asdict(_SETTINGS)))
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(unknown)}; expected some of {sorted(asdict(_SETTINGS))}.")
    previous = _SETTINGS
    _SETTINGS = replace(_SETTINGS, **changes)
    return previous


def default_rep_cache_dir() -> str:
    s = _SETTINGS
    return s.rep_cache_dir or os.path.join(s.output_root, "rep_cache")
