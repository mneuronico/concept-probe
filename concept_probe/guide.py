"""The guides the package ships (for people and for agents): ``guide()`` lists them, ``guide(name)``
returns one. They are Markdown files in ``concept_probe/guides``."""
from importlib.resources import files

NAMES = ("workflow", "datasets", "steering", "delegating")


def guide(name=None):
    """A guide's text (``"workflow"``, ``"datasets"``, ``"steering"`` or ``"delegating"``), or, with no
    name, the list of guides with their first lines."""
    folder = files("concept_probe").joinpath("guides")
    if name is None:
        lines = ["concept-probe guides (concept_probe.guide(name)):", ""]
        for n in NAMES:
            first = folder.joinpath(f"{n}.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
            lines.append(f"- {n}: {first}")
        return "\n".join(lines)
    if name not in NAMES:
        raise ValueError(f"no guide called {name!r}; the guides are {list(NAMES)}")
    return folder.joinpath(f"{name}.md").read_text(encoding="utf-8")
