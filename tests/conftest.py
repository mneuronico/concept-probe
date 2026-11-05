"""Plots are written to files only: use the non-interactive backend, whatever the machine's default."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

# Tests do not write to the working directory or print automatic choices unless they ask to: the
# session defaults of the library (save, console) are turned off here.
import concept_probe.settings  # noqa: E402

concept_probe.settings.configure(save=False, console=False)
