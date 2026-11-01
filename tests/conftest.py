"""Plots are written to files only: use the non-interactive backend, whatever the machine's default."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
