"""Parameter folders for the parameter-set tests.

``linked_param_folder`` links mcpu08's files into a new folder: the same
files in another place, so the same signature. ``changed_param_folder`` does
the same with one table changed, so another signature.
``register_set_like_mcpu08`` registers a second parameter set with mcpu08's
layout for one test, so a run can use another set name.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from pymcpu import params

#: The table changed_param_folder changes: nine float32 energies, small.
CHANGED_TABLE = "mcpu_params/aromatic_potentials.bin"


def linked_param_folder(root: Path) -> Path:
    """A folder that links every mcpu08 file the set's layout names."""
    src = Path(params.ensure_params("mcpu08"))
    for rel in params.signed_files("mcpu08"):
        if (src / rel).is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).symlink_to(src / rel)
    return root


def changed_param_folder(root: Path) -> Path:
    """:func:`linked_param_folder` with the aromatic table's first energy
    raised by 1 (a table of its own, not a link)."""
    linked_param_folder(root)
    table = root / CHANGED_TABLE
    values = np.fromfile(table, dtype=np.float32)
    values[0] += 1.0
    table.unlink()
    values.tofile(table)
    return root


def register_set_like_mcpu08(monkeypatch: pytest.MonkeyPatch, name: str = "mcpu08_copy") -> str:
    """Register ``name``, a set with mcpu08's layout and no tables of its own
    (pass ``param_dir``), until the test ends."""
    registry = copy.deepcopy(params._load_registry())
    entry = copy.deepcopy(registry["sets"]["mcpu08"])
    entry.pop("local_source", None)
    registry["sets"][name] = entry
    monkeypatch.setattr(params, "_load_registry", lambda: registry)
    return name
