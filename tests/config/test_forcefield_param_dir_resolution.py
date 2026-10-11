"""``MCPUForceField`` uses an explicit ``param_dir`` as given.

Without ``param_dir`` the force field calls ``ensure_params()``, which every
other force-field build in the suite goes through. This checks the other
branch: a directory passed in is used as-is and ``ensure_params()`` is never
called.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md

import pymcpu.forcefields.mcpu as mcpu_ff
from pymcpu.params import ensure_params
from pymcpu.runners import default_example_pdb


def test_explicit_param_dir_is_used_without_calling_ensure_params(monkeypatch) -> None:
    param_dir = ensure_params("mcpu08")

    def _refuse(*_args, **_kwargs):
        raise AssertionError("ensure_params must not be called when param_dir is set")

    monkeypatch.setattr(mcpu_ff, "ensure_params", _refuse)
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = mcpu_ff.MCPUForceField(heavy, param_dir=str(param_dir))
    assert forcefield.param_dir == Path(param_dir)
