"""Two KORP engine defects that short tests could not see.

1. The rigid-pivot moved-moved elision in OrientationalPairPotential is not
   exact for a nearest-bin table under float32 rotation: a co-moving pair near a
   bin edge can change bin with nothing entering delta_E. It is now OFF by
   default. The failure is rare (~1e-4 per accepted move) but individually large,
   so the guard is a LONG, HOT, pivot-only chain whose incrementally maintained
   total must keep matching a full recompute -- a few hundred steps of
   MCPU_VERIFY_PHYSICS never sees an event.

2. OrientationalPairMap references its table by raw pointer. The table used to
   be kept alive only through the map's Python wrapper (py::keep_alive), while
   potentials own the map by shared_ptr and outlive that wrapper -- and
   KORPForceField holds only its latest map. A table with no other owner (one
   swapped in for a single system) was freed under a live potential. Run in a
   subprocess, because the failure mode is a segfault.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from pymcpu import mcpu_core


def _map_path() -> Path:
    path = os.environ.get("KORP_MAP_PATH")
    if not path:
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")
    return Path(path)


def _structure(rel: str) -> Path:
    pdb = _map_path().parent / rel
    if not pdb.is_file():
        pytest.skip(f"{rel} not found next to the map (needs the KORP bundle)")
    return pdb


def _forcefield(n_res: int = 60):
    import mdtraj as md
    from pymcpu.forcefields.korp import KORPForceField
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = md.load(str(_structure("CASP12DCsel20/T0860D1.pdb")))
    traj = traj.atom_slice(traj.topology.select(f"resid < {n_res}"))
    return KORPForceField(traj, map_path=_map_path()), traj


def test_rigid_skip_is_off_by_default():
    ff, traj = _forcefield()
    ff.create_system(traj.topology)
    assert ff.pair_potential.rigid_skip_enabled is False   # bound as a property


def test_long_hot_pivot_chain_keeps_the_running_total_exact():
    """Incremental total vs full recompute over 40k hot, pivot-only steps.

    With the elision on, runs like this drift by units to tens of units from
    a single hidden bin flip. With exact dE the delta adds the same float
    pair terms as the full sum, in double, so the residual is double rounding.
    """
    ff, traj = _forcefield()
    context = mcpu_core.Context(ff.create_system(traj.topology))
    context.set_positions(np.ascontiguousarray((ff.coords[0] * 10.0).T, dtype=np.float32))
    ff.apply_energy_weights(context)
    context.calculate_total_energy(-1)                 # seed the running total
    integrator = mcpu_core.Integrator(temperature=8.0, step_size_rad=0.05)
    integrator.set_seed(1)
    integrator.set_move_weights(1.0, 0.0, 0.0)         # every move a rigid pivot
    worst = 0.0
    for _ in range(20):
        integrator.run(context, 2000)
        incremental = context.get_state().current_energy
        worst = max(worst, abs(incremental - context.calculate_total_energy(-1)))
    assert worst < 1e-6, f"running total drifted {worst:.4g} from a full recompute"


def test_table_outlives_its_python_map_object():
    helpers_ok = _structure("CASP12DCsel20/T0860D1.pdb")      # skips cleanly if absent
    script = textwrap.dedent(f"""
        import dataclasses, gc, warnings
        import numpy as np, mdtraj as md
        from pymcpu import mcpu_core
        from pymcpu.forcefields.korp import KORPForceField
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = md.load({str(helpers_ok)!r})
        ff = KORPForceField(t, map_path={str(_map_path())!r})
        base = ff.korp_map
        fresh = np.array(base.table, copy=True)       # owned by nothing but this system's map
        ff.korp_map = dataclasses.replace(base, table=fresh)
        ctx = mcpu_core.Context(ff.create_system(t.topology))
        ctx.set_positions(np.ascontiguousarray((ff.coords[0] * 10.0).T, dtype=np.float32))
        ff.apply_energy_weights(ctx)
        e_ref = ctx.calculate_total_energy(-1)
        ff.korp_map = base
        del fresh
        ff.create_system(t.topology)                  # replaces ff._engine_map: old wrapper dropped
        gc.collect()
        junk = np.full(base.table.size, 7.0e3, dtype=np.float32)   # encourage reuse of freed pages
        e_after = ctx.calculate_total_energy(-1)
        print(repr(e_ref), repr(e_after))
        assert e_after == e_ref, (e_ref, e_after)
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={**os.environ}, timeout=600)
    assert proc.returncode == 0, (
        f"table freed under a live potential (exit {proc.returncode}):\n"
        f"{proc.stdout[-500:]}\n{proc.stderr[-1500:]}")
