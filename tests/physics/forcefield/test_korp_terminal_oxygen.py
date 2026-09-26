"""A C-terminal OXT must not put an atom in the topology the engine lacks.

`_BACKBONE_SELECTION` admits OXT/OCT so a residue with no plain O can still
supply one. When a terminus carries BOTH, only one is used -- and the other
used to remain in `output_topology` with no engine slot, making
`inverse_mapping` sparse. The XTC reporter sizes its output from
`max(mapping) + 1` and zero-fills the rest, so that produced either a file with
a stray atom at the origin that loads cleanly (when the unused atom does not
sort last) or one that will not load at all (when it does).

Neither shipped structure fixture has a terminal OXT, so nothing caught it.
These build their own, and cover both orderings plus the fallback the selection
exists for.
"""

from __future__ import annotations

import os
import warnings

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.korp import KORPForceField

_CA_SPACING = 3.8      # A; well clear of the 3.2 A steric floor at |i-j| >= 3


def _pdb_text(n_res=6, *, terminal="O+OXT", oxt_first=False):
    """An extended poly-ALA backbone, optionally with a C-terminal oxygen.

    ``terminal`` is "O", "O+OXT" or "OXT" (the last exercising the fallback:
    a residue with an OXT and no plain O).  ``oxt_first`` writes OXT ahead of
    N/CA in the final residue, which is how CLN025 is laid out and which is
    what makes the unused atom sort mid-file rather than last.
    """
    lines, serial = [], 1

    def atom(name, res_i, xyz):
        nonlocal serial
        lines.append(
            # cols: 1-6 record, 7-11 serial, 13-16 name, 17 altLoc,
            # 18-20 resName, 22 chain, 23-26 resSeq, 31-54 xyz
            f"ATOM  {serial:5d} {name:<4s} ALA A{res_i + 1:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           "
            f"{name[0]}  "
        )
        serial += 1

    for i in range(n_res):
        x = i * _CA_SPACING
        n_xyz = (x - 1.20, 0.60, 0.0)
        ca_xyz = (x, 0.0, 0.0)
        c_xyz = (x + 1.20, 0.60, 0.0)
        o_xyz = (x + 1.30, 1.80, 0.0)
        oxt_xyz = (x + 2.30, 0.10, 0.0)
        last = i == n_res - 1

        if last and oxt_first:
            if terminal in ("O", "O+OXT"):
                atom("O", i, o_xyz)
            if terminal in ("OXT", "O+OXT"):
                atom("OXT", i, oxt_xyz)
            atom("N", i, n_xyz)
            atom("CA", i, ca_xyz)
            atom("C", i, c_xyz)
        else:
            atom("N", i, n_xyz)
            atom("CA", i, ca_xyz)
            atom("C", i, c_xyz)
            if not last or terminal in ("O", "O+OXT"):
                atom("O", i, o_xyz)
            if last and terminal in ("OXT", "O+OXT"):
                atom("OXT", i, oxt_xyz)

    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _load(tmp_path, **kw):
    pdb = tmp_path / "chain.pdb"
    pdb.write_text(_pdb_text(**kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # mdtraj is noisy about PDB columns
        return md.load(str(pdb))


@pytest.fixture(autouse=True)
def _needs_map():
    if not os.environ.get("KORP_MAP_PATH"):
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")


@pytest.mark.parametrize("oxt_first", [False, True], ids=["oxt_last", "oxt_first"])
def test_output_topology_holds_exactly_the_engine_atoms(tmp_path, oxt_first):
    ff = KORPForceField(_load(tmp_path, terminal="O+OXT", oxt_first=oxt_first))
    assert ff.output_topology.n_atoms == ff.n_atoms == 4 * ff.n_res
    assert not [a for a in ff.output_topology.atoms if a.name in ("OXT", "OCT")]


@pytest.mark.parametrize("oxt_first", [False, True], ids=["oxt_last", "oxt_first"])
def test_inverse_mapping_is_a_dense_permutation(tmp_path, oxt_first):
    ff = KORPForceField(_load(tmp_path, terminal="O+OXT", oxt_first=oxt_first))
    assert sorted(ff.inverse_mapping) == list(range(ff.n_atoms))


@pytest.mark.parametrize("oxt_first", [False, True], ids=["oxt_last", "oxt_first"])
def test_xtc_round_trips_and_leaves_nothing_at_the_origin(tmp_path, oxt_first):
    """The symptom the fix exists for, in both orderings.

    oxt_first=False used to write a file with fewer atoms than the topology
    (mdtraj refused it); oxt_first=True used to write a stray atom at the
    origin in a file that loaded cleanly.
    """
    traj = _load(tmp_path, terminal="O+OXT", oxt_first=oxt_first)
    ff = KORPForceField(traj)
    sim = mc.Simulation(ff.output_topology, ff.create_system(traj.topology),
                        mcpu_core.Integrator(temperature=1.0, step_size_rad=0.05))
    sim.integrator.set_seed(5)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ff.apply_energy_weights(sim.context)
    sim.integrator.set_move_weights(0.5, 0.5, 0.0)

    out = tmp_path / "korp.xtc"
    sim.add_xtc_reporter(str(out), interval=5, inverse_mapping=ff.inverse_mapping)
    sim.step(10)
    sim.flush_reporters()

    loaded = md.load(str(out), top=ff.output_topology)
    assert loaded.n_atoms == ff.n_atoms
    assert not np.any(np.all(loaded.xyz[-1] == 0.0, axis=1)), "atom left at the origin"

    engine = np.asarray(sim.context.coords).T / 10.0
    np.testing.assert_allclose(loaded.xyz[-1][ff.inverse_mapping], engine, atol=2e-3)


def test_a_structure_without_a_terminal_oxt_is_unaffected(tmp_path):
    """The clean path must not move: this fix is a no-op there."""
    ff = KORPForceField(_load(tmp_path, terminal="O"))
    assert ff.output_topology.n_atoms == ff.n_atoms
    assert ff.inverse_mapping == list(ff.ordered_indices)


def test_oxt_still_serves_as_the_oxygen_when_there_is_no_plain_o(tmp_path):
    """Guards the reason OXT is in the selection at all.

    Stripping OXT wholesale would look like a simpler fix and would break this.
    """
    ff = KORPForceField(_load(tmp_path, terminal="OXT"))
    assert ff.output_topology.n_atoms == ff.n_atoms == 4 * ff.n_res
    assert sorted(ff.inverse_mapping) == list(range(ff.n_atoms))
