"""`KORPForceField`: the backbone-only layout and the things it refuses.

Most of these are about failure modes that are silent if unguarded. Dropping
sidechains changes the engine's atom set, and the two ways that can go wrong --
a sparse `inverse_mapping` (which makes the XTC reporter write atoms at the
origin) and a `sc_start` that is a valid index with no count (which makes the
KIC move transform the wrong atoms) -- both produce output that loads cleanly
and is quietly wrong.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.korp import KORPForceField
from pymcpu.forcefields.korp_map import KorpMapError

REFERENCE_ENERGIES = {
    "CASP12DCsel20/T0860D1.pdb": -3693.586739,
    "rcd6/1CEO.pdb": -11463.957486,
}


def _map_path():
    path = os.environ.get("KORP_MAP_PATH")
    if not path:
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")
    return Path(path)


def _load(rel):
    pdb = _map_path().parent / rel
    if not pdb.is_file():
        pytest.skip(f"{rel} not found next to the map (needs the KORP bundle)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # mdtraj is noisy about PDB columns
        return md.load(str(pdb))


@pytest.fixture(scope="module")
def chain_traj():
    return _load("CASP12DCsel20/T0860D1.pdb")


def _simulate(forcefield, topology_source):
    system = forcefield.create_system(topology_source.topology)
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
    sim = mc.Simulation(forcefield.output_topology, system, integrator)
    sim.context.set_positions(
        (forcefield.coords[0] * 10.0).T.astype(np.float32))
    forcefield.apply_energy_weights(sim.context)
    return sim


def test_layout_is_backbone_only(chain_traj):
    ff = KORPForceField(chain_traj)
    assert ff.total_sc_atoms == 0
    assert ff.total_h_atoms == 0
    assert ff.total_bb_atoms == 3 * ff.n_res
    assert ff.total_o_atoms == ff.n_res
    assert ff.n_atoms == 4 * ff.n_res
    assert ff.output_topology.n_atoms == ff.n_atoms


def test_sidechain_block_indices_are_absent_not_empty(chain_traj):
    """`sc_start` must be -1, never a valid index with `sc_count == 0`.

    The KIC move has a fallback for the second case that builds a span
    reaching into other residues' atoms and rigid-transforms them. It stays in
    bounds, so nothing crashes -- the geometry is simply wrong.
    """
    ff = KORPForceField(chain_traj)
    for block in ff.blocks:
        assert block.sc_start == -1
        assert block.sc_count == 0
        assert block.h_start == -1


def test_downstream_cache_makes_the_sidechain_span_empty(chain_traj):
    """Pivot rotates [first_sc_of_residue[r], sc_end) without checking."""
    ff = KORPForceField(chain_traj)
    assert np.all(np.asarray(ff.downstream.first_sc_of_residue) == ff.n_atoms)
    assert np.all(np.asarray(ff.downstream.first_h_of_residue) == ff.n_atoms)


def test_inverse_mapping_is_a_dense_permutation(chain_traj):
    """What keeps the XTC reporter honest.

    It sizes its output from max(mapping) + 1 and zero-fills anything
    unmapped, so a mapping with holes writes a file that loads fine and has
    atoms piled at the origin.
    """
    ff = KORPForceField(chain_traj)
    mapping = ff.inverse_mapping
    assert -1 not in mapping
    assert sorted(mapping) == list(range(ff.output_topology.n_atoms))


def test_every_atom_has_one_slot(chain_traj):
    """Every atom has one slot, glycine's CA included."""
    ff = KORPForceField(chain_traj)
    assert "GLY" in ff.res_names          # the fixture must actually test this
    assert len(set(ff.ordered_indices)) == len(ff.ordered_indices)


@pytest.mark.parametrize("rel,expected", sorted(REFERENCE_ENERGIES.items()))
def test_energy_from_a_plain_pdb_matches_reference_korpe(rel, expected):
    traj = _load(rel)
    ff = KORPForceField(traj)
    sim = _simulate(ff, traj)
    energy = sim.context.energy_breakdown(weighted=False)["by_group"][7]
    assert energy == pytest.approx(expected, rel=1e-6)


def test_a_native_structure_passes_the_steric_guard(chain_traj):
    """The guard must not veto real structures.

    The default 3.2 A floor is set from measurement: across 1CEO, 1DOS,
    T0860D1, actin and chignolin the closest CA-CA contact at >= 3 apart in
    sequence is 3.53 A, and those are genuine packing contacts.
    """
    ff = KORPForceField(chain_traj)
    sim = _simulate(ff, chain_traj)
    assert sim.context.energy_breakdown(weighted=False)["by_group"][8] == 0.0


def test_monte_carlo_runs_and_proposes_no_sidechain_moves(chain_traj):
    ff = KORPForceField(chain_traj)
    sim = _simulate(ff, chain_traj)
    sim.integrator.set_seed(7)
    sim.integrator.set_move_weights(0.5, 0.5, 0.0)
    sim.step(200)
    stats = sim.integrator.move_stats()
    assert stats["num_propose_sc"] == 0
    assert stats["num_propose_rotamer"] == 0
    assert stats["num_propose_pivot"] + stats["num_propose_kic"] == 200


def test_the_default_move_mix_is_refused(chain_traj):
    """These residues have no chi angles, so the engine must say so."""
    ff = KORPForceField(chain_traj)
    sim = _simulate(ff, chain_traj)
    with pytest.raises(ValueError, match="no residue"):
        sim.step(10)


def test_trajectory_round_trips_against_output_topology(chain_traj, tmp_path):
    ff = KORPForceField(chain_traj)
    sim = _simulate(ff, chain_traj)
    sim.integrator.set_seed(1)
    sim.integrator.set_move_weights(0.5, 0.5, 0.0)
    out = tmp_path / "korp.xtc"
    sim.add_xtc_reporter(str(out), interval=25,
                         inverse_mapping=ff.inverse_mapping)
    sim.step(100)
    sim.flush_reporters()

    loaded = md.load(str(out), top=ff.output_topology)
    assert loaded.n_atoms == ff.n_atoms
    assert loaded.n_frames >= 1
    # Nothing parked at the origin, which is what a sparse mapping produces.
    assert not np.any(np.all(np.abs(loaded.xyz[0]) < 1e-9, axis=-1))


def test_missing_backbone_atoms_are_reported_not_skipped(chain_traj):
    stripped = chain_traj.atom_slice(
        [a.index for a in chain_traj.topology.atoms
         if not (a.name == "C" and a.residue.index == 5)])
    with pytest.raises(ValueError, match="backbone is incomplete"):
        KORPForceField(stripped)


def test_non_increasing_residue_numbering_is_rejected(chain_traj):
    """KORP takes sequence separation from PDB numbers, not array position."""
    traj = md.Trajectory(chain_traj.xyz.copy(), chain_traj.topology.copy())
    residues = list(traj.topology.residues)
    residues[10].resSeq = residues[9].resSeq      # a duplicate number
    with pytest.raises(ValueError, match="not increasing"):
        KORPForceField(traj)
    # ... and it can be waived deliberately.
    KORPForceField(traj, strict_residue_numbering=False)


def test_a_clashing_input_fails_at_construction(chain_traj):
    """Not mid-run, where the message is about the delta path instead."""
    traj = md.Trajectory(chain_traj.xyz.copy(), chain_traj.topology.copy())
    residues = list(traj.topology.residues)
    ca_a = next(a.index for a in residues[4].atoms if a.name == "CA")
    ca_b = next(a.index for a in residues[60].atoms if a.name == "CA")
    traj.xyz[0, ca_b] = traj.xyz[0, ca_a] + 0.01   # 0.1 A apart
    with pytest.raises(ValueError, match="already violates"):
        KORPForceField(traj)


def test_a_missing_map_says_how_to_get_one(chain_traj, monkeypatch, tmp_path):
    monkeypatch.delenv("KORP_MAP_PATH", raising=False)
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path))
    with pytest.raises(KorpMapError) as excinfo:
        KORPForceField(chain_traj, map_path=tmp_path / "absent.bin")
    message = str(excinfo.value)
    assert "chaconlab.org" in message
    assert "331777205" in message
    assert "KORP_MAP_PATH" in message


def _two_chain_energy(tmp_path, b_first):
    from tests.physics.forcefield.test_korp_chain_identity import _two_chains

    traj = _two_chains(tmp_path, b_first=b_first)
    ff = KORPForceField(traj, map_path=_map_path())
    energy = _simulate(ff, traj).context.energy_breakdown(weighted=False)["by_group"][7]
    return ff, energy


def test_two_chains_are_scored_as_two_chains(tmp_path):
    """Pairs on different chains are non-bonded. The slice used to drop the
    chain IDs, so the copy was scored as a continuation of chain A."""
    from pymcpu.forcefields.korp_map import score_structure

    ff, energy = _two_chain_energy(tmp_path, b_first=11)
    frames = (ff.coords[0, : 3 * ff.n_res] * 10.0).reshape(ff.n_res, 3, 3)
    two = score_structure(ff.korp_map, frames, ff.res_names, ff.res_seq, ff.chain_ids)
    one = score_structure(ff.korp_map, frames, ff.res_names, ff.res_seq, [" "] * ff.n_res)
    assert energy == pytest.approx(two, rel=1e-6)
    assert abs(two - one) > 1.0


def test_renumbering_another_chain_does_not_change_the_energy(tmp_path):
    _, from_one = _two_chain_energy(tmp_path, b_first=1)
    _, from_eleven = _two_chain_energy(tmp_path, b_first=11)
    assert from_one == from_eleven


@pytest.mark.parametrize(("rel", "inter_chain"), [
    ("rcd6/1M4J.pdb", 78.266785),
    ("rcd6/1DOS.pdb", -295.255619),
])
def test_the_inter_chain_energy_matches_korpe(rel, inter_chain):
    """E(AB) - E(A) - E(B) from the reference korpe binary, on real
    two-chain structures from the KORP bundle. (The per-chain totals are
    not compared: chain A differs from korpe by a pre-existing amount that
    has nothing to do with chains.)"""
    traj = _load(rel)

    def energy(t):
        ff = KORPForceField(t, map_path=_map_path())
        return _simulate(ff, t).context.energy_breakdown(weighted=False)["by_group"][7]

    chains = [traj.atom_slice(traj.topology.select(f"chainid {i}")) for i in range(2)]
    assert energy(traj) - energy(chains[0]) - energy(chains[1]) == pytest.approx(inter_chain, abs=1e-2)
