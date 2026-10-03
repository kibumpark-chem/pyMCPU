"""A full energy recompute also brings Mu's live contact list back in line.

A move's Mu energy change reads the old contacts off a live list. A rigid
pivot does not re-decide the pairs it carries, and its rounding can carry
one across its contact cutoff by about 1e-6 A, so neither the running energy
nor the list sees the change. calculate_total_energy(-1) corrected the
energy but kept the stale entry, and the next move that separated the pair
subtracted it: the energy went one contact off the other way. The recompute
now rewrites the list from the same pass.

Construction: TYR1 CD2 and GLY6 CA, a contact pair, placed exactly on their
contact distance; residue 9 fixed, so every pivot turns the N-terminal side
and pivots at residues 7 and 8 carry the pair rigidly. After each step the
energy is recomputed (as Simulation does), and a move that pulls GLY6 CA
0.05 A away from the pair is checked against the full energy.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

CONTACT_R = 4.914  # 1.35 * (r_i + r_j) for this pair's atom types


def _atom(ff, residue: int, name: str) -> int:
    return next(k for k, a in enumerate(ff.ordered_atom_list) if a.residue_index == residue and a.name == name)


def _radius(ff, atom: int) -> float:
    a = ff.ordered_atom_list[atom]
    residue = "GLY" if (a.residue_name == "GLY" and a.name == "CA") else (
        "XXX" if a.name in ("N", "CA", "C", "O", "OXT") else a.residue_name)
    return ff.atom_type_lookup[(residue, a.name)][1]


def test_a_recompute_refreshes_the_contact_list() -> None:
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    i, j = _atom(ff, 1, "CD2"), _atom(ff, 6, "CA")
    assert abs(1.35 * (_radius(ff, i) + _radius(ff, j)) - CONTACT_R) < 1e-6
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    away = start[:, j].astype(np.float64) - start[:, i]
    away /= np.linalg.norm(away)
    on_cutoff = start.copy()
    on_cutoff[:, j] = (start[:, i] + CONTACT_R * away).astype(np.float32)

    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
    integ.set_seed(1)
    integ.set_move_weights(1.0, 0.0, 0.0)
    integ.set_fixed_residues([9], 10)
    ctx.set_positions(on_cutoff)
    ctx.calculate_total_energy(-1)

    crossings = 0
    for step in range(2000):
        integ.run(ctx, 1, step)
        running = float(ctx.get_state().current_energy)
        crossings += abs(running - ctx.calculate_total_energy(-1)) > 1e-3

        coords = np.asarray(ctx.coords, dtype=np.float32)
        d = coords[:, j].astype(np.float64) - coords[:, i]
        r = np.linalg.norm(d)
        pulled = coords.copy()
        pulled[:, j] = (coords[:, i] + (r + 0.05) * d / r).astype(np.float32)
        old = ctx.get_state()
        new = mcpu_core.State(old)
        new.coords = pulled
        patch = mcpu_core.ProposalPatch(coords.shape[1])
        patch.mark_moved(j)
        patch.is_valid = True
        check = mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old, new, patch, 1, 1e-3)
        assert check.passed, (step, check.message)

        if abs(r - CONTACT_R) > 1e-4:  # a move separated the pair: put it back on the cutoff
            ctx.set_positions(on_cutoff)
            ctx.calculate_total_energy(-1)
        if crossings >= 3:
            break
    assert crossings >= 3  # rigid carries really did round the pair across its cutoff


def test_a_recompute_under_a_mask_leaves_no_masked_list_behind() -> None:
    """Under a residue energy mask, pairs with a masked residue score 0 in the
    full energy and moves do not read the list. A recompute there must not
    refill the list from that pass, or the moves that read it again once the
    mask is cleared would miss every masked contact."""
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    ctx = mcpu_core.Context(system)
    ctx.set_positions(np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32)))
    ctx.calculate_total_energy(-1)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
    integ.set_seed(1)
    integ.run(ctx, 50, 0)  # the list now exists
    ctx.calculate_total_energy(-1)

    system.set_energy_ignored_residues([0, 1, 2], "ignore_all")
    ctx.energy_breakdown()
    ctx.calculate_total_energy(-1)
    system.clear_energy_ignored_residues()

    # Pull each atom near the masked residues 1 A away and check the move.
    coords = np.asarray(ctx.coords, dtype=np.float32)
    atoms = ff.ordered_atom_list
    masked = [k for k, a in enumerate(atoms) if a.residue_index <= 2]
    tried = 0
    for j, a in enumerate(atoms):
        if a.residue_index < 5:
            continue
        near = [k for k in masked if np.linalg.norm(coords[:, k] - coords[:, j]) < 5.5]
        if not near:
            continue
        away = coords[:, j] - coords[:, near].mean(axis=1)
        pulled = coords.copy()
        pulled[:, j] = coords[:, j] + away / np.linalg.norm(away)
        old = ctx.get_state()
        new = mcpu_core.State(old)
        new.coords = pulled
        patch = mcpu_core.ProposalPatch(coords.shape[1])
        patch.mark_moved(j)
        patch.is_valid = True
        check = mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old, new, patch, 1, 1e-3)
        assert check.passed, (j, a.name, check.message)
        tried += 1
    assert tried > 10
