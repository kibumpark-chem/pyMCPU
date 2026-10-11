"""Regression tests for the KIC loop-closure and pivot geometry fixes.

Every move in pyMCPU is meant to keep bond lengths and bond angles fixed. Before
this fix they did not:

* The KIC solver's back-substitution (as half-tangent ratios) became rounding
  noise when a torsion was near 180 degrees, and returned "closed" windows with
  N-CA-C wrong by up to ~40 degrees. Nothing checked them, and each move took its
  targets from the current coordinates, so every bad closure became the next
  move's target. N-CA-C random-walked without bound.
* KIC changed proline phi.
* N-terminal psi pivots swung the carbonyl O(r) out of the peptide plane.
* KIC's Jacobian depended on how the molecule sat in the lab frame.
* KIC left stale cached backbone torsions on neighbouring residues, so the
  incremental energy of later moves was wrong.
* KIC carried each window residue's O, CB and sidechain with frame maths done
  in float32, which stretched sidechain bonds and shortened CA-CB a little on
  every accepted move.

Each test below fails on the unfixed engine. All run on chignolin (10 residues,
PRO at index 3) and take seconds.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.physics.helpers.minimal_system_builders import setup_minimal_bb_system

SEED = 20260928
PRO = 3                  # chignolin is G Y D P E T G T W G
# Float32 storage: every rigid rotation re-rounds the atoms it moves (~1e-6 A), a
# random walk that reaches ~0.005 deg and ~1e-4 A over the 60k steps below. The
# bugs these guard against moved N-CA-C by 0.3 deg and O=C-N by 37-72 deg in the
# same run, so the bounds sit well clear of both.
ANGLE_TOL_DEG = 0.03
LENGTH_TOL_A = 1e-3


def _chignolin(*, virtual_amide_h: bool = True):
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, virtual_amide_h=virtual_amide_h)
    system = forcefield.create_system(heavy.topology)
    context = mcpu_core.Context(system)
    start = (forcefield.coords[0] * 10.0).T.astype(np.float32)
    context.set_positions(start)
    context.calculate_total_energy(-1)
    assert system.is_proline(PRO)
    return system, context, start


def _integrator(weights, temperature=1.0, seed=SEED):
    integrator = mcpu_core.Integrator(temperature=temperature, step_size_rad=0.1)
    integrator.set_seed(seed)
    integrator.set_move_weights(*weights)
    return integrator


def _atoms(system):
    blocks = system.get_block_indices()
    return {
        "N": np.array([b.bb_start for b in blocks]),
        "CA": np.array([b.ca_atom() for b in blocks]),
        "C": np.array([b.c_atom() for b in blocks]),
        "O": np.array([b.o_start for b in blocks]),
        "H": np.array([b.h_start for b in blocks]),
    }


def _angle(a, b, c):
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    y = np.linalg.norm(b2, axis=-1) * np.sum(b1 * n2, axis=-1)
    return np.degrees(np.arctan2(y, np.sum(n1 * n2, axis=-1)))


def _geometry(coords, idx):
    """Every backbone bond length, bond angle and omega, plus amide H if explicit."""
    X = np.asarray(coords, dtype=np.float64).T
    N, CA, C, O = X[idx["N"]], X[idx["CA"]], X[idx["C"]], X[idx["O"]]
    a, b = slice(None, -1), slice(1, None)
    lengths = {
        "N-CA": np.linalg.norm(CA - N, axis=1),
        "CA-C": np.linalg.norm(C - CA, axis=1),
        "C=O": np.linalg.norm(O - C, axis=1),
        "C-N": np.linalg.norm(N[b] - C[a], axis=1),
    }
    angles = {
        "N-CA-C": _angle(N, CA, C),
        "CA-C=O": _angle(CA, C, O),
        "CA-C-N": _angle(CA[a], C[a], N[b]),
        "C-N-CA": _angle(C[a], N[b], CA[b]),
        "O=C-N": _angle(O[a], C[a], N[b]),
    }
    dihedrals = {"omega": _dihedral(CA[a], C[a], N[b], CA[b])}
    h = np.flatnonzero(idx["H"] >= 0)
    h = h[h >= 1]
    if len(h):
        H = X[idx["H"][h]]
        lengths["N-H"] = np.linalg.norm(H - N[h], axis=1)
        angles["C-N-H"] = _angle(C[h - 1], N[h], H)
        angles["CA-N-H"] = _angle(CA[h], N[h], H)
    return lengths, angles, dihedrals


def _worst_change(start, now):
    worst = {}
    for kind in range(3):
        for name, v0 in start[kind].items():
            d = np.abs(now[kind][name] - v0)
            if kind == 2:
                d = np.minimum(d, 360.0 - d)
            worst[name] = float(d.max())
    return worst


@pytest.mark.parametrize("virtual_amide_h", [True, False])
def test_accepted_moves_keep_backbone_geometry(virtual_amide_h):
    """N-CA-C, O=C-N and every other bond length and angle stay at their start
    values, to float precision, over thousands of accepted pivot and KIC moves.

    Before the fix N-CA-C walked away through bad KIC closures and O=C-N broke on
    every accepted N-terminal psi pivot (chignolin's residues 1-4 take that
    branch). With explicit amide H, the pivot and the phi driver also left H
    behind.
    """
    system, context, start = _chignolin(virtual_amide_h=virtual_amide_h)
    idx = _atoms(system)
    assert (idx["H"] >= 0).any() == (not virtual_amide_h)
    reference = _geometry(start, idx)
    integrator = _integrator((0.5, 0.5, 0.0))

    worst = {}
    for _ in range(30):
        integrator.run(context, 2000)
        now = _geometry(context.coords, idx)
        for name, value in _worst_change(reference, now).items():
            worst[name] = max(worst.get(name, 0.0), value)

    stats = integrator.move_stats()
    assert stats["num_accept_kic"] > 200 and stats["num_accept_pivot"] > 200, stats
    angles_deg = {k: v for k, v in worst.items() if k not in ("N-CA", "CA-C", "C=O", "C-N", "N-H")}
    lengths_a = {k: v for k, v in worst.items() if k in ("N-CA", "CA-C", "C=O", "C-N", "N-H")}
    assert max(angles_deg.values()) < ANGLE_TOL_DEG, angles_deg
    assert max(lengths_a.values()) < LENGTH_TOL_A, lengths_a


def test_kic_keeps_the_bonds_it_carries():
    """Every heavy-atom bond length stays at its start value over 200k KIC-only
    steps (about 19k accepted moves).

    KIC moves a window residue's dependent atoms (O, the sidechain, H) rigidly
    with its backbone frame. With that frame maths in float32 the carried bonds
    drifted steadily, not as a random walk: sidechain bonds grew by up to 1e-3 A
    and C=O changed by up to 6e-4 A here. Done in double and rounded once, they
    move by about 2e-5 A, the same float noise as every other atom."""
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    heavy.topology.create_standard_bonds()
    forcefield = MCPUForceField(heavy)
    engine = {a.original_index: k for k, a in enumerate(forcefield.ordered_atom_list)}
    bonds = np.array([(engine[a.index], engine[b.index]) for a, b in heavy.topology.bonds])
    context = mcpu_core.Context(forcefield.create_system(heavy.topology))
    start = (forcefield.coords[0] * 10.0).T.astype(np.float32)
    context.set_positions(start)
    context.calculate_total_energy(-1)

    def lengths(coords):
        X = np.asarray(coords, dtype=np.float64)
        return np.linalg.norm(X[:, bonds[:, 0]] - X[:, bonds[:, 1]], axis=0)

    integrator = mcpu_core.Integrator(temperature=0.6)
    integrator.set_seed(1)
    integrator.set_move_weights(0.0, 1.0, 0.0)
    integrator.run(context, 200_000)
    assert integrator.get_kic_accepted() > 15_000
    worst = float(np.abs(lengths(context.coords) - lengths(start)).max())
    assert worst < 1e-4, f"a bond changed by {worst:.2e} A under KIC"


def test_kic_never_changes_proline_phi():
    """KIC must skip any window whose phi it would change at a proline: r, r+1,
    r+2, and r+3 for the phi driver (legacy loop.h refuses the same set)."""
    system, context, start = _chignolin()
    idx = _atoms(system)

    def pro_phi(coords):
        X = np.asarray(coords, dtype=np.float64).T
        return float(_dihedral(X[idx["C"][PRO - 1]], X[idx["N"][PRO]],
                               X[idx["CA"][PRO]], X[idx["C"][PRO]]))

    phi0 = pro_phi(start)
    integrator = _integrator((0.0, 1.0, 0.0))
    worst = 0.0
    for _ in range(20):
        integrator.run(context, 1000)
        d = abs(pro_phi(context.coords) - phi0)
        worst = max(worst, min(d, 360.0 - d))

    assert worst < 1e-4, f"proline phi moved by {worst:.4f} deg under KIC"
    assert integrator.get_kic_accepted() > 100
    assert integrator.get_kic_proline_skipped() > 0
    assert integrator.move_stats()["kic_proline_skipped"] == integrator.get_kic_proline_skipped()


def _twist_jacobian(n, a, c):
    """1/|det| of the six window torsion axes' twists (u, p x u): the
    orientation-free loop-closure Jacobian, written independently in NumPy."""
    cols = []
    for i in range(3):
        for p, q in ((n[i], a[i]), (a[i], c[i])):
            u = (q - p) / np.linalg.norm(q - p)
            cols.append(np.concatenate([u, np.cross(q - a[0], u)]))
    return 1.0 / abs(np.linalg.det(np.array(cols).T))


def test_kic_jacobian_is_orientation_free():
    """The Jacobian must not depend on how the molecule sits in the lab frame.

    The old body used the lab x/y components of the CA3->C3 bond and returned
    J_true / |u_z|; a global rotation changed a phi-driver move's log-weight by
    up to 0.39.
    """
    system, _, start = _chignolin()
    idx = _atoms(system)
    X = np.asarray(start, dtype=np.float64).T
    solver = mcpu_core.TripeptideSolver()
    rng = np.random.default_rng(SEED)

    def jacobian(n, a, c):
        sol = mcpu_core.Solution()
        sol.r_n, sol.r_a, sol.r_c = list(n), list(a), list(c)
        return solver.calculate_jacobian(sol)

    worst_rotation, worst_numpy, n_windows = 0.0, 0.0, 0
    for r in range(system.get_num_residues() - 2):
        n = X[idx["N"][r:r + 3]]
        a = X[idx["CA"][r:r + 3]]
        c = X[idx["C"][r:r + 3]]
        j0 = jacobian(n, a, c)
        assert j0 > 0.0
        worst_numpy = max(worst_numpy, abs(j0 / _twist_jacobian(n, a, c) - 1.0))
        for _ in range(25):
            q, rr = np.linalg.qr(rng.normal(size=(3, 3)))
            q = q @ np.diag(np.sign(np.diag(rr)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1.0
            t = rng.uniform(-50.0, 50.0, size=3)
            j = jacobian(n @ q.T + t, a @ q.T + t, c @ q.T + t)
            worst_rotation = max(worst_rotation, abs(j / j0 - 1.0))
        n_windows += 1

    assert n_windows == 8
    assert worst_rotation < 1e-9, worst_rotation
    assert worst_numpy < 1e-9, worst_numpy


def test_incremental_energy_matches_full_recompute_after_kic():
    """After every accepted move, the engine's own dE equals the change of a full
    recompute, and no cached backbone torsion is stale.

    KIC moves O(r) (phi driver) and O(r-1), N(r+2) (psi driver), and residue k's
    cached pCA/bCA read N, CA, O of k-1 and k+1. The old refresh set missed r-1
    (phi) and r-2, r+3 (psi), so the next move touching them was billed the
    difference -- by more than 0.1 on 4-9 % of KIC moves.
    """
    system, context, _ = _chignolin()
    n_res = system.get_num_residues()
    calculator = mcpu_core.Context(system)

    def full(coords):
        calculator.set_positions(coords)
        energy = calculator.calculate_total_energy(-1)
        return energy, np.array([[t.phi, t.psi, t.p_ca, t.b_ca]
                                 for t in calculator.get_state().backbone_torsions])

    integrator = _integrator((0.25, 0.75, 0.0))
    coords = np.asarray(context.coords, dtype=np.float32).copy()
    energy, _ = full(coords)
    accepted = integrator.get_bb_accepted() + integrator.get_kic_accepted()
    worst_de, worst_cache, n_kic = 0.0, 0.0, 0
    for step in range(3000):
        kic_before = integrator.get_kic_accepted()
        integrator.run(context, 1, step)
        now = integrator.get_bb_accepted() + integrator.get_kic_accepted()
        if now == accepted:
            continue
        accepted = now
        n_kic += integrator.get_kic_accepted() - kic_before
        coords = np.asarray(context.coords, dtype=np.float32).copy()
        new_energy, fresh = full(coords)
        worst_de = max(worst_de, abs(integrator.last_delta_energy() - (new_energy - energy)))
        energy = new_energy
        cached = np.array([[t.phi, t.psi, t.p_ca, t.b_ca]
                           for t in context.get_state().backbone_torsions])
        d = np.abs(cached - fresh)[1:n_res - 1]
        worst_cache = max(worst_cache, float(np.minimum(d, 2.0 * np.pi - d).max()))

    assert n_kic > 50, n_kic
    assert worst_cache < 1e-4, f"stale cached backbone torsion, off by {worst_cache:.3g} rad"
    assert worst_de < 1e-2, f"incremental dE off by {worst_de:.3g} from a full recompute"


def test_reverse_check_refuses_windows_that_do_not_match_the_start():
    """A window the solver cannot reproduce from the start-structure targets
    cannot be reached by KIC, so a move out of it has no reverse. Those moves are
    refused and counted in ``kic_reverse_missing``.

    Moving C(5) by 0.05 A gives residues 4-6 windows whose lengths and angles no
    longer match the start; everything else still matches.
    """
    system, context, start = _chignolin()
    integrator = _integrator((0.0, 1.0, 0.0))
    integrator.run(context, 4000)
    clean = integrator.get_kic_reverse_missing()
    clean_rate = clean / integrator.get_kic_attempted()

    idx = _atoms(system)
    distorted = start.copy()
    distorted[0, idx["C"][5]] += 0.05
    system2, context2, _ = _chignolin()
    context2.set_positions(distorted)
    context2.calculate_total_energy(-1)
    integrator2 = _integrator((0.0, 1.0, 0.0))
    integrator2.run(context2, 4000)
    refused = integrator2.get_kic_reverse_missing()

    assert clean_rate < 1e-2, (clean, integrator.get_kic_attempted())
    assert refused > 10 * max(clean, 1), (refused, clean)
    assert integrator2.move_stats()["kic_reverse_missing"] == refused
    # A refused move writes nothing: the distortion is still exactly there.
    X0 = np.asarray(distorted, dtype=np.float64).T
    X1 = np.asarray(context2.coords, dtype=np.float64).T
    ca5, c5 = idx["CA"][5], idx["C"][5]
    assert abs(np.linalg.norm(X1[c5] - X1[ca5]) - np.linalg.norm(X0[c5] - X0[ca5])) < 1e-4


def test_create_system_stores_the_start_structure_targets():
    """The closure targets come from the start coordinates (double precision on
    the float32 array every replica is positioned with) and never move."""
    system, context, start = _chignolin()
    assert system.has_kic_reference()
    idx = _atoms(system)
    X = np.asarray(start, dtype=np.float64).T
    N, CA, C = X[idx["N"]], X[idx["CA"]], X[idx["C"]]
    a, b = slice(None, -1), slice(1, None)
    expected = {
        "len_na": np.linalg.norm(CA - N, axis=1),
        "len_ac": np.linalg.norm(C - CA, axis=1),
        "ang_nac": np.radians(_angle(N, CA, C)),
        "len_cn": np.linalg.norm(N[b] - C[a], axis=1),
        "ang_acn": np.radians(_angle(CA[a], C[a], N[b])),
        "ang_cna": np.radians(_angle(C[a], N[b], CA[b])),
        "omega": np.radians(_dihedral(CA[a], C[a], N[b], CA[b])),
    }
    before = system.get_kic_reference()
    for key, value in expected.items():
        assert len(before[key]) == len(value), key
        d = np.asarray(before[key]) - value
        if key == "omega":                      # wrap at +-pi
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
        assert np.abs(d).max() < 1e-9, (key, np.abs(d).max())

    # Moving the chain (and so every replica swap or checkpoint restore, which
    # only replace positions) leaves the stored targets bit-identical.
    _integrator((0.5, 0.5, 0.0)).run(context, 2000)
    context.set_positions(start)
    after = system.get_kic_reference()
    for key in expected:
        assert list(after[key]) == list(before[key]), key


def test_kic_refuses_a_system_without_start_targets():
    """A hand-built System has no start structure to take targets from. KIC
    raises instead of measuring them from whatever the chain looks like now."""
    n_res = 8
    n_atoms = 4 * n_res
    system, context = setup_minimal_bb_system(n_res, n_atoms)
    coords = np.zeros((3, n_atoms), dtype=np.float32)
    for r in range(n_res):
        coords[0, 3 * r:3 * r + 3] = [3.8 * r, 3.8 * r + 1.2, 3.8 * r + 2.4]
        coords[0, 3 * n_res + r] = 3.8 * r + 2.9
    context.set_positions(coords)
    assert not system.has_kic_reference()
    with pytest.raises(RuntimeError, match="set_kic_reference"):
        _integrator((0.0, 1.0, 0.0)).run(context, 50)
    with pytest.raises(ValueError):
        system.set_kic_reference(coords[:, :-1])
    system.set_kic_reference(coords)
    assert system.has_kic_reference()
