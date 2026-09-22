"""NativeContactsBiasPotential: bound C++ potential vs. hand-derived harmonic energy.

Builds a minimal 4-residue system, attaches ``NativeContactsBiasPotential`` for a
hard-cutoff native-contact set of 2 (see the derivation below), and checks
the bound potential reproduces ``U = 0.5 * k * (N - N0)^2`` exactly at native
geometry -- a first-principles physics self-consistency check with no legacy
reference involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import Integrator, NativeContactsBiasPotential, Simulation
from pymcpu.sampling.collective_variables import NativeContactsCV
from tests.physics.helpers.minimal_system_builders import setup_minimal_bb_system

# Reference geometry: 4 residues on a line, spaced 5 Angstrom apart, treated
# as 4 contact (CA) atoms. contact_cutoff=11 + min_seq_sep=2 keeps pairs
# (0,2) and (1,3) (d=10 each) but excludes (0,3) (d=15) -> n_contacts == 2.
_N_RES = 4
_CA_IDX = np.array([0, 1, 2, 3], dtype=np.int64)
_REF_XYZ = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0], [15.0, 0, 0]], dtype=np.float64)
_K_SPRING = 4.0
_N0 = 0.0


def _build_bound_simulation() -> Simulation:
    system, _ = setup_minimal_bb_system(_N_RES, _N_RES * 4)
    cv = NativeContactsCV(
        _CA_IDX, _REF_XYZ, contact_cutoff=11.0, min_seq_sep=2, mode="hard", q_cutoff=11.0
    )
    assert cv.n_contacts == 2  # matches the hand-derivation above

    atom_i, atom_j = cv.atom_pair_indices()
    potential = NativeContactsBiasPotential(atom_i, atom_j, float(cv.q_cutoff))
    potential.set_energy_group(6)
    system.add_potential(potential)

    integ = Integrator(1.0, 0.05)
    # Topology argument is unused by Simulation's constructor for this
    # smoke-level test; a bare placeholder keeps the fixture minimal.
    sim = Simulation(object(), system, integ)  # type: ignore[arg-type]
    coords = np.zeros((3, system.get_num_atoms()), dtype=np.float32)
    coords[0, :4] = [0.0, 5.0, 10.0, 15.0]
    sim.context.set_positions(coords)
    return sim


def test_harmonic_bias_energy_at_native_geometry() -> None:
    sim = _build_bound_simulation()
    sim.context.set_native_contacts_bias(_K_SPRING, _N0)
    # At native geometry N=2 -> U = 0.5 * 4 * (2-0)^2 = 8.
    e = float(sim.context.calculate_total_energy(6))
    assert e == pytest.approx(8.0, abs=1e-4)


def test_zero_spring_constant_gives_zero_bias_energy() -> None:
    sim = _build_bound_simulation()
    sim.context.set_native_contacts_bias(0.0, _N0)
    e0 = float(sim.context.calculate_total_energy(6))
    assert e0 == pytest.approx(0.0, abs=1e-6)


def _build_bound_simulation_explicit_pairs() -> Simulation:
    """Same 4-residue-on-a-line geometry as ``_build_bound_simulation``, but
    the native-contact set is hand-specified via ``native_contact_pairs``
    instead of derived from ``contact_cutoff``/``min_seq_sep``. Pairs (0, 2)
    and (1, 3) are the same two pairs (d=10 each) the cutoff derivation picks
    -- q_cutoff=11.0 must still be passed explicitly, since contact_cutoff's
    role as the q_cutoff fallback is independent of pair-selection mode.
    """
    system, _ = setup_minimal_bb_system(_N_RES, _N_RES * 4)
    cv = NativeContactsCV(
        _CA_IDX,
        _REF_XYZ,
        mode="hard",
        q_cutoff=11.0,
        native_contact_pairs=[[0, 2], [1, 3]],
    )
    assert cv.n_contacts == 2  # matches the hand-derivation above

    atom_i, atom_j = cv.atom_pair_indices()
    potential = NativeContactsBiasPotential(atom_i, atom_j, float(cv.q_cutoff))
    potential.set_energy_group(6)
    system.add_potential(potential)

    integ = Integrator(1.0, 0.05)
    # Topology argument is unused by Simulation's constructor for this
    # smoke-level test; a bare placeholder keeps the fixture minimal.
    sim = Simulation(object(), system, integ)  # type: ignore[arg-type]
    coords = np.zeros((3, system.get_num_atoms()), dtype=np.float32)
    coords[0, :4] = [0.0, 5.0, 10.0, 15.0]
    sim.context.set_positions(coords)
    return sim


def test_harmonic_bias_energy_at_native_geometry_with_explicit_pairs() -> None:
    sim = _build_bound_simulation_explicit_pairs()
    sim.context.set_native_contacts_bias(_K_SPRING, _N0)
    # At native geometry N=2 -> U = 0.5 * 4 * (2-0)^2 = 8, same as the
    # cutoff-derived test above.
    e = float(sim.context.calculate_total_energy(6))
    assert e == pytest.approx(8.0, abs=1e-4)
