"""Energy terms carry their names, so nothing downstream keeps its own table.

Each force field names its terms where it assigns their energy groups.
``System.energy_terms()``, ``energy_breakdown()["by_name"]`` and the timing
keys of ``step_stats()`` all read those names back.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

import pymcpu as mc  # noqa: E402
from pymcpu import mcpu_core  # noqa: E402
from pymcpu.runners import default_example_pdb  # noqa: E402

MCPU_TERMS = {
    1: "mu",
    2: "backbone_torsion",
    3: "sidechain_torsion",
    4: "hydrogen_bond",
    5: "aromatic",
}


@pytest.fixture(scope="module")
def heavy():
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture()
def mcpu_sim(heavy):
    ff = mc.MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(1)
    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    return sim, system, integrator


def _bias_potential(group: int = 9):
    potential = mcpu_core.NativeContactsBiasPotential([0], [5], 5.0)
    potential.set_energy_group(group)
    return potential


def test_mcpu_terms_are_named(mcpu_sim) -> None:
    _, system, _ = mcpu_sim
    assert system.energy_terms() == MCPU_TERMS


def test_by_name_matches_by_group_and_totals(mcpu_sim) -> None:
    sim, system, _ = mcpu_sim
    sim.step(50)
    for weighted, total_key in [(True, "weighted_total"), (False, "raw_total")]:
        bd = sim.context.energy_breakdown(weighted=weighted)
        assert list(bd["by_name"]) == list(MCPU_TERMS.values())
        for group, name in MCPU_TERMS.items():
            assert bd["by_name"][name] == bd["by_group"][group]
        assert sum(bd["by_name"].values()) == pytest.approx(bd[total_key], abs=1e-4)


def test_step_stats_timings_are_keyed_by_name(mcpu_sim) -> None:
    sim, _, integrator = mcpu_sim
    sim.step(20)
    assert list(integrator.step_stats()["energy_delta_ns"]) == list(MCPU_TERMS.values())


def test_terms_are_sorted_by_group_not_by_registration(mcpu_sim) -> None:
    """A term added last but with the lowest group comes first everywhere --
    the dict order is what the CSV columns and by_name follow."""
    sim, system, _ = mcpu_sim
    extra = _bias_potential(group=0)
    extra.set_name("extra_term")
    system.add_potential(extra)

    assert list(system.energy_terms().items()) == [(0, "extra_term"), *MCPU_TERMS.items()]
    sim.step(5)
    assert list(sim.context.energy_breakdown()["by_name"]) == ["extra_term", *MCPU_TERMS.values()]


def test_unnamed_group_reports_as_group_n(mcpu_sim) -> None:
    _, system, _ = mcpu_sim
    system.add_potential(_bias_potential(group=9))
    assert system.energy_terms()[9] == "group_9"


def test_same_group_same_name_is_allowed(mcpu_sim) -> None:
    _, system, _ = mcpu_sim
    for _ in range(2):
        p = _bias_potential(group=6)
        p.set_name("native_contacts_bias")
        system.add_potential(p)
    assert system.energy_terms()[6] == "native_contacts_bias"


def test_conflicting_names_are_rejected_at_add_time(mcpu_sim) -> None:
    _, system, _ = mcpu_sim
    n_before = len(system.get_potentials())

    other_name_same_group = _bias_potential(group=1)
    other_name_same_group.set_name("contacts")
    with pytest.raises(ValueError, match="energy group 1 has two names"):
        system.add_potential(other_name_same_group)

    same_name_other_group = _bias_potential(group=9)
    same_name_other_group.set_name("mu")
    with pytest.raises(ValueError, match="'mu' is used by groups 1 and 9"):
        system.add_potential(same_name_other_group)

    # A rejected potential is not left behind in the System.
    assert len(system.get_potentials()) == n_before
    assert system.energy_terms() == MCPU_TERMS


@pytest.mark.parametrize(
    "name",
    ["Mu", "9x", "hydrogen-bond", "", "step", "total", "walker_id", "group_3",
     "pivot_accepted", "kic_attempted"],
)
def test_invalid_or_reserved_names_are_rejected(name: str) -> None:
    potential = _bias_potential()
    if name == "":
        potential.set_name(name)  # empty clears the name; allowed
        assert potential.get_name() == ""
        return
    with pytest.raises(ValueError):
        potential.set_name(name)


@pytest.mark.skipif(
    not os.environ.get("KORP_MAP_PATH"),
    reason="set KORP_MAP_PATH to the korp6Dv1.bin energy map",
)
def test_korp_terms_are_named() -> None:
    from pymcpu.forcefields.korp import KORPForceField

    traj = md.load(str(default_example_pdb()))
    ff = KORPForceField(traj, map_path=os.environ["KORP_MAP_PATH"])
    system = ff.create_system(traj.topology)
    # KORP adds the steric guard (8) before the KORP term (7); output is by group.
    assert list(system.energy_terms().items()) == [(7, "korp_6d"), (8, "calpha_excluded_volume")]
