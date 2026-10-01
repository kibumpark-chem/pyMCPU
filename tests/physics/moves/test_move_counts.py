"""Per-kind move counts: each move counted once, labelled by kind.

The slot getters (``get_bb_*``, ``get_sc_*``, ``get_kic_*``) count the
knowledge-based sub-kinds inside their slot, and the continuous moves have no
counter of their own. ``move_counts()`` splits them into pivot, rama_pivot,
kic, sidechain and rotamer, and reports only the kinds the current settings
can propose.
"""

from __future__ import annotations

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

import pymcpu as mc  # noqa: E402
from pymcpu.runners import default_example_pdb  # noqa: E402

ALL_KINDS = ["pivot", "rama_pivot", "kic", "sidechain", "rotamer"]


@pytest.fixture(scope="module")
def heavy():
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _sim(heavy, *, weights=None, mode=None, rama_p=None, seed=3):
    ff = mc.MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(seed)
    if weights is not None:
        integrator.set_move_weights(*weights)
    if mode is not None:
        integrator.set_sidechain_move_mode(mode)
    if rama_p is not None:
        integrator.set_pivot_rama_probability(rama_p)
    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    return sim, integrator


def test_kinds_sum_to_the_slot_getters(heavy) -> None:
    sim, integ = _sim(heavy, rama_p=0.5)
    sim.step(1500)
    counts = integ.move_counts(include_unused=True)
    assert list(counts) == ALL_KINDS
    for (acc, att), slot in [
        (np.add(counts["pivot"], counts["rama_pivot"]), (integ.get_bb_accepted(), integ.get_bb_attempted())),
        (counts["kic"], (integ.get_kic_accepted(), integ.get_kic_attempted())),
        (np.add(counts["sidechain"], counts["rotamer"]), (integ.get_sc_accepted(), integ.get_sc_attempted())),
    ]:
        assert (int(acc), int(att)) == slot
    assert counts["rama_pivot"][1] > 0 and counts["pivot"][1] > 0
    assert sum(att for _, att in counts.values()) == 1500


def test_default_mode_reports_rotamer_not_sidechain(heavy) -> None:
    sim, integ = _sim(heavy)
    assert list(integ.move_counts()) == ["pivot", "rama_pivot", "kic", "rotamer"]
    sim.step(200)
    assert integ.move_counts()["rotamer"][1] == integ.get_sc_attempted() > 0


def test_continuous_mode_reports_sidechain_not_rotamer(heavy) -> None:
    sim, integ = _sim(heavy, mode="continuous")
    sim.step(200)
    counts = integ.move_counts()
    assert list(counts) == ["pivot", "rama_pivot", "kic", "sidechain"]
    assert counts["sidechain"][1] == integ.get_sc_attempted() > 0


def test_rama_probability_one_moves_all_pivots_to_rama_pivot(heavy) -> None:
    sim, integ = _sim(heavy, rama_p=1.0)
    sim.step(300)
    counts = integ.move_counts()
    assert counts["pivot"] == (0, 0)
    assert counts["rama_pivot"][1] == integ.get_bb_attempted() > 0


def test_backbone_only_weights_report_backbone_kinds_only(heavy) -> None:
    sim, integ = _sim(heavy, weights=(0.5, 0.5, 0.0))
    sim.step(200)
    assert list(integ.move_counts()) == ["pivot", "rama_pivot", "kic"]


def test_a_kind_with_counts_stays_visible_after_weights_change(heavy) -> None:
    sim, integ = _sim(heavy)
    sim.step(200)
    rotamer_before = integ.move_counts()["rotamer"]
    integ.set_move_weights(1.0, 0.0, 0.0)
    counts = integ.move_counts()
    assert counts["rotamer"] == rotamer_before
    assert "sidechain" not in counts


def test_simulation_reporter_prints_the_kinds_in_use(heavy, capfd) -> None:
    sim, _ = _sim(heavy, weights=(0.5, 0.5, 0.0))
    sim.add_simulation_reporter(100)
    sim.step(100)
    out = capfd.readouterr().out
    # Compare whole line labels: "rama_pivot:" contains "pivot:" as a substring.
    labels = {line.split(":", 1)[0] for line in out.splitlines() if ":" in line}
    assert {"pivot", "rama_pivot", "kic"} <= labels
    assert not labels & {"rotamer", "sidechain"}
