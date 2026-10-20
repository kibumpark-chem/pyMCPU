"""The hard-core factor is a force-field option, ``clash_alpha``.

Two atoms that can clash may come no closer than ``clash_alpha`` times the sum
of their radii. mcpu08's 0.75 is the default (for a parameter set whose
``constants/paramset.json`` gives no factor: test_paramset_manifest.py
covers one that does); the option moves the hard core
alone, and the contact cutoff (1.8 x 0.75 = 1.35 radius sums) stays where it
is. Probed with one pair of chignolin, TYR1 CB and TRP8's last side-chain atom
(a contact pair), placed on the side of CB facing away from its neighbours so
that no other pair comes near a clash.
"""

from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

#: The full energy judges a state 0.0015 A (the move cutoff's rounding margin)
#: plus 0.001 A (STATE_CLASH_BUFFER_A) inside the hard core.
STATE_MARGIN = 0.0015 + float(mcpu_core.STATE_CLASH_BUFFER_A)


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _radius(ff: MCPUForceField, index: int) -> float:
    atom = ff.ordered_atom_list[index]
    if atom.residue_name == "GLY" and atom.name == "CA":
        residue = "GLY"
    elif atom.name in ("N", "CA", "C", "O", "OXT", "OCT"):
        residue = "XXX"
    else:
        residue = atom.residue_name
    return ff.atom_type_lookup[(residue, atom.name)][1]


class _Probe:
    """A context of chignolin under ``clash_alpha``, and its probe pair."""

    def __init__(self, heavy: md.Trajectory, **options) -> None:
        ff = MCPUForceField(heavy, **options)
        self.start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
        self.cb = ff.blocks[1].sc_start  # TYR1 CB
        block = ff.blocks[8]
        self.tip = block.sc_start + block.sc_count - 1  # TRP8's last side-chain atom
        assert ff.ordered_atom_list[self.cb].name == "CB"
        self.radius_sum = _radius(ff, self.cb) + _radius(ff, self.tip)
        xyz = self.start.astype(np.float64)
        near = np.linalg.norm(xyz - xyz[:, [self.cb]], axis=0) < 6.0
        near[self.cb] = False
        away = (xyz[:, [self.cb]] - xyz[:, near]).sum(axis=1)
        self.away = away / np.linalg.norm(away)
        self.ctx = mcpu_core.Context(ff.create_system(heavy.topology))

    def _set(self, r: float) -> float:
        coords = self.start.copy()
        xyz = self.start[:, self.cb].astype(np.float64)
        coords[:, self.tip] = (xyz + r * self.away).astype(np.float32)
        self.ctx.set_positions(coords)
        return self.ctx.calculate_total_energy(-1)

    def clashes(self, r: float) -> bool:
        self._set(r)
        return bool(self.ctx.has_steric_clash())

    def energy(self, r: float) -> float:
        energy = self._set(r)
        assert not self.ctx.has_steric_clash()
        return energy

    def clash_distance(self) -> float:
        """The distance under which the full energy calls the pair a clash."""
        lo, hi = 1.0, 4.5
        assert self.clashes(lo) and not self.clashes(hi)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.clashes(mid):
                lo = mid
            else:
                hi = mid
        return hi


def test_the_default_is_mcpu08s(heavy: md.Trajectory) -> None:
    assert MCPUForceField(heavy).clash_alpha == 0.75
    assert MCPUForceField(heavy, clash_alpha=None).clash_alpha == 0.75


@pytest.mark.parametrize("alpha", [0.75, 0.85])
def test_the_hard_core_is_clash_alpha_radius_sums(heavy: md.Trajectory, alpha: float) -> None:
    probe = _Probe(heavy, clash_alpha=alpha)
    expected = alpha * probe.radius_sum - STATE_MARGIN
    assert probe.clash_distance() == pytest.approx(expected, abs=2e-4)


def test_the_contact_cutoff_does_not_move(heavy: md.Trajectory) -> None:
    """Crossing 1.35 radius sums changes the energy by the pair's contact
    energy under both factors: the contact cutoff is not alpha's."""
    steps = []
    for alpha in (0.75, 0.85):
        probe = _Probe(heavy, clash_alpha=alpha)
        cutoff = 1.35 * probe.radius_sum
        steps.append(probe.energy(cutoff - 0.005) - probe.energy(cutoff + 0.005))
    assert steps[0] < -0.1  # the pair makes a contact inside the cutoff
    assert steps[1] == steps[0]


# None is not here: like hbond_beta_favor, it asks for the parameter set's own.
@pytest.mark.parametrize(
    "value",
    [0, 0.0, -0.5, 1.35, 1.5, math.nan, math.inf, True, "0.85"],
    ids=repr,
)
def test_clash_alpha_is_checked(heavy: md.Trajectory, value: object) -> None:
    with pytest.raises(ValueError, match="clash_alpha must be a number above 0 and below 1.35"):
        MCPUForceField(heavy, clash_alpha=value)


def test_a_numpy_value_is_taken_as_a_float(heavy: md.Trajectory) -> None:
    alpha = MCPUForceField(heavy, clash_alpha=np.float32(0.85)).clash_alpha
    assert type(alpha) is float and alpha == float(np.float32(0.85))
