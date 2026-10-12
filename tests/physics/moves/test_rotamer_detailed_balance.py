"""Detailed balance of the default rotamer move (``dunbrack2002_sym``),
checked exactly on the engine's own transitions.

The move proposes chi from q and accepts with
min(1, exp(-dE/T) q(x_old) / q(x_new)). For every ordered pair of chi states
(a, b) of a residue, the engine is put in state a, the move is forced to b
(``debug_force_rotamer_to``) and dE and the Hastings term are read back; then
the same from b to a. Detailed balance,

    pi(a) q(b) alpha(a, b) = pi(b) q(a) alpha(b, a),

is then checked in log form with q the density of the move's sampler,
computed independently from ``bbind02.May.lib``
(``tests/fixtures/rotamer2002.reference_log_q``). That q is the sampler's
density is checked on the C++ sampler in ``test_rotamer_library_tables.py``
and, for the engine's own random proposals, below.

Residues of actin cover one to four chi, the symmetric chi of ASP, GLU, PHE
and TYR, and both chain ends. A deliberately wrong Hastings term (none, the
wrong sign, and for a symmetric chi period 2 pi or the ``dunbrack2002``
density) makes the same check fail by orders of magnitude.
"""

from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest
from scipy import stats

from pymcpu import mcpu_core
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX
from pymcpu.forcefields.builders.rotamer_builder import load_rotamer_library
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures import rotamer2002 as ref
from tests.fixtures.context_builders import resolve_test_pdb

T = 1.0
TOL = 2e-3  # float32 energies and densities


class Engine:
    """Actin with the dunbrack2002_sym library attached, and the numpy helpers
    to set and read a residue's chi."""

    def __init__(self):
        pdb = resolve_test_pdb()
        if pdb.name != "acta.pdb":
            pytest.skip("needs the actin example structure")
        traj = md.load(str(pdb))
        heavy = traj.atom_slice(traj.topology.select("not element H"))
        self.ff = MCPUForceField(heavy)
        self.top = heavy.topology
        self.system = self.ff.create_system(self.top)
        self.system.set_rotamer_library(load_rotamer_library("dunbrack2002_sym"))
        self.ctx = mcpu_core.Context(self.system)
        self.ctx.set_use_legacy_weights(True)
        self.native = (self.ff.coords[0] * 10.0).T.astype(np.float64)
        self.set_coords(self.native)
        self.integ = mcpu_core.Integrator(T, 0.1)
        self.integ.set_seed(2024)
        self.n_res = self.top.n_residues
        self.names = [r.name for r in self.top.residues]

    def set_coords(self, coords: np.ndarray) -> None:
        self.ctx.set_positions(coords.astype(np.float32))
        self.ctx.calculate_total_energy(-1)

    def coords(self) -> np.ndarray:
        return np.asarray(self.ctx.coords, dtype=np.float64).T

    def chi(self, r: int) -> np.ndarray:
        """Residue r's chi from the current coordinates, as the move computes it."""
        xyz = self.coords()
        n = ref.N_CHI[self.names[r]]
        return np.array([ref.dihedral(*(xyz[i] for i in quad))
                         for quad in self.ff.chi_atom_indices[r][:n]])

    @staticmethod
    def _rotate(xyz: np.ndarray, moved, b: int, c: int, delta: float) -> None:
        """Rotates xyz[moved] by delta about the axis from atom b to atom c."""
        axis = (xyz[c] - xyz[b]) / np.linalg.norm(xyz[c] - xyz[b])
        k_mat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + math.sin(delta) * k_mat + (1 - math.cos(delta)) * k_mat @ k_mat
        xyz[moved] = (xyz[moved] - xyz[c]) @ rot.T + xyz[c]

    def with_chi(self, coords: np.ndarray, r: int, target) -> np.ndarray:
        """Coordinates with residue r's chi set to `target`, rotating each
        chi's distal atoms about its bond, proximal to distal."""
        xyz = coords.T.copy()
        for k, (quad, (lo, hi)) in enumerate(zip(self.ff.chi_atom_indices[r],
                                                 self.ff.chi_moved_atom_ranges[r])):
            if k >= len(target):
                break
            a, b, c, d = quad
            self._rotate(xyz, slice(lo, hi), b, c,
                         target[k] - ref.dihedral(xyz[a], xyz[b], xyz[c], xyz[d]))
        return xyz.T

    def force(self, r: int, target) -> tuple[float, float]:
        chi4 = [float(x) for x in target] + [0.0] * (4 - len(target))
        assert self.integ.debug_force_rotamer_to(self.ctx, r, chi4)
        return self.integ.last_delta_energy(), self.integ.last_log_jacobian_weight()


@pytest.fixture(scope="module")
def engine():
    return Engine()


def _residues(eng: Engine) -> dict[str, int]:
    """Test residues: one per kind, and both chain ends."""
    def first(names):
        return next(r for r in range(1, eng.n_res - 1) if eng.names[r] in names)

    chosen = {
        "1 chi": first({"VAL", "THR", "SER", "CYS"}),
        "2 chi": first({"ILE", "LEU", "HIS", "ASN", "TRP"}),
        "3 chi": first({"MET", "GLN"}),
        "4 chi": first({"LYS", "ARG"}),
        "symmetric chi2, TYR": first({"TYR"}),
        "symmetric chi2, PHE": first({"PHE"}),
        "symmetric chi2, ASP": first({"ASP"}),
        "symmetric chi3, GLU": first({"GLU"}),
    }
    for label, r in (("N-terminus", 0), ("C-terminus", eng.n_res - 1)):
        if eng.names[r] in ref.N_CHI:
            chosen[label] = r
    return chosen


def _states(eng: Engine, r: int, count: int = 5) -> list[np.ndarray]:
    """Chi states of residue r without a hard-core clash: the native chi, then
    candidates near the row means (most probable rows first), each followed
    by its 180 degree image for a symmetric chi, then random points.
    Clashing candidates (states of zero weight) are skipped."""
    res = eng.names[r]
    sym = ref.SYMMETRIC.get(res)
    w, mu, _ = ref.reference_rows(res)
    rng = np.random.default_rng(r)
    candidates = []
    for k in np.argsort(-w):
        x = mu[k] + rng.normal(0.0, 0.05, len(mu[k]))
        candidates.append(x)
        if sym is not None:
            flipped = x.copy()
            flipped[sym] += math.pi
            candidates.append(flipped)
    candidates += [rng.uniform(-math.pi, math.pi, len(mu[0])) for _ in range(20)]
    states = [eng.chi(r)]
    for x in candidates:
        x = np.angle(np.exp(1j * x))
        if eng.force(r, x)[0] < 100.0:
            states.append(x)
        if len(states) == count:
            break
    assert len(states) >= 3, f"residue {r}: too few clash-free states"
    return states


def _measure(eng: Engine, r: int, states: list[np.ndarray]):
    """dE[a][b], hastings[a][b] and the engine's chi in each state a, from
    the native coordinates with r's chi set."""
    m = len(states)
    d_e = np.zeros((m, m))
    hastings = np.zeros((m, m))
    chi_at = []
    for a in range(m):
        eng.set_coords(eng.with_chi(eng.native, r, states[a]))
        chi_at.append(eng.chi(r))
        for b in range(m):
            if a != b:
                d_e[a, b], hastings[a, b] = eng.force(r, states[b])
    eng.set_coords(eng.native)
    return d_e, hastings, chi_at


def _residual(d_e, hastings, log_q_state, log_q_target) -> float:
    """max over pairs of |log[pi(a) q(b) alpha(a,b)] - log[pi(b) q(a) alpha(b,a)]|."""
    m = len(d_e)
    worst = 0.0
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            log_alpha_ab = min(0.0, -d_e[a, b] / T + hastings[a, b])
            log_alpha_ba = min(0.0, -d_e[b, a] / T + hastings[b, a])
            lhs = d_e[a, b] / T + log_q_target[b] + log_alpha_ab  # log pi(a) - log pi(b) = dE_ab / T
            rhs = log_q_state[a] + log_alpha_ba
            worst = max(worst, abs(lhs - rhs))
    return worst


@pytest.fixture(scope="module")
def measured(engine):
    out = {}
    for label, r in _residues(engine).items():
        states = _states(engine, r)
        out[label] = (r, states, *_measure(engine, r, states))
    return out


def test_detailed_balance_holds_exactly_on_the_engine_transitions(engine, measured):
    for label, (r, states, d_e, hastings, chi_at) in measured.items():
        res = engine.names[r]
        q_state = [ref.reference_log_q(res, x) for x in chi_at]
        q_target = [ref.reference_log_q(res, x) for x in states]
        assert np.allclose(d_e, -d_e.T, atol=1e-3), f"{label}: energy is not a state function"
        assert _residual(d_e, hastings, q_state, q_target) < TOL, (label, res, r)


def test_a_wrong_hastings_term_breaks_the_same_check(engine, measured):
    for label, (r, states, d_e, hastings, chi_at) in measured.items():
        res = engine.names[r]
        q_state = [ref.reference_log_q(res, x) for x in chi_at]
        q_target = [ref.reference_log_q(res, x) for x in states]

        def hastings_from(f):
            m = len(states)
            return np.array([[f(chi_at[a]) - f(states[b]) if a != b else 0.0
                              for b in range(m)] for a in range(m)])

        broken = {"no Hastings term": np.zeros_like(hastings), "wrong sign": -hastings}
        # Without a symmetric chi the two proposals have the same density (up to
        # the truncation: at most 2e-5 of a row's mass, for the widest 2002
        # row), so only a residue with one tells them apart.
        if res in ref.SYMMETRIC:
            broken["period 2 pi"] = hastings_from(
                lambda x, n=res: ref.reference_log_q(n, x, period_pi=False))
            broken["the dunbrack2002 density"] = hastings_from(
                lambda x, n=res: ref.legacy_log_q(n, x))
        for what, wrong in broken.items():
            assert _residual(d_e, wrong, q_state, q_target) > 50 * TOL, (label, what)


def test_engine_proposals_come_from_the_density_of_the_table(engine):
    """The engine's random proposals: log q(new) = log q(old) - hastings has the
    distribution of log q(X), X drawn by the library's sampler (two-sample KS
    test)."""
    lib = engine.system.get_rotamer_library()
    for label, r in _residues(engine).items():
        amino = AMINO_INDEX[engine.names[r]]
        old = [float(x) for x in engine.chi(r)] + [0.0] * (4 - len(engine.chi(r)))
        log_q_old = lib.log_density(amino, old)
        engine_side = []
        for _ in range(3000):
            assert engine.integ.debug_force_rotamer(engine.ctx, r)
            engine_side.append(log_q_old - engine.integ.last_log_jacobian_weight())
        _, draws = lib.sample_chi(amino, 5 + r, 3000)
        library_side = [lib.log_density(amino, row.tolist()) for row in draws]
        assert stats.ks_2samp(engine_side, library_side).pvalue > 1e-3, (label, r)


def test_rotamer_moves_use_current_coordinates_between_backbone_moves(engine):
    """Backbone moves (accepted and rejected pivots and KIC) interleaved with
    rotamer moves: the Hastings term always matches the reference from the
    chi of the current coordinates."""
    residues = list(_residues(engine).values())
    engine.integ.set_move_weights(0.4, 0.4, 0.2)
    for burst in range(4):
        engine.integ.run(engine.ctx, 300)
        for r in residues:
            res = engine.names[r]
            target = engine.chi(r) + 0.7
            _, hastings = engine.force(r, target)
            want = (ref.reference_log_q(res, engine.chi(r))
                    - ref.reference_log_q(res, np.angle(np.exp(1j * target))))
            assert hastings == pytest.approx(want, abs=TOL), (burst, r)
    engine.set_coords(engine.native)
