"""Independent NumPy transliteration of legacy MCPU's hydrogen-bond energy
(``dbfold_actin/MCPU/src_mpi_umbrella/hbonds.h``), used as a parity oracle
for pyMCPU's ``HBondPotential``.

Deliberately reads the legacy *text* parameter files directly
(``jPL3h.energy``, ``seq_dep_hb_mu_low.energy``) rather than pyMCPU's
compiled ``.bin`` tables, so this oracle is independent of pyMCPU's own
parameter-loading code — it cannot pass by construction.

Every formula below is transcribed directly from ``hbonds.h`` with an
inline citation of the exact legacy lines it mirrors; see
``docs/hbond_legacy_parity.md`` for the full atom-offset derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np

# legacy pdb_util.h:444-486 GetAminoNumber() -- exact alphabetical order.
AMINO_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AMINO_INDEX = {name: i for i, name in enumerate(AMINO_ORDER)}

# legacy define.h
HBOND_WEIGHT = 1.35
CUT_SECSTR = 4
RDTHREE_CON = 2.0
BETA_FAVOR = 3.0
HB_CUTOFF = 2.5
HB_INNER = 2.5  # == HB_CUTOFF, so the HB_PENALTY branch is unreachable
HB_PENALTY = 1.0

HBOND_DIM = 9
BIN_SIZE_DEG = 20.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """legacy vector.h Angle() -- radians, via acos of the normalized dot."""
    c = np.clip(np.dot(_unit(v1), _unit(v2)), -1.0, 1.0)
    return float(np.arccos(c))


def _dihedral_from_points(p1, p2, p3, p4) -> float:
    """Dihedral in radians for p1-p2-p3-p4, same convention as legacy's
    struct_calc_dih_ang(b1, b2, b3) called on consecutive bond vectors
    b1=p2-p1, b2=p3-p2, b3=p4-p3 (hbonds.h ~416-432)."""
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    x = np.dot(n1, n2)
    y = np.dot(b1, n2) * np.linalg.norm(b2)
    return float(np.arctan2(y, x))


def load_jpl3h_table(path: str | Path) -> np.ndarray:
    """Parse jPL3h.energy (``jj1..jj7 E ...``) into a dense (3,9,9,9,9,9,9)
    table, legacy default 0 for unlisted bins (hbonds.h:341-349)."""
    table = np.zeros((3, HBOND_DIM, HBOND_DIM, HBOND_DIM, HBOND_DIM, HBOND_DIM, HBOND_DIM), dtype=np.float64)
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        jj = tuple(int(x) for x in parts[:7])
        e = float(parts[7])
        table[jj] = e
    return table


def load_seq_dep_table(path: str | Path) -> np.ndarray:
    """Parse seq_dep_hb_mu_low.energy (``type aa_i aa_j value count1 count2``)
    into a (3,20,20) multiplier table, sign-flipped per hbonds.h:216-223
    (``seq_hb = -1.0 * tmp_val``, so favorable/negative entries become
    positive multipliers)."""
    table = np.zeros((3, 20, 20), dtype=np.float64)
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        hs, i, j = int(parts[0]), int(parts[1]), int(parts[2])
        value = float(parts[3])
        table[hs, i, j] = -1.0 * value
    return table


@dataclass
class DonorAtoms:
    N: np.ndarray
    CA: np.ndarray
    C: np.ndarray
    prev_N: np.ndarray
    prev_CA: np.ndarray
    prev_C: np.ndarray
    next_CA: np.ndarray
    H: np.ndarray


@dataclass
class AcceptorAtoms:
    N: np.ndarray
    CA: np.ndarray
    C: np.ndarray
    O: np.ndarray
    next_N: np.ndarray
    next_CA: np.ndarray
    next_C: np.ndarray
    prev_CA: np.ndarray


def _virtual_amide_h(N, CA, prev_C) -> np.ndarray:
    """Legacy hbonds.h ~409-414: H = N - normalize(normalize(N-CA) + normalize(N-prev_C))."""
    v1 = _unit(N - CA)
    v2 = _unit(N - prev_C)
    h_dir = _unit(v1 + v2)
    return N + h_dir


def _residue_atoms(top: md.Topology, xyz_ang: np.ndarray, r: int) -> dict[str, np.ndarray]:
    """Per-residue heavy-atom coordinate lookup by name, Angstrom."""
    out: dict[str, np.ndarray] = {}
    for atom in top.residue(r).atoms:
        out[atom.name] = xyz_ang[atom.index]
    return out


class LegacyHBondOracle:
    """Reproduces legacy MCPU's ``HydrogenBonds()``/``FoldHydrogenBonds()``.

    Every one of the four correctness fixes this session found is an
    independent, named toggle so the oracle can express every rung of the
    validation ladder in ``docs`` / the plan file.
    """

    def __init__(
        self,
        jpl3h_table: np.ndarray,
        seq_dep_table: np.ndarray | None = None,
        *,
        caca_fixed: bool = True,
        rama_gates: bool = True,
        seq_dep: bool = True,
        beta_favor: bool = True,
    ):
        self.table = jpl3h_table
        self.seq_dep_table = seq_dep_table
        self.caca_fixed = caca_fixed
        self.rama_gates = rama_gates
        self.seq_dep = seq_dep
        self.beta_favor = beta_favor

    def energy(
        self,
        traj: md.Trajectory,
        *,
        secstr: str | None = None,
        frame: int = 0,
    ) -> tuple[float, int]:
        """Returns (raw energy incl. /1000*RDTHREE_CON, n_hbonds)."""
        top = traj.topology
        n_res = top.n_residues
        xyz_ang = traj.xyz[frame] * 10.0  # nm -> Angstrom
        if secstr is None:
            secstr = "C" * n_res
        assert len(secstr) == n_res

        atoms = [_residue_atoms(top, xyz_ang, r) for r in range(n_res)]
        amino = [AMINO_INDEX.get(top.residue(r).name, 0) for r in range(n_res)]

        total = 0.0
        n_bonds = 0
        for i in range(1, n_res - 1):  # donor, hbonds.h: skip termini
            if not {"N", "CA", "C"} <= atoms[i].keys():
                continue
            don = DonorAtoms(
                N=atoms[i]["N"], CA=atoms[i]["CA"], C=atoms[i]["C"],
                prev_N=atoms[i - 1]["N"], prev_CA=atoms[i - 1]["CA"], prev_C=atoms[i - 1]["C"],
                next_CA=atoms[i + 1]["CA"],
                H=np.zeros(3),
            )
            don.H = _virtual_amide_h(don.N, don.CA, don.prev_C)

            for j in range(1, n_res - 1):  # acceptor
                if not {"N", "CA", "C", "O"} <= atoms[j].keys():
                    continue
                acc = AcceptorAtoms(
                    N=atoms[j]["N"], CA=atoms[j]["CA"], C=atoms[j]["C"], O=atoms[j]["O"],
                    next_N=atoms[j + 1]["N"], next_CA=atoms[j + 1]["CA"], next_C=atoms[j + 1]["C"],
                    prev_CA=atoms[j - 1]["CA"],
                )

                e, is_bond, hs = self._pair_energy(i, j, don, acc, secstr, amino)
                if is_bond:
                    total += e
                    n_bonds += 1

        total = total / 1000.0 * RDTHREE_CON
        return total, n_bonds

    def _pair_energy(self, i, j, don, acc, secstr, amino):
        res_dif = abs(i - j)

        # hbonds.h ~394-397: skip[]=4 in hydrogen_jPL3h.data -> |i-j|>=4 required.
        if res_dif < 4:
            return 0.0, False, 0

        # H...O cutoff (hbonds.h ~403-407)
        d_ho2 = float(np.sum((don.H - acc.O) ** 2))
        if d_ho2 > HB_CUTOFF * HB_CUTOFF:
            return 0.0, False, 0

        # CA-CA orientation prefilter (hbonds.h ~409-423: d_CA_n0=D2(donor5,acceptor3),
        # d_CA_0p=D2(donor2,acceptor6), d_CA_np=D2(donor5,acceptor6), d_CA_00=D2(donor2,acceptor3);
        # donor5=don.prev_CA, donor2=don.CA, acceptor3=acc.CA, acceptor6=acc.next_CA.
        # Matches the current (already-correct) C++ is_hydrogen_bond() exactly.)
        min1 = min(
            float(np.sum((don.prev_CA - acc.CA) ** 2)),
            float(np.sum((don.prev_CA - acc.next_CA) ** 2)),
        )
        min2 = min(
            float(np.sum((don.CA - acc.next_CA) ** 2)),
            float(np.sum((don.CA - acc.CA) ** 2)),
        )
        min3 = min(min1, min2)
        if res_dif == 4:
            if min1 > 5.8 * 5.8 or min2 > 5.8 * 5.8:
                return 0.0, False, 0
            if min3 > 5.5 * 5.5:
                return 0.0, False, 0
        else:
            if min1 > 6.0 * 6.0 or min2 > 6.0 * 6.0:
                return 0.0, False, 0
            if min3 > 5.4 * 5.4:
                return 0.0, False, 0
            if secstr[i] == "H" or secstr[j] == "H":
                return 0.0, False, 0

        # Ramachandran gates (hbonds.h ~424-460)
        if self.rama_gates:
            deg = np.degrees(1.0)
            d_phi = _dihedral_from_points(don.prev_C, don.N, don.CA, don.C) * deg + 180.0
            d_psi = _dihedral_from_points(don.prev_N, don.prev_CA, don.prev_C, don.N) * deg + 180.0
            a_phi = _dihedral_from_points(acc.C, acc.next_N, acc.next_CA, acc.next_C) * deg + 180.0
            a_psi = _dihedral_from_points(acc.N, acc.CA, acc.C, acc.next_N) * deg + 180.0

            if res_dif == 4:
                if secstr[i] in ("E", "L") or secstr[j] in ("E", "L"):
                    if d_phi < 180.0 and d_psi < 180.0:
                        return 0.0, False, 0
                    if a_phi < 180.0 and a_psi < 180.0:
                        return 0.0, False, 0
            else:
                if d_phi > 150.0:
                    return 0.0, False, 0
                if 30.0 < d_psi < 210.0:
                    return 0.0, False, 0
                if a_phi > 150.0:
                    return 0.0, False, 0
                if 30.0 < a_psi < 210.0:
                    return 0.0, False, 0
                if secstr[i] == "L" or secstr[j] == "L":
                    return 0.0, False, 0

        # helix/sheet type + CA-CA chain-axis angle (hbonds.h ~453-461)
        if res_dif == 4:
            hs = 0
        else:
            if self.caca_fixed:
                v1 = don.next_CA - don.prev_CA
                v2 = acc.next_CA - acc.prev_CA
            else:
                # pyMCPU's current (buggy) call: cross-chain operands.
                v1 = don.prev_CA - acc.prev_CA
                v2 = don.next_CA - acc.next_CA
            ang_caca = _angle(v1, v2)
            threshold = np.pi / 2.0 if self.caca_fixed else 0.5
            hs = 1 if ang_caca < threshold else 2

        # 7-index table lookup (hbonds.h's jj1..jj7 via ang_PCA/bCA of the
        # (donor,acceptor) pair, its (prev,next) shifted pair, and the
        # (H,O)-centered pair -- matches HydrogenBondPotential.cpp's
        # hydrogen_bond_indices exactly; both codebases share this geometry.)
        idx = self._table_indices(hs, don, acc)
        e_table = float(self.table[idx])

        aa_i, aa_j = amino[i], amino[j]
        seq_factor = 1.0
        if self.seq_dep and self.seq_dep_table is not None:
            seq_factor = float(self.seq_dep_table[hs, aa_i, aa_j])

        e = e_table * seq_factor
        if self.beta_favor and res_dif > 4:
            e *= BETA_FAVOR
        return e, True, hs

    @staticmethod
    def _table_indices(hs, don, acc) -> tuple[int, int, int, int, int, int, int]:
        def bisector(v1, v2):
            return _unit(_unit(v1) + _unit(v2))

        def a_bca(N1, CA1, C1, N2, CA2, C2):
            b1 = bisector(N1 - CA1, C1 - CA1)
            b2 = bisector(N2 - CA2, C2 - CA2)
            return _angle(b1, b2)

        def a_pca(N1, CA1, C1, N2, CA2, C2):
            n1 = _unit(np.cross(N1 - CA1, C1 - CA1))
            n2 = _unit(np.cross(N2 - CA2, C2 - CA2))
            return _angle(n1, n2)

        bin_size = np.radians(BIN_SIZE_DEG)

        interplanar = a_pca(don.N, don.CA, don.C, acc.N, acc.CA, acc.C)
        bisector_a = a_bca(don.N, don.CA, don.C, acc.N, acc.CA, acc.C)

        interplanar_prev = a_pca(don.prev_N, don.prev_CA, don.prev_C, acc.next_N, acc.next_CA, acc.next_C)
        bisector_prev = a_bca(don.prev_N, don.prev_CA, don.prev_C, acc.next_N, acc.next_CA, acc.next_C)

        interplanar_ho = a_pca(don.prev_C, don.N, don.CA, acc.CA, acc.C, acc.next_N)
        bisector_ho = a_bca(don.prev_C, don.N, don.CA, acc.CA, acc.C, acc.next_N)

        j1 = min(int(interplanar / bin_size), HBOND_DIM - 1)
        j2 = min(int(bisector_a / bin_size), HBOND_DIM - 1)
        j3 = min(int(interplanar_prev / bin_size), HBOND_DIM - 1)
        j4 = min(int(bisector_prev / bin_size), HBOND_DIM - 1)
        j5 = min(int(interplanar_ho / bin_size), HBOND_DIM - 1)
        j6 = min(int(bisector_ho / bin_size), HBOND_DIM - 1)
        return (hs, j1, j2, j3, j4, j5, j6)


def legacy_hbond_energy(
    traj: md.Trajectory,
    jpl3h_table: np.ndarray,
    seq_dep_table: np.ndarray | None = None,
    secstr: str | None = None,
    *,
    caca_fixed: bool = True,
    rama_gates: bool = True,
    seq_dep: bool = True,
    beta_favor: bool = True,
    frame: int = 0,
) -> tuple[float, int]:
    """Convenience wrapper around :class:`LegacyHBondOracle`."""
    oracle = LegacyHBondOracle(
        jpl3h_table, seq_dep_table,
        caca_fixed=caca_fixed, rama_gates=rama_gates, seq_dep=seq_dep, beta_favor=beta_favor,
    )
    return oracle.energy(traj, secstr=secstr, frame=frame)
