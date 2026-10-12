"""The ``dunbrack2002_sym`` tables of ``mcpu_core.RotamerLibrary``: built
from ``bbind02.May.lib`` row for row, the density ``log_density`` evaluates,
and that ``sample_chi`` (the move's own sampler) draws from it.

The reference density in ``tests/fixtures/rotamer2002`` reads the file with
its own parser and does not use the builder or the C++ table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pymcpu import mcpu_core
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX
from pymcpu.forcefields.builders.rotamer_builder import load_rotamer_library
from tests.fixtures import rotamer2002 as ref

Lib = mcpu_core.RotamerLibrary


@pytest.fixture(scope="module")
def library():
    return load_rotamer_library("dunbrack2002_sym")


def test_the_table_holds_the_file_rows_with_a_positive_probability(library):
    assert library.has_truncated_tables()
    for res, n in ref.N_CHI.items():
        amino = AMINO_INDEX[res]
        lines = ref.file_rows(res)
        kept = [line for line in lines if line[0] > 0.0]
        assert library.num_rows(amino) == len(kept), res
        rows = [library.table_row(amino, k) for k in range(len(kept))]
        weights = [row[0] for row in rows]
        # Stored by weight, largest first (equal weights draw within 2^-32).
        assert all(a >= b - 1e-9 for a, b in zip(weights, weights[1:])), res
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
        total = sum(line[0] for line in kept)
        # Each stored row is one line: its probability (to the 32-bit draw),
        # its means and widths on the residue's chi.
        unmatched = [(p / total, np.radians(mu[:n]), np.radians(sd[:n])) for p, mu, sd in kept]
        for w, mu, sg in rows:
            for i, (p, mu_ref, sg_ref) in enumerate(unmatched):
                same_angle = np.allclose(np.angle(np.exp(1j * (np.array(mu[:n]) - mu_ref))), 0.0,
                                         atol=1e-6)
                if abs(w - p) < 1e-9 and same_angle and np.allclose(sg[:n], sg_ref, rtol=1e-6):
                    unmatched.pop(i)
                    break
            else:
                raise AssertionError(f"{res}: stored row {w, mu, sg} matches no line of the file")
        assert not unmatched
    # No table for residue types without chi, nor for proline (ring geometry).
    for res in ("GLY", "ALA", "PRO"):
        assert library.num_rows(AMINO_INDEX[res]) == 0
    # LYS and ARG are the types with zero-probability lines.
    assert len(ref.file_rows("LYS")) - library.num_rows(AMINO_INDEX["LYS"]) == 6
    assert len(ref.file_rows("ARG")) - library.num_rows(AMINO_INDEX["ARG"]) == 7


def test_symmetric_chi_have_period_pi(library):
    rng = np.random.default_rng(1)
    for res, n in ref.N_CHI.items():
        amino = AMINO_INDEX[res]
        for _ in range(5):
            chi = rng.uniform(-math.pi, math.pi, 4)
            sym = ref.SYMMETRIC.get(res, n - 1)  # for the others, flip the last chi
            flipped = chi.copy()
            flipped[sym] = math.remainder(flipped[sym] + math.pi, 2 * math.pi)
            a = library.log_density(amino, chi.astype(np.float32).tolist())
            b = library.log_density(amino, flipped.astype(np.float32).tolist())
            if res in ref.SYMMETRIC:
                assert a == pytest.approx(b, abs=1e-4), res
            elif max(a, b) > -8.0:
                assert a != pytest.approx(b, abs=1e-2), res


def test_log_density_matches_the_independent_reference(library):
    rng = np.random.default_rng(3)
    for res, n in ref.N_CHI.items():
        for _ in range(20):
            chi = rng.uniform(-math.pi, math.pi, 4).astype(np.float32)
            got = library.log_density(AMINO_INDEX[res], chi.tolist())
            want = ref.reference_log_q(res, chi[:n].astype(np.float64))
            assert got == pytest.approx(want, abs=2e-5 + 2e-6 * abs(want)), res


@pytest.mark.parametrize("res", ["SER", "TYR", "ASP", "HIS"])
def test_density_integrates_to_one(library, res):
    amino = AMINO_INDEX[res]
    m = 720 if ref.N_CHI[res] == 1 else 240
    x = -math.pi + (np.arange(m) + 0.5) * (2 * math.pi / m)
    if ref.N_CHI[res] == 1:
        vals = [library.log_density(amino, [a, 0, 0, 0]) for a in x]
        total = np.exp(vals).sum() * (2 * math.pi / m)
    else:
        vals = [library.log_density(amino, [a, b, 0, 0]) for a in x for b in x]
        total = np.exp(vals).sum() * (2 * math.pi / m) ** 2
    assert total == pytest.approx(1.0, abs=2e-3), res


def _chi2_pvalue(observed: np.ndarray, expected_prob: np.ndarray) -> float:
    expected = expected_prob / expected_prob.sum() * observed.sum()
    keep = expected > 5
    rest_obs, rest_exp = observed[~keep].sum(), expected[~keep].sum()
    obs = np.append(observed[keep], rest_obs)
    exp = np.append(expected[keep], rest_exp)
    if rest_exp <= 0:
        obs, exp = obs[:-1], exp[:-1]
    return stats.chisquare(obs, exp).pvalue


def _marginal(lib, res, dims, bins, sub=6):
    """Bin masses of the marginal of chi[dims] from the stored rows (weights,
    means, widths), integrated on a sub-grid of each bin."""
    amino = AMINO_INDEX[res]
    symmetric = ref.SYMMETRIC.get(res)
    edges = np.linspace(-math.pi, math.pi, bins + 1)
    fine = (edges[:-1, None] + (np.arange(sub) + 0.5)[None] * (2 * math.pi / bins / sub)).ravel()
    out = 0.0
    for k in range(lib.num_rows(amino)):
        w, mu, sg = lib.table_row(amino, k)
        factor = None
        for d in dims:
            period = math.pi if d == symmetric else 2 * math.pi
            delta = (fine - mu[d] + period / 2) % period - period / 2
            g = np.exp(-0.5 * (delta / sg[d]) ** 2)
            g = g.reshape(bins, sub).sum(axis=1)
            factor = g if factor is None else np.multiply.outer(factor, g)
        out = out + w * factor / factor.sum()
    return out


@pytest.mark.parametrize("res", ["SER", "TYR", "GLU", "ARG"])
def test_sampler_draws_from_the_stored_rows(library, res):
    """Rows drawn with their weights; 1D marginals of every chi and 2D
    marginals of every chi pair match the stored rows."""
    amino = AMINO_INDEX[res]
    n = ref.N_CHI[res]
    rows, chi = library.sample_chi(amino, 12345, 200_000)
    weights = np.array([library.table_row(amino, k)[0] for k in range(library.num_rows(amino))])
    assert _chi2_pvalue(np.bincount(rows, minlength=len(weights)), weights) > 1e-4
    for i in range(n):
        hist = np.histogram(chi[:, i], bins=36, range=(-math.pi, math.pi))[0]
        assert _chi2_pvalue(hist, _marginal(library, res, [i], 36)) > 1e-4, (res, i)
    for i in range(n):
        for j in range(i + 1, n):
            hist = np.histogram2d(chi[:, i], chi[:, j], bins=18, range=[(-math.pi, math.pi)] * 2)[0]
            expect = _marginal(library, res, [i, j], 18)
            assert _chi2_pvalue(hist.ravel(), expect.ravel()) > 1e-4, (res, i, j)


@pytest.mark.parametrize("res", ["TYR", "HIS"])
def test_sampler_draws_from_the_evaluated_density(library, res):
    """2D histogram of the move's draws against the density log_density
    evaluates, integrated over each bin: the truncation, the periods and the
    random image of a symmetric chi are the same in both."""
    amino = AMINO_INDEX[res]
    bins, sub = 24, 5
    centres = -math.pi + (np.arange(bins * sub) + 0.5) * (2 * math.pi / (bins * sub))
    dens = np.array([[math.exp(library.log_density(amino, [a, b, 0, 0])) for b in centres]
                     for a in centres])
    expect = dens.reshape(bins, sub, bins, sub).sum(axis=(1, 3))
    _, chi = library.sample_chi(amino, 777, 300_000)
    hist = np.histogram2d(chi[:, 0], chi[:, 1], bins=bins, range=[(-math.pi, math.pi)] * 2)[0]
    assert _chi2_pvalue(hist.ravel(), expect.ravel()) > 1e-4, res


@pytest.mark.parametrize("res", ["TYR", "PHE", "ASP", "GLU"])
def test_symmetric_chi_is_proposed_on_both_halves(library, res):
    """The 2002 rows cover one half of the symmetric chi; the image gives the
    other half of the proposals."""
    sym = ref.SYMMETRIC[res]
    _, chi = library.sample_chi(AMINO_INDEX[res], 99, 100_000)
    _, mu, _ = ref.reference_rows(res)
    centre = np.angle(np.mean(np.exp(2j * mu[:, sym]))) / 2  # middle of the covered half
    near = np.abs(np.angle(np.exp(1j * (chi[:, sym] - centre)))) < math.pi / 2
    assert near.mean() == pytest.approx(0.5, abs=0.01), res


def test_truncated_tables_are_validated():
    lib = Lib()
    ok = dict(weights=np.full(2, 0.5), means=np.zeros((2, 1)), sigmas=np.full((2, 1), 0.2),
              periods=np.array([2 * math.pi]))
    with pytest.raises(ValueError, match="period"):
        lib.add_truncated_table(15, **{**ok, "periods": np.array([1.0])})
    with pytest.raises(ValueError, match="positive"):
        lib.add_truncated_table(15, **{**ok, "weights": np.array([1.0, 0.0])})
    with pytest.raises(ValueError, match="sigmas"):
        lib.add_truncated_table(15, **{**ok, "sigmas": np.zeros((2, 1))})
    with pytest.raises(ValueError, match=r"\[row\]\[chi\]"):
        lib.add_truncated_table(15, **{**ok, "means": np.zeros((3, 1))})
    with pytest.raises(ValueError, match="too small"):
        lib.add_truncated_table(15, **{**ok, "weights": np.array([1.0, 1e-12])})
    with pytest.raises(IndexError):
        lib.add_truncated_table(20, **ok)
    assert not lib.has_truncated_tables()
    lib.add_truncated_table(15, **ok)
    assert lib.has_truncated_tables() and lib.num_rows(15) == 2 and lib.num_rows(16) == 0
    with pytest.raises(RuntimeError, match="do not mix"):
        lib.add_residue_type(16, [1.0], [[0.0] * 4], [[0.2] * 4])
    rows = Lib()
    rows.add_residue_type(16, [1.0], [[0.0] * 4], [[0.2] * 4])
    with pytest.raises(RuntimeError, match="do not mix"):
        rows.add_truncated_table(15, **ok)
    with pytest.raises(IndexError):
        rows.log_density(16, [0.0] * 4)  # a row table has no truncated density


def test_each_library_is_built_once_and_says_what_it_is():
    sym, legacy = load_rotamer_library("dunbrack2002_sym"), load_rotamer_library("dunbrack2002")
    assert load_rotamer_library() is sym  # the default
    assert load_rotamer_library(" DUNBRACK2002 ") is legacy
    md5 = __import__("hashlib").md5(ref.BBIND_FILE.read_bytes()).hexdigest()
    assert sym.fingerprint == f"dunbrack2002_sym md5={md5} builder=1"
    assert legacy.fingerprint == f"dunbrack2002 md5={md5}"
    assert not legacy.has_truncated_tables()
    assert legacy.num_rows(AMINO_INDEX["LYS"]) == len(ref.file_rows("LYS"))
    with pytest.raises(ValueError, match="rotamer_library must be one of"):
        load_rotamer_library("dunbrack2010")
