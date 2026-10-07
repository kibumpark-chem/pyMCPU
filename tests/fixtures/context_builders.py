"""Shared pyMCPU ``Context``/``MCPUForceField`` builders for the physics test
suite. Public (not underscore-prefixed) since they're used directly by test
modules that need more control than the ``chignolin_context``/
``chignolin_with_qbias`` fixtures expose (e.g. reorder-mode or virtual-amide-H
knobs).

Default test structure is actin (``examples/actin/input_pdb/acta.pdb``), with
chignolin (``1uao.pdb``) as a fallback when actin isn't present. Fixture names
keep the historical ``chignolin_*`` prefix even though actin is the default,
so existing call sites don't need renaming.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import PACKAGE_ROOT, mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling.collective_variables import (
    NativeContactsCV,
    attach_native_contacts_bias_potential,
    build_ca_index,
    reference_ca_from_pdb,
)

REPO_ROOT = Path(PACKAGE_ROOT).parent

TEST_PDB = REPO_ROOT / "examples" / "actin" / "input_pdb" / "acta.pdb"
FALLBACK_PDBS = (REPO_ROOT / "examples" / "chignolin" / "input_pdb" / "1uao.pdb",)

#: Absolute tolerance for internal (non-legacy) physics self-consistency
#: checks -- e.g. incremental delta-energy vs. a full recomputation. This is
#: NOT a legacy-comparison tolerance; see tests/legacy_parity/framework.py
#: for that.
ATOL = 1e-3


def require_safe_math_for_accept_determinism() -> None:
    """No-op under this project's SAFE-math-only build configuration (kept
    as a call site marker: tests that assert bit-identical accept-bit
    streams call this first, documenting that the assertion assumes
    SAFE-math, in case a future build ever offers an unsafe-math mode where
    it wouldn't hold)."""
    return


def resolve_test_pdb() -> Path:
    if TEST_PDB.exists():
        return TEST_PDB
    for candidate in FALLBACK_PDBS:
        if candidate.exists():
            return candidate
    pytest.skip(f"Test PDB not found at {TEST_PDB}")


def build_test_context(with_qbias: bool = False):
    """Build a ``Context`` the "normal" way: heavy atoms only, default
    forcefield settings, positions loaded from ``resolve_test_pdb()``.

    With ``with_qbias=True``, also attaches a native-contacts bias potential
    targeting half the reference structure's native contact count.
    """
    pdb_path = resolve_test_pdb()
    traj = md.load(str(pdb_path))
    heavy = traj.atom_slice(traj.topology.select("not element H"))

    forcefield = MCPUForceField(heavy)
    system = forcefield.create_system(heavy.topology)

    n0 = None
    if with_qbias:
        ca_idx = build_ca_index(forcefield)
        ref_ca = reference_ca_from_pdb(str(pdb_path))
        q_cv = NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref_ca,
            contact_cutoff=8.0,
            min_seq_sep=3,
            mode="hard",
        )
        attach_native_contacts_bias_potential(system, q_cv)
        n0 = 0.5 * q_cv.n_contacts

    context = mcpu_core.Context(system)
    coords_angstroms = forcefield.coords[0] * 10.0
    context.set_positions(coords_angstroms.T.astype(np.float32))
    context.calculate_total_energy(-1)

    if with_qbias:
        # n0 is a hard native-contact count, not a fraction of Q.
        if hasattr(context, "set_native_contacts_bias"):
            context.set_native_contacts_bias(10.0, float(n0))
        else:
            context.set_q_bias(10.0, float(n0))

    return context, forcefield


def build_raw_context(*, virtual_amide_h: bool = False, reorder: str = "off"):
    """Build a ``Context`` via the from-scratch (non-qbias) construction
    path, exposing ``virtual_amide_h`` and ``set_atom_reorder_mode`` directly
    -- used by tests (KIC counters, proline-skip policy) that need those
    knobs and don't want ``build_test_context``'s qbias/native-contacts
    wiring.
    """
    pdb_path = resolve_test_pdb()
    traj = md.load(str(pdb_path))
    heavy = traj.atom_slice(traj.topology.select("not element H"))

    forcefield = MCPUForceField(heavy, virtual_amide_h=virtual_amide_h)
    system = forcefield.create_system(heavy.topology)
    context = mcpu_core.Context(system)
    context.set_use_legacy_weights(True)
    context.set_atom_reorder_mode(reorder)
    coords_angstroms = (forcefield.coords[0] * 10.0).T.astype(np.float32)
    context.set_positions(coords_angstroms)
    context.calculate_total_energy(-1)
    return context, heavy.topology


@pytest.fixture
def chignolin_context():
    context, _ = build_test_context(with_qbias=False)
    return context


@pytest.fixture
def chignolin_with_qbias():
    context, forcefield = build_test_context(with_qbias=True)
    return context, forcefield


@pytest.fixture(scope="session")
def chignolin_pdb_path() -> str:
    """Path to chignolin (PDB 1uao, 10 residues, 77 atoms) for folding-metric
    tests that need a real, small protein structure independent of the
    actin-default fixtures above.

    Resolved from the INSTALLED package via
    :func:`pymcpu.runners.default_example_pdb`, not from the repo layout, so
    this fixture is correct for a test run against a wheel -- which is what
    an out-of-tree integration's own test suite does.

    ``pymcpu/data/1uao.pdb`` and ``examples/data/1uao.pdb`` are byte-identical
    (md5 8a0e4b6abd7c59145794cec45e031ceb), so switching the source changed no
    measured value.

    This deliberately RAISES rather than falling back. The previous version
    fabricated a 2-atom GLY stub when it could not find a real structure,
    which meant a suite run outside the repo silently measured its physics --
    RMSDs, native-contact fractions, energies -- against a two-atom molecule
    and still went green.
    """
    from pymcpu.runners import default_example_pdb

    installed = default_example_pdb()
    if installed.is_file():
        return str(installed.resolve())

    in_repo = REPO_ROOT / "examples" / "data" / "1uao.pdb"
    if in_repo.is_file():
        return str(in_repo.resolve())

    raise FileNotFoundError(
        f"chignolin structure not found at {installed} (installed package) "
        f"nor {in_repo} (repo). Tests needing a real structure cannot run; "
        f"reinstall pymcpu so pymcpu/data/1uao.pdb is present."
    )


@pytest.fixture(scope="session")
def minimal_pdb_path(tmp_path_factory) -> str:
    """Smallest practical protein PDB for MPI/checkpointing smoke tests.

    Priority: ``examples/data/1uao.pdb`` > ``examples/chignolin/input_pdb/1uao.pdb``
    > ``examples/chignolin_folding/input_pdb/1uao_processed.pdb`` > a
    generated minimal 5-atom glycine PDB.
    """
    candidates = (
        REPO_ROOT / "examples" / "data" / "1uao.pdb",
        REPO_ROOT / "examples" / "chignolin" / "input_pdb" / "1uao.pdb",
        REPO_ROOT / "examples" / "chignolin_folding" / "input_pdb" / "1uao_processed.pdb",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    pdb_dir = tmp_path_factory.mktemp("pdb")
    pdb_path = pdb_dir / "minimal.pdb"
    pdb_path.write_text(
        "ATOM      1  N   GLY A   1       1.000   1.000   1.000  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       1.460   2.400   1.000  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1       2.980   2.400   1.000  1.00  0.00           C\n"
        "ATOM      4  O   GLY A   1       3.600   2.400   2.000  1.00  0.00           O\n"
        "ATOM      5  CB  GLY A   1       0.960   3.200   2.200  1.00  0.00           C\n"
        "END\n"
    )
    return str(pdb_path)
