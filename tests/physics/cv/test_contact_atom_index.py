"""CA/CB "contact atom mode" index and reference-coordinate construction.

Covers mode-string normalization/validation, building per-residue
contact-atom index arrays (CA vs. CB, with GLY falling back to its backbone
CA), loading matching reference
coordinates from a PDB, and constructing/attaching a NativeContactsCV+bias
potential in CB mode. Pure physics_internal: every assertion checks
pyMCPU's own forcefield-derived indices/coordinates against pyMCPU's
own outputs, with no legacy MCPU comparison. The YAML config-plumbing half of
the original combined file lives separately in
``tests/config/test_contact_atom_mode_config.py``.
"""

from __future__ import annotations

import os

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.sampling.collective_variables import (
    NativeContactsCV,
    attach_native_contacts_bias_potential,
    build_ca_index,
    build_contact_atom_index,
    normalize_contact_atom_mode,
    reference_ca_from_pdb,
    reference_contact_from_pdb,
)
from tests.fixtures.context_builders import build_test_context, resolve_test_pdb


def test_normalize_defaults_to_ca() -> None:
    assert normalize_contact_atom_mode(None) == "ca"
    assert normalize_contact_atom_mode("") == "ca"
    assert normalize_contact_atom_mode("CB") == "cb"


def test_normalize_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="contact_atom_mode"):
        normalize_contact_atom_mode("cg")


def test_ca_mode_matches_build_ca_index() -> None:
    _, ff = build_test_context(with_qbias=False)
    a = build_contact_atom_index(ff, mode="ca")
    b = build_ca_index(ff)
    assert np.array_equal(a, b)
    assert a.shape[0] == ff.n_res


def test_cb_mode_uses_cb_or_falls_back_to_bb_ca() -> None:
    _, ff = build_test_context(with_qbias=False)
    ca_idx = build_contact_atom_index(ff, mode="ca")
    cb_idx = build_contact_atom_index(ff, mode="cb")
    assert cb_idx.shape == ca_idx.shape
    atoms = ff.ordered_atom_list
    n_diff = 0
    for r in range(ff.n_res):
        atom = atoms[int(cb_idx[r])]
        assert atom.residue_index == r
        if atom.name == "CB":
            n_diff += 1
            assert int(cb_idx[r]) != int(ca_idx[r])
        else:
            assert atom.name == "CA"
            assert int(cb_idx[r]) == int(ca_idx[r])
    assert n_diff > 0, "test PDB should contain at least one residue with a real CB"


class _BackboneOnlyForceField:
    """Stands in for KORPForceField, which needs its 316 MiB map to build:
    per-residue N, CA, C blocks and no sidechain atoms."""

    def __init__(self, residue_names: list[str]) -> None:
        self.blocks = []
        for r in range(len(residue_names)):
            block = mcpu_core.BlockIndices()
            block.bb_start = 3 * r
            block.c_start = 3 * r + 2
            self.blocks.append(block)  # sc_start stays -1
        self.n_res = len(residue_names)
        self.output_topology = md.Topology()
        chain = self.output_topology.add_chain()
        for name in residue_names:
            self.output_topology.add_residue(name, chain)


def test_ca_mode_needs_only_the_residue_blocks() -> None:
    ff = _BackboneOnlyForceField(["ALA", "GLY", "LEU"])
    assert build_contact_atom_index(ff, mode="ca").tolist() == [1, 4, 7]


def test_cb_mode_refuses_a_force_field_without_cbs() -> None:
    """Returning the CA instead would be compared against the reference's
    CB coordinates, so the native contacts would be wrong without any error."""
    ff = _BackboneOnlyForceField(["GLY", "ALA"])
    with pytest.raises(ValueError, match=r"residue 1 \(ALA\).*contact_atom_mode='ca'"):
        build_contact_atom_index(ff, mode="cb")


def test_cb_mode_uses_the_ca_of_a_glycine() -> None:
    ff = _BackboneOnlyForceField(["GLY", "GLY"])
    assert build_contact_atom_index(ff, mode="cb").tolist() == [1, 4]


@pytest.mark.skipif(not os.environ.get("KORP_MAP_PATH"), reason="set KORP_MAP_PATH")
def test_a_korp_session_computes_native_contacts(engine_spec_factory) -> None:
    """At its own native structure every native contact is formed."""
    from pymcpu.sampling import EngineSession

    native_q = engine_spec_factory().cv[0]

    def session(contact_atom_mode: str) -> EngineSession:
        return EngineSession(engine_spec_factory(
            forcefield="korp",
            forcefield_options={"map_path": os.environ["KORP_MAP_PATH"]},
            move_weights=(0.5, 0.5, 0.0),
            cv=(dict(native_q, contact_atom_mode=contact_atom_mode),),
        ))

    korp = session("ca")
    native = korp.coords_from_auxref(korp.spec.pdb)
    assert korp.compute_cv(native).tolist() == [1.0]
    cb = session("cb")
    with pytest.raises(ValueError, match="contact_atom_mode='ca'"):
        cb.compute_cv(native)


def test_gly_cb_mode_uses_backbone_ca() -> None:
    _, ff = build_test_context(with_qbias=False)
    cb_idx = build_contact_atom_index(ff, mode="cb")
    gly_found = False
    for r in range(ff.n_res):
        res_name = next(a.residue_name for a in ff.ordered_atom_list if a.residue_index == r)
        if res_name != "GLY":
            continue
        gly_found = True
        idx = int(cb_idx[r])
        assert ff.ordered_atom_list[idx].name == "CA"
        assert idx == ff.blocks[r].bb_start + 1
    assert gly_found, "test PDB should contain GLY"


def test_reference_contact_from_pdb_ca_matches_reference_ca_from_pdb() -> None:
    pdb = str(resolve_test_pdb())
    a = reference_contact_from_pdb(pdb, mode="ca")
    b = reference_ca_from_pdb(pdb)
    assert np.allclose(a, b)
    assert a.shape[1] == 3


def test_reference_contact_from_pdb_cb_finite_and_gly_uses_ca() -> None:
    pdb = str(resolve_test_pdb())
    xyz = reference_contact_from_pdb(pdb, mode="cb")
    assert np.all(np.isfinite(xyz))
    ref = md.load(pdb)
    assert xyz.shape[0] == ref.n_residues
    for i, res in enumerate(ref.topology.residues):
        if res.name == "GLY":
            ca = [a.index for a in res.atoms if a.name == "CA"][0]
            assert np.allclose(xyz[i], ref.xyz[0, ca] * 10.0)


def test_cb_vs_ca_cv_pair_atoms() -> None:
    _, ff = build_test_context(with_qbias=False)
    pdb = str(resolve_test_pdb())
    ca_idx = build_contact_atom_index(ff, "ca")
    cb_idx = build_contact_atom_index(ff, "cb")
    ref_ca = reference_contact_from_pdb(pdb, "ca")
    ref_cb = reference_contact_from_pdb(pdb, "cb")
    n = min(ca_idx.shape[0], ref_ca.shape[0])
    ca_cv = NativeContactsCV(
        ca_idx[:n], ref_ca[:n], contact_cutoff=8.0, min_seq_sep=3, contact_atom_mode="ca"
    )
    cb_cv = NativeContactsCV(
        cb_idx[:n], ref_cb[:n], contact_cutoff=8.0, min_seq_sep=3, contact_atom_mode="cb"
    )
    assert ca_cv.contact_atom_mode == "ca"
    assert cb_cv.contact_atom_mode == "cb"
    ai, aj = cb_cv.atom_pair_indices()
    atoms = ff.ordered_atom_list
    for a in np.concatenate([ai, aj]):
        assert atoms[int(a)].name in ("CB", "CA")


def test_cb_mode_bias_potential_attaches_and_evaluates_finite() -> None:
    ctx, ff = build_test_context(with_qbias=False)
    pdb = str(resolve_test_pdb())
    idx = build_contact_atom_index(ff, "cb")
    ref = reference_contact_from_pdb(pdb, "cb")
    n = min(idx.shape[0], ref.shape[0], ctx.get_system().get_num_residues())
    cv = NativeContactsCV(
        idx[:n], ref[:n], contact_cutoff=8.0, min_seq_sep=3, contact_atom_mode="cb"
    )
    # Attach to the live system (same atom ordering as indices).
    potential = attach_native_contacts_bias_potential(ctx.get_system(), cv)
    assert potential is not None
    ctx.calculate_total_energy(-1)
    e = float(ctx.calculate_total_energy(6))
    assert np.isfinite(e)


def test_cb_mode_explicit_pairs_bias_potential_attaches_and_evaluates_finite() -> None:
    """Same end-to-end path as the derived-pairs test above, but with a
    hand-specified ``native_contact_pairs`` list instead of
    ``contact_cutoff``/``min_seq_sep`` derivation."""
    ctx, ff = build_test_context(with_qbias=False)
    pdb = str(resolve_test_pdb())
    idx = build_contact_atom_index(ff, "cb")
    ref = reference_contact_from_pdb(pdb, "cb")
    n = min(idx.shape[0], ref.shape[0], ctx.get_system().get_num_residues())
    pairs = [[0, n - 1], [1, n - 2]]
    cv = NativeContactsCV(
        idx[:n],
        ref[:n],
        contact_atom_mode="cb",
        native_contact_pairs=pairs,
    )
    assert cv.n_contacts == len(pairs)
    assert cv.native_contact_pairs is not None
    # Attach to the live system (same atom ordering as indices).
    potential = attach_native_contacts_bias_potential(ctx.get_system(), cv)
    assert potential is not None
    ctx.calculate_total_energy(-1)
    e = float(ctx.calculate_total_energy(6))
    assert np.isfinite(e)


def test_cb_mode_explicit_pair_referencing_gly_resolves_to_backbone_ca() -> None:
    """An explicit pair that names a GLY residue resolves to that residue's
    backbone CA under ``contact_atom_mode="cb"`` -- same behavior as
    :func:`test_gly_cb_mode_uses_backbone_ca`, exercised through the
    explicit-pairs path (no new resolution logic)."""
    _, ff = build_test_context(with_qbias=False)
    pdb = str(resolve_test_pdb())
    idx = build_contact_atom_index(ff, "cb")
    ref = reference_contact_from_pdb(pdb, "cb")
    n = min(idx.shape[0], ref.shape[0], ff.n_res)

    gly_res = None
    for r in range(n):
        res_name = next(a.residue_name for a in ff.ordered_atom_list if a.residue_index == r)
        if res_name == "GLY":
            gly_res = r
            break
    assert gly_res is not None, "test PDB should contain GLY"
    other = 0 if gly_res != 0 else n - 1

    cv = NativeContactsCV(
        idx[:n],
        ref[:n],
        contact_atom_mode="cb",
        native_contact_pairs=[[gly_res, other]],
    )
    ai, aj = cv.atom_pair_indices()
    gly_atom_idx = int(ai[0])  # pairs_i[0] == gly_res, so ai[0] is its resolved atom
    assert ff.ordered_atom_list[gly_atom_idx].name == "CA"
    assert gly_atom_idx == ff.blocks[gly_res].bb_start + 1
