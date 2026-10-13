"""``EngineSession.fingerprint`` covers the parameter files and the sheet
factor, as checkpoints do.

The fingerprint lets an external sampler refuse a restart state built from a
different system. It now includes the parameter set the force field was
built from (``forcefield_options`` can name another than ``spec.param_set``)
and the signature of its files, so the same files elsewhere give the same
fingerprint and other tables another one, and the hydrogen-bond sheet factor
the force field resolved, which checkpoints check too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pymcpu.params import ensure_params, param_signature
from pymcpu.sampling import EngineSession, compute_fingerprint
from tests.helpers.param_folders import (
    changed_param_folder,
    linked_param_folder,
    register_set_like_mcpu08,
)


def test_the_fingerprint_covers_the_parameter_files(
    engine_spec_factory, tmp_path: Path, monkeypatch
) -> None:
    spec = engine_spec_factory()
    session = EngineSession(spec)
    n_atoms = session.coords().shape[1]
    signature = param_signature(ensure_params("mcpu08"))
    assert session.fingerprint == compute_fingerprint(
        spec.pdb, "mcpu08", n_atoms, param_signature=signature, hbond_beta_favor=2.0
    )

    linked = str(linked_param_folder(tmp_path / "linked"))
    assert EngineSession(engine_spec_factory(param_dir=linked)).fingerprint == session.fingerprint
    changed = str(changed_param_folder(tmp_path / "changed"))
    assert EngineSession(engine_spec_factory(param_dir=changed)).fingerprint != session.fingerprint

    name = register_set_like_mcpu08(monkeypatch)
    other_set = EngineSession(
        engine_spec_factory(forcefield_options={"param_set": name, "param_dir": linked})
    )
    assert other_set.fingerprint == compute_fingerprint(
        spec.pdb, name, n_atoms, param_signature=signature, hbond_beta_favor=2.0
    )


def test_the_fingerprint_covers_the_sheet_factor(engine_spec_factory) -> None:
    """Sessions that differ only in the sheet factor the force field
    resolved have different fingerprints, as a checkpoint written with one
    is refused by a run with the other; the same factor, given or taken
    from the set (mcpu08: 2.0), gives the same fingerprint."""
    def fingerprint(**options) -> str:
        return EngineSession(engine_spec_factory(forcefield_options=options)).fingerprint

    default = fingerprint()
    assert fingerprint(hbond_beta_favor=2.0) == default
    assert fingerprint(hbond_beta_favor=3.0) != default
    # Compared as the engine's float32, as checkpoints compare it.
    assert compute_fingerprint("a.pdb", "mcpu08", 9, hbond_beta_favor=0.1) == (
        compute_fingerprint("a.pdb", "mcpu08", 9, hbond_beta_favor=float(np.float32(0.1))))
    assert compute_fingerprint("a.pdb", "mcpu08", 9, hbond_beta_favor=3.0) != (
        compute_fingerprint("a.pdb", "mcpu08", 9))


def test_without_a_signature_or_factor_the_fingerprint_is_the_old_one(
    chignolin_pdb_path: str,
) -> None:
    payload = f"{Path(chignolin_pdb_path).resolve()}|mcpu08|77|False|C"
    assert compute_fingerprint(chignolin_pdb_path, "mcpu08", 77) == (
        hashlib.sha256(payload.encode()).hexdigest()[:16]
    )
