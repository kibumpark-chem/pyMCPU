"""``EngineSession.fingerprint`` covers the parameter files, as checkpoints do.

The fingerprint lets an external sampler refuse a restart state built from a
different system. It now includes the parameter set the force field was
built from (``forcefield_options`` can name another than ``spec.param_set``)
and the signature of its files, so the same files elsewhere give the same
fingerprint and other tables another one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
        spec.pdb, "mcpu08", n_atoms, param_signature=signature
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
        spec.pdb, name, n_atoms, param_signature=signature
    )


def test_without_a_signature_the_fingerprint_is_the_old_one(chignolin_pdb_path: str) -> None:
    payload = f"{Path(chignolin_pdb_path).resolve()}|mcpu08|77|False|C"
    assert compute_fingerprint(chignolin_pdb_path, "mcpu08", 77) == (
        hashlib.sha256(payload.encode()).hexdigest()[:16]
    )
