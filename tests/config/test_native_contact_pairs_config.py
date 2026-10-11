"""YAML round-trip checks for ``native_contact_pairs``.

Pure config-plumbing: verifies ``yaml_dict_to_config`` parses/validates the
new ``native_contact_pairs`` top-level YAML key the way ``pymcpu/config.py``'s
``ReplicaExchangeConfig`` dataclass defines it, and that the YAML schema
knows the key (so a valid config isn't rejected as having an unknown key). No
physics computation is involved here -- deep semantic validation (index
range, self-pairs, duplicates, energy-ignored residues) needs ``n_res`` and
is deliberately deferred to ``NativeContactsCV`` at construction time (see
``tests/physics/cv/test_native_contacts_cv.py``); the YAML-round-trip tests
only cover what ``yaml_dict_to_config`` itself can and does check (list-of-2
arity) plus the plain dataclass plumbing. Mirrors
``test_contact_atom_mode_config.py``'s style/fixtures.

That the field reaches the serial and MPI replica-exchange drivers through
``run_from_config`` is checked in ``test_run_from_config_remd.py``.
"""

from __future__ import annotations

import warnings

import pytest

from pymcpu.config import load_yaml_config, yaml_dict_to_config

# Minimal valid base kwargs shared by every case below; each test overrides
# only the field(s) it's actually exercising.
_BASE_YAML = {
    "pdb": "examples/data/1uao.pdb",
    "temperatures": [0.4, 0.5],
    "num_cycles": 1,
    "mc_replica_steps": 10,
}


def test_native_contact_pairs_roundtrips_into_config() -> None:
    cfg = yaml_dict_to_config(
        {**_BASE_YAML, "native_contact_pairs": [[0, 5], [1, 6], [2, 8]]}
    )
    assert cfg.replica_exchange is not None
    assert cfg.replica_exchange.native_contact_pairs == [[0, 5], [1, 6], [2, 8]]


def test_native_contact_pairs_defaults_to_none() -> None:
    cfg = yaml_dict_to_config(dict(_BASE_YAML))
    assert cfg.replica_exchange is not None
    # Matches ReplicaExchangeConfig.native_contact_pairs's dataclass default.
    assert cfg.replica_exchange.native_contact_pairs is None


def test_native_contact_pairs_wrong_arity_raises_at_parse_time() -> None:
    """A malformed entry (e.g. a 3-element tuple instead of a pair) must be
    rejected by ``yaml_dict_to_config`` itself, before it ever reaches
    ``NativeContactsCV`` -- this is arity checking only, not the deeper
    range/self-pair/duplicate semantic validation that needs ``n_res`` and
    lives in ``NativeContactsCV`` instead."""
    with pytest.raises(ValueError, match="exactly 2"):
        yaml_dict_to_config(
            {**_BASE_YAML, "native_contact_pairs": [[1, 2, 3]]}
        )


def test_native_contact_pairs_key_does_not_warn(tmp_path) -> None:
    f = tmp_path / "native_contact_pairs.yaml"
    f.write_text(
        "pdb: examples/data/1uao.pdb\n"
        "temperatures: [0.4, 0.5]\n"
        "num_cycles: 1\n"
        "mc_replica_steps: 10\n"
        "native_contact_pairs: [[0, 5], [1, 6]]\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_yaml_config(f)
