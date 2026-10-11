"""YAML round-trip for ``contact_atom_mode`` on ReplicaExchangeConfig.

Pure config_plumbing: verifies ``yaml_dict_to_config`` parses/validates
``contact_atom_mode`` (and its companion ``contact_cutoff``) the way
``pymcpu/config.py``'s ``ReplicaExchangeConfig`` dataclass defines them. No
physics computation is involved -- this is software correctness for the CLI/
config layer, split out from the physics half of the original combined file
(now ``tests/physics/cv/test_contact_atom_index.py``).
"""

from __future__ import annotations

import pytest

from pymcpu.config import yaml_dict_to_config


# Minimal valid base kwargs shared by every case below; each test overrides
# only the field(s) it's actually exercising.
_BASE_YAML = {
    "pdb": "examples/data/1uao.pdb",
    "temperatures": [0.4, 0.5],
    "num_cycles": 1,
    "mc_replica_steps": 10,
}


def test_contact_atom_mode_cb_roundtrips_with_cutoff() -> None:
    cfg = yaml_dict_to_config(
        {**_BASE_YAML, "contact_atom_mode": "cb", "contact_cutoff": 7.5}
    )
    assert cfg.replica_exchange is not None
    assert cfg.replica_exchange.contact_atom_mode == "cb"
    assert cfg.replica_exchange.contact_cutoff == 7.5


def test_invalid_contact_atom_mode_raises() -> None:
    with pytest.raises(ValueError, match="contact_atom_mode"):
        yaml_dict_to_config({**_BASE_YAML, "contact_atom_mode": "cg"})


def test_contact_atom_mode_defaults_to_ca() -> None:
    cfg = yaml_dict_to_config(dict(_BASE_YAML))
    # Matches ReplicaExchangeConfig.contact_atom_mode's dataclass default.
    assert cfg.replica_exchange.contact_atom_mode == "ca"
