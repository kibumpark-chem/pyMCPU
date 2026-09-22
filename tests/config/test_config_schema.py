"""Tests for ``pymcpu.config``'s JSON schema: ``config_from_dict``,
``load_config``, ``validate_config``.

Two kinds of coverage live here:

1. Parsing the two shipped example configs (``examples/configs/folding.json``
   and ``replica_exchange_2d.json``) into the expected dataclass fields. These
   assertions intentionally mirror those JSON files' literal field values --
   they are fixture-echo checks, not derived/physics constants -- so each one
   is commented with the source field it mirrors.
2. Schema validation logic (invalid ``mode``, a ``replica_exchange_2d`` config
   missing its ``replica_exchange`` block) using inline dicts, since that
   logic doesn't depend on the shipped example files at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu import PACKAGE_ROOT
from pymcpu.config import config_from_dict, load_config, validate_config

REPO_ROOT = Path(PACKAGE_ROOT).parent
CONFIGS = REPO_ROOT / "examples" / "configs"


class TestExampleConfigsParse:
    """``validate_config``/``load_config`` against the two shipped example JSON files."""

    @pytest.mark.parametrize("name", ["folding.json", "replica_exchange_2d.json"])
    def test_example_config_validates(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(REPO_ROOT)  # both configs' pdb field is repo-relative
        cfg = validate_config(CONFIGS / name)
        # both example configs point at the same test structure, examples/data/1uao.pdb
        assert cfg.pdb.endswith("1uao.pdb")
        assert cfg.mode in ("folding", "replica_exchange_2d")

    def test_folding_mode_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(REPO_ROOT)
        cfg = load_config(CONFIGS / "folding.json")
        assert cfg.mode == "folding"
        assert cfg.integrator.temperature == 0.6  # mirrors folding.json integrator.temperature
        assert cfg.integrator.steps == 1000  # mirrors folding.json integrator.steps
        assert cfg.outputs.output_dir == "./out_folding"  # mirrors folding.json outputs.output_dir

    def test_rex_mode_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(REPO_ROOT)
        cfg = load_config(CONFIGS / "replica_exchange_2d.json")
        assert cfg.mode == "replica_exchange_2d"
        assert cfg.replica_exchange is not None
        assert cfg.replica_exchange.temperatures == [0.5, 0.6]  # mirrors rex json's temperatures
        # mirrors rex json's native_contact_targets
        assert cfg.replica_exchange.native_contact_targets == [0, 5, 10]
        # effective_k_bias() passes through k_native_contacts:1.0 since k_bias is unset in the JSON
        assert cfg.replica_exchange.effective_k_bias() == 1.0
        # effective_mc_steps() falls back to steps_per_cycle:100 since swap_interval is unset
        assert cfg.replica_exchange.effective_mc_steps() == 100
        assert cfg.replica_exchange.backend == "serial"  # only supported backend currently


class TestSchemaValidation:
    """Validation logic that doesn't depend on any shipped example file."""

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            config_from_dict({"mode": "nope", "pdb": "x.pdb"})

    def test_rex_requires_replica_exchange_block(self) -> None:
        with pytest.raises(ValueError, match="replica_exchange"):
            config_from_dict(
                {
                    "mode": "replica_exchange_2d",
                    "pdb": "examples/data/1uao.pdb",
                }
            )
