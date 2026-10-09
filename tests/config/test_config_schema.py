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


class TestJsonUmbrellaAndFolding:
    """A replica_exchange block without targets is plain temperature REMD (it
    used to get three umbrella windows at N = 0, 5 and 10, with k = 1); the
    ``folding`` block holds the folding-only settings."""

    @staticmethod
    def _rex(**block) -> dict:
        return {"mode": "replica_exchange_2d", "pdb": "x.pdb",
                "replica_exchange": {"temperatures": [0.5, 0.6], **block}}

    def test_temperatures_only_has_no_targets_and_no_umbrella(self) -> None:
        rex = config_from_dict(self._rex()).replica_exchange
        assert rex.native_contact_targets is None and rex.q_targets is None
        assert rex.effective_k_bias() == 0.0

    def test_targets_without_k_keep_the_default(self) -> None:
        assert config_from_dict(self._rex(q_targets=[0.5])).replica_exchange.effective_k_bias() == 1.0

    def test_both_kinds_of_target_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            config_from_dict(self._rex(native_contact_targets=[0], q_targets=[0.5]))

    def test_folding_block(self) -> None:
        cfg = config_from_dict(
            {"mode": "folding", "pdb": "x.pdb",
             "folding": {"steps_per_cycle": 50, "q_threshold": 0.9, "contact_cutoff": 7.0}}
        )
        assert (cfg.folding.steps_per_cycle, cfg.folding.q_threshold) == (50, 0.9)
        assert (cfg.folding.contact_cutoff, cfg.folding.min_seq_sep) == (7.0, 4)

    def test_folding_defaults(self) -> None:
        folding = config_from_dict({"mode": "folding", "pdb": "x.pdb"}).folding
        assert folding.steps_per_cycle is None  # one report_interval
        assert folding.q_threshold is None  # no early stop

    def test_folding_block_is_refused_in_replica_exchange(self) -> None:
        with pytest.raises(ValueError, match="'folding' block applies to mode 'folding' only"):
            config_from_dict({**self._rex(), "folding": {"q_threshold": 0.9}})
