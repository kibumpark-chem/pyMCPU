"""API/config-contract checks for checkpointing: not checkpoint I/O itself,
just that checkpoint-related parameters and config fields are actually wired
through the public entry points that callers use.

Split out of the old ``test_checkpointing.py`` because these two tests check
signatures/config-object plumbing (software wiring), not any checkpoint save
or load behavior -- they belong with the rest of the config-plumbing suite,
not with ``tests/integration/checkpointing``.
"""

from __future__ import annotations

import inspect

import pytest

from pymcpu import checkpointing
from pymcpu.config import (
    CheckpointConfig,
    IntegratorConfig,
    OutputsConfig,
    SimulationConfig,
    config_from_dict,
    yaml_dict_to_config,
)
from pymcpu.runners import run_replica_exchange_2d
from pymcpu.sampling.replica_exchange import ReplicaExchange


def test_config_checkpoint_config_is_the_runtime_type() -> None:
    """pymcpu.config.CheckpointConfig must not be a separate mirror class --
    a SimulationConfig loaded from YAML/JSON should already carry the real
    pymcpu.checkpointing.CheckpointConfig (resolved_resume_path() etc.),
    with no reconstruction step needed."""
    assert CheckpointConfig is checkpointing.CheckpointConfig
    cfg = SimulationConfig(
        mode="folding",
        pdb="dummy.pdb",
        integrator=IntegratorConfig(),
        outputs=OutputsConfig(),
    )
    assert hasattr(cfg.checkpoint, "resolved_resume_path")


def test_checkpoint_kwargs_reach_runner_init_and_run() -> None:
    """Regression: checkpoint args must not be dropped between the runner
    function and ``ReplicaExchange.__init__``/``.run()``."""
    sig_runner = inspect.signature(run_replica_exchange_2d)
    sig_init = inspect.signature(ReplicaExchange.__init__)
    sig_run = inspect.signature(ReplicaExchange.run)
    for name in ("checkpoint_dir", "checkpoint_interval", "resume"):
        assert name in sig_runner.parameters
    for name in ("checkpoint_dir", "checkpoint_interval"):
        assert name in sig_init.parameters
    for name in ("checkpoint_dir", "checkpoint_interval", "resume"):
        assert name in sig_run.parameters


# The checkpoint upload (cloud_sync / cloud_bucket / cloud_sync_cmd) was
# removed. Copies of the old template carry those keys at their defaults, so
# they load with a warning; turning the upload on is an error.
_CLOUD_DEFAULTS = {"cloud_sync": False, "cloud_bucket": "", "cloud_sync_cmd": "aws s3 cp"}


@pytest.mark.parametrize("nested", [True, False])
def test_removed_cloud_keys_load_with_a_warning(nested: bool) -> None:
    data: dict = {"pdb": "dummy.pdb", "temperatures": [0.5]}
    data.update({"checkpointing": dict(_CLOUD_DEFAULTS)} if nested else _CLOUD_DEFAULTS)
    with pytest.warns(UserWarning, match="checkpoint upload was removed"):
        cfg = yaml_dict_to_config(data)
    assert not hasattr(cfg.checkpoint, "cloud_sync")


@pytest.mark.parametrize("nested", [True, False])
def test_cloud_sync_on_is_an_error(nested: bool) -> None:
    data: dict = {"pdb": "dummy.pdb", "temperatures": [0.5]}
    data.update({"checkpointing": {"cloud_sync": True}} if nested else {"cloud_sync": True})
    with pytest.raises(ValueError, match="no longer uploads checkpoints"):
        yaml_dict_to_config(data)


def test_json_config_handles_the_removed_cloud_keys() -> None:
    base = {"mode": "folding", "pdb": "dummy.pdb"}
    with pytest.warns(UserWarning, match="checkpoint upload was removed"):
        cfg = config_from_dict({**base, "checkpoint": {**_CLOUD_DEFAULTS, "checkpoint_interval": 7}})
    assert cfg.checkpoint.checkpoint_interval == 7
    with pytest.raises(ValueError, match="no longer uploads checkpoints"):
        config_from_dict({**base, "checkpoint": {"cloud_sync": True}})
