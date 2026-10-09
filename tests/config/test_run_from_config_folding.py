"""A config-driven folding run, end to end, with nothing mocked.

``run_from_config`` passed the rama-pivot settings to ``run_folding``, whose
signature did not accept them, so every ``mcpu run`` of a folding config
raised ``TypeError``. The dispatch tests mock ``run_folding`` and could not
see it. This one runs a short simulation and reads its CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu import runners  # noqa: E402
from pymcpu.config import (  # noqa: E402
    CheckpointConfig,
    IntegratorConfig,
    OutputsConfig,
    SimulationConfig,
)


@pytest.mark.parametrize(
    "rama",
    [
        {"pivot_rama_probability": 1.0},
        # The default temperature (0.6) is above t_high, so the schedule gives p_max.
        {"pivot_rama_schedule": {"t_low": 0.1, "t_high": 0.2, "p_min": 0.0, "p_max": 1.0}},
    ],
    ids=["probability", "schedule"],
)
def test_folding_config_runs_and_applies_move_settings(
    tmp_path: Path, monkeypatch, rama: dict
) -> None:
    monkeypatch.chdir(tmp_path)
    steps = 40
    cfg = SimulationConfig(
        mode="folding",
        pdb=str(runners.default_example_pdb()),
        integrator=IntegratorConfig(
            steps=steps,
            report_interval=20,
            move_weights=(1.0, 0.0, 0.0),
            **rama,
        ),
        outputs=OutputsConfig(output_dir=str(tmp_path / "out")),
        checkpoint=CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
    )

    out = Path(runners.run_from_config(cfg, verbose=False))

    with (out / "folding_data.csv").open() as fh:
        last = list(csv.DictReader(fh))[-1]
    assert int(last["step"]) == steps
    # move_weights=(1, 0, 0) with an effective rama p of 1: every step is a
    # rama pivot, and only the pivot-slot kinds get columns.
    assert int(last["rama_pivot_attempted"]) == steps
    assert int(last["pivot_attempted"]) == 0
    assert "kic_attempted" not in last
    assert "rotamer_attempted" not in last


def test_a_yaml_folding_config_runs_steps_in_total(tmp_path: Path, monkeypatch) -> None:
    """``steps`` in a one-temperature YAML config is the total: 60 steps run
    as 3 cycles of mc_replica_steps 20 (it used to run 10 cycles of 60)."""
    from pymcpu.config import yaml_dict_to_config

    monkeypatch.chdir(tmp_path)
    cfg = yaml_dict_to_config(
        {
            "pdb": str(runners.default_example_pdb()),
            "temperatures": [0.5],
            "steps": 60,
            "mc_replica_steps": 20,
            "output_prefix": str(tmp_path / "out" / "fold"),
            "checkpointing": {"checkpoint_dir": str(tmp_path / "ck"), "checkpoint_interval": 1},
        }
    )

    out = Path(runners.run_from_config(cfg, verbose=False))

    with (out / "fold_data.csv").open() as fh:
        steps = [int(row["step"]) for row in csv.DictReader(fh)]
    assert steps == [0, 20, 40, 60]  # log_interval defaults to one cycle
    # one checkpoint per cycle; cycles count from 0
    assert sorted(p.name for p in (tmp_path / "ck").glob("checkpoint_cycle_*.chk")) == [
        f"checkpoint_cycle_{c:06d}.chk" for c in range(3)
    ]


class _Stop(Exception):
    pass


def test_folding_settings_reach_the_runner(tmp_path: Path, monkeypatch) -> None:
    """The config's cycle length, reference structure, native-contact
    definition and early stop all reach FoldingRunner."""
    from pymcpu.config import yaml_dict_to_config
    import pymcpu.sampling.folding as folding_module

    seen: dict = {}

    class Recorder:
        def __init__(self, pdb, **kwargs) -> None:
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(folding_module, "FoldingRunner", Recorder)
    monkeypatch.chdir(tmp_path)
    pdb = str(runners.default_example_pdb())
    cfg = yaml_dict_to_config(
        {
            "pdb": pdb,
            "reference_pdb": pdb,
            "temperatures": [0.5],
            "steps": 900,
            "mc_replica_steps": 300,
            "q_threshold": 0.7,
            "convergence_window": 3,
            "contact_cutoff": 7.5,
            "min_seq_sep": 5,
            "contact_atom_mode": "cb",
            "native_contact_pairs": [[0, 7]],
            "output_prefix": str(tmp_path / "out" / "fold"),
            "checkpointing": {"enabled": False},
        }
    )

    with pytest.raises(_Stop):
        runners.run_from_config(cfg, verbose=False)

    assert seen["steps_per_cycle"] == 300
    assert Path(seen["reference_pdb"]) == Path(pdb).resolve()
    assert (seen["q_threshold"], seen["convergence_window"]) == (0.7, 3)
    assert (seen["contact_cutoff_ang"], seen["min_seq_sep"]) == (7.5, 5)
    assert seen["contact_atom_mode"] == "cb"
    assert seen["native_contact_pairs"] == [[0, 7]]
    assert seen["checkpoint_config"].enabled is False
