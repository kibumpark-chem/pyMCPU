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


def test_folding_config_runs_and_applies_move_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    steps = 40
    cfg = SimulationConfig(
        mode="folding",
        pdb=str(runners.default_example_pdb()),
        integrator=IntegratorConfig(
            steps=steps,
            report_interval=20,
            move_weights=(1.0, 0.0, 0.0),
            pivot_rama_probability=1.0,
        ),
        outputs=OutputsConfig(output_dir=str(tmp_path / "out")),
        checkpoint=CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
    )

    out = Path(runners.run_from_config(cfg, verbose=False))

    with (out / "folding_data.csv").open() as fh:
        last = list(csv.DictReader(fh))[-1]
    assert int(last["Step"]) == steps
    # move_weights=(1, 0, 0): every step is a pivot-slot move.
    assert int(last["PivotAttempted"]) == steps
    assert int(last["SidechainAttempted"]) == 0
    assert int(last["KicAttempted"]) == 0
