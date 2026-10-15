"""A checkpoint resumes only with the clash options it was written with.

``clash_alpha``, ``clash_bond_exclusion`` and ``clash_start_exemption``
(``forcefield_options``) are part of the energy function, as the sheet factor
is, so checkpoints record them and a resume with other values is refused. A
checkpoint written before they were recorded came from a build that used
mcpu08's (0.75, 0, True). A checkpoint of other options is format version 4,
which a pyMCPU that cannot check them refuses. The MPI driver's case is in
``tests/integration/mpi/test_mpi_checkpointing.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj")

from pymcpu import checkpointing  # noqa: E402
from pymcpu.checkpointing import (  # noqa: E402
    CheckpointState,
    checkpoint_clash_options_error,
    clash_options_record,
    load_checkpoint,
    save_checkpoint,
)
from pymcpu.sampling import EngineSession, compute_fingerprint  # noqa: E402
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402
from pymcpu.sampling.replica_exchange import ReplicaExchange  # noqa: E402

LEGACY = {"clash_alpha": 0.75, "clash_bond_exclusion": 0, "clash_start_exemption": True}
HARD = {"clash_alpha": 0.85, "clash_bond_exclusion": 3}
KEYS = tuple(LEGACY)


def test_the_check_compares_the_saved_options_with_the_runs() -> None:
    assert checkpoint_clash_options_error(dict(LEGACY), dict(LEGACY)) is None
    assert checkpoint_clash_options_error({}, dict(LEGACY)) is None  # written before they were recorded
    assert checkpoint_clash_options_error({"clash_alpha": 0.85}, dict.fromkeys(KEYS)) is None
    message = checkpoint_clash_options_error(
        {**LEGACY, "clash_alpha": 0.85}, dict(LEGACY), "run/last.chk")
    assert message == (
        "checkpoint run/last.chk was written with clash_alpha 0.85, but this run uses "
        "clash_alpha 0.75; resume it with forcefield_options: {clash_alpha: 0.85}, or start "
        "a new run without resume"
    )
    both = checkpoint_clash_options_error(
        {**LEGACY, "clash_bond_exclusion": 3, "clash_start_exemption": False}, dict(LEGACY))
    assert ("written with clash_bond_exclusion 3, clash_start_exemption false, but this run "
            "uses clash_bond_exclusion 0, clash_start_exemption true") in both
    assert "{clash_bond_exclusion: 3, clash_start_exemption: false}" in both
    # A checkpoint without them counts as mcpu08's.
    assert "written with clash_alpha 0.75, but this run uses clash_alpha 0.85" in (
        checkpoint_clash_options_error({}, {**LEGACY, "clash_alpha": 0.85}))
    # The engine holds clash_alpha as float32; a dataclass state reads the same.
    assert checkpoint_clash_options_error(
        CheckpointState(clash_alpha=0.85), {**LEGACY, "clash_alpha": np.float32(0.85)}) is None


def test_the_record_names_the_force_fields_options() -> None:
    class WithOptions:
        clash_alpha, clash_bond_exclusion, clash_start_exemption = 0.85, 3, False

    assert clash_options_record(WithOptions()) == {
        "clash_alpha": 0.85, "clash_bond_exclusion": 3, "clash_start_exemption": False}
    assert clash_options_record(object()) == dict.fromkeys(KEYS)
    assert checkpointing.LEGACY_CLASH_OPTIONS == LEGACY


def _folding(pdb: str, root: Path, *, resume: bool = False, **options) -> FoldingRunner:
    return FoldingRunner(
        pdb, output_dir=str(root / "out"), checkpoint_dir=str(root / "ck"), seed=5,
        report_interval=5, steps_per_cycle=5, checkpoint_interval=1, resume=resume,
        forcefield_options=options or None, verbose=False,
    )


def _attempted(runner: FoldingRunner) -> int:
    integ = runner.simulation.integrator
    return integ.get_bb_attempted() + integ.get_kic_attempted() + integ.get_sc_attempted()


def _saved(root: Path) -> dict:
    return load_checkpoint(root / "ck" / "last.chk")


def test_a_default_run_records_mcpu08s_options(chignolin_pdb_path: str, tmp_path: Path) -> None:
    _folding(chignolin_pdb_path, tmp_path).run(n_cycles=1)
    saved = _saved(tmp_path)
    assert {key: saved[key] for key in KEYS} == LEGACY
    assert saved["format_version"] == 2  # the shipped mcpu08 files, mcpu08's options


def test_folding_records_the_options_and_refuses_others(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    _folding(chignolin_pdb_path, tmp_path, **HARD).run(n_cycles=2)
    saved = _saved(tmp_path)
    assert {key: saved[key] for key in KEYS} == {**HARD, "clash_start_exemption": True}
    assert saved["format_version"] == 4

    default = _folding(chignolin_pdb_path, tmp_path, resume=True)
    with pytest.raises(ValueError, match=(
        "written with clash_alpha 0.85, clash_bond_exclusion 3, but this run uses "
        "clash_alpha 0.75, clash_bond_exclusion 0; resume it with forcefield_options: "
        r"\{clash_alpha: 0.85, clash_bond_exclusion: 3\}"
    )):
        default.run(n_cycles=4)
    assert _attempted(default) == 0

    same = _folding(chignolin_pdb_path, tmp_path, resume=True, **HARD)
    same.run(n_cycles=4)
    assert _attempted(same) == 4 * 5  # the counters continue from the checkpoint


def test_an_old_checkpoint_resumes_with_mcpu08s_options(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    _folding(chignolin_pdb_path, tmp_path).run(n_cycles=2)
    for chk in (tmp_path / "ck").glob("*.chk"):
        state = load_checkpoint(chk)
        for key in KEYS:
            assert state.pop(key) == LEGACY[key]  # saved by this version...
        save_checkpoint(state, chk.parent, filename=chk.name)  # ...and now removed

    harder = _folding(chignolin_pdb_path, tmp_path, resume=True, clash_alpha=0.85)
    with pytest.raises(ValueError, match="written with clash_alpha 0.75, but this run uses"):
        harder.run(n_cycles=4)

    legacy = _folding(chignolin_pdb_path, tmp_path, resume=True)
    legacy.run(n_cycles=4)
    assert _attempted(legacy) == 4 * 5


def test_a_relaxation_checkpoint_refuses_the_exemption(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    relaxing = {**HARD, "clash_start_exemption": False}
    _folding(chignolin_pdb_path, tmp_path, **relaxing).run(n_cycles=2)
    assert _saved(tmp_path)["clash_start_exemption"] is False
    exempting = _folding(chignolin_pdb_path, tmp_path, resume=True, **HARD)
    with pytest.raises(ValueError, match=(
        "written with clash_start_exemption false, but this run uses clash_start_exemption "
        r"true; resume it with forcefield_options: \{clash_start_exemption: false\}"
    )):
        exempting.run(n_cycles=4)
    _folding(chignolin_pdb_path, tmp_path, resume=True, **relaxing).run(n_cycles=4)


def test_serial_replica_exchange_refuses_another_core(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def rex(**options) -> ReplicaExchange:
        return ReplicaExchange(
            chignolin_pdb_path, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
            log_interval=5, output_prefix="rex", output_dir=tmp_path / "out", seed=3,
            checkpoint_dir=tmp_path / "ck", checkpoint_interval=1,
            forcefield_options=options or None,
        )

    rex(clash_alpha=0.85).run(1, 5, verbose=False, checkpoint_dir=tmp_path / "ck",
                              checkpoint_interval=1)
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")
    assert (saved["clash_alpha"], saved["format_version"]) == (0.85, 4)
    with pytest.raises(ValueError, match="written with clash_alpha 0.85, but this run uses"):
        rex().run(2, 5, verbose=False, checkpoint_dir=tmp_path / "ck",
                  checkpoint_interval=1, resume=tmp_path / "ck" / "last.chk")


def test_the_fingerprint_covers_options_other_than_mcpu08s(engine_spec_factory) -> None:
    def fingerprint(**options) -> str:
        return EngineSession(engine_spec_factory(forcefield_options=options)).fingerprint

    default = fingerprint()
    assert fingerprint(**LEGACY) == default
    assert fingerprint(clash_alpha=0.85) != default
    assert fingerprint(clash_bond_exclusion=3) != default
    assert fingerprint(clash_alpha=0.85) != fingerprint(**HARD)
    # Without the options, or with mcpu08's, the hash is the one before they existed.
    assert compute_fingerprint("a.pdb", "mcpu08", 9, clash_options=dict(LEGACY)) == (
        compute_fingerprint("a.pdb", "mcpu08", 9))
    assert compute_fingerprint("a.pdb", "mcpu08", 9, clash_options={**LEGACY, "clash_alpha": 0.85}) != (
        compute_fingerprint("a.pdb", "mcpu08", 9))
