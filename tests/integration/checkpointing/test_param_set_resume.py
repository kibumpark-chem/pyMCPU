"""A checkpoint resumes only with the parameter set and the files it was
written with.

Checkpoints record ``param_set``, ``param_signature`` (the sha256 of the
parameter files, ``MCPUForceField.param_signature``) and ``param_dir``. A
resume under another set, or with files whose signature differs, is refused
with an error that says what to pass; the same files in another folder
resume. A checkpoint written before these were recorded counts as mcpu08 and
resumes under it with one warning. The MPI driver's cases are in
``tests/integration/mpi/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mdtraj")

from pymcpu import checkpointing  # noqa: E402
from pymcpu.checkpointing import (  # noqa: E402
    CheckpointState,
    checkpoint_param_set_error,
    load_checkpoint,
    param_set_record,
    save_checkpoint,
)
from pymcpu.params import ensure_params, param_signature  # noqa: E402
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402
from pymcpu.sampling.replica_exchange import ReplicaExchange  # noqa: E402
from tests.helpers.param_folders import (  # noqa: E402
    changed_param_folder,
    linked_param_folder,
    register_set_like_mcpu08,
)
from tests.helpers.resume_outputs import assert_same_outputs  # noqa: E402

PARAM_KEYS = ("param_set", "param_signature", "param_dir")
OLD = "a" * 64
NEW = "b" * 64


def _strip_param_fields(checkpoint_dir: Path) -> None:
    """Make every checkpoint in the folder look written before the fields existed."""
    for chk in checkpoint_dir.glob("*.chk"):
        state = load_checkpoint(chk)
        for key in PARAM_KEYS:
            assert state.pop(key)  # saved by this version...
        save_checkpoint(state, chk.parent, filename=chk.name)  # ...and now removed


def _old_checkpoint_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "records no parameter set" in r.getMessage()]


def test_the_check_says_what_differs_and_what_to_pass() -> None:
    current = {"param_set": "mcpu08", "param_signature": NEW, "param_dir": "/new"}
    # The same files in another folder resume; a force field without a set is not checked.
    assert checkpoint_param_set_error({**current, "param_dir": "/old"}, current) is None
    assert checkpoint_param_set_error(
        {"param_set": "mcpu26"}, {"param_set": "", "param_signature": "", "param_dir": ""}
    ) is None

    message = checkpoint_param_set_error(
        {"param_set": "mcpu26", "param_signature": OLD, "param_dir": "/old"}, current, "run/last.chk"
    )
    assert message == (
        "checkpoint run/last.chk was written with param_set 'mcpu26', but this run uses "
        "'mcpu08'; resume it with param_set: mcpu26, or start a new run without resume"
    )

    message = checkpoint_param_set_error(
        {"param_set": "mcpu08", "param_signature": OLD, "param_dir": "/old"}, current, "run/last.chk"
    )
    assert message.startswith(
        "checkpoint run/last.chk was written with param_set 'mcpu08' from parameter files "
        f"with signature {OLD} (/old), but this run's files have signature {NEW} (/new): "
        "a table or constants file differs; resume it with param_dir: /old"
    )
    # The same folder, whose files changed after the checkpoint was written.
    message = checkpoint_param_set_error(
        {"param_set": "mcpu08", "param_signature": OLD, "param_dir": "/new"}, current
    )
    assert "the files in /new have changed since" in message
    # A checkpoint read into a CheckpointState is checked like the raw dict.
    assert "param_set 'mcpu26'" in checkpoint_param_set_error(CheckpointState(param_set="mcpu26"), current)


def test_one_folder_reached_by_two_paths_is_the_same_folder(tmp_path: Path) -> None:
    """A folder reached through a link is the folder it links to: the record
    names its real path, and when the files there changed, the error says so
    instead of advising a resume from that same folder."""
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    forcefield = SimpleNamespace(param_set="mcpu08", param_signature=NEW, param_dir=alias)
    assert param_set_record(forcefield)["param_dir"] == str(real)

    current = {"param_set": "mcpu08", "param_signature": NEW, "param_dir": str(alias)}
    for saved_dir in (real, alias):  # written through either path, as 1e7896e recorded it
        saved = {"param_set": "mcpu08", "param_signature": OLD, "param_dir": str(saved_dir)}
        message = checkpoint_param_set_error(saved, current)
        assert f"the files in {alias} have changed since" in message
        assert "resume it with param_dir:" not in message
    # Another folder is still named as the one to resume from.
    other = {"param_set": "mcpu08", "param_signature": OLD, "param_dir": str(tmp_path / "old")}
    assert f"resume it with param_dir: {tmp_path / 'old'} " in checkpoint_param_set_error(
        other, current)


def test_a_checkpoint_without_a_parameter_set_counts_as_mcpu08(caplog) -> None:
    current = {"param_set": "mcpu08", "param_signature": NEW, "param_dir": "/new"}
    with caplog.at_level(logging.WARNING, logger="pymcpu.checkpointing"):
        assert checkpoint_param_set_error({}, current, "old.chk") is None
    (line,) = _old_checkpoint_warnings(caplog)
    assert line == (
        "checkpoint old.chk records no parameter set, as it was written before they were "
        "recorded: it counts as mcpu08, and its parameter files are not checked against "
        "this run's (/new)"
    )
    message = checkpoint_param_set_error({}, {**current, "param_set": "mcpu26"}, "old.chk")
    assert message == (
        "checkpoint old.chk records no parameter set, so it counts as 'mcpu08' (the only set "
        "when it was written), but this run uses param_set 'mcpu26'; resume it with "
        "param_set: mcpu08, or start a new run without resume"
    )


def _folding(pdb: str, root: Path, *, resume: bool = False, **options) -> FoldingRunner:
    return FoldingRunner(
        pdb, output_dir=str(root / "out"), checkpoint_dir=str(root / "ck"), seed=5,
        report_interval=5, steps_per_cycle=5, checkpoint_interval=1, resume=resume,
        verbose=False, **options,
    )


def _attempted(runner: FoldingRunner) -> int:
    integ = runner.simulation.integrator
    return integ.get_bb_attempted() + integ.get_kic_attempted() + integ.get_sc_attempted()


def test_folding_records_the_files_and_refuses_others(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    first = _folding(chignolin_pdb_path, tmp_path)
    first.run(n_cycles=2)
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")
    folder = ensure_params("mcpu08")
    assert saved["param_set"] == "mcpu08"
    assert saved["param_signature"] == param_signature(folder) == first.forcefield.param_signature
    assert saved["param_dir"] == os.path.realpath(first.forcefield.param_dir)

    other = _folding(chignolin_pdb_path, tmp_path, resume=True,
                     param_dir=str(changed_param_folder(tmp_path / "changed")))
    with pytest.raises(ValueError) as refused:
        other.run(n_cycles=4)
    message = str(refused.value)
    assert f"signature {saved['param_signature']} ({saved['param_dir']})" in message
    assert f"signature {other.forcefield.param_signature} ({tmp_path / 'changed'})" in message
    assert f"resume it with param_dir: {saved['param_dir']} " in message
    assert _attempted(other) == 0


def test_folding_refuses_another_parameter_set(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch
) -> None:
    _folding(chignolin_pdb_path, tmp_path).run(n_cycles=2)
    name = register_set_like_mcpu08(monkeypatch)
    # The same files under another set name: the set alone decides.
    other = _folding(chignolin_pdb_path, tmp_path, resume=True, param_set=name,
                     param_dir=str(linked_param_folder(tmp_path / "linked")))
    with pytest.raises(ValueError, match=(
        f"written with param_set 'mcpu08', but this run uses '{name}'; "
        "resume it with param_set: mcpu08,"
    )):
        other.run(n_cycles=4)
    assert _attempted(other) == 0


def test_folding_resumes_from_the_same_files_in_another_folder(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    """Only the files count, not where they are, and such a resume continues
    the run byte for byte."""
    first, second = (str(linked_param_folder(tmp_path / f)) for f in ("first", "second"))
    _folding(chignolin_pdb_path, tmp_path / "straight", param_dir=first).run(n_cycles=4)
    _folding(chignolin_pdb_path, tmp_path / "resumed", param_dir=first).run(n_cycles=2)
    resumed = _folding(chignolin_pdb_path, tmp_path / "resumed", resume=True, param_dir=second)
    resumed.run(n_cycles=4)
    assert _attempted(resumed) == 4 * 5
    assert load_checkpoint(tmp_path / "resumed" / "ck" / "last.chk")["param_dir"] == second
    assert_same_outputs(
        tmp_path / "straight" / "out", tmp_path / "resumed" / "out",
        tmp_path / "straight" / "ck" / "last.chk", tmp_path / "resumed" / "ck" / "last.chk",
    )


def test_an_old_folding_checkpoint_resumes_as_mcpu08(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch, caplog
) -> None:
    _folding(chignolin_pdb_path, tmp_path).run(n_cycles=2)
    _strip_param_fields(tmp_path / "ck")

    name = register_set_like_mcpu08(monkeypatch)
    other = _folding(chignolin_pdb_path, tmp_path, resume=True, param_set=name,
                     param_dir=str(linked_param_folder(tmp_path / "linked")))
    with pytest.raises(ValueError, match="records no parameter set, so it counts as 'mcpu08'"):
        other.run(n_cycles=4)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pymcpu.checkpointing"):
        legacy = _folding(chignolin_pdb_path, tmp_path, resume=True)
        legacy.run(n_cycles=4)
    assert len(_old_checkpoint_warnings(caplog)) == 1
    assert _attempted(legacy) == 4 * 5  # the counters continue from the checkpoint
    assert load_checkpoint(tmp_path / "ck" / "last.chk")["param_set"] == "mcpu08"


# The signature of the mcpu08 files pyMCPU 0.1.0 shipped, wherever they are.
SHIPPED_MCPU08 = "615479a19e4e2a80d89278c8898a762afe558bb2d644cfc5382e61a94a73fc90"


def test_the_shipped_mcpu08_files_sign_as_older_pymcpu_shipped_them() -> None:
    """Checkpoints of these files stay at format version 2 (below)."""
    assert param_signature(ensure_params("mcpu08")) == SHIPPED_MCPU08


def test_a_checkpoint_older_pymcpu_cannot_check_is_format_3(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch
) -> None:
    """A checkpoint of parameter files other than the mcpu08 files pyMCPU
    0.1.0 shipped is written as format version 3. A pyMCPU that reads at
    most version 2 (0.1.0 to f52ac23) ignores the parameter record, so it
    would resume such a run under its own mcpu08; it refuses a version 3
    file as newer than it understands. A checkpoint of the shipped files
    stays at version 2, which it resumes as before; this pyMCPU reads both."""
    changed = str(changed_param_folder(tmp_path / "changed"))
    _folding(chignolin_pdb_path, tmp_path / "shipped").run(n_cycles=1)
    _folding(chignolin_pdb_path, tmp_path / "other", param_dir=changed).run(n_cycles=1)
    shipped = tmp_path / "shipped" / "ck" / "last.chk"
    other = tmp_path / "other" / "ck" / "last.chk"
    assert load_checkpoint(shipped)["format_version"] == 2
    assert load_checkpoint(other)["format_version"] == 3
    resumed = _folding(chignolin_pdb_path, tmp_path / "other", resume=True, param_dir=changed)
    resumed.run(n_cycles=2)
    assert load_checkpoint(other)["cycle"] == 1

    monkeypatch.setattr(checkpointing, "CHECKPOINT_FORMAT_VERSION", 2)  # an older reader
    assert load_checkpoint(shipped)["cycle"] == 0
    with pytest.raises(ValueError, match="format_version=3 is newer than supported 2"):
        load_checkpoint(other)


def test_the_format_version_follows_the_recorded_files() -> None:
    def version(param_set: str, signature: str) -> int:
        return checkpointing.checkpoint_format_version(
            {"param_set": param_set, "param_signature": signature, "param_dir": "/p"})

    assert version("", "") == 2  # a force field without a parameter set
    assert version("mcpu08", SHIPPED_MCPU08) == 2
    assert version("mcpu08", OLD) == 3
    assert version("mcpu26", SHIPPED_MCPU08) == 3
    assert version("mcpu08", "") == 3
    assert checkpointing.CHECKPOINT_FORMAT_VERSION == 3


def _rex(pdb: str, root: Path, **options) -> ReplicaExchange:
    return ReplicaExchange(
        pdb, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0, log_interval=5,
        output_prefix="rex", output_dir=root / "out", seed=3, checkpoint_dir=root / "ck",
        checkpoint_interval=1, **options,
    )


def _rex_run(rex: ReplicaExchange, root: Path, cycles: int, resume: bool) -> None:
    rex.run(cycles, 5, verbose=False, checkpoint_dir=root / "ck", checkpoint_interval=1,
            resume=root / "ck" / "last.chk" if resume else None)


def test_serial_replica_exchange_refuses_other_files_and_another_set(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.chdir(tmp_path)
    first = _rex(chignolin_pdb_path, tmp_path)
    _rex_run(first, tmp_path, 1, resume=False)
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")
    assert saved["param_set"] == "mcpu08"
    assert saved["param_signature"] == first.forcefield.param_signature
    assert saved["format_version"] == 2

    changed = str(changed_param_folder(tmp_path / "changed"))
    with pytest.raises(ValueError, match=(
        "a table or constants file differs; resume it with param_dir: "
        f"{saved['param_dir']} "
    )):
        _rex_run(_rex(chignolin_pdb_path, tmp_path, param_dir=changed), tmp_path, 2, resume=True)
    # A run of the other files writes a checkpoint older pyMCPU refuse.
    other_root = tmp_path / "other"
    _rex_run(_rex(chignolin_pdb_path, other_root, param_dir=changed), other_root, 1, resume=False)
    assert load_checkpoint(other_root / "ck" / "last.chk")["format_version"] == 3

    name = register_set_like_mcpu08(monkeypatch)
    linked = str(linked_param_folder(tmp_path / "linked"))
    with pytest.raises(ValueError, match=f"but this run uses '{name}'; resume it with param_set: mcpu08,"):
        _rex_run(_rex(chignolin_pdb_path, tmp_path, param_set=name, param_dir=linked),
                 tmp_path, 2, resume=True)

    _strip_param_fields(tmp_path / "ck")
    with pytest.raises(ValueError, match="records no parameter set, so it counts as 'mcpu08'"):
        _rex_run(_rex(chignolin_pdb_path, tmp_path, param_set=name, param_dir=linked),
                 tmp_path, 2, resume=True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pymcpu.checkpointing"):
        _rex_run(_rex(chignolin_pdb_path, tmp_path), tmp_path, 2, resume=True)
    assert len(_old_checkpoint_warnings(caplog)) == 1
    resumed = load_checkpoint(tmp_path / "ck" / "last.chk")
    assert resumed["cycle"] == 2 and resumed["param_signature"] == saved["param_signature"]
