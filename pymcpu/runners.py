"""Shared folding / replica-exchange runners used by CLI and examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pymcpu.checkpointing import CheckpointConfig
from pymcpu.config import SimulationConfig, resolve_path
from pymcpu.sampling.replica_exchange import ReplicaExchange, RunSummary


def default_example_pdb() -> Path:
    """Path to the bundled 1UAO (chignolin) structure used by the examples.

    Resolved from the INSTALLED package, not from a repo layout. The previous
    implementation returned ``<package parent>/examples/data/1uao.pdb``, which
    is the repo root only for an editable install; from a wheel it pointed at
    a nonexistent ``site-packages/examples/...`` and every caller -- including
    the README quickstart -- failed with ``OSError: No such file``.

    ``pymcpu/data/1uao.pdb`` ships in the wheel, so this works for a plain
    ``pip install pymcpu`` from any working directory.
    """
    try:
        from importlib.resources import files as _files
    except ImportError:  # Python 3.8 and older; requires-python is >=3.9
        from importlib_resources import files as _files  # type: ignore[import-not-found]

    return Path(str(_files("pymcpu.data").joinpath("1uao.pdb")))


def run_folding(
    *,
    pdb: str | Path,
    temperature: float = 0.6,
    steps: int = 1000,
    report_interval: int = 100,
    output_dir: str | Path = "./out_folding",
    seed: int = 42,
    param_dir: str | Path | None = None,
    param_set: str = "mcpu08",
    step_size_rad: float = 0.1,
    sidechain_move_mode: str = "rotamer_library",
    move_weights: tuple[float, float, float] | None = None,
    pivot_rama_probability: float = 0.0,
    pivot_rama_schedule: dict[str, float] | None = None,
    prefix: str = "folding",
    verbose: bool = True,
    fixed_residues: list[int] | None = None,
    linker_residues: list[int] | None = None,
    linker_energy_mode: str = "ignore_all",
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 50,
    resume: str | Path | bool | None = None,
    keep_last_n: int | None = 3,
    cloud_sync: bool = False,
    cloud_bucket: str = "",
    cloud_sync_cmd: str = "aws s3 cp",
) -> Path:
    """Run a single-temperature MC folding trajectory (OpenMM-style)."""
    from pymcpu.sampling.folding import FoldingRunner

    pdb_path = Path(pdb)
    if not pdb_path.is_absolute():
        pdb_path = resolve_path(pdb_path)

    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval=int(checkpoint_interval),
        resume=resume if resume is not None else False,
        keep_last_n=keep_last_n,
        cloud_sync=bool(cloud_sync),
        cloud_bucket=str(cloud_bucket or ""),
        cloud_sync_cmd=str(cloud_sync_cmd or "aws s3 cp"),
    )
    resume_path = ckpt_cfg.resolved_resume_path()
    ckpt_cfg.resume = True if resume_path else False

    runner = FoldingRunner(
        pdb_path,
        temperature=float(temperature),
        report_interval=int(report_interval),
        output_dir=output_dir,
        seed=int(seed),
        param_dir=param_dir,
        param_set=param_set,
        step_size_rad=float(step_size_rad),
        sidechain_move_mode=sidechain_move_mode,
        move_weights=move_weights,
        pivot_rama_probability=pivot_rama_probability,
        pivot_rama_schedule=pivot_rama_schedule,
        prefix=prefix,
        fixed_residues=fixed_residues,
        linker_residues=linker_residues,
        linker_energy_mode=linker_energy_mode,
        checkpoint_config=ckpt_cfg,
        verbose=verbose,
    )
    return runner.run(
        steps=int(steps),
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval=int(checkpoint_interval),
        keep_last_n=keep_last_n,
        resume=ckpt_cfg.resume,
        cloud_sync=bool(cloud_sync),
        cloud_bucket=str(cloud_bucket or ""),
        cloud_sync_cmd=str(cloud_sync_cmd or "aws s3 cp"),
    )


def run_replica_exchange_2d(
    *,
    pdb: str | Path,
    reference_pdb: str | Path | None = None,
    temperatures: list[float] | np.ndarray,
    n_targets: list[float] | np.ndarray | None = None,
    q_targets: list[float] | np.ndarray | None = None,
    k_bias: float = 1.0,
    cycles: int = 10,
    steps_per_cycle: int = 100,
    swap_interval: int | None = None,
    output_dir: str | Path = "./out_rex",
    seed: int = 42,
    backend: str = "serial",
    log_interval: int = 100,
    contact_cutoff: float = 6.0,
    min_seq_sep: int = 4,
    contact_atom_mode: str = "ca",
    native_contact_pairs: list[list[int]] | None = None,
    hdf5_path: str | Path | None = None,
    prefix: str = "rex",
    verbose: bool = True,
    fixed_residues: list[int] | None = None,
    linker_residues: list[int] | None = None,
    linker_energy_mode: str = "ignore_all",
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 50,
    resume: str | Path | bool | None = None,
    keep_last_n: int | None = 3,
    cloud_sync: bool = False,
    cloud_bucket: str = "",
    cloud_sync_cmd: str = "aws s3 cp",
    exchange_log: str = "none",
    state_log_interval: int = 0,
    log_walker_in_data_csv: bool = True,
    step_size_rad: float = 0.1,
    move_weights: tuple[float, float, float] | None = None,
    sidechain_move_mode: str = "rotamer_library",
    pivot_rama_probability: float = 0.0,
    pivot_rama_schedule: dict[str, float] | None = None,
) -> RunSummary:
    """Run 2D temperature × native-contact replica exchange (serial backend)."""
    if backend != "serial":
        raise ValueError(f"Only backend='serial' is supported (got {backend!r})")

    pdb_path = Path(pdb)
    if not pdb_path.is_absolute():
        pdb_path = resolve_path(pdb_path)
    ref = Path(reference_pdb) if reference_pdb is not None else pdb_path
    if not ref.is_absolute():
        ref = resolve_path(ref)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mc_steps = int(swap_interval if swap_interval is not None else steps_per_cycle)

    analysis = None
    if hdf5_path is not None:
        analysis = Path(hdf5_path)
        if not analysis.is_absolute():
            analysis = out / analysis

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval=int(checkpoint_interval),
        resume=resume if resume is not None else False,
        keep_last_n=keep_last_n,
        cloud_sync=bool(cloud_sync),
        cloud_bucket=str(cloud_bucket or ""),
        cloud_sync_cmd=str(cloud_sync_cmd or "aws s3 cp"),
    )
    resume_path = ckpt_cfg.resolved_resume_path()

    if verbose:
        print(f"PDB: {pdb_path}")
        print(f"Reference: {ref}")
        print(f"Temperatures: {list(np.asarray(temperatures))}")
        print(f"N targets: {None if n_targets is None else list(np.asarray(n_targets))}")
        print(f"Q targets: {None if q_targets is None else list(np.asarray(q_targets))}")
        print(f"k_bias={k_bias}  cycles={cycles}  mc_steps={mc_steps}")
        if checkpoint_dir:
            print(f"Checkpoint: dir={checkpoint_dir}  interval={checkpoint_interval}")
        if resume_path:
            print(f"Resuming from: {resume_path}")

    rex = ReplicaExchange(
        str(pdb_path),
        reference_pdb=str(ref),
        temperatures=np.asarray(temperatures, dtype=np.float64),
        n_targets=None if n_targets is None else np.asarray(n_targets, dtype=np.float64),
        q_targets=None if q_targets is None else np.asarray(q_targets, dtype=np.float64),
        k_bias=float(k_bias),
        contact_cutoff=float(contact_cutoff),
        min_seq_sep=int(min_seq_sep),
        log_interval=int(log_interval),
        output_prefix=prefix,
        output_dir=out,
        seed=int(seed),
        fixed_residues=fixed_residues,
        linker_residues=linker_residues,
        linker_energy_mode=linker_energy_mode,
        contact_atom_mode=contact_atom_mode,
        native_contact_pairs=native_contact_pairs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        keep_last_n=keep_last_n,
        checkpoint_config=ckpt_cfg,
        exchange_log=exchange_log,
        state_log_interval=state_log_interval,
        log_walker_in_data_csv=log_walker_in_data_csv,
        step_size_rad=float(step_size_rad),
        move_weights=move_weights,
        sidechain_move_mode=sidechain_move_mode,
        pivot_rama_probability=pivot_rama_probability,
        pivot_rama_schedule=pivot_rama_schedule,
    )
    if verbose:
        print(rex.describe())

    return rex.run(
        int(cycles),
        mc_steps,
        verbose=verbose,
        hdf5_path=analysis,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        resume=resume_path,
        keep_last_n=keep_last_n,
    )


def run_mpi_replica_exchange_2d(
    *,
    comm: Any,
    pdb: str | Path,
    reference_pdb: str | Path | None = None,
    temperatures: list[float] | np.ndarray | None = None,
    n_targets: list[float] | np.ndarray | None = None,
    q_targets: list[float] | np.ndarray | None = None,
    k_bias: float = 1.0,
    cycles: int = 10,
    steps_per_cycle: int = 100,
    swap_interval: int | None = None,
    output_dir: str | Path = "./out_rex",
    seed: int = 42,
    log_interval: int = 100,
    contact_cutoff: float = 6.0,
    min_seq_sep: int = 4,
    contact_atom_mode: str = "ca",
    native_contact_pairs: list[list[int]] | None = None,
    hdf5_path: str | Path | None = None,
    prefix: str = "rex",
    verbose: bool = True,
    fixed_residues: list[int] | None = None,
    linker_residues: list[int] | None = None,
    linker_energy_mode: str = "ignore_all",
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 50,
    resume: str | Path | bool | None = None,
    keep_last_n: int | None = 3,
    cloud_sync: bool = False,
    cloud_bucket: str = "",
    cloud_sync_cmd: str = "aws s3 cp",
    temp_min: float = 0.1,
    temp_step: float = 0.05,
    n_temps: int = 4,
    n_q_windows: int = 1,
    q_step: float = 0.1,
    exchange_log: str = "none",
    state_log_interval: int = 0,
    log_walker_in_data_csv: bool = True,
    step_size_rad: float = 0.1,
    move_weights: tuple[float, float, float] | None = None,
    sidechain_move_mode: str = "rotamer_library",
    pivot_rama_probability: float = 0.0,
    pivot_rama_schedule: dict[str, float] | None = None,
) -> RunSummary | None:
    """Run 2D temperature × N umbrella replica exchange under MPI.

    ``CheckpointConfig`` (all 7 fields) is built here and passed into
    :class:`~pymcpu.sampling.mpi_replica_exchange.MPIReplicaExchange`.
    """
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange

    pdb_path = Path(pdb)
    if not pdb_path.is_absolute():
        pdb_path = resolve_path(pdb_path)
    ref = Path(reference_pdb) if reference_pdb is not None else pdb_path
    if not ref.is_absolute():
        ref = resolve_path(ref)

    out = Path(output_dir)
    rank = comm.Get_rank()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    mc_steps = int(swap_interval if swap_interval is not None else steps_per_cycle)

    analysis = None
    if hdf5_path is not None:
        analysis = Path(hdf5_path)
        if not analysis.is_absolute():
            analysis = out / analysis

    if checkpoint_dir is not None and rank == 0:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    ckpt_cfg = CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval=int(checkpoint_interval),
        resume=resume if resume is not None else False,
        keep_last_n=keep_last_n,
        cloud_sync=bool(cloud_sync),
        cloud_bucket=str(cloud_bucket or ""),
        cloud_sync_cmd=str(cloud_sync_cmd or "aws s3 cp"),
    )
    # Resolve resume to a concrete path/bool before constructing RE
    resume_path = ckpt_cfg.resolved_resume_path()
    ckpt_cfg.resume = True if resume_path else False

    if verbose and rank == 0:
        print(f"PDB: {pdb_path}")
        print(f"Reference: {ref}")
        if checkpoint_dir:
            print(f"Checkpoint: dir={checkpoint_dir}  interval={checkpoint_interval}")
        if resume_path:
            print(f"Resuming from: {resume_path}")

    rex_kwargs: dict[str, Any] = {
        "reference_pdb": str(ref),
        "k_bias": float(k_bias),
        "contact_cutoff": float(contact_cutoff),
        "min_seq_sep": int(min_seq_sep),
        "contact_atom_mode": contact_atom_mode,
        "native_contact_pairs": native_contact_pairs,
        "log_interval": int(log_interval),
        "output_prefix": prefix,
        "output_dir": out,
        "seed": int(seed),
        "fixed_residues": fixed_residues,
        "linker_residues": linker_residues,
        "linker_energy_mode": linker_energy_mode,
        "checkpoint_config": ckpt_cfg,
        "exchange_log": exchange_log,
        "state_log_interval": state_log_interval,
        "log_walker_in_data_csv": log_walker_in_data_csv,
        "step_size_rad": float(step_size_rad),
        "move_weights": move_weights,
        "sidechain_move_mode": sidechain_move_mode,
        "pivot_rama_probability": pivot_rama_probability,
        "pivot_rama_schedule": pivot_rama_schedule,
    }
    if temperatures is not None:
        rex_kwargs["temperatures"] = np.asarray(temperatures, dtype=np.float64)
    else:
        rex_kwargs["temp_min"] = float(temp_min)
        rex_kwargs["temp_step"] = float(temp_step)
        rex_kwargs["n_temps"] = int(n_temps)
    if n_targets is not None:
        rex_kwargs["n_targets"] = np.asarray(n_targets, dtype=np.float64)
    elif q_targets is not None:
        rex_kwargs["q_targets"] = np.asarray(q_targets, dtype=np.float64)
    else:
        rex_kwargs["n_q_windows"] = int(n_q_windows)
        rex_kwargs["q_step"] = float(q_step)

    rex = MPIReplicaExchange(comm, str(pdb_path), **rex_kwargs)
    if verbose and rank == 0:
        print(rex.describe())

    return rex.run(
        int(cycles),
        mc_steps,
        verbose=verbose,
        hdf5_path=analysis,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval=int(checkpoint_interval),
        keep_last_n=keep_last_n,
        resume=ckpt_cfg.resume,
        cloud_sync=bool(cloud_sync),
        cloud_bucket=str(cloud_bucket or ""),
        cloud_sync_cmd=str(cloud_sync_cmd or "aws s3 cp"),
    )


def run_from_config(
    cfg: SimulationConfig, *, comm: Any = None, verbose: bool = True
) -> Path | RunSummary | None:
    """Run a loaded :class:`SimulationConfig`.

    ``comm`` is an MPI communicator (e.g. ``mpi4py.MPI.COMM_WORLD``), supplied
    by the caller when the process was launched under ``mpirun``. It is never
    read from ``cfg`` itself -- MPI-ness is a launch-time/infrastructure
    decision, not a physics setting (see docs/running_remd.md), so passing
    ``comm`` is what selects the MPI-parallel replica-exchange runner.
    """
    pdb = cfg.resolve_pdb()
    fixed = cfg.constraints.fixed_residues or None
    linker = cfg.constraints.linker_residues or None
    linker_mode = cfg.constraints.linker_energy_mode or "ignore_all"
    if cfg.mode == "folding":
        if comm is not None:
            raise ValueError("MPI is not supported for mode='folding'; run without --mpi")
        return run_folding(
            pdb=pdb,
            temperature=cfg.integrator.temperature,
            steps=cfg.integrator.steps,
            report_interval=cfg.integrator.report_interval,
            output_dir=cfg.outputs.output_dir,
            seed=cfg.integrator.seed,
            param_dir=cfg.param_dir,
            param_set=cfg.param_set,
            step_size_rad=cfg.integrator.step_size_rad,
            sidechain_move_mode=cfg.integrator.sidechain_move_mode,
            move_weights=cfg.integrator.move_weights,
            pivot_rama_probability=cfg.integrator.pivot_rama_probability,
            pivot_rama_schedule=cfg.integrator.pivot_rama_schedule,
            prefix=cfg.outputs.prefix if cfg.outputs.prefix != "rex" else "folding",
            verbose=verbose,
            fixed_residues=fixed,
            linker_residues=linker,
            linker_energy_mode=linker_mode,
            checkpoint_dir=cfg.checkpoint.checkpoint_dir,
            checkpoint_interval=cfg.checkpoint.checkpoint_interval,
            resume=cfg.checkpoint.resume,
            keep_last_n=cfg.checkpoint.keep_last_n,
            cloud_sync=cfg.checkpoint.cloud_sync,
            cloud_bucket=cfg.checkpoint.cloud_bucket,
            cloud_sync_cmd=cfg.checkpoint.cloud_sync_cmd,
        )

    if cfg.mode == "replica_exchange_2d":
        assert cfg.replica_exchange is not None
        rex = cfg.replica_exchange
        shared_kwargs: dict[str, Any] = dict(
            pdb=pdb,
            reference_pdb=cfg.resolve_reference_pdb(),
            temperatures=rex.temperatures,
            n_targets=rex.native_contact_targets,
            q_targets=rex.q_targets,
            k_bias=rex.effective_k_bias(),
            cycles=rex.cycles,
            steps_per_cycle=rex.steps_per_cycle,
            swap_interval=rex.swap_interval,
            output_dir=cfg.outputs.output_dir,
            seed=cfg.integrator.seed,
            log_interval=rex.log_interval,
            contact_cutoff=rex.contact_cutoff,
            min_seq_sep=rex.min_seq_sep,
            contact_atom_mode=rex.contact_atom_mode,
            native_contact_pairs=rex.native_contact_pairs,
            hdf5_path=cfg.outputs.hdf5,
            prefix=cfg.outputs.prefix,
            verbose=verbose,
            fixed_residues=fixed,
            linker_residues=linker,
            linker_energy_mode=linker_mode,
            checkpoint_dir=cfg.checkpoint.checkpoint_dir,
            checkpoint_interval=cfg.checkpoint.checkpoint_interval,
            resume=cfg.checkpoint.resume,
            keep_last_n=cfg.checkpoint.keep_last_n,
            cloud_sync=cfg.checkpoint.cloud_sync,
            cloud_bucket=cfg.checkpoint.cloud_bucket,
            cloud_sync_cmd=cfg.checkpoint.cloud_sync_cmd,
            exchange_log=rex.exchange_log,
            state_log_interval=rex.state_log_interval,
            log_walker_in_data_csv=rex.log_walker_in_data_csv,
            step_size_rad=cfg.integrator.step_size_rad,
            move_weights=cfg.integrator.move_weights,
            sidechain_move_mode=cfg.integrator.sidechain_move_mode,
            pivot_rama_probability=cfg.integrator.pivot_rama_probability,
            pivot_rama_schedule=cfg.integrator.pivot_rama_schedule,
        )
        if comm is not None:
            return run_mpi_replica_exchange_2d(comm=comm, **shared_kwargs)
        return run_replica_exchange_2d(backend=rex.backend, **shared_kwargs)

    raise ValueError(f"Unsupported mode: {cfg.mode!r}")
