#!/usr/bin/env python3
"""CLI entry point for pyMCPU replica exchange / parallel tempering.

Usage with config file (preferred):
    bash scripts/submit.sh inputs/template.yaml
    mpirun -n N_REPLICAS python run_mcpu_replica_exchange.py --mpi -c input.yaml
    python run_mcpu_replica_exchange.py -c input.yaml

Usage with CLI flags (legacy):
    mpirun -n N_REPLICAS python run_mcpu_replica_exchange.py --mpi --pdb struct.pdb ...

See docs/running_remd.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pymcpu.config import parse_float_list, resolve_path
from pymcpu.runners import default_example_pdb, run_from_config
from pymcpu.utils.cli import add_checkpoint_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temperature replica exchange / parallel tempering with pyMCPU."
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to YAML or JSON config file. Overrides all other flags except --mpi.",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "Use mpi4py. Launch with mpirun -n <ranks>. "
            "Ranks can be fewer than replicas; each rank then runs "
            "multiple trajectories sequentially before exchanges."
        ),
    )
    parser.add_argument(
        "--pdb",
        default=None,
        help="Input structure (also used as the native reference for N/Q).",
    )
    parser.add_argument(
        "--reference-pdb",
        default=None,
        help="Native reference PDB (defaults to --pdb).",
    )
    parser.add_argument("--output-prefix", default="rex", help="Output file prefix.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--num-cycles", type=int, default=10, help="Exchange cycles.")
    parser.add_argument(
        "--mc-replica-steps",
        type=int,
        default=1000,
        help="MC steps per replica between exchange attempts.",
    )
    parser.add_argument("--log-interval", type=int, default=100, help="Reporter interval.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for exchanges.")
    parser.add_argument(
        "--hdf5",
        default=None,
        help="Optional analysis output path (HDF5 / NPZ fallback).",
    )
    parser.add_argument(
        "--fixed-residue-indices",
        default=None,
        help="Comma-separated residue indices to fix.",
    )
    parser.add_argument(
        "--linker-residue-indices",
        default=None,
        help="Comma-separated movable ghost residue indices (energy-masked).",
    )
    parser.add_argument(
        "--linker-energy-mode",
        default="ignore_all",
        choices=["ignore_all", "clash_only"],
        help="Energy mask mode for linker residues (default: ignore_all).",
    )

    parser.add_argument("--temp-min", type=float, default=0.1)
    parser.add_argument("--temp-step", type=float, default=0.05)
    parser.add_argument("--n-temps", type=int, default=4, help="Number of temperature replicas.")
    parser.add_argument(
        "--temperatures",
        default=None,
        help="Optional comma-separated temperatures (overrides temp-min/step/n-temps).",
    )

    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=6.0,
        help="Reference CA-CA distance cutoff (A) defining native pairs.",
    )
    parser.add_argument(
        "--q-cutoff",
        type=float,
        default=None,
        help="Current-structure CA-CA cutoff (A) for a formed contact (defaults to contact-cutoff).",
    )
    parser.add_argument(
        "--min-seq-sep",
        type=int,
        default=4,
        help="Skip residue pairs with sequence separation smaller than this value.",
    )
    parser.add_argument(
        "--contact-atom-mode",
        default="ca",
        choices=["ca", "cb"],
        help="Native-contact atoms: ca (CA–CA) or cb (CB–CB, GLY→CA). Default: ca.",
    )

    parser.add_argument(
        "--n-targets",
        default=None,
        help="Comma-separated hard N0 umbrella centers.",
    )
    parser.add_argument(
        "--n-q-windows",
        type=int,
        default=1,
        help="Number of umbrella windows along Q (1 = temperature RE only).",
    )
    parser.add_argument(
        "--q-step",
        type=float,
        default=0.1,
        help="Spacing between Q umbrella centers when n-q-windows > 1.",
    )
    parser.add_argument(
        "--q-targets",
        default=None,
        help="Comma-separated fraction-Q centers (converted to N).",
    )
    parser.add_argument(
        "--k-bias",
        type=float,
        default=0.0,
        help="Harmonic umbrella strength for N parallel tempering (0 = no bias).",
    )

    # None defaults: "not set on the CLI" so the --config path only overrides
    # the YAML-loaded CheckpointConfig for flags the user actually passed.
    add_checkpoint_args(
        parser,
        checkpoint_interval_default=None,
        checkpoint_dir_default=None,
        keep_last_n_default=None,
    )
    return parser.parse_args()


def _rex_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "reference_pdb": args.reference_pdb,
        "k_bias": args.k_bias,
        "contact_cutoff": args.contact_cutoff,
        "q_cutoff": args.q_cutoff,
        "min_seq_sep": args.min_seq_sep,
        "contact_atom_mode": args.contact_atom_mode,
        "log_interval": args.log_interval,
        "output_prefix": args.output_prefix,
        "seed": args.seed,
    }
    if args.temperatures:
        kwargs["temperatures"] = parse_float_list(args.temperatures)
    else:
        kwargs["temp_min"] = args.temp_min
        kwargs["temp_step"] = args.temp_step
        kwargs["n_temps"] = args.n_temps

    if args.n_targets:
        kwargs["n_targets"] = parse_float_list(args.n_targets)
    elif args.q_targets:
        kwargs["q_targets"] = parse_float_list(args.q_targets)
    else:
        kwargs["n_q_windows"] = args.n_q_windows
        kwargs["q_step"] = args.q_step

    if args.fixed_residue_indices:
        from pymcpu.config import parse_int_list
        kwargs["fixed_residues"] = parse_int_list(args.fixed_residue_indices)
    if args.linker_residue_indices:
        from pymcpu.config import parse_int_list
        kwargs["linker_residues"] = parse_int_list(args.linker_residue_indices)
        kwargs["linker_energy_mode"] = args.linker_energy_mode

    from pymcpu.checkpointing import CheckpointConfig

    kwargs["checkpoint_config"] = CheckpointConfig(
        checkpoint_dir=args.checkpoint_dir or "checkpoints",
        checkpoint_interval=int(args.checkpoint_interval)
        if args.checkpoint_interval is not None
        else 50,
        keep_last_n=int(args.keep_last_n) if args.keep_last_n is not None else 3,
        resume=bool(args.resume),
        cloud_sync=bool(args.cloud_sync),
        cloud_bucket=str(args.cloud_bucket or ""),
        cloud_sync_cmd=str(args.cloud_sync_cmd or "aws s3 cp"),
    )

    return kwargs


def _run_from_config_path(args: argparse.Namespace) -> None:
    """Config-file path (preferred): go through the same SimulationConfig /
    run_from_config path as ``mcpu run``, instead of hand-rolling a second
    config-to-kwargs adapter. MPI-ness still comes only from ``--mpi``
    (never from a ``mpi:`` key in the config -- see docs/running_remd.md)."""
    from pymcpu.cli import _apply_checkpoint_cli_overrides
    from pymcpu.config import load_config_auto

    cfg = load_config_auto(args.config)
    if cfg.replica_exchange is None:
        raise ValueError("Config does not define a replica exchange simulation")
    _apply_checkpoint_cli_overrides(cfg, args)
    if args.hdf5 is not None:
        cfg.outputs.hdf5 = args.hdf5

    comm = None
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    run_from_config(cfg, comm=comm, verbose=True)


def _run_from_legacy_flags(args: argparse.Namespace) -> None:
    """Legacy CLI-flag path (no --config): unchanged, not routed through
    SimulationConfig -- --n-q-windows/--q-step/temp-min/temp-step/n-temps
    ladder generation have no equivalent in ReplicaExchangeConfig today."""
    kwargs = _rex_kwargs(args)
    pdb = str(resolve_path(args.pdb or str(default_example_pdb())))
    if args.reference_pdb:
        kwargs["reference_pdb"] = str(resolve_path(args.reference_pdb))
    num_cycles = args.num_cycles
    mc_steps = args.mc_replica_steps
    hdf5_path = args.hdf5

    if args.mpi:
        from mpi4py import MPI

        from pymcpu.sampling import MPIReplicaExchange

        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            kwargs["output_prefix"] = str(out / Path(kwargs.get("output_prefix", "rex")).name)

        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            print(f"Loading structure from {pdb}")
        rex = MPIReplicaExchange(comm, pdb, **kwargs)
        if comm.Get_rank() == 0:
            print(rex.describe())

        # Merge CLI checkpoint flags into the stored CheckpointConfig (None = keep YAML).
        ckpt = rex.checkpoint_config
        if args.checkpoint_dir is not None:
            ckpt.checkpoint_dir = args.checkpoint_dir
        if args.checkpoint_interval is not None:
            ckpt.checkpoint_interval = int(args.checkpoint_interval)
        if args.keep_last_n is not None:
            ckpt.keep_last_n = int(args.keep_last_n)
        if args.resume:
            ckpt.resume = True
        if args.cloud_sync:
            ckpt.cloud_sync = True
        if args.cloud_bucket:
            ckpt.cloud_bucket = args.cloud_bucket
        if args.cloud_sync_cmd:
            ckpt.cloud_sync_cmd = args.cloud_sync_cmd

        rex.run(
            num_cycles,
            mc_steps,
            hdf5_path=hdf5_path,
            checkpoint_dir=ckpt.checkpoint_dir,
            checkpoint_interval=ckpt.checkpoint_interval,
            keep_last_n=ckpt.keep_last_n,
            resume=ckpt.resolved_resume_path(),
            cloud_sync=ckpt.cloud_sync,
            cloud_bucket=ckpt.cloud_bucket,
            cloud_sync_cmd=ckpt.cloud_sync_cmd,
        )
        return

    from pymcpu.sampling import ReplicaExchange

    if args.output_dir:
        kwargs["output_dir"] = args.output_dir

    print(f"Loading structure from {pdb}")
    rex = ReplicaExchange(pdb, **kwargs)
    print(rex.describe())
    rex.run(num_cycles, mc_steps, hdf5_path=hdf5_path)


def main() -> None:
    args = parse_args()
    if not args.mpi:
        # Serial: let the exception propagate normally.
        if args.config:
            _run_from_config_path(args)
        else:
            _run_from_legacy_flags(args)
        return

    # Under MPI a fatal error on ONE rank must take the whole job down.
    #
    # Without this, an exception here kills only the raising rank while the
    # other N-1 stay blocked in the next collective (the exchange gather), and
    # the job then burns its full allocation doing nothing until SLURM's
    # walltime or an operator kills it. That is not hypothetical: the four
    # p18.8.8 NBD1 runs each raised StericClashError on a single rank and then
    # hung -- hsa for 8h16m, pab 4h43m, sce 4h17m, all at 60 ranks, roughly
    # 1300 core-hours of pure waste. Their logs contain the traceback and no
    # abort of any kind.
    #
    # MPI.Abort() is the only reliable way out: once one rank has left the
    # collective sequence the communicator cannot be resynchronised, so there
    # is nothing to gain by trying to shut down cleanly. Print the traceback
    # first (tagged and flushed, since output from a dying MPI job is easily
    # lost or interleaved) so the log still says what happened.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        if args.config:
            _run_from_config_path(args)
        else:
            _run_from_legacy_flags(args)
    except BaseException:  # noqa: BLE001 -- deliberately includes KeyboardInterrupt/SystemExit
        import traceback

        rank = comm.Get_rank()
        sys.stderr.write(
            f"\n=== FATAL on MPI rank {rank}/{comm.Get_size()} -- aborting all ranks ===\n"
        )
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        comm.Abort(1)
        raise  # unreachable; Abort does not return


if __name__ == "__main__":
    main()
