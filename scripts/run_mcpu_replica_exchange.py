#!/usr/bin/env python3
"""CLI entry point for pyMCPU replica exchange / parallel tempering.

Usage:
    bash scripts/submit.sh CONFIG.yaml
    mpirun -n N_RANKS python run_mcpu_replica_exchange.py --mpi -c input.yaml
    python run_mcpu_replica_exchange.py -c input.yaml

See docs/running_remd.md.
"""

from __future__ import annotations

import argparse
import sys

from pymcpu.config import load_config_auto
from pymcpu.runners import run_from_config
from pymcpu.utils.cli import add_checkpoint_args, apply_checkpoint_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temperature replica exchange / parallel tempering with pyMCPU."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to the YAML or JSON config file.",
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
        "--hdf5",
        default=None,
        help="Optional analysis output path (HDF5 / NPZ fallback); overrides the config.",
    )
    # None defaults: "not set on the CLI", so only the flags the user actually
    # passed override the config's checkpointing.
    add_checkpoint_args(
        parser,
        checkpoint_interval_default=None,
        checkpoint_dir_default=None,
        keep_last_n_default=None,
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Go through the same SimulationConfig / run_from_config path as
    ``mcpu run``. MPI-ness comes only from ``--mpi`` (never from a ``mpi:``
    key in the config -- see docs/running_remd.md)."""
    cfg = load_config_auto(args.config)
    if cfg.replica_exchange is None:
        raise ValueError("Config does not define a replica exchange simulation")
    apply_checkpoint_args(cfg.checkpoint, args)
    if args.hdf5 is not None:
        cfg.outputs.hdf5 = args.hdf5

    comm = None
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    run_from_config(cfg, comm=comm, verbose=True)


def main() -> None:
    args = parse_args()
    if not args.mpi:
        # Serial: let the exception propagate normally.
        _run(args)
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
        _run(args)
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
