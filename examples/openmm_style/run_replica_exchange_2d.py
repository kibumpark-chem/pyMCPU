#!/usr/bin/env python3
"""OpenMM-style 2D temperature × native-contact replica exchange example."""

from __future__ import annotations

import argparse

from pymcpu.config import parse_float_list, parse_int_list, resolve_path
from pymcpu.runners import default_example_pdb, run_replica_exchange_2d
from pymcpu.utils.cli import add_checkpoint_args


def _default_pdb() -> str:
    return str(default_example_pdb())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2D replica exchange (temperature × N umbrella) with pyMCPU."
    )
    parser.add_argument(
        "--pdb",
        default=_default_pdb(),
        help="Input PDB (default: examples/data/1uao.pdb)",
    )
    parser.add_argument(
        "--reference-pdb",
        default=None,
        help="Native reference PDB (defaults to --pdb).",
    )
    parser.add_argument(
        "--temperatures",
        default="0.5,0.6",
        help="Comma-separated temperature ladder, e.g. 0.5,0.6",
    )
    parser.add_argument(
        "--n-targets",
        default=None,
        help="Comma-separated N0 ladder, e.g. 0,5,10 (default if --q-targets omitted)",
    )
    parser.add_argument(
        "--q-targets",
        default=None,
        help="Comma-separated fraction-Q centers (converted to N0 = Q * n_contacts)",
    )
    parser.add_argument(
        "--k-bias",
        "--k-native-contacts",
        dest="k_bias",
        type=float,
        default=1.0,
        help="Harmonic N umbrella strength (aliases: --k-native-contacts)",
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--steps-per-cycle", type=int, default=100)
    parser.add_argument(
        "--swap-interval",
        type=int,
        default=None,
        help="MC steps between exchange attempts (defaults to --steps-per-cycle)",
    )
    parser.add_argument("--output-dir", default="./out_rex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        default="serial",
        choices=["serial"],
        help="Execution backend (serial only for now)",
    )
    parser.add_argument(
        "--hdf5",
        default="rex_samples.h5",
        help="Analysis sample output under output-dir (HDF5, or NPZ fallback)",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--fixed-residues",
        default=None,
        help="Comma-separated 0-based engine residue indices to fix, e.g. 0,1,2",
    )

    add_checkpoint_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pdb = resolve_path(args.pdb)
    ref = resolve_path(args.reference_pdb) if args.reference_pdb else pdb

    temps = parse_float_list(args.temperatures)
    n_targets = None
    q_targets = None
    if args.q_targets is not None:
        q_targets = parse_float_list(args.q_targets)
    elif args.n_targets is not None:
        n_targets = parse_float_list(args.n_targets)
    else:
        n_targets = [0.0, 5.0, 10.0]

    fixed = parse_int_list(args.fixed_residues) if args.fixed_residues else None

    run_replica_exchange_2d(
        pdb=pdb,
        reference_pdb=ref,
        temperatures=temps,
        n_targets=n_targets,
        q_targets=q_targets,
        k_bias=args.k_bias,
        cycles=args.cycles,
        steps_per_cycle=args.steps_per_cycle,
        swap_interval=args.swap_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        backend=args.backend,
        log_interval=args.log_interval,
        hdf5_path=args.hdf5,
        fixed_residues=fixed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        keep_last_n=args.keep_last_n,
        cloud_sync=args.cloud_sync,
        cloud_bucket=args.cloud_bucket,
        cloud_sync_cmd=args.cloud_sync_cmd,
    )


if __name__ == "__main__":
    main()
