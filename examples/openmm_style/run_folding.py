#!/usr/bin/env python3
"""OpenMM-style single-temperature MC folding example."""

from __future__ import annotations

import argparse

from pymcpu.config import parse_int_list, resolve_path
from pymcpu.runners import default_example_pdb, run_folding
from pymcpu.utils.cli import add_checkpoint_args


def _default_pdb() -> str:
    return str(default_example_pdb())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-temperature pyMCPU folding simulation."
    )
    parser.add_argument(
        "--pdb",
        default=_default_pdb(),
        help="Input PDB (default: examples/data/1uao.pdb)",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--output-dir", default="./out_folding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param-dir", default=None)
    parser.add_argument("--param-set", default="mcpu08")
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
    fixed = parse_int_list(args.fixed_residues) if args.fixed_residues else None
    run_folding(
        pdb=pdb,
        temperature=args.temperature,
        steps=args.steps,
        report_interval=args.report_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        param_dir=args.param_dir,
        param_set=args.param_set,
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
