#!/usr/bin/env python3
"""Single-temperature MC folding (argparse wrapper around pymcpu.runners)."""

from __future__ import annotations

import argparse

from pymcpu.config import resolve_path
from pymcpu.runners import default_example_pdb, run_folding
from pymcpu.utils.cli import add_checkpoint_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pyMCPU folding simulation.")
    parser.add_argument(
        "--pdb",
        default=str(default_example_pdb()),
        help="Input PDB (default: examples/data/1uao.pdb)",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--output-dir", default="./out_folding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param-dir", default=None)
    parser.add_argument("--param-set", default="mcpu08")

    add_checkpoint_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_folding(
        pdb=resolve_path(args.pdb),
        temperature=args.temperature,
        steps=args.steps,
        report_interval=args.report_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        param_dir=args.param_dir,
        param_set=args.param_set,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        keep_last_n=args.keep_last_n,
    )


if __name__ == "__main__":
    main()
