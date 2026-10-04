#!/usr/bin/env python3
"""GROMACS-style entry point for pyMCPU simulations.

Usage
-----
    python run.py --input input.yaml [--resume --checkpoint-dir checkpoints]
                  [--output-dir ./results] [--dry-run]

The input YAML fully specifies the simulation. See inputs/template.yaml
for all available fields and their default values.

This script is a thin adapter:
  (a) Parses the YAML
  (b) Constructs the SAME Python objects as the OpenMM-style API
  (c) Calls the SAME run() method
Zero duplicated logic between the two interfaces.
"""

from __future__ import annotations

import argparse
import sys

from pymcpu.utils.cli import add_checkpoint_args, apply_checkpoint_args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pyMCPU simulation from YAML input (GROMACS-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Restart from an explicit checkpoint file (overrides --resume)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides YAML value)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate YAML without running the simulation",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    # Unset flags stay None, so the YAML's checkpointing block is kept.
    add_checkpoint_args(
        parser,
        checkpoint_interval_default=None,
        checkpoint_dir_default=None,
        keep_last_n_default=None,
    )
    args = parser.parse_args()

    from pymcpu.utils.yaml_parser import simulation_from_yaml

    try:
        sim = simulation_from_yaml(
            args.input,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    apply_checkpoint_args(sim.config.checkpoint, args)
    if args.checkpoint:
        sim.load_checkpoint(args.checkpoint)

    if args.dry_run:
        print("[dry-run] YAML parsed successfully. Simulation object created.")
        sim.describe()
        return 0

    try:
        sim.run()
    except Exception as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
