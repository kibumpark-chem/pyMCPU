"""``mcpu`` command-line entry point.

Subcommands: ``version``, ``download-params``, ``materialize-params``,
``run``, ``validate``, ``, ``, ````. See docs/cli.rst.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pymcpu.utils.cli import add_checkpoint_args, apply_checkpoint_args


def _add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    # Unset flags stay None, so they leave the config's checkpointing alone.
    add_checkpoint_args(
        parser,
        checkpoint_interval_default=None,
        checkpoint_dir_default=None,
        keep_last_n_default=None,
    )


def _cmd_download_params(args: argparse.Namespace) -> int:
    from pymcpu.params import ParamsError, download_params

    try:
        path = download_params(
            args.set,
            dest=args.dir,
        )
    except ParamsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(path)
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    from pymcpu import __version__

    print(__version__)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from pymcpu.config import load_config_auto

    try:
        cfg = load_config_auto(args.config)
    except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"config validation failed: {exc}", file=sys.stderr)
        return 1
    apply_checkpoint_args(cfg.checkpoint, args)
    print(f"OK  mode={cfg.mode}  pdb={cfg.pdb}")
    if cfg.mpi:
        print("    mpi=true")
    if cfg.checkpoint.checkpoint_dir:
        print(f"    checkpoint_dir={cfg.checkpoint.checkpoint_dir}")
        print(f"    checkpoint_interval={cfg.checkpoint.checkpoint_interval}")
    return 0


def _cmd_materialize_params(args: argparse.Namespace) -> int:
    """Decode the in-wheel compact tables once and print the params root.

    Exists so an MPI launcher can serialize the decode. `pymcpu.params`
    documents the recipe as one line before `mpirun`:

        export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"

    which turns an N-way race between ranks cold-starting against a shared
    $HOME into a single call, and pins every rank to one identical root.
    """
    from pymcpu.params import ParamsError, materialize_from_wheel

    try:
        path = materialize_from_wheel(
            args.set,
            timeout=args.timeout,
            verify=not args.no_verify,
        )
    except ParamsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(path)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pymcpu.config import load_config_auto
    from pymcpu.runners import run_from_config

    try:
        cfg = load_config_auto(args.config)
        apply_checkpoint_args(cfg.checkpoint, args)
        run_from_config(cfg, verbose=not args.quiet)
    except Exception as exc:
        print(f"mcpu run failed: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcpu",
        description="pyMCPU command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ver = sub.add_parser("version", help="Print package version")
    p_ver.set_defaults(func=_cmd_version)

    p_dl = sub.add_parser(
        "download-params",
        help="Resolve/download pretrained parameters and print the final path",
    )
    p_dl.add_argument(
        "--set",
        default="mcpu08",
        help="Parameter set name from the registry (default: mcpu08)",
    )
    p_dl.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Optional staging directory to copy resolved params into",
    )
    p_dl.set_defaults(func=_cmd_download_params)

    p_mat = sub.add_parser(
        "materialize-params",
        help=(
            "Decode the in-wheel compact tables into a params root and print "
            "it (run once before mpirun to avoid an N-way decode race)"
        ),
    )
    p_mat.add_argument(
        "--set",
        default="mcpu08",
        help="Parameter set name from the registry (default: mcpu08)",
    )
    p_mat.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for another process holding the lock (default: 900)",
    )
    p_mat.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the per-table sha256 check (faster; not recommended)",
    )
    p_mat.set_defaults(func=_cmd_materialize_params)

    p_run = sub.add_parser("run", help="Run a simulation config (JSON or YAML)")
    p_run.add_argument("config", type=Path, help="Path to config file (.json/.yaml/.yml)")
    p_run.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout chatter",
    )
    _add_checkpoint_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="Validate a simulation config (JSON or YAML)")
    p_val.add_argument("config", type=Path, help="Path to config file (.json/.yaml/.yml)")
    _add_checkpoint_args(p_val)
    p_val.set_defaults(func=_cmd_validate)


    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
