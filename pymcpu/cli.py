"""``mcpu`` command-line entry point.

Subcommands: ``version``, ``download-params``, ``materialize-params``,
``run``, ``validate``, ``, ``, ````. See docs/cli.rst.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    _apply_checkpoint_cli_overrides(cfg, args)
    print(f"OK  mode={cfg.mode}  pdb={cfg.pdb}")
    if cfg.mpi:
        print("    mpi=true")
    if cfg.checkpoint.checkpoint_dir:
        print(f"    checkpoint_dir={cfg.checkpoint.checkpoint_dir}")
        print(f"    checkpoint_interval={cfg.checkpoint.checkpoint_interval}")
    return 0


def _apply_checkpoint_cli_overrides(cfg, args: argparse.Namespace) -> None:
    """Merge CLI checkpoint flags into ``cfg.checkpoint`` when provided."""
    chk = cfg.checkpoint
    if getattr(args, "checkpoint_dir", None) is not None:
        chk.checkpoint_dir = args.checkpoint_dir
    if getattr(args, "checkpoint_interval", None) is not None:
        chk.checkpoint_interval = int(args.checkpoint_interval)
    if getattr(args, "keep_last_n", None) is not None:
        chk.keep_last_n = int(args.keep_last_n)
    if getattr(args, "resume", False):
        chk.resume = True
    if getattr(args, "cloud_sync", False):
        chk.cloud_sync = True
    if getattr(args, "cloud_bucket", None):
        chk.cloud_bucket = str(args.cloud_bucket)
    if getattr(args, "cloud_sync_cmd", None):
        chk.cloud_sync_cmd = str(args.cloud_sync_cmd)


def _cmd_materialize_params(args: argparse.Namespace) -> int:
    """Decode the in-wheel compact tables once and print the params root.

    Exists so an MPI launcher can serialize the decode. `pymcpu.params`
    documents the recipe as one line before `mpirun`:

        export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu_v1)"

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
        _apply_checkpoint_cli_overrides(cfg, args)
        run_from_config(cfg, verbose=not args.quiet)
    except Exception as exc:
        print(f"mcpu run failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    checkpoint_group = parser.add_argument_group("checkpointing")
    checkpoint_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        metavar="N",
        help="Save a checkpoint every N cycles. Default: 50 (from CheckpointConfig).",
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to write checkpoint files. Default: checkpoints/",
    )
    checkpoint_group.add_argument(
        "--keep-last-n",
        type=int,
        default=None,
        help="Number of versioned checkpoints to keep. Default: 3.",
    )
    checkpoint_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the latest checkpoint in --checkpoint-dir.",
    )
    checkpoint_group.add_argument(
        "--cloud-sync",
        action="store_true",
        default=False,
        help="Upload last.chk to cloud after each save.",
    )
    checkpoint_group.add_argument(
        "--cloud-bucket",
        type=str,
        default="",
        metavar="URI",
        help="Cloud destination URI. Example: s3://my-bucket/run-01/",
    )
    checkpoint_group.add_argument(
        "--cloud-sync-cmd",
        type=str,
        default="aws s3 cp",
        help="Cloud sync command. Default: 'aws s3 cp'.",
    )


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
        default="mcpu_v1",
        help="Parameter set name from the registry (default: mcpu_v1)",
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
        default="mcpu_v1",
        help="Parameter set name from the registry (default: mcpu_v1)",
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
