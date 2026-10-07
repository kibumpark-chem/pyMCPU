"""Shared argparse building blocks for pyMCPU CLI entry-point scripts.

Kept intentionally tiny: this module only assembles ``argparse`` groups that
are otherwise copy-pasted across scripts/ and examples/ entry points. It must
never import anything from the physics/engine layers.
"""

from __future__ import annotations

import argparse
from typing import Any


def add_checkpoint_args(
    parser: argparse.ArgumentParser,
    *,
    checkpoint_interval_default: int | None = 50,
    checkpoint_dir_default: str | None = "checkpoints",
    keep_last_n_default: int | None = 3,
) -> argparse._ArgumentGroup:
    """Add the shared ``--checkpoint-*`` / ``--resume`` group.

    Identical across every pyMCPU CLI entry point (scripts/run_mcpu_folding.py,
    scripts/run_mcpu_replica_exchange.py, examples/openmm_style/*.py,
    ``mcpu run``); factored out so the flags/help text can't
    silently drift between scripts.

    Pass ``None`` for a default to mean "not set on the CLI, defer to whatever
    CheckpointConfig/YAML already supplies" -- this is what
    scripts/run_mcpu_replica_exchange.py's ``--config`` path relies on, since it
    only overrides the YAML-loaded CheckpointConfig for flags the user actually
    passed on the command line.
    """
    checkpoint_group = parser.add_argument_group("checkpointing")
    interval_help = (
        "Save a checkpoint every N cycles. Default: 50 (from CheckpointConfig)."
        if checkpoint_interval_default is None
        else f"Save a checkpoint every N cycles. Default: {checkpoint_interval_default}."
    )
    checkpoint_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=checkpoint_interval_default,
        metavar="N",
        help=interval_help,
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=checkpoint_dir_default,
        help="Directory to write checkpoint files. Default: checkpoints/",
    )
    checkpoint_group.add_argument(
        "--keep-last-n",
        type=int,
        default=keep_last_n_default,
        help="Number of versioned checkpoints to keep. Default: 3.",
    )
    checkpoint_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from the latest checkpoint in --checkpoint-dir.",
    )
    return checkpoint_group


def apply_checkpoint_args(checkpoint: Any, args: argparse.Namespace) -> None:
    """Override a loaded ``CheckpointConfig`` with the checkpoint flags that
    were given on the command line, leaving the config's other settings alone.

    Use it with ``add_checkpoint_args(..., checkpoint_interval_default=None,
    checkpoint_dir_default=None, keep_last_n_default=None)``, so that a flag
    left out reads as "not set" rather than as its default.
    """
    if getattr(args, "checkpoint_dir", None) is not None:
        checkpoint.checkpoint_dir = args.checkpoint_dir
    if getattr(args, "checkpoint_interval", None) is not None:
        checkpoint.checkpoint_interval = int(args.checkpoint_interval)
    if getattr(args, "keep_last_n", None) is not None:
        checkpoint.keep_last_n = int(args.keep_last_n)
    if getattr(args, "resume", False):
        checkpoint.resume = True
