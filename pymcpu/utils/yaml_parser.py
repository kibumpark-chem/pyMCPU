"""YAML → Python API bridge (GROMACS-style).

This module provides a thin adapter that parses a YAML input file and
constructs the *same* Simulation/runner objects used by the OpenMM-style
Python API.  Zero duplicated physics logic — all heavy lifting is delegated
to :mod:`pymcpu.config` and :mod:`pymcpu.runners`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pymcpu.config import SimulationConfig, check_yaml_keys, load_yaml_config

REQUIRED_FIELDS = [
    "pdb",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file, returning the raw dict."""
    p = Path(path)
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(cfg).__name__}")
    _validate(cfg, str(p))
    return cfg


def _validate(cfg: dict[str, Any], source: str) -> None:
    """Check that the required fields are present and every key is known."""
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        raise ValueError(
            f"[yaml_parser] The following required fields are missing "
            f"in {source}:\n  " + "\n  ".join(missing)
        )
    check_yaml_keys(cfg, source)


def config_from_yaml(path: str | Path) -> SimulationConfig:
    """Parse YAML into a validated SimulationConfig (no simulation objects yet)."""
    # Validate first, so that errors about missing or unknown keys name the file.
    load_yaml(path)
    return load_yaml_config(path)


def simulation_from_yaml(
    path: str | Path,
    *,
    output_dir: str | None = None,
    verbose: bool = True,
    dry_run: bool = False,
) -> "SimulationHandle":
    """Construct a fully configured simulation handle from a YAML input file.

    This is the GROMACS-style entry point.  Internally uses the same Python
    API as the OpenMM-style interface (via :func:`pymcpu.runners.run_from_config`).

    Parameters
    ----------
    path
        Path to YAML input file.
    output_dir
        Override the output directory from the YAML.
    verbose
        Print simulation parameters before running.
    dry_run
        If True, print a description and return without further setup.
    """
    cfg = load_yaml_config(path)
    if output_dir is not None:
        cfg.outputs.output_dir = output_dir
    handle = SimulationHandle(cfg, verbose=verbose)
    if dry_run:
        handle.describe()
    return handle


class SimulationHandle:
    """Thin wrapper around SimulationConfig for GROMACS-style CLI usage.

    This object does NOT duplicate any physics logic.  ``run()`` delegates
    entirely to :func:`pymcpu.runners.run_from_config`.
    """

    def __init__(self, config: SimulationConfig, *, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._checkpoint_path: str | None = None

    def describe(self) -> None:
        """Print all simulation parameters without running."""
        cfg = self.config
        print(f"Mode:         {cfg.mode}")
        print(f"PDB:          {cfg.pdb}")
        print(f"Param set:    {cfg.param_set}")
        if cfg.param_dir:
            print(f"Param dir:    {cfg.param_dir}")
        print(f"Temperature:  {cfg.integrator.temperature}")
        print(f"Steps:        {cfg.integrator.steps}")
        print(f"Seed:         {cfg.integrator.seed}")
        print(f"Step size:    {cfg.integrator.step_size_rad} rad")
        print(f"Report int:   {cfg.integrator.report_interval}")
        print(f"Output dir:   {cfg.outputs.output_dir}")
        if cfg.constraints.fixed_residues:
            print(f"Fixed res:    {len(cfg.constraints.fixed_residues)} residues")
        if cfg.constraints.linker_residues:
            print(
                f"Linker res:   {len(cfg.constraints.linker_residues)} residues "
                f"(mode={cfg.constraints.linker_energy_mode})"
            )
        if cfg.replica_exchange is not None:
            rex = cfg.replica_exchange
            print(f"  Temperatures: {rex.temperatures}")
            if rex.q_targets:
                print(f"  Q targets:    {rex.q_targets}")
            if rex.native_contact_targets:
                print(f"  N targets:    {rex.native_contact_targets}")
            print(f"  k_bias:       {rex.effective_k_bias()}")
            print(f"  Cycles:       {rex.cycles}")
            print(f"  Steps/cycle:  {rex.steps_per_cycle}")
            print(f"  Contact mode: {rex.contact_atom_mode}")
            print(f"  Contact cut:  {rex.contact_cutoff} A (min_seq_sep={rex.min_seq_sep})")
        if cfg.checkpoint.checkpoint_dir:
            print(f"Checkpoint:   {cfg.checkpoint.checkpoint_dir}")

    def load_checkpoint(self, path: str) -> None:
        """Validate and set checkpoint path for resumption."""
        p = Path(path)
        if p.is_dir():
            cand = p / "last.chk"
            if not cand.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {cand}")
            resolved = str(cand)
        else:
            if not p.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            resolved = str(p)
        self._checkpoint_path = resolved
        self.config.checkpoint.resume = resolved

    def run(self) -> Any:
        """Execute the simulation via :func:`pymcpu.runners.run_from_config`."""
        from pymcpu.runners import run_from_config

        return run_from_config(self.config, verbose=self.verbose)
