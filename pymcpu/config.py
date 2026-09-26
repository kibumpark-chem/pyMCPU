"""JSON simulation config schema (stdlib dataclasses; pydantic optional)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pymcpu.checkpointing import CheckpointConfig

# The supported surface of this module. It is declared explicitly because
# external code depends on it: `scripts/`, `examples/` and the WESTPA add-on
# all import from here, and without an `__all__` every name was public only by
# accident -- there was no way for a caller to tell an intended API from an
# implementation detail it happened to be able to reach.
__all__ = [
    # Config objects
    "CheckpointConfig",
    "ConstraintsConfig",
    "EngineSpec",
    "IntegratorConfig",
    "OutputsConfig",
    "ReplicaExchangeConfig",
    "SimulationConfig",
    # Enumerated value sets
    "ContactAtomMode",
    "LinkerEnergyMode",
    "Mode",
    "SidechainMoveMode",
    "VALID_CONTACT_ATOM_MODES",
    "VALID_LINKER_ENERGY_MODES",
    "VALID_SIDECHAIN_MOVE_MODES",
    # Normalizers and validators. Anything building a pyMCPU run from its own
    # config format should go through these rather than reimplement the rules,
    # so that every entry point rejects the same inputs identically.
    "apply_linker_energy_mask",
    "normalize_linker_energy_mode",
    "normalize_move_weights",
    "normalize_pivot_rama_probability",
    "normalize_pivot_rama_schedule",
    "normalize_sidechain_move_mode",
    "validate_fixed_linker_disjoint",
    # Paths and scalar parsing
    "parse_float_list",
    "parse_int_list",
    "repo_root",
    "replica_grid_dims",
    "resolve_path",
    # Loaders
    "config_from_dict",
    "load_config",
    "load_config_auto",
    "load_yaml_config",
    "validate_config",
    "yaml_dict_to_config",
]


Mode = Literal["folding", "replica_exchange_2d"]
ContactAtomMode = Literal["ca", "cb"]
VALID_CONTACT_ATOM_MODES: tuple[str, ...] = ("ca", "cb")


def _normalize_contact_atom_mode_cfg(mode: str | None) -> ContactAtomMode:
    if mode is None or mode == "":
        return "ca"
    m = str(mode).strip().lower()
    if m not in VALID_CONTACT_ATOM_MODES:
        raise ValueError(
            f"contact_atom_mode must be one of {VALID_CONTACT_ATOM_MODES}, got {mode!r}"
        )
    return m  # type: ignore[return-value]


SidechainMoveMode = Literal["continuous", "rotamer_library"]
VALID_SIDECHAIN_MOVE_MODES: tuple[str, ...] = ("continuous", "rotamer_library")


def normalize_sidechain_move_mode(mode: str | None) -> SidechainMoveMode:
    if mode is None or mode == "":
        return "rotamer_library"
    m = str(mode).strip().lower()
    if m not in VALID_SIDECHAIN_MOVE_MODES:
        raise ValueError(
            f"sidechain_move_mode must be one of {VALID_SIDECHAIN_MOVE_MODES}, got {mode!r}"
        )
    return m  # type: ignore[return-value]


def normalize_pivot_rama_probability(p: float | None) -> float:
    if p is None:
        return 0.05
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"pivot_rama_probability must be in [0, 1], got {p!r}")
    return p


def normalize_move_weights(
    weights: Sequence[float] | None,
) -> tuple[float, float, float]:
    """Validate and normalize the (pivot, kic, sidechain) move-slot weights.

    ``None`` keeps the engine default (0.25, 0.25, 0.50). Normalizing here as
    well as in C++ means a config file reports its own error, naming the key,
    rather than surfacing it from the engine mid-run.
    """
    if weights is None:
        return (0.25, 0.25, 0.50)
    values = tuple(float(w) for w in weights)
    if len(values) != 3:
        raise ValueError(
            f"move_weights must be [pivot, kic, sidechain], got {weights!r}"
        )
    if any(w < 0.0 or w != w for w in values):
        raise ValueError(f"move_weights must be non-negative, got {weights!r}")
    total = sum(values)
    if not (total > 0.0) or total == float("inf"):
        raise ValueError(
            f"move_weights must have a positive, finite sum, got {weights!r}"
        )
    return (values[0] / total, values[1] / total, values[2] / total)


def normalize_pivot_rama_schedule(
    schedule: Mapping[str, float] | None,
) -> dict[str, float] | None:
    """Validates the optional {t_low, t_high, p_min, p_max} schedule dict.
    When present, overrides pivot_rama_probability (see
    MCIntegrator::set_pivot_rama_schedule)."""
    if schedule is None:
        return None
    required = ("t_low", "t_high", "p_min", "p_max")
    missing = [k for k in required if k not in schedule]
    if missing:
        raise ValueError(f"pivot_rama_schedule missing keys: {missing}")
    t_low, t_high = float(schedule["t_low"]), float(schedule["t_high"])
    p_min, p_max = float(schedule["p_min"]), float(schedule["p_max"])
    if not (t_low < t_high):
        raise ValueError("pivot_rama_schedule: t_low must be < t_high")
    if not (0.0 <= p_min <= 1.0 and 0.0 <= p_max <= 1.0):
        raise ValueError("pivot_rama_schedule: p_min/p_max must be in [0, 1]")
    return {"t_low": t_low, "t_high": t_high, "p_min": p_min, "p_max": p_max}


@dataclass
class IntegratorConfig:
    temperature: float = 0.6
    steps: int = 1000
    seed: int = 42
    report_interval: int = 100
    step_size_rad: float = 0.1
    sidechain_move_mode: SidechainMoveMode = "rotamer_library"
    pivot_rama_probability: float = 0.0
    pivot_rama_schedule: dict[str, float] | None = None
    #: (pivot, kic, sidechain) move-slot probabilities. Set the third to 0.0
    #: for a force field whose residues have no chi angles, or half the run is
    #: spent on sidechain proposals that cannot do anything.
    move_weights: tuple[float, float, float] = (0.25, 0.25, 0.50)

    def __post_init__(self) -> None:
        self.move_weights = normalize_move_weights(self.move_weights)


@dataclass
class OutputsConfig:
    output_dir: str = "./out"
    hdf5: str | None = None
    prefix: str = "rex"


@dataclass
class ReplicaExchangeConfig:
    temperatures: list[float] = field(default_factory=lambda: [0.5, 0.6])
    native_contact_targets: list[float] | None = field(
        default_factory=lambda: [0.0, 5.0, 10.0]
    )
    q_targets: list[float] | None = None
    k_native_contacts: float = 1.0
    k_bias: float | None = None  # alias for k_native_contacts
    cycles: int = 10
    steps_per_cycle: int = 100
    swap_interval: int | None = None
    backend: Literal["serial"] = "serial"
    log_interval: int = 100
    contact_cutoff: float = 6.0
    min_seq_sep: int = 4
    contact_atom_mode: Literal["ca", "cb"] = "ca"
    native_contact_pairs: list[list[int]] | None = None
    exchange_log: Literal["none", "all"] = "none"
    state_log_interval: int = 0
    log_walker_in_data_csv: bool = True

    def effective_k_bias(self) -> float:
        if self.k_bias is not None:
            return float(self.k_bias)
        return float(self.k_native_contacts)

    def effective_mc_steps(self) -> int:
        if self.swap_interval is not None:
            return int(self.swap_interval)
        return int(self.steps_per_cycle)


LinkerEnergyMode = Literal["ignore_all", "clash_only"]
VALID_LINKER_ENERGY_MODES: tuple[str, ...] = ("ignore_all", "clash_only")


@dataclass
class ConstraintsConfig:
    fixed_residues: list[int] = field(default_factory=list)
    linker_residues: list[int] = field(default_factory=list)
    linker_energy_mode: LinkerEnergyMode = "ignore_all"


def normalize_linker_energy_mode(mode: str | None) -> LinkerEnergyMode:
    """Validate and normalize linker energy mask mode (default: ignore_all)."""
    if mode is None or mode == "":
        return "ignore_all"
    m = str(mode).strip().lower()
    if m not in VALID_LINKER_ENERGY_MODES:
        raise ValueError(
            f"linker_energy_mode must be one of {VALID_LINKER_ENERGY_MODES}, got {mode!r}"
        )
    return m  # type: ignore[return-value]


def validate_fixed_linker_disjoint(
    fixed_residues: Sequence[int] | None,
    linker_residues: Sequence[int] | None,
) -> None:
    """Reject residues listed as both fixed and linker (contradictory constraints)."""
    fixed = set(int(r) for r in (fixed_residues or []))
    linker = set(int(r) for r in (linker_residues or []))
    overlap = sorted(fixed & linker)
    if overlap:
        raise ValueError(
            "fixed_residues and linker_residues must be disjoint; "
            f"overlapping indices: {overlap}"
        )


def apply_linker_energy_mask(
    system: Any,
    linker_residues: Sequence[int] | None,
    linker_energy_mode: str | None = "ignore_all",
    *,
    fixed_residues: Sequence[int] | None = None,
) -> None:
    """Validate overlap and apply System.set_energy_ignored_residues if linkers given."""
    validate_fixed_linker_disjoint(fixed_residues, linker_residues)
    if not linker_residues:
        return
    mode = normalize_linker_energy_mode(linker_energy_mode)
    system.set_energy_ignored_residues(list(int(r) for r in linker_residues), mode)


@dataclass
class SimulationConfig:
    mode: Mode
    pdb: str
    param_set: str = "mcpu_v1"
    param_dir: str | None = None
    #: Which force field to build: "mcpu08" (all-atom, the default and what
    #: every existing config means) or "korp" (backbone-only). See
    #: pymcpu.forcefields.available_forcefields().
    forcefield: str = "mcpu08"
    #: Constructor arguments for that force field. The two take different
    #: arguments -- MCPU a parameter set, KORP an energy map -- so they are
    #: passed through rather than flattened into one schema that would be
    #: half-irrelevant whichever you pick. Unknown keys raise at build time.
    forcefield_options: dict[str, Any] = field(default_factory=dict)
    reference_pdb: str | None = None
    mpi: bool = False
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    replica_exchange: ReplicaExchangeConfig | None = None
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def resolve_pdb(self, *, repo_root: Path | None = None, cwd: Path | None = None) -> Path:
        return resolve_path(self.pdb, repo_root=repo_root, cwd=cwd)

    def resolve_reference_pdb(
        self, *, repo_root: Path | None = None, cwd: Path | None = None
    ) -> Path:
        ref = self.reference_pdb or self.pdb
        return resolve_path(ref, repo_root=repo_root, cwd=cwd)



@dataclass
class EngineSpec:
    """Everything needed to build one pyMCPU engine, and nothing else.

    This is the seam an external sampling framework builds against: map your
    own configuration onto an ``EngineSpec`` and hand it to
    :class:`pymcpu.sampling.EngineSession`. See
    ``docs/integrating_pymcpu.md``.

    Deliberately FLAT rather than composed of :class:`IntegratorConfig` and
    :class:`ConstraintsConfig`, and the reason is a correctness one rather
    than taste. ``IntegratorConfig`` carries ``seed``, ``steps`` and
    ``report_interval``; an engine driven by an external sampler must not
    honour any of them, because the caller seeds each trajectory segment
    itself (from :func:`pymcpu.sampling.derive_seed`) and decides its own
    step counts. Seeding at construction time would silently change the
    first random number of every segment. Leaving those fields out of the
    type means that cannot be done by accident, which is stronger than a
    test asserting nobody did it.

    The fields and their defaults are exactly the engine-relevant subset of
    what the WESTPA integration's own config block has always carried, with
    the same values, so promoting this type changed no behaviour.
    """

    pdb: str
    cv: tuple[dict, ...] = ()
    param_set: str = "mcpu_v1"
    param_dir: str | None = None
    #: Which force field to build: "mcpu08" (all-atom, the default and what
    #: every existing config means) or "korp" (backbone-only). See
    #: pymcpu.forcefields.available_forcefields().
    forcefield: str = "mcpu08"
    #: Constructor arguments for that force field. The two take different
    #: arguments -- MCPU a parameter set, KORP an energy map -- so they are
    #: passed through rather than flattened into one schema that would be
    #: half-irrelevant whichever you pick. Unknown keys raise at build time.
    forcefield_options: dict[str, Any] = field(default_factory=dict)

    compute_dssp: bool = False
    dssp_coil_state: str = "C"
    temperature: float = 0.6
    step_size_rad: float = 0.1
    sidechain_move_mode: str = "rotamer_library"
    pivot_rama_probability: float = 0.0
    pivot_rama_schedule: dict[str, float] | None = None
    move_weights: tuple[float, float, float] = (0.25, 0.25, 0.50)
    fixed_residues: tuple[int, ...] = ()
    linker_residues: tuple[int, ...] = ()
    linker_energy_mode: str = "ignore_all"

    def __post_init__(self) -> None:
        self.linker_energy_mode = normalize_linker_energy_mode(self.linker_energy_mode)
        self.sidechain_move_mode = normalize_sidechain_move_mode(self.sidechain_move_mode)
        self.pivot_rama_probability = normalize_pivot_rama_probability(
            self.pivot_rama_probability
        )
        self.pivot_rama_schedule = normalize_pivot_rama_schedule(self.pivot_rama_schedule)
        self.move_weights = normalize_move_weights(self.move_weights)
        # Resolved here rather than at build time, so a typo'd name is
        # reported against the config that carries it.
        from pymcpu.forcefields import get_forcefield

        get_forcefield(self.forcefield)
        self.forcefield_options = dict(self.forcefield_options or {})
        validate_fixed_linker_disjoint(self.fixed_residues, self.linker_residues)

        self.fixed_residues = tuple(int(r) for r in self.fixed_residues)
        self.linker_residues = tuple(int(r) for r in self.linker_residues)
        self.cv = tuple(dict(spec) for spec in self.cv)

        # Fail here rather than deep inside a worker: an external sampler
        # typically builds the spec in a master process and the engine in a
        # forked worker, where a missing file surfaces as one dead segment
        # among thousands.
        if not Path(self.pdb).is_file():
            raise FileNotFoundError(f"EngineSpec.pdb not found: {self.pdb}")
        if self.param_dir is not None and not Path(self.param_dir).is_dir():
            raise FileNotFoundError(f"EngineSpec.param_dir not found: {self.param_dir}")
        for spec in self.cv:
            for key in ("reference_pdb", "reference_a", "reference_b"):
                ref = spec.get(key)
                if ref is not None and not Path(ref).is_file():
                    raise FileNotFoundError(f"CV spec {key} not found: {ref}")

    @classmethod
    def from_simulation_config(cls, cfg: SimulationConfig) -> "EngineSpec":
        """Build a spec from pyMCPU's own JSON/YAML config.

        Exists so that a caller who already has a ``SimulationConfig`` does
        not have to restate it, and as the demonstration that this type is
        not shaped around any one external framework.

        ``cfg.integrator.seed``, ``.steps`` and ``.report_interval`` are
        NOT copied -- see the class docstring.
        """
        return cls(
            pdb=cfg.pdb,
            param_set=cfg.param_set,
            param_dir=cfg.param_dir,
            temperature=cfg.integrator.temperature,
            step_size_rad=cfg.integrator.step_size_rad,
            sidechain_move_mode=cfg.integrator.sidechain_move_mode,
            pivot_rama_probability=cfg.integrator.pivot_rama_probability,
            pivot_rama_schedule=cfg.integrator.pivot_rama_schedule,
            move_weights=cfg.integrator.move_weights,
            forcefield=cfg.forcefield,
            forcefield_options=dict(cfg.forcefield_options),
            fixed_residues=tuple(cfg.constraints.fixed_residues),
            linker_residues=tuple(cfg.constraints.linker_residues),
            linker_energy_mode=cfg.constraints.linker_energy_mode,
        )


def repo_root() -> Path:
    """Best-effort repo root, for resolving relative paths in config files.

    This is ``<package parent>``, which is the repo root for an editable
    install and ``site-packages`` for a wheel. That is inherent -- an installed
    package has no repo -- so callers must not rely on it to find shipped data;
    use ``importlib.resources`` (see ``runners.default_example_pdb``). It stays
    as the last-resort base for user-authored config paths only.
    """
    return Path(__file__).resolve().parents[1]


_default_repo_root = repo_root  # captured before resolve_path's `repo_root` param shadows it


def resolve_path(
    path: str | Path,
    *,
    repo_root: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve a path relative to CWD first, then the repository root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    cand = (base_cwd / p).resolve()
    if cand.exists():
        return cand
    root = _default_repo_root() if repo_root is None else Path(repo_root)
    return (root / p).resolve()


def _merge_dataclass(cls: type, data: Mapping[str, Any] | None, *, label: str):
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown fields in {label}: {sorted(unknown)}")
    return cls(**{k: data[k] for k in data if k in known})


def _validate_mode(mode: Any) -> Mode:
    if mode not in ("folding", "replica_exchange_2d"):
        raise ValueError(
            f"mode must be 'folding' or 'replica_exchange_2d', got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def config_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    if not isinstance(data, Mapping):
        raise ValueError("Config root must be a JSON object")
    known = {f.name for f in fields(SimulationConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown top-level fields: {sorted(unknown)}")
    if "mode" not in data or "pdb" not in data:
        raise ValueError("Config requires 'mode' and 'pdb'")

    mode = _validate_mode(data["mode"])
    integrator = _merge_dataclass(
        IntegratorConfig, data.get("integrator"), label="integrator"
    )
    outputs = _merge_dataclass(OutputsConfig, data.get("outputs"), label="outputs")
    constraints = _merge_dataclass(
        ConstraintsConfig, data.get("constraints"), label="constraints"
    )
    checkpoint = _merge_dataclass(
        CheckpointConfig, data.get("checkpoint"), label="checkpoint"
    )

    rex: ReplicaExchangeConfig | None = None
    if mode == "replica_exchange_2d":
        rex_raw = data.get("replica_exchange")
        if rex_raw is None:
            raise ValueError("replica_exchange_2d mode requires 'replica_exchange'")
        rex = _merge_dataclass(
            ReplicaExchangeConfig, rex_raw, label="replica_exchange"
        )
        if not rex.temperatures:
            raise ValueError("replica_exchange.temperatures must be non-empty")
        if rex.native_contact_targets is None and rex.q_targets is None:
            raise ValueError(
                "Provide replica_exchange.native_contact_targets or q_targets"
            )
        if rex.backend != "serial":
            raise ValueError(
                f"Only backend='serial' is supported for now (got {rex.backend!r})"
            )
    elif data.get("replica_exchange") is not None:
        rex = _merge_dataclass(
            ReplicaExchangeConfig, data["replica_exchange"], label="replica_exchange"
        )

    return SimulationConfig(
        mode=mode,
        pdb=str(data["pdb"]),
        param_set=str(data.get("param_set", "mcpu_v1")),
        param_dir=data.get("param_dir"),
        forcefield=str(data.get("forcefield", "mcpu08")),
        forcefield_options=dict(data.get("forcefield_options") or {}),
        reference_pdb=data.get("reference_pdb"),
        integrator=integrator,
        outputs=outputs,
        replica_exchange=rex,
        constraints=constraints,
        checkpoint=checkpoint,
    )


def load_config(path: str | Path) -> SimulationConfig:
    p = Path(path)
    with p.open() as fh:
        data = json.load(fh)
    cfg = config_from_dict(data)
    # Soft-check paths exist relative to CWD / repo (do not require outputs).
    pdb = cfg.resolve_pdb()
    if not pdb.exists():
        raise FileNotFoundError(f"pdb not found: {cfg.pdb} (resolved {pdb})")
    if cfg.reference_pdb:
        ref = cfg.resolve_reference_pdb()
        if not ref.exists():
            raise FileNotFoundError(
                f"reference_pdb not found: {cfg.reference_pdb} (resolved {ref})"
            )
    if cfg.param_dir is not None:
        pd = resolve_path(cfg.param_dir)
        if not pd.exists():
            raise FileNotFoundError(f"param_dir not found: {cfg.param_dir}")
    return cfg


def validate_config(path: str | Path) -> SimulationConfig:
    """Load and validate a JSON config file."""
    return load_config(path)


def parse_float_list(text: str) -> list[float]:
    """Parse comma-separated floats, e.g. ``'0.5,0.6'``."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected a non-empty comma-separated float list")
    return [float(p) for p in parts]


def parse_int_list(text: str) -> list[int]:
    """Parse comma-separated ints, e.g. ``'0,1,2'``."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("expected a non-empty comma-separated int list")
    return [int(p) for p in parts]


# ---------------------------------------------------------------------------
# YAML flat-format loader
# ---------------------------------------------------------------------------

def _expand_temperatures(data: Mapping[str, Any]) -> list[float]:
    """Expand temp_min/temp_step/n_temps into an explicit list, or use 'temperatures'."""
    if "temperatures" in data:
        return [float(t) for t in data["temperatures"]]
    temp_min = data.get("temp_min")
    temp_step = data.get("temp_step")
    n_temps = data.get("n_temps")
    if temp_min is not None and n_temps is not None:
        step = float(temp_step) if temp_step is not None else 0.025
        return [round(float(temp_min) + i * step, 6) for i in range(int(n_temps))]
    if temp_min is not None:
        return [float(temp_min)]
    raise ValueError("Config must provide either 'temperatures' or 'temp_min'+'n_temps'")


def replica_grid_dims(data: Mapping[str, Any]) -> tuple[int, int]:
    """Return (n_replicas, n_temps) implied by a raw YAML/JSON config dict.

    Lightweight, dependency-free companion to :func:`yaml_dict_to_config`: used
    by launch tooling (e.g. scripts/submit.sh) to size ``--ntasks`` without
    building a full SimulationConfig or touching the structure/forcefield.

    Formula mirrors ReplicaExchange/MPIReplicaExchange's actual replica grid:
        n_replicas = len(temperatures) * n_windows
    where n_windows = len(q_targets) if set, else len(n_targets|native_contact_targets)
    if set, else n_q_windows (default 1).
    """
    temperatures = _expand_temperatures(data)
    q_targets = data.get("q_targets")
    n_targets = data.get("n_targets", data.get("native_contact_targets"))
    if q_targets is not None:
        n_windows = len(q_targets)
    elif n_targets is not None:
        n_windows = len(n_targets)
    else:
        n_windows = int(data.get("n_q_windows", 1))
    return len(temperatures) * n_windows, len(temperatures)


def _infer_output_dir_and_prefix(output_prefix: str) -> tuple[str, str]:
    """Split an output_prefix path into (directory, filename_prefix)."""
    p = Path(output_prefix)
    return str(p.parent), p.name


def _checkpoint_from_yaml(data: Mapping[str, Any]) -> CheckpointConfig:
    """Build CheckpointConfig from flat keys and/or nested ``checkpointing``."""
    nested = data.get("checkpointing")
    src: Mapping[str, Any] = nested if isinstance(nested, Mapping) else data
    flat_fallback = data if isinstance(nested, Mapping) else {}
    return CheckpointConfig(
        checkpoint_dir=src.get(
            "checkpoint_dir", flat_fallback.get("checkpoint_dir", "checkpoints")
        ),
        checkpoint_interval=int(
            src.get(
                "checkpoint_interval",
                flat_fallback.get("checkpoint_interval", 50),
            )
        ),
        keep_last_n=src.get("keep_last_n", flat_fallback.get("keep_last_n", 3)),
        resume=src.get("resume", flat_fallback.get("resume", False)),
        cloud_sync=bool(src.get("cloud_sync", flat_fallback.get("cloud_sync", False))),
        cloud_bucket=str(
            src.get("cloud_bucket", flat_fallback.get("cloud_bucket", "")) or ""
        ),
        cloud_sync_cmd=str(
            src.get(
                "cloud_sync_cmd",
                flat_fallback.get("cloud_sync_cmd", "aws s3 cp"),
            )
            or "aws s3 cp"
        ),
        enabled=bool(src.get("enabled", flat_fallback.get("enabled", True))),
    )


def load_yaml_config(path: str | Path) -> SimulationConfig:
    """Load a flat YAML config and map it to SimulationConfig.

    This accepts the flat schema used by pyMCPU production runs:
        pdb, output_prefix, num_cycles, mc_replica_steps, seed,
        fixed_residue_indices, temp_min/temp_step/n_temps (or temperatures),
        n_targets/native_contact_targets or q_targets, k_bias, contact_cutoff,
        min_seq_sep, checkpointing / checkpoint_dir, log_interval, …
    """
    import yaml

    p = Path(path)
    with p.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")

    return yaml_dict_to_config(data)


def yaml_dict_to_config(data: Mapping[str, Any]) -> SimulationConfig:
    """Convert a flat YAML dict into a SimulationConfig."""
    data = dict(data)
    if "pdb" not in data:
        raise ValueError("YAML config requires 'pdb'")

    temperatures = _expand_temperatures(data)
    mode: Mode = "replica_exchange_2d" if len(temperatures) > 1 else "folding"

    seed = int(data.get("seed", 42))
    mc_steps = int(data.get("mc_replica_steps", data.get("steps", 1000)))
    num_cycles = int(data.get("num_cycles", 10))
    log_interval = int(data.get("log_interval", mc_steps))
    step_size_rad = float(data.get("step_size_rad", 0.1))

    output_prefix_raw = data.get("output_prefix", "./out/sim")
    out_dir, out_name = _infer_output_dir_and_prefix(str(output_prefix_raw))
    if "output_dir" in data:
        out_dir = str(data["output_dir"])

    outputs = OutputsConfig(
        output_dir=out_dir,
        hdf5=data.get("hdf5"),
        prefix=out_name,
    )

    integrator = IntegratorConfig(
        temperature=float(temperatures[0]),
        steps=mc_steps * num_cycles if mode == "folding" else mc_steps,
        seed=seed,
        report_interval=log_interval,
        step_size_rad=step_size_rad,
        sidechain_move_mode=normalize_sidechain_move_mode(
            data.get("sidechain_move_mode", "rotamer_library")
        ),
        pivot_rama_probability=normalize_pivot_rama_probability(
            data.get("pivot_rama_probability")
        ),
        pivot_rama_schedule=normalize_pivot_rama_schedule(data.get("pivot_rama_schedule")),
        move_weights=normalize_move_weights(data.get("move_weights")),
    )

    rex: ReplicaExchangeConfig | None = None
    if mode == "replica_exchange_2d":
        q_targets = data.get("q_targets")
        n_targets = data.get("n_targets", data.get("native_contact_targets"))
        if q_targets is not None and n_targets is not None:
            raise ValueError(
                "Provide either 'n_targets'/'native_contact_targets' or "
                "'q_targets', not both"
            )
        k_bias = None
        if "k_bias" in data:
            k_bias = float(data["k_bias"])
        elif "k_native_contacts" in data:
            k_bias = float(data["k_native_contacts"])
        ex_log = str(data.get("exchange_log", "none")).strip().lower()
        if ex_log not in ("none", "all"):
            raise ValueError("exchange_log must be 'none' or 'all'")
        native_contact_pairs_raw = data.get("native_contact_pairs")
        native_contact_pairs: list[list[int]] | None = None
        if native_contact_pairs_raw is not None:
            native_contact_pairs = []
            for pair in native_contact_pairs_raw:
                pair_list = list(pair)
                if len(pair_list) != 2:
                    raise ValueError(
                        "native_contact_pairs entries must have exactly 2 "
                        f"elements, got {pair!r}"
                    )
                native_contact_pairs.append([int(pair_list[0]), int(pair_list[1])])
        rex = ReplicaExchangeConfig(
            temperatures=temperatures,
            native_contact_targets=(
                [float(n) for n in n_targets] if n_targets is not None else None
            ),
            q_targets=(
                [float(q) for q in q_targets] if q_targets is not None else None
            ),
            k_bias=k_bias,
            cycles=num_cycles,
            steps_per_cycle=mc_steps,
            log_interval=log_interval,
            contact_cutoff=float(data.get("contact_cutoff", 6.0)),
            min_seq_sep=int(data.get("min_seq_sep", 4)),
            contact_atom_mode=_normalize_contact_atom_mode_cfg(
                data.get("contact_atom_mode", "ca")
            ),
            native_contact_pairs=native_contact_pairs,
            exchange_log=ex_log,  # type: ignore[arg-type]
            state_log_interval=int(data.get("state_log_interval", 0)),
            log_walker_in_data_csv=bool(data.get("log_walker_in_data_csv", True)),
        )

    fixed = data.get("fixed_residue_indices", data.get("fixed_residues", [])) or []
    linker = data.get("linker_residue_indices", data.get("linker_residues", [])) or []
    linker_mode = normalize_linker_energy_mode(data.get("linker_energy_mode", "ignore_all"))
    constraints = ConstraintsConfig(
        fixed_residues=[int(r) for r in fixed],
        linker_residues=[int(r) for r in linker],
        linker_energy_mode=linker_mode,
    )
    validate_fixed_linker_disjoint(constraints.fixed_residues, constraints.linker_residues)
    checkpoint = _checkpoint_from_yaml(data)

    return SimulationConfig(
        mode=mode,
        pdb=str(data["pdb"]),
        param_set=str(data.get("param_set", "mcpu_v1")),
        param_dir=data.get("param_dir"),
        forcefield=str(data.get("forcefield", "mcpu08")),
        forcefield_options=dict(data.get("forcefield_options") or {}),
        reference_pdb=data.get("reference_pdb"),
        integrator=integrator,
        outputs=outputs,
        replica_exchange=rex,
        constraints=constraints,
        checkpoint=checkpoint,
    )


def load_config_auto(path: str | Path) -> SimulationConfig:
    """Auto-detect JSON or YAML and load accordingly."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_yaml_config(p)
    return load_config(p)
