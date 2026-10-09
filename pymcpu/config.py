"""JSON simulation config schema (stdlib dataclasses; pydantic optional)."""

from __future__ import annotations

import difflib
import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pymcpu.checkpointing import CheckpointConfig

# The supported surface of this module. It is declared explicitly because
# external code depends on it: `scripts/`, `examples/` and external
# integrations all import from here, and without an `__all__` every name was public only by
# accident -- there was no way for a caller to tell an intended API from an
# implementation detail it happened to be able to reach.
__all__ = [
    # Config objects
    "CheckpointConfig",
    "ConstraintsConfig",
    "EngineSpec",
    "FoldingConfig",
    "IntegratorConfig",
    "OutputsConfig",
    "ReplicaExchangeConfig",
    "SimulationConfig",
    # Enumerated value sets and shared defaults
    "ContactAtomMode",
    "DEFAULT_CONTACT_ATOM_MODE",
    "DEFAULT_CONTACT_CUTOFF",
    "DEFAULT_KIC_STEP_SIZE_RAD",
    "DEFAULT_MIN_SEQ_SEP",
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
    "check_yaml_keys",
    "normalize_kic_step_size_rad",
    "normalize_linker_energy_mode",
    "normalize_move_weights",
    "check_move_weights",
    "normalize_pivot_rama_probability",
    "normalize_pivot_rama_schedule",
    "normalize_sidechain_move_mode",
    "resolve_k_bias",
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

#: The default native-contact definition, the same for every entry point
#: (YAML and JSON configs, the runners, ``FoldingRunner``, both replica
#: exchange drivers, ``NativeContactsCV`` and ``build_cv``): a pair of
#: residues at least ``DEFAULT_MIN_SEQ_SEP`` apart in sequence whose CA atoms
#: are closer than ``DEFAULT_CONTACT_CUTOFF`` Angstrom in the reference.
DEFAULT_CONTACT_CUTOFF: float = 6.0
DEFAULT_MIN_SEQ_SEP: int = 4
DEFAULT_CONTACT_ATOM_MODE: ContactAtomMode = "ca"

# Umbrella strength a replica exchange config gets when it sets targets but no
# k_bias. Without targets the default is 0: no umbrella.
_DEFAULT_K_BIAS_WITH_TARGETS = 1.0


def resolve_k_bias(k_bias: float | None, *, has_targets: bool) -> float:
    """The umbrella strength on N for a run that leaves ``k_bias`` unset.

    ``k_bias`` itself when given. Otherwise 1.0 when the run has umbrella
    targets, and 0.0, no umbrella, when it has none: without targets the one
    window sits at N = 0, so a default bias would pull every replica toward
    unfolded structures.
    """
    if k_bias is not None:
        return float(k_bias)
    return _DEFAULT_K_BIAS_WITH_TARGETS if has_targets else 0.0


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


#: Width (Gaussian std-dev, radians) of the KIC driver, the torsion that moves
#: one end of the three-residue window before the closure. ``step_size_rad``
#: sets the pivot only. 0.1 rad is the width the driver shared with the pivot
#: before it had its own; the engine's ``Integrator`` has the same default.
DEFAULT_KIC_STEP_SIZE_RAD = 0.1


def normalize_kic_step_size_rad(sigma: float | None) -> float:
    """Validate the KIC driver width; ``None`` means the default."""
    if sigma is None:
        return DEFAULT_KIC_STEP_SIZE_RAD
    if isinstance(sigma, bool):
        raise ValueError(f"kic_step_size_rad must be a positive number of radians, got {sigma!r}")
    value = float(sigma)
    if not (value > 0.0) or value == float("inf"):
        raise ValueError(f"kic_step_size_rad must be a positive number of radians, got {sigma!r}")
    return value


def normalize_pivot_rama_probability(p: float | None) -> float:
    # None means "not set": the engine's default, 0.0, because the rama pivot
    # is opt-in (see MCIntegrator::set_pivot_rama_probability).
    if p is None:
        return 0.0
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


def check_move_weights(
    forcefield: Any, move_weights: Sequence[float] | None, name: str = ""
) -> None:
    """Refuse sidechain moves on a chain that has no sidechain to move.

    The engine refuses them too, but only when it first steps -- after a
    replica-exchange run has opened its output files. A system without
    sidechain atoms (an all-glycine chain, say), or whose residues have no
    chi angle a sidechain move can change (glycine and alanine have none,
    and proline's are never moved), needs a sidechain weight of zero; it is
    not zeroed silently, because that would change the pivot/KIC mix the
    config asked for.
    """
    weights = normalize_move_weights(move_weights)
    if weights[2] <= 0.0:
        return
    label = repr(name) if name else type(forcefield).__name__
    fix = (
        "set move_weights (top-level in a flat config, or "
        "integrator.move_weights in a nested one) to [pivot, kic, 0.0], "
        "e.g. [0.5, 0.5, 0.0]"
    )
    if int(getattr(forcefield, "total_sc_atoms", 1)) == 0:
        raise ValueError(
            f"forcefield {label} has no sidechains, so sidechain moves are "
            f"impossible, but move_weights gives them {weights[2]:.3g} of the "
            f"moves; {fix}"
        )
    movable = getattr(forcefield, "sidechain_move_residues", None)
    if movable is not None and len(movable) == 0:
        raise ValueError(
            f"no residue of this chain has a chi angle a sidechain move can change "
            f"(glycine and alanine have none, and proline's are never moved), so "
            f"sidechain moves are impossible under forcefield {label}, but "
            f"move_weights gives them {weights[2]:.3g} of the moves; {fix}"
        )


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
    #: Width (radians) of the pivot move's torsion change.
    step_size_rad: float = 0.1
    #: Width (radians) of the KIC driver; see DEFAULT_KIC_STEP_SIZE_RAD.
    kic_step_size_rad: float = DEFAULT_KIC_STEP_SIZE_RAD
    sidechain_move_mode: SidechainMoveMode = "rotamer_library"
    pivot_rama_probability: float = 0.0
    pivot_rama_schedule: dict[str, float] | None = None
    #: (pivot, kic, sidechain) move-slot probabilities. Set the third to 0.0
    #: for a force field whose residues have no chi angles, or half the run is
    #: spent on sidechain proposals that cannot do anything.
    move_weights: tuple[float, float, float] = (0.25, 0.25, 0.50)
    #: Full energy recompute and check every this many MC steps
    #: (pymcpu.Simulation.full_energy_every_steps).
    full_energy_every_steps: int = 1_000_000

    def __post_init__(self) -> None:
        self.move_weights = normalize_move_weights(self.move_weights)
        self.kic_step_size_rad = normalize_kic_step_size_rad(self.kic_step_size_rad)
        steps = self.full_energy_every_steps
        if isinstance(steps, bool) or not isinstance(steps, (int, float)) or steps != int(steps) or steps < 1:
            raise ValueError(
                f"full_energy_every_steps must be a positive whole number of MC steps, got {steps!r}"
            )
        self.full_energy_every_steps = int(steps)


@dataclass
class OutputsConfig:
    output_dir: str = "./out"
    hdf5: str | None = None
    prefix: str = "rex"


@dataclass
class ReplicaExchangeConfig:
    temperatures: list[float] = field(default_factory=lambda: [0.5, 0.6])
    #: Umbrella centres in native contacts N. With neither these nor
    #: ``q_targets`` there is one window and, by default, no umbrella.
    native_contact_targets: list[float] | None = None
    q_targets: list[float] | None = None
    #: Umbrella strength on N. ``None`` (the default): 1.0 when targets are
    #: set, 0.0 (no umbrella) when they are not; see :func:`resolve_k_bias`.
    k_native_contacts: float | None = None
    k_bias: float | None = None  # alias for k_native_contacts
    cycles: int = 10
    steps_per_cycle: int = 100
    swap_interval: int | None = None
    backend: Literal["serial"] = "serial"
    log_interval: int = 100
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP
    contact_atom_mode: Literal["ca", "cb"] = DEFAULT_CONTACT_ATOM_MODE
    native_contact_pairs: list[list[int]] | None = None
    exchange_log: Literal["none", "all"] = "none"
    state_log_interval: int = 0
    log_walker_in_data_csv: bool = True

    def has_targets(self) -> bool:
        """True when the config sets umbrella targets, in N or in Q."""
        return self.native_contact_targets is not None or self.q_targets is not None

    def effective_k_bias(self) -> float:
        k = self.k_bias if self.k_bias is not None else self.k_native_contacts
        return resolve_k_bias(k, has_targets=self.has_targets())

    def effective_mc_steps(self) -> int:
        if self.swap_interval is not None:
            return int(self.swap_interval)
        return int(self.steps_per_cycle)


@dataclass
class FoldingConfig:
    """Settings only a folding run (mode ``"folding"``) reads.

    A folding run is divided into cycles of ``steps_per_cycle`` MC steps;
    ``checkpoint_interval`` counts them. ``None`` makes a cycle one
    ``integrator.report_interval``.

    ``q_threshold`` turns on an early stop: the run ends once Q, the fraction
    of native contacts formed, has stayed at or above it for
    ``convergence_window`` cycles in a row. ``None`` (the default) never stops
    early. Q counts the native contacts of the config's ``reference_pdb`` (by
    default its ``pdb``), defined by the last four fields, which have the
    same meaning and defaults as in :class:`ReplicaExchangeConfig`.
    """

    steps_per_cycle: int | None = None
    q_threshold: float | None = None
    convergence_window: int = 10
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF
    min_seq_sep: int = DEFAULT_MIN_SEQ_SEP
    contact_atom_mode: Literal["ca", "cb"] = DEFAULT_CONTACT_ATOM_MODE
    native_contact_pairs: list[list[int]] | None = None

    def __post_init__(self) -> None:
        if self.steps_per_cycle is not None:
            self.steps_per_cycle = _positive_int(self.steps_per_cycle, "steps_per_cycle")
        if self.q_threshold is not None:
            q = self.q_threshold
            if isinstance(q, bool) or not (0.0 < float(q) <= 1.0):
                raise ValueError(
                    f"q_threshold is a fraction of native contacts, in (0, 1], got "
                    f"{self.q_threshold!r}; leave it unset to run every step"
                )
            self.q_threshold = float(q)
        self.convergence_window = _positive_int(self.convergence_window, "convergence_window")
        self.contact_atom_mode = _normalize_contact_atom_mode_cfg(self.contact_atom_mode)


def _positive_int(value: Any, name: str) -> int:
    """``value`` as an int, refusing booleans, fractions and values below 1."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if ok:
        try:
            ok = value == int(value) and value >= 1
        except (OverflowError, ValueError):  # inf, nan
            ok = False
    if not ok:
        raise ValueError(f"{name} must be a positive whole number, got {value!r}")
    return int(value)


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


def normalize_move_settings(
    *,
    move_weights: Sequence[float] | None = None,
    sidechain_move_mode: str | None = "rotamer_library",
    pivot_rama_probability: float | None = 0.0,
    pivot_rama_schedule: Mapping[str, float] | None = None,
    kic_step_size_rad: float | None = DEFAULT_KIC_STEP_SIZE_RAD,
) -> dict[str, Any]:
    """Validate the move settings a sampler passes to :func:`configure_integrator`.

    Returns them as keyword arguments for that function. Every default is the
    engine's own, so an all-default call configures nothing new.
    """
    return {
        "move_weights": normalize_move_weights(move_weights),
        "sidechain_move_mode": normalize_sidechain_move_mode(sidechain_move_mode),
        "pivot_rama_probability": normalize_pivot_rama_probability(pivot_rama_probability),
        "pivot_rama_schedule": normalize_pivot_rama_schedule(pivot_rama_schedule),
        "kic_step_size_rad": normalize_kic_step_size_rad(kic_step_size_rad),
    }


def configure_integrator(
    integrator: Any,
    *,
    move_weights: Sequence[float],
    sidechain_move_mode: str,
    pivot_rama_probability: float,
    pivot_rama_schedule: Mapping[str, float] | None,
    kic_step_size_rad: float = DEFAULT_KIC_STEP_SIZE_RAD,
) -> None:
    """Apply already-validated move settings to a new Integrator.

    Takes the output of :func:`normalize_move_settings`, or the equivalent
    fields of a config that normalized them itself. A schedule, when given,
    overrides ``pivot_rama_probability``.
    """
    integrator.set_kic_step_size_rad(kic_step_size_rad)
    integrator.set_move_weights(*move_weights)
    integrator.set_sidechain_move_mode(sidechain_move_mode)
    if pivot_rama_schedule is not None:
        integrator.set_pivot_rama_schedule(**pivot_rama_schedule)
    else:
        integrator.set_pivot_rama_probability(pivot_rama_probability)


@dataclass
class SimulationConfig:
    mode: Mode
    pdb: str
    param_set: str = "mcpu08"
    param_dir: str | None = None
    #: Which force field to build: "mcpu08" (all-atom MCPU, the default; alias
    #: "mcpu"). See pymcpu.forcefields.available_forcefields().
    forcefield: str = "mcpu08"
    #: Constructor arguments for that force field, passed through rather than
    #: flattened into one schema shared by every force field. Unknown keys
    #: raise at build time.
    forcefield_options: dict[str, Any] = field(default_factory=dict)
    reference_pdb: str | None = None
    #: Accepted and not used: a run uses MPI when it is launched that way
    #: (``--mpi`` under ``mpirun``), whatever the config says.
    mpi: bool = False
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    replica_exchange: ReplicaExchangeConfig | None = None
    #: Read by mode "folding" only.
    folding: FoldingConfig = field(default_factory=FoldingConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        # A typo'd name is reported when the config loads, not after the
        # run has set up its output directory.
        from pymcpu.forcefields import get_forcefield
        get_forcefield(self.forcefield)
        self.forcefield_options = dict(self.forcefield_options or {})

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
    """

    pdb: str
    cv: tuple[dict, ...] = ()
    param_set: str = "mcpu08"
    param_dir: str | None = None
    #: Which force field to build: "mcpu08" (all-atom MCPU, the default; alias
    #: "mcpu"). See pymcpu.forcefields.available_forcefields().
    forcefield: str = "mcpu08"
    #: Constructor arguments for that force field, passed through rather than
    #: flattened into one schema shared by every force field. Unknown keys
    #: raise at build time.
    forcefield_options: dict[str, Any] = field(default_factory=dict)

    compute_dssp: bool = False
    dssp_coil_state: str = "C"
    temperature: float = 0.6
    step_size_rad: float = 0.1
    kic_step_size_rad: float = DEFAULT_KIC_STEP_SIZE_RAD
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
        self.kic_step_size_rad = normalize_kic_step_size_rad(self.kic_step_size_rad)
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
            kic_step_size_rad=cfg.integrator.kic_step_size_rad,
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
    checkpoint_data = data.get("checkpoint")
    if isinstance(checkpoint_data, Mapping):
        _check_removed_cloud_keys(checkpoint_data, "the checkpoint block")
        checkpoint_data = {
            k: v for k, v in checkpoint_data.items() if k not in _REMOVED_CLOUD_KEYS
        }
    checkpoint = _merge_dataclass(CheckpointConfig, checkpoint_data, label="checkpoint")

    rex: ReplicaExchangeConfig | None = None
    if mode == "replica_exchange_2d":
        if data.get("folding") is not None:
            raise ValueError(
                "The 'folding' block applies to mode 'folding' only; a "
                "replica_exchange_2d config takes its settings in 'replica_exchange'"
            )
        rex_raw = data.get("replica_exchange")
        if rex_raw is None:
            raise ValueError("replica_exchange_2d mode requires 'replica_exchange'")
        rex = _merge_dataclass(
            ReplicaExchangeConfig, rex_raw, label="replica_exchange"
        )
        if not rex.temperatures:
            raise ValueError("replica_exchange.temperatures must be non-empty")
        _check_targets(
            rex.native_contact_targets,
            rex.q_targets,
            n_name="replica_exchange.native_contact_targets",
            q_name="replica_exchange.q_targets",
        )
        if rex.backend != "serial":
            raise ValueError(
                f"Only backend='serial' is supported for now (got {rex.backend!r})"
            )
    elif data.get("replica_exchange") is not None:
        rex = _merge_dataclass(
            ReplicaExchangeConfig, data["replica_exchange"], label="replica_exchange"
        )
    folding = _merge_dataclass(FoldingConfig, data.get("folding"), label="folding")

    return SimulationConfig(
        mode=mode,
        pdb=str(data["pdb"]),
        param_set=str(data.get("param_set", "mcpu08")),
        param_dir=data.get("param_dir"),
        forcefield=str(data.get("forcefield", "mcpu08")),
        forcefield_options=dict(data.get("forcefield_options") or {}),
        reference_pdb=data.get("reference_pdb"),
        integrator=integrator,
        outputs=outputs,
        replica_exchange=rex,
        folding=folding,
        constraints=constraints,
        checkpoint=checkpoint,
    )


def _check_targets(
    n_targets: Sequence[float] | None,
    q_targets: Sequence[float] | None,
    *,
    n_name: str,
    q_name: str,
) -> None:
    """Refuse both kinds of umbrella target at once, and an empty list."""
    if n_targets is not None and q_targets is not None:
        raise ValueError(f"Provide either {n_name!r} or {q_name!r}, not both")
    for name, values in ((n_name, n_targets), (q_name, q_targets)):
        if values is not None and len(values) == 0:
            raise ValueError(
                f"{name!r} is empty; list at least one umbrella target, or leave "
                "it out for one window with no umbrella"
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
    if set, else 1.
    """
    temperatures = _expand_temperatures(data)
    q_targets = data.get("q_targets")
    n_targets = data.get("n_targets", data.get("native_contact_targets"))
    if q_targets is not None:
        n_windows = len(q_targets)
    elif n_targets is not None:
        n_windows = len(n_targets)
    else:
        n_windows = 1
    return len(temperatures) * n_windows, len(temperatures)


def _infer_output_dir_and_prefix(output_prefix: str) -> tuple[str, str]:
    """Split an output_prefix path into (directory, filename_prefix)."""
    p = Path(output_prefix)
    return str(p.parent), p.name


# Keys of a YAML config's ``checkpointing`` block. They are also accepted at
# the top level of the flat schema.
_YAML_CHECKPOINT_KEYS = frozenset({
    "checkpoint_dir", "checkpoint_interval", "keep_last_n", "resume", "enabled",
})

# Every top-level key of the flat YAML schema (see yaml_dict_to_config).
# ``mpi``, ``title`` and ``description`` are accepted and not used.
_YAML_KEYS = frozenset({
    # structure and force field
    "pdb", "reference_pdb", "param_set", "param_dir", "forcefield",
    "forcefield_options",
    # temperatures
    "temperatures", "temp_min", "temp_step", "n_temps",
    # run length and moves
    "seed", "mc_replica_steps", "steps", "num_cycles", "log_interval",
    "step_size_rad", "kic_step_size_rad", "sidechain_move_mode",
    "pivot_rama_probability", "pivot_rama_schedule", "move_weights",
    "full_energy_every_steps",
    # folding only: the early stop
    "q_threshold", "convergence_window",
    # replica exchange
    "q_targets", "n_targets", "native_contact_targets", "k_bias",
    "k_native_contacts", "contact_cutoff", "min_seq_sep", "contact_atom_mode",
    "native_contact_pairs", "exchange_log", "state_log_interval",
    "log_walker_in_data_csv",
    # constraints
    "fixed_residue_indices", "fixed_residues", "linker_residue_indices",
    "linker_residues", "linker_energy_mode",
    # outputs
    "output_prefix", "output_dir", "hdf5",
    # checkpointing
    "checkpointing", *_YAML_CHECKPOINT_KEYS,
    # accepted, not used
    "mpi", "title", "description",
})

# Keys that older configs carry and that no loader ever read. They load with a
# warning, so that existing configs keep working.
_RETIRED_YAML_KEYS = {
    "output_layout": "the output location comes from output_prefix and output_dir",
    "mode": (
        "a YAML config runs replica exchange when it lists more than one "
        "temperature or sets targets, and folding otherwise"
    ),
}

_JSON_BLOCK = "a block of the JSON schema; in YAML its settings are top-level keys"

# What to write instead of a key that a YAML config does not have but that a
# user may well try: names from the JSON schema, and keys of older configs.
# difflib's guess is wrong or missing for most of these.
_YAML_KEY_HINTS = {
    "temperature": "write 'temperatures: [T]', a list even for one temperature",
    "integrator": _JSON_BLOCK,
    "outputs": _JSON_BLOCK,
    "replica_exchange": _JSON_BLOCK,
    "constraints": _JSON_BLOCK,
    "checkpoint": "the YAML block is called 'checkpointing'",
    "report_interval": "the YAML name is 'log_interval'",
    "steps_per_cycle": "the YAML name is 'mc_replica_steps'",
    "swap_interval": "the YAML name is 'mc_replica_steps'",
}

# Settings of the checkpoint upload, which was removed. Copies of the old
# template still carry them at their defaults, so they load with a warning;
# `cloud_sync: true` is an error, because nothing would be uploaded.
_REMOVED_CLOUD_KEYS = frozenset({"cloud_sync", "cloud_bucket", "cloud_sync_cmd"})


def _check_removed_cloud_keys(block: Mapping[str, Any], where: str) -> None:
    if block.get("cloud_sync"):
        raise ValueError(
            f"'cloud_sync' is on in {where}, but pyMCPU no longer uploads "
            "checkpoints; it writes them only to checkpoint_dir. Delete the "
            "cloud keys."
        )
    for key in sorted(_REMOVED_CLOUD_KEYS.intersection(block)):
        warnings.warn(
            f"{key!r} in {where} has no effect (checkpoint upload was removed); "
            "you can delete it",
            UserWarning,
            stacklevel=1,
        )


def _unknown_keys(data: Mapping[str, Any], known: frozenset[str], prefix: str = "") -> list[str]:
    found = []
    for key in sorted((k for k in data if k not in known), key=str):
        hint = None if prefix else _YAML_KEY_HINTS.get(key)
        if hint is None:
            close = difflib.get_close_matches(str(key), sorted(known), n=1)
            hint = f"did you mean {close[0]!r}?" if close else None
        name = f"{prefix}{key}"
        found.append(f"{name!r} ({hint})" if hint else repr(name))
    return found


def check_yaml_keys(data: Mapping[str, Any], source: str = "the YAML config") -> None:
    """Raise ValueError if a flat YAML config has keys the schema does not.

    Checks the top level and the ``checkpointing`` block and names every
    unknown key at once, with the closest known key or what to write instead,
    so that a misspelled key is reported instead of silently ignored. Keys
    that older configs carry and that do nothing (``output_layout``,
    ``mode``, and the removed ``cloud_*`` settings) only warn, except
    ``cloud_sync: true``. ``source`` names the config in the messages.
    """
    for key, reason in _RETIRED_YAML_KEYS.items():
        if key in data:
            warnings.warn(
                f"{key!r} in {source} has no effect ({reason}); you can delete it",
                UserWarning,
                stacklevel=1,
            )
    known = _YAML_KEYS | frozenset(_RETIRED_YAML_KEYS) | _REMOVED_CLOUD_KEYS
    unknown = _unknown_keys(data, known)
    block = data.get("checkpointing")
    if isinstance(block, Mapping):
        unknown += _unknown_keys(
            block, _YAML_CHECKPOINT_KEYS | _REMOVED_CLOUD_KEYS, prefix="checkpointing."
        )
    if unknown:
        noun = "key" if len(unknown) == 1 else "keys"
        raise ValueError(f"Unknown {noun} in {source}: {', '.join(unknown)}")
    _check_removed_cloud_keys(data, source)
    if isinstance(block, Mapping):
        _check_removed_cloud_keys(block, f"the checkpointing block of {source}")
    if block is not None and not isinstance(block, Mapping):
        raise ValueError(
            f"'checkpointing' in {source} must be a mapping of checkpoint "
            f"settings, got {block!r}"
        )


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
        enabled=bool(src.get("enabled", flat_fallback.get("enabled", True))),
    )


def load_yaml_config(path: str | Path) -> SimulationConfig:
    """Load a flat YAML config and map it to SimulationConfig.

    This accepts the flat schema used by pyMCPU production runs:
        pdb, output_prefix, num_cycles, mc_replica_steps, steps, seed,
        fixed_residue_indices, temp_min/temp_step/n_temps (or temperatures),
        n_targets/native_contact_targets or q_targets, k_bias, contact_cutoff,
        min_seq_sep, checkpointing / checkpoint_dir, log_interval, ...

    A config runs replica exchange when it lists more than one temperature or
    sets targets, and folding otherwise; see :func:`yaml_dict_to_config`. An
    unknown key raises ValueError (see :func:`check_yaml_keys`).
    """
    import yaml

    p = Path(path)
    with p.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")

    return yaml_dict_to_config(data, source=str(p))


# Cycle length of a run that does not set one, in MC steps.
_DEFAULT_CYCLE_STEPS = 1000
_DEFAULT_NUM_CYCLES = 10

# Keys only a folding run reads.
_YAML_FOLDING_ONLY_KEYS = ("q_threshold", "convergence_window")


def _yaml_count(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    return None if value is None else _positive_int(value, key)


def _folding_run_length(data: Mapping[str, Any]) -> tuple[int, int]:
    """Total MC steps and cycle length of a folding YAML config.

    ``steps`` is the total. The cycle length is ``mc_replica_steps`` when set,
    else ``steps / num_cycles`` when both are set, else 1000 steps (or all of
    ``steps``, if fewer). Without ``steps`` the total is
    ``num_cycles x mc_replica_steps``, as in replica exchange.
    """
    steps = _yaml_count(data, "steps")
    num_cycles = _yaml_count(data, "num_cycles")
    per_cycle = _yaml_count(data, "mc_replica_steps")
    if steps is None:
        per_cycle = per_cycle or _DEFAULT_CYCLE_STEPS
        return (num_cycles or _DEFAULT_NUM_CYCLES) * per_cycle, per_cycle
    if per_cycle is None:
        if num_cycles is None:
            return steps, min(steps, _DEFAULT_CYCLE_STEPS)
        if steps % num_cycles:
            raise ValueError(
                f"In a folding config (one temperature, no targets) steps is the "
                f"total number of MC steps, split into num_cycles cycles, so it "
                f"must be a multiple of num_cycles; got steps={steps}, "
                f"num_cycles={num_cycles}"
            )
        return steps, steps // num_cycles
    if num_cycles is not None and num_cycles * per_cycle != steps:
        raise ValueError(
            f"In a folding config (one temperature, no targets) steps is the "
            f"total number of MC steps, so it must equal num_cycles x "
            f"mc_replica_steps when all three are set; got steps={steps}, "
            f"num_cycles={num_cycles}, mc_replica_steps={per_cycle} "
            f"(= {num_cycles * per_cycle}). Set two of them."
        )
    return steps, min(per_cycle, steps)


def _native_contacts_from_yaml(data: Mapping[str, Any]) -> dict[str, Any]:
    """The native-contact definition of a YAML config, either mode."""
    pairs_raw = data.get("native_contact_pairs")
    pairs: list[list[int]] | None = None
    if pairs_raw is not None:
        pairs = []
        for pair in pairs_raw:
            pair_list = list(pair)
            if len(pair_list) != 2:
                raise ValueError(
                    "native_contact_pairs entries must have exactly 2 "
                    f"elements, got {pair!r}"
                )
            pairs.append([int(pair_list[0]), int(pair_list[1])])
    return {
        "contact_cutoff": float(data.get("contact_cutoff", DEFAULT_CONTACT_CUTOFF)),
        "min_seq_sep": int(data.get("min_seq_sep", DEFAULT_MIN_SEQ_SEP)),
        "contact_atom_mode": _normalize_contact_atom_mode_cfg(
            data.get("contact_atom_mode", DEFAULT_CONTACT_ATOM_MODE)
        ),
        "native_contact_pairs": pairs,
    }


def yaml_dict_to_config(
    data: Mapping[str, Any], *, source: str = "the YAML config"
) -> SimulationConfig:
    """Convert a flat YAML dict into a SimulationConfig.

    The config runs replica exchange (``replica_exchange_2d``) when it lists
    more than one temperature or sets targets (``n_targets``,
    ``native_contact_targets`` or ``q_targets``); one temperature with targets
    is umbrella sampling at that temperature. Otherwise it runs folding.

    Replica exchange runs ``num_cycles`` cycles of ``mc_replica_steps`` MC
    steps (``steps`` is another name for ``mc_replica_steps`` there). In a
    folding config ``steps`` is the total number of MC steps, and a cycle is
    ``mc_replica_steps`` steps when that is set, ``steps / num_cycles`` when
    those two are set, and otherwise 1000 steps (or all of ``steps``, if
    fewer); a config that sets all three must have ``steps == num_cycles *
    mc_replica_steps``. In both modes ``log_interval`` defaults to the cycle
    length and ``checkpoint_interval`` counts cycles.

    ``source`` names the config in error messages, for example its path.
    """
    data = dict(data)
    check_yaml_keys(data, source)
    if "pdb" not in data:
        raise ValueError("YAML config requires 'pdb'")

    temperatures = _expand_temperatures(data)
    q_targets = data.get("q_targets")
    n_key = "n_targets" if data.get("n_targets") is not None else "native_contact_targets"
    n_targets = data.get(n_key)
    _check_targets(n_targets, q_targets, n_name=n_key, q_name="q_targets")
    has_targets = q_targets is not None or n_targets is not None
    mode: Mode = (
        "replica_exchange_2d" if len(temperatures) > 1 or has_targets else "folding"
    )

    seed = int(data.get("seed", 42))
    if mode == "folding":
        total_steps, cycle_steps = _folding_run_length(data)
    else:
        folding_keys = [k for k in _YAML_FOLDING_ONLY_KEYS if k in data]
        if folding_keys:
            raise ValueError(
                f"{', '.join(map(repr, folding_keys))} in {source} only applies to a "
                "folding run (one temperature and no targets); replica exchange "
                "always runs num_cycles cycles"
            )
        cycle_steps = int(data.get("mc_replica_steps", data.get("steps", _DEFAULT_CYCLE_STEPS)))
        total_steps = cycle_steps
        num_cycles = int(data.get("num_cycles", _DEFAULT_NUM_CYCLES))
    log_interval = int(data.get("log_interval", cycle_steps))
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
        steps=total_steps,
        seed=seed,
        report_interval=log_interval,
        step_size_rad=step_size_rad,
        kic_step_size_rad=normalize_kic_step_size_rad(data.get("kic_step_size_rad")),
        sidechain_move_mode=normalize_sidechain_move_mode(
            data.get("sidechain_move_mode", "rotamer_library")
        ),
        pivot_rama_probability=normalize_pivot_rama_probability(
            data.get("pivot_rama_probability")
        ),
        pivot_rama_schedule=normalize_pivot_rama_schedule(data.get("pivot_rama_schedule")),
        move_weights=normalize_move_weights(data.get("move_weights")),
        full_energy_every_steps=data.get("full_energy_every_steps", 1_000_000),
    )

    contacts = _native_contacts_from_yaml(data)
    rex: ReplicaExchangeConfig | None = None
    folding = FoldingConfig()
    if mode == "replica_exchange_2d":
        k_bias = None
        if "k_bias" in data:
            k_bias = float(data["k_bias"])
        elif "k_native_contacts" in data:
            k_bias = float(data["k_native_contacts"])
        ex_log = str(data.get("exchange_log", "none")).strip().lower()
        if ex_log not in ("none", "all"):
            raise ValueError("exchange_log must be 'none' or 'all'")
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
            steps_per_cycle=cycle_steps,
            log_interval=log_interval,
            exchange_log=ex_log,  # type: ignore[arg-type]
            state_log_interval=int(data.get("state_log_interval", 0)),
            log_walker_in_data_csv=bool(data.get("log_walker_in_data_csv", True)),
            **contacts,
        )
    else:
        folding = FoldingConfig(
            steps_per_cycle=cycle_steps,
            q_threshold=data.get("q_threshold"),
            convergence_window=data.get("convergence_window", 10),
            **contacts,
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
        param_set=str(data.get("param_set", "mcpu08")),
        param_dir=data.get("param_dir"),
        forcefield=str(data.get("forcefield", "mcpu08")),
        forcefield_options=dict(data.get("forcefield_options") or {}),
        reference_pdb=data.get("reference_pdb"),
        integrator=integrator,
        outputs=outputs,
        replica_exchange=rex,
        folding=folding,
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
