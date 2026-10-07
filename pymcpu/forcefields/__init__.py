"""Force-field registry.

pyMCPU has two force fields and they are alternatives, not layers: an
all-atom MCPU system and a backbone-only KORP system are built differently,
score differently, and never appear together. This module is what lets a
config file name one instead of every call site hard-coding
:class:`~pymcpu.forcefields.mcpu.MCPUForceField`.

Deliberately a plain dict rather than entry points or plugin discovery. There
are two implementations and a handful of call sites; ``include/pymcpu/forces/
README.md``'s rule against adding a tier speculatively applies to the Python
side too.

The two constructors do not take the same arguments -- MCPU wants a parameter
set, KORP wants an energy map -- so options are passed through as a mapping
rather than flattened into one config schema that is half-irrelevant whichever
force field you pick. Unknown options raise, so a typo in a YAML file is not
silently ignored.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pymcpu.forcefields.base import BaseForceField

if TYPE_CHECKING:  # avoid importing mdtraj / the extension just to use names
    import mdtraj as md

__all__ = [
    "BaseForceField",
    "available_forcefields",
    "build_forcefield",
    "get_forcefield",
    "load_forcefield",
    "register_forcefield",
]

#: Tag -> class. The tags match the fit-directory names under
#: ``include/pymcpu/forces/`` and the ``"forcefield"`` value a parameter set
#: declares, so one name identifies the same thing everywhere.
_REGISTRY: dict[str, type] = {}


def register_forcefield(name: str, cls: type) -> None:
    """Register a force-field class under ``name``."""
    if not issubclass(cls, BaseForceField):
        raise TypeError(f"{cls.__name__} does not implement BaseForceField")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"force field {name!r} is already registered to "
            f"{existing.__name__}; pick a different tag"
        )
    _REGISTRY[name] = cls


def available_forcefields() -> list[str]:
    return sorted(_REGISTRY)


def get_forcefield(name: str) -> type:
    """Look up a force-field class, listing the known ones if it is missing."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown force field {name!r}; known: {available_forcefields()}"
        ) from None


def build_forcefield(
    name: str,
    trajectory: "md.Trajectory",
    options: Mapping[str, Any] | None = None,
) -> BaseForceField:
    """Construct the named force field from ``trajectory`` and ``options``.

    ``options`` is checked against the constructor's signature rather than
    being forwarded blindly, so a misspelled key in a config file is reported
    against the force field that would have received it.
    """
    cls = get_forcefield(name)
    options = dict(options or {})

    # Checked before any work on the trajectory: validating a mapping is free,
    # and slicing a structure only to reject a misspelled key afterwards wastes
    # the expensive step and reports the error late.
    if options:
        parameters = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if not accepts_kwargs:
            unknown = sorted(set(options) - set(parameters))
            if unknown:
                known = sorted(
                    p for p in parameters if p not in ("self", "trajectory"))
                raise ValueError(
                    f"force field {name!r} does not accept {unknown}; "
                    f"it takes {known}"
                )

    return cls(cls.prepare_trajectory(trajectory), **options)


def load_forcefield(
    structure: "str | Path | md.Trajectory",
    name: str = "mcpu08",
    options: Mapping[str, Any] | None = None,
    *,
    param_set: str = "mcpu08",
    param_dir: str | Path | None = None,
    compute_dssp: bool = False,
    dssp_coil_state: str = "C",
) -> BaseForceField:
    """Build the force field a run configuration names, from a PDB path or a
    loaded trajectory.

    The one place a config becomes a force field: the folding runner, both
    replica-exchange engines and :class:`~pymcpu.sampling.EngineSession` all
    call it, so they cannot disagree about which force field a config means.

    ``param_set``, ``param_dir`` and the DSSP settings are top-level config
    fields because they predate ``forcefield_options``, but they belong to
    MCPU. They are forwarded to an MCPU force field (an explicit entry in
    ``options`` wins) and refused for any other: KORP has no parameter set,
    and silently dropping a ``param_dir`` someone set would hide a mistake.
    """
    import mdtraj as md
    from pymcpu.forcefields.mcpu import MCPUForceField

    cls = get_forcefield(name)
    merged: dict[str, Any] = dict(options or {})
    if issubclass(cls, MCPUForceField):
        merged.setdefault("param_set", param_set)
        merged.setdefault("compute_dssp", compute_dssp)
        merged.setdefault("dssp_coil_state", dssp_coil_state)
        if param_dir is not None:
            merged.setdefault("param_dir", str(param_dir))
    else:
        if param_dir is not None or param_set != "mcpu08":
            raise ValueError(
                f"param_set/param_dir configure the MCPU force field and do "
                f"not apply to forcefield {name!r}; remove them (its own "
                f"options go in forcefield_options)"
            )
        if compute_dssp:
            raise ValueError(
                f"compute_dssp applies to the MCPU force field, not {name!r}")
    trajectory = structure if isinstance(structure, md.Trajectory) else md.load(str(structure))
    return build_forcefield(name, trajectory, merged)


def _register_builtins() -> None:
    # Imported here rather than at module scope: both pull in mdtraj and the
    # compiled extension, and `pymcpu.forcefields` is imported by
    # `pymcpu/__init__.py` before those are necessarily ready.
    from pymcpu.forcefields.korp import KORPForceField
    from pymcpu.forcefields.mcpu import MCPUForceField

    # "mcpu08" is the fit tag: it matches forces/mcpu/mcpu08/ and the tag a
    # parameter set declares. "mcpu" is kept as an alias because it is the
    # obvious thing to type and the lineage currently has one fit.
    register_forcefield("mcpu08", MCPUForceField)
    register_forcefield("mcpu", MCPUForceField)
    register_forcefield("korp", KORPForceField)


_register_builtins()
