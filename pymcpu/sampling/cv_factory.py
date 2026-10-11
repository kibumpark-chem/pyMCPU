"""Build a collective variable from a declarative spec.

``pymcpu.sampling.collective_variables`` holds the numerics; this module is
the thin layer that turns a list of plain dicts into a CV object. That
matters for anything driving pyMCPU from its own configuration format --
a YAML/JSON block, a ``west.cfg`` section, a CLI flag -- because it means
a CV can be *declared* rather than constructed in Python.

Every CV here computes directly from live engine coordinates
(``coords_3xn``, shape ``(3, n_atoms)``, Angstrom, engine atom order) with
**no dynamics run** -- cheap enough to call on every sub-step of a
trajectory, and before any Monte Carlo has happened at all.

``type: custom`` takes a dotted import path to your own factory, so a
framework-specific CV needs no change here and no registry: see
:func:`custom_cv_from_spec`.

Failure mode note: these raise rather than returning a sentinel. An RMSD or
a Q of exactly 0.0 has a specific physical meaning -- "perfectly aligned to
the reference", "no native contacts" -- so substituting it for a
computation failure would corrupt a free-energy or flux estimate with no
visible symptom.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from pymcpu.config import DEFAULT_CONTACT_ATOM_MODE, DEFAULT_CONTACT_CUTOFF, DEFAULT_MIN_SEQ_SEP
from pymcpu.sampling.collective_variables import (
    CARMSDCV,
    NativeContactsCV,
    build_contact_atom_index,
    reference_contact_from_pdb,
)


__all__ = [
    "CaRmsd",
    "CollectiveVariable",
    "CompositeCV",
    "NativeContactsN",
    "NativeContactsQ",
    "TwoStateDelta",
    "TwoStateRmsd",
    "build_cv",
    "custom_cv_from_spec",
]


@runtime_checkable
class CollectiveVariable(Protocol):
    """The structural contract a CV satisfies.

    ``@runtime_checkable``, so a consumer can accept any object with these
    three members -- including one built by a ``type: custom`` factory it
    has never heard of.
    """

    ndim: int
    labels: tuple[str, ...]

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray: ...


@dataclass
class NativeContactsQ:
    """Native-contact fraction Q -- the usual folding coordinate."""

    cv: NativeContactsCV
    ndim: int = 1
    labels: tuple[str, ...] = ("Q",)

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.array([self.cv.compute_Q(coords_3xn)], dtype=np.float64)


@dataclass
class NativeContactsN:
    """Native-contact count N (matches the C++ ``NativeContactsBiasPotential``
    convention exactly, useful if the same bias is ever enabled)."""

    cv: NativeContactsCV
    ndim: int = 1
    labels: tuple[str, ...] = ("N",)

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.array([self.cv.compute_N(coords_3xn)], dtype=np.float64)


@dataclass
class CaRmsd:
    """CA-RMSD (Angstrom) to a single reference structure, after optimal
    (Kabsch) superposition."""

    cv: CARMSDCV
    ndim: int = 1
    labels: tuple[str, ...] = ("RMSD",)

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.array([self.cv.compute(coords_3xn)], dtype=np.float64)


@dataclass
class TwoStateRmsd:
    """``[rmsd_to_a, rmsd_to_b]`` — the general two-state transition-path
    coordinate: no reference is privileged as "native"."""

    cv_a: CARMSDCV
    cv_b: CARMSDCV
    ndim: int = 2
    labels: tuple[str, ...] = ("RMSD_A", "RMSD_B")

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.array(
            [self.cv_a.compute(coords_3xn), self.cv_b.compute(coords_3xn)], dtype=np.float64
        )


@dataclass
class TwoStateDelta:
    """``rmsd_to_a - rmsd_to_b`` — a 1-D projection of :class:`TwoStateRmsd`,
    useful when a single scalar transition coordinate is preferred (e.g. for
    a 1-D bin mapper)."""

    cv_a: CARMSDCV
    cv_b: CARMSDCV
    ndim: int = 1
    labels: tuple[str, ...] = ("RMSD_A_minus_B",)

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.array(
            [self.cv_a.compute(coords_3xn) - self.cv_b.compute(coords_3xn)], dtype=np.float64
        )


@dataclass
class CompositeCV:
    """Concatenation of several CVs, in order."""

    components: tuple[Any, ...]

    def __post_init__(self) -> None:
        self.ndim = sum(int(c.ndim) for c in self.components)
        labels: list[str] = []
        for c in self.components:
            labels.extend(c.labels)
        self.labels = tuple(labels)

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.concatenate([np.atleast_1d(c(coords_3xn)) for c in self.components])


def _load_dotted(path: str) -> Any:
    """Import ``module.submodule.factory_name`` and return the attribute."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"custom CV factory path must be 'module.attr', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def custom_cv_from_spec(spec: dict) -> Any:
    """Build a user-supplied CV: ``{"type": "custom", "factory": "pkg.mod.make_cv", "kwargs": {...}}``.
    The factory must return an object exposing ``ndim``, ``labels``, and
    ``__call__(coords_3xn) -> np.ndarray``."""
    factory = _load_dotted(spec["factory"])
    return factory(**spec.get("kwargs", {}))


def build_cv(spec: Sequence[dict], forcefield: Any) -> Any:
    """Build a CV -- possibly composite -- from a list of spec dicts.

    Each dict carries a ``type`` key selecting a built-in CV plus that CV's
    kwargs; ``type: custom`` dispatches to :func:`custom_cv_from_spec`.

    A single-element spec returns that CV **directly**, not wrapped in a
    one-item :class:`CompositeCV`, so ``.ndim`` and ``.labels`` are exactly
    what the CV itself reports.
    """
    components = [_build_one(s, forcefield) for s in spec]
    if len(components) == 1:
        return components[0]
    return CompositeCV(tuple(components))


def _ca_index_and_ref(forcefield: Any, ref_path: str, mode: str) -> tuple[Any, Any]:
    """``(build_contact_atom_index, reference_contact_from_pdb)`` pair,
    shared by every CV family below that needs a contact-atom index plus a
    single reference structure."""
    ca_idx = build_contact_atom_index(forcefield, mode=mode)
    ref_ca = reference_contact_from_pdb(ref_path, mode=mode)
    return ca_idx, ref_ca


def _build_one(spec: dict, forcefield: Any) -> Any:
    kind = spec["type"]
    if kind == "custom":
        return custom_cv_from_spec(spec)

    contact_atom_mode = spec.get("contact_atom_mode", DEFAULT_CONTACT_ATOM_MODE)

    if kind in ("native_contacts_q", "native_contacts_n"):
        ca_idx, ref_ca = _ca_index_and_ref(forcefield, spec["reference_pdb"], contact_atom_mode)
        cv = NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref_ca,
            contact_cutoff=float(spec.get("contact_cutoff", DEFAULT_CONTACT_CUTOFF)),
            min_seq_sep=int(spec.get("min_seq_sep", DEFAULT_MIN_SEQ_SEP)),
            mode="hard",
            contact_atom_mode=contact_atom_mode,
            native_contact_pairs=spec.get("native_contact_pairs"),
        )
        return NativeContactsQ(cv=cv) if kind == "native_contacts_q" else NativeContactsN(cv=cv)

    if kind == "ca_rmsd":
        ca_idx, ref_ca = _ca_index_and_ref(forcefield, spec["reference_pdb"], "ca")
        return CaRmsd(cv=CARMSDCV(ca_internal_idx=ca_idx, ref_ca_xyz=ref_ca))

    if kind in ("two_state_rmsd", "two_state_delta"):
        ca_idx, ref_a = _ca_index_and_ref(forcefield, spec["reference_a"], "ca")
        _, ref_b = _ca_index_and_ref(forcefield, spec["reference_b"], "ca")
        cv_a = CARMSDCV(ca_internal_idx=ca_idx, ref_ca_xyz=ref_a)
        cv_b = CARMSDCV(ca_internal_idx=ca_idx, ref_ca_xyz=ref_b)
        return TwoStateRmsd(cv_a, cv_b) if kind == "two_state_rmsd" else TwoStateDelta(cv_a, cv_b)

    raise ValueError(
        f"unknown CV type {kind!r}; expected one of "
        "native_contacts_q, native_contacts_n, ca_rmsd, two_state_rmsd, two_state_delta, custom"
    )
