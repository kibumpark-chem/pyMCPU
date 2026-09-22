"""Determinism guarantees of :class:`pymcpu.sampling.EngineSession`.

These justify every restart design built on top of pyMCPU: a trajectory
segment must be resumable from nothing but ``(coordinates, rng_state)``,
and re-seeding that state on a *freshly constructed* engine must reproduce
the original trajectory bitwise.

This is the only place in the repo that asserts that property. The
``test_rng_state*`` modules reach their checkpoint by replaying the warmup
rather than via ``set_positions``, so they do not cover reconstitution --
which is exactly what ``pymcpu/checkpointing.py`` and
``FoldingRunner``/``ReplicaExchange`` resume depend on, and what an
external sampler relies on for every segment.

Nothing here compares against legacy MCPU; it is pyMCPU's own engine being
self-consistent under teardown and rebuild. Needs the C++ engine, no
external framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pymcpu.config import EngineSpec
from pymcpu.sampling import EngineSession, derive_seed

_BASE_SEED = 42


def _build_spec(pdb_path: str, **overrides: Any) -> EngineSpec:
    """Minimal spec for these determinism checks. Kept local rather than
    using ``engine_spec_factory`` because these tests pin the CV arguments
    (``contact_cutoff``, ``min_seq_sep``) explicitly -- a change to the
    shared factory's defaults must not quietly change what is asserted
    here."""
    pdb = str(pdb_path)
    data: dict[str, Any] = {
        "pdb": pdb,
        "cv": (
            {
                "type": "native_contacts_q",
                "reference_pdb": pdb,
                "contact_cutoff": 10.0,
                "min_seq_sep": 3,
            },
        ),
    }
    data.update(overrides)
    return EngineSpec(**data)


@pytest.fixture
def spec(chignolin_pdb_path: str) -> EngineSpec:
    return _build_spec(chignolin_pdb_path)


def test_engine_state_completeness_round_trip(spec: EngineSpec) -> None:
    """(coords, rng_state) fully capture engine state -- a full teardown and
    rebuild, restoring only those two things, continues bitwise identically."""
    eng_a = EngineSession(spec)
    eng_a.set_seed(123)
    eng_a.step(500)
    snap_coords = eng_a.coords()
    snap_rng = eng_a.get_rng_state()
    snap_step = eng_a.current_step
    eng_a.step(500)
    final_a = eng_a.coords()

    eng_b = EngineSession(spec)  # fully independent engine
    eng_b.set_coords(snap_coords)
    eng_b.restore_rng_state(snap_rng)
    eng_b.current_step = snap_step
    eng_b.step(500)
    final_b = eng_b.coords()

    assert np.array_equal(final_a, final_b)


def test_engine_pcoord_matches_native(spec: EngineSpec) -> None:
    eng = EngineSession(spec)
    q = eng.compute_cv(eng.coords_from_auxref(spec.pdb))
    # Definitional, not empirical: Q of the reference structure against
    # itself is 1.0 by construction of the native-contacts fraction.
    np.testing.assert_allclose(q, [1.0])


def test_same_parent_and_seed_gives_identical_result(spec: EngineSpec) -> None:
    """The production determinism guarantee: given the same parent
    coordinates and the same derived seed, two independent engines produce
    bitwise-identical results -- required for a segment to be exactly
    reproducible from (n_iter, seg_id, base_seed, parent coords) alone."""
    parent_coords = EngineSession(spec).coords_from_auxref(spec.pdb)
    # positional: derive_seed(base_seed, round_index, stream_index) --
    # here the WE iteration and segment id.
    seed = derive_seed(_BASE_SEED, 3, 5)

    eng1 = EngineSession(spec)
    eng1.set_coords(parent_coords)
    eng1.set_seed(seed)
    eng1.step(300)

    eng2 = EngineSession(spec)
    eng2.set_coords(parent_coords)
    eng2.set_seed(seed)
    eng2.step(300)

    assert np.array_equal(eng1.coords(), eng2.coords())
