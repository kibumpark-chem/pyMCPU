"""Clones of one state must sample independently.

Any framework that splits a walker -- weighted ensemble, forward flux,
adaptive seeding -- hands every child the *same* parent state. If a child's
Monte Carlo stream comes from restoring the parent's saved RNG state rather
than from :func:`pymcpu.sampling.derive_seed`, every sibling executes a
**bitwise identical** trajectory. The split still "succeeds": the children
are distinct objects, the weights divide correctly, every log line looks
healthy. The ensemble is simply one walker counted N times, and only the
statistics show it.

So this lives in ``tests/physics/`` and not with any one framework's tests:
it guards a core promise that core's public API now makes. The last test
here deliberately demonstrates the wrong design producing identical
children, which is the clearest evidence in the repo for why the seeding
scheme exists.

See ``tests/physics/test_engine_determinism.py`` for the bare-engine
determinism guarantees this relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.config import EngineSpec
from pymcpu.sampling import EngineSession, derive_seed

# An arbitrary run-level seed. In WESTPA terms this is `base_seed`; the core
# API calls the three arguments (base_seed, round_index, stream_index).
_BASE_SEED = 99


@pytest.fixture
def spec(engine_spec_factory) -> EngineSpec:
    return engine_spec_factory()


def test_siblings_with_same_parent_diverge(spec: EngineSpec) -> None:
    parent_coords = EngineSession(spec).coords_from_auxref(spec.pdb)

    seed_a = derive_seed(_BASE_SEED, 5, 10)
    seed_b = derive_seed(_BASE_SEED, 5, 11)
    assert seed_a != seed_b  # distinct stream index must derive distinct seeds

    eng_a = EngineSession(spec)
    eng_a.set_coords(parent_coords)
    eng_a.set_seed(seed_a)
    eng_a.step(300)

    eng_b = EngineSession(spec)
    eng_b.set_coords(parent_coords)
    eng_b.set_seed(seed_b)
    eng_b.step(300)

    assert not np.array_equal(eng_a.coords(), eng_b.coords())


def test_many_siblings_are_pairwise_distinct(spec: EngineSpec) -> None:
    """A split into N > 2 children must not just avoid *one* collision --
    check several siblings pairwise."""
    parent_coords = EngineSession(spec).coords_from_auxref(spec.pdb)
    round_index = 7
    finals = []
    for stream_index in range(6):
        eng = EngineSession(spec)
        eng.set_coords(parent_coords)
        eng.set_seed(derive_seed(_BASE_SEED, round_index, stream_index))
        eng.step(200)
        finals.append(eng.coords())

    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            assert not np.array_equal(finals[i], finals[j]), (
                f"siblings {i} and {j} produced identical trajectories "
                "-- this is exactly the silent-clone failure mode this test guards against"
            )


def test_inheriting_parent_rng_state_would_have_cloned(spec: EngineSpec) -> None:
    """Demonstrates *why* the design derives a fresh seed instead of
    restoring the parent's saved RNG stream: doing the latter (the naive,
    wrong design) does produce bitwise-identical children, which is exactly
    the bug this project's seeding design avoids in production."""
    eng_parent = EngineSession(spec)
    eng_parent.set_seed(1)
    eng_parent.step(300)
    parent_coords = eng_parent.coords()
    parent_rng = eng_parent.get_rng_state()

    eng_child_a = EngineSession(spec)
    eng_child_a.set_coords(parent_coords)
    eng_child_a.restore_rng_state(parent_rng)
    eng_child_a.step(200)

    eng_child_b = EngineSession(spec)
    eng_child_b.set_coords(parent_coords)
    eng_child_b.restore_rng_state(parent_rng)  # naive: same RNG state as sibling A
    eng_child_b.step(200)

    assert np.array_equal(eng_child_a.coords(), eng_child_b.coords())
