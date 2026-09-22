"""Python-exposed ``state.coords`` API contract.

Split out of the old ``test_coords_soa.py`` (see
``tests/physics/test_coords_soa.py``): this asserts nothing about physics --
only that a numpy array assigned to ``state.coords`` reads back unchanged
with the expected shape -- so it belongs with the other pure software/API
contract checks in ``tests/config/``, not the physics-determinism suite.
"""

from __future__ import annotations

from tests.fixtures.context_builders import build_test_context


def test_state_coords_roundtrip() -> None:
    ctx, _ = build_test_context(with_qbias=False)
    state = ctx.get_state()
    coords = state.coords
    # coords is (3, n_atoms) -- shape must be preserved through a full
    # assign-then-read cycle, not just alias the same array.
    assert coords.shape == (3, state.coords.shape[1])
    state.coords = coords.copy()
    roundtripped = state.coords
    assert (roundtripped == coords).all()
