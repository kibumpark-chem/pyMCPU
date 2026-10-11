"""A minimal, valid :class:`~pymcpu.config.EngineSpec` for core tests.

Several suites need an engine built from a real small protein and differ
only in a field or two. Factoring construction here means a future required
``EngineSpec`` field is added in one place rather than in six.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from pymcpu.config import EngineSpec


@pytest.fixture
def engine_spec_factory(chignolin_pdb_path: str) -> Callable[..., EngineSpec]:
    """Builder for an ``EngineSpec`` on chignolin, using the same structure
    as both the system and its own native-contacts-Q reference. Callers pass
    only the overrides their test cares about."""

    def _build(**overrides: Any) -> EngineSpec:
        pdb = str(chignolin_pdb_path)
        data: dict[str, Any] = {
            "pdb": pdb,
            "cv": ({"type": "native_contacts_q", "reference_pdb": pdb},),
        }
        data.update(overrides)
        return EngineSpec(**data)

    return _build
