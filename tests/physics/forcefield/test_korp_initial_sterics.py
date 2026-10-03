"""KORPForceField refuses a clashing input with the engine's own floor.

The engine judges a whole state against min_distance less
``mcpu_core.STATE_CLASH_BUFFER_A`` (0.001 A), so construction does too: an
input it accepted could otherwise fail the first full-energy check, or one
the engine would run be refused. Needs no map (see test_korp_chain_identity).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.physics.forcefield.test_korp_chain_identity import _layout, _two_chains


def _with_b11_at(tmp_path, distance: float):
    """Chain B's residue 11 moved so its CA is ``distance`` A from CA(A10),
    on the far side of A10 from the rest of chain A."""
    traj = _two_chains(tmp_path, b_first=11)
    top = traj.topology
    ca_a = top.select("chainid 0 and name CA")
    ca_a10 = top.select("chainid 0 and resSeq 10 and name CA")[0]
    b11 = top.select("chainid 1 and resSeq 11")
    ca_b11 = top.select("chainid 1 and resSeq 11 and name CA")[0]
    outward = traj.xyz[0, ca_a10] - traj.xyz[0, ca_a].mean(axis=0)
    target = traj.xyz[0, ca_a10] + (distance / 10.0) * outward / np.linalg.norm(outward)  # nm
    traj.xyz[0, b11] += target - traj.xyz[0, ca_b11]
    return traj


def test_a_pair_inside_the_buffer_is_accepted(tmp_path) -> None:
    _layout(_with_b11_at(tmp_path, 3.2 - 0.0005))


def test_a_pair_beyond_the_buffer_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="closer than 3.199 A"):
        _layout(_with_b11_at(tmp_path, 3.2 - 0.0015))
