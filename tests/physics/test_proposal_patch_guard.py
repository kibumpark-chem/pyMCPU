"""ProposalPatch rejects bad moved-atom lists at the Python boundary.

``moved_indices`` must list each moved atom once, with every index inside
``[0, num_atoms)``. Mu's delta compares a per-cell count of moved atoms with
the cell's occupancy, so a duplicate makes a cell that still holds an unmoved
atom look fully moved; an out-of-range index writes past ``moving_atoms``.
The engine's own moves never produce either, so the release engine does not
check; the bindings do.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core

N_ATOMS = 8


def test_mark_moved_twice_is_a_no_op():
    patch = mcpu_core.ProposalPatch(N_ATOMS)
    patch.mark_moved(3)
    patch.mark_moved(3)
    patch.mark_moved(5)
    assert list(patch.moved_indices) == [3, 5]
    assert list(patch.moving_atoms) == [0, 0, 0, 1, 0, 1, 0, 0]


@pytest.mark.parametrize("bad", [-1, N_ATOMS, N_ATOMS + 100])
def test_mark_moved_out_of_range_raises_index_error(bad):
    patch = mcpu_core.ProposalPatch(N_ATOMS)
    with pytest.raises(IndexError):
        patch.mark_moved(bad)
    assert list(patch.moved_indices) == []
    assert sum(patch.moving_atoms) == 0


def test_moved_indices_setter_accepts_a_unique_list():
    patch = mcpu_core.ProposalPatch(N_ATOMS)
    patch.moved_indices = [0, 7, 2]
    assert list(patch.moved_indices) == [0, 7, 2]


def test_moved_indices_setter_rejects_duplicates():
    patch = mcpu_core.ProposalPatch(N_ATOMS)
    patch.moved_indices = [1, 2]
    with pytest.raises(ValueError, match="listed twice"):
        patch.moved_indices = [4, 6, 4]
    # A rejected list leaves the previous one in place.
    assert list(patch.moved_indices) == [1, 2]


@pytest.mark.parametrize("bad", [[-1], [0, N_ATOMS], [2, 3, 1000]])
def test_moved_indices_setter_rejects_out_of_range(bad):
    patch = mcpu_core.ProposalPatch(N_ATOMS)
    with pytest.raises(IndexError, match="outside"):
        patch.moved_indices = bad
    assert list(patch.moved_indices) == []
