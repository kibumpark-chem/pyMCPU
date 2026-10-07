"""The triplet tables are memory-mapped read-only, not read into memory.

A mapping lets every process on a node share one copy of the 633 MiB
sidechain table through the page cache. These tests pin that the loader
returns a read-only map with the same values a full read gives.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.forcefields.builders.triplet_builder import TripletPotentialBuilder


def _write_table(path, n: int) -> np.ndarray:
    values = np.arange(n, dtype=np.float32) * np.float32(0.25)
    values.tofile(path)
    return values


def test_triplet_table_is_a_read_only_map(tmp_path) -> None:
    cls = TripletPotentialBuilder
    shape = (cls.BB_DIM_RES,) * 3 + (cls.BLOCK_SIZE,)
    path = tmp_path / "triplet_potentials.bin"
    values = _write_table(path, int(np.prod(shape)))

    table = cls.load_parameters(str(path))

    assert isinstance(table, np.memmap)
    assert not table.flags.writeable
    assert table.shape == shape
    np.testing.assert_array_equal(table, values.reshape(shape))
    with pytest.raises(ValueError):
        table[0, 0, 0, 0] = 1.0


def test_triplet_table_of_the_wrong_size_is_rejected(tmp_path) -> None:
    path = tmp_path / "triplet_potentials.bin"
    _write_table(path, 1000)
    with pytest.raises(ValueError):
        TripletPotentialBuilder.load_parameters(str(path))
