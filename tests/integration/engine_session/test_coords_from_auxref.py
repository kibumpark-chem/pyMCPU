"""``EngineSession.coords_from_auxref`` -- starting-state loading.

No physics content: these check I/O format handling (PDB vs. ``.npz``
agreement), cache identity semantics, and input validation.

The ``.npz`` case is deliberately written with a bare ``np.savez``. The
loader used to call the WESTPA add-on's ``load_segment_state``, which tied
core to one external framework's restart format for no benefit -- it read
nothing from the payload except ``coords``. It now accepts *any* ``.npz``
carrying a ``coords`` array, which is why the add-on's own segment states
still load (they write ``coords=``) without core knowing anything about
them. The pymcpu-westpa repo's own
``tests/test_core_interop.py`` pins that direction from its side.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.config import EngineSpec
from pymcpu.sampling import EngineSession


@pytest.fixture
def spec(engine_spec_factory) -> EngineSpec:
    return engine_spec_factory()


def test_coords_from_auxref_pdb_and_npz_agree(spec: EngineSpec, tmp_path) -> None:
    eng = EngineSession(spec)
    coords = eng.coords_from_auxref(spec.pdb)

    npz_path = tmp_path / "seed_state.npz"
    np.savez(npz_path, coords=coords, step=0)
    coords_from_npz = eng.coords_from_auxref(str(npz_path))
    np.testing.assert_allclose(coords, coords_from_npz)


def test_coords_from_auxref_accepts_either_orientation(spec: EngineSpec, tmp_path) -> None:
    """``(n_atoms, 3)`` is normalized to the engine's ``(3, n_atoms)``.

    Worth pinning separately: a transposed array is not an error, it is a
    silently wrong structure, and an external sampler writing its own
    restart files is exactly who would get the orientation the other way
    round.
    """
    eng = EngineSession(spec)
    coords = eng.coords_from_auxref(spec.pdb)
    assert coords.shape[0] == 3

    transposed = tmp_path / "transposed.npz"
    np.savez(transposed, coords=coords.T)
    np.testing.assert_array_equal(eng.coords_from_auxref(str(transposed)), coords)


def test_coords_from_auxref_rejects_npz_without_coords(spec: EngineSpec, tmp_path) -> None:
    bad = tmp_path / "no_coords.npz"
    np.savez(bad, positions=np.zeros((3, 4), dtype=np.float32))
    eng = EngineSession(spec)
    with pytest.raises(ValueError, match="no 'coords' array"):
        eng.coords_from_auxref(str(bad))


def test_coords_from_auxref_is_cached(spec: EngineSpec) -> None:
    eng = EngineSession(spec)
    c1 = eng.coords_from_auxref(spec.pdb)
    c2 = eng.coords_from_auxref(spec.pdb)
    assert c1 is not c2  # each call returns an independent copy
    np.testing.assert_array_equal(c1, c2)
    assert spec.pdb in eng._auxref_cache


def test_coords_from_auxref_rejects_unknown_extension(spec: EngineSpec, tmp_path) -> None:
    bogus = tmp_path / "state.bogus"
    bogus.write_text("nope")
    eng = EngineSession(spec)
    with pytest.raises(ValueError, match="unsupported basis-state auxref"):
        eng.coords_from_auxref(str(bogus))
