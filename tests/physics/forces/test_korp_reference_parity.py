"""pyMCPU's KORP energy against the reference ``korpe`` binary.

This is the test that actually establishes the implementation *is* KORP. Every
other KORP test checks a piece in isolation -- the map decode, the frame
algebra, the binning -- and each of those can be individually right while the
assembled energy is wrong, because the pieces only combine correctly under one
consistent set of conventions. A single number agreeing with upstream pins all
of them at once: the binary layout, the frame convention (where the paper and
the released code disagree), the six coordinates, the nearest-bin lookup, the
one-letter-alphabetical residue ordering, and the sequence-separation split.

Needs the ~316 MiB map, which pyMCPU does not ship, so it skips by default.
To run it::

    export KORP_MAP_PATH=/path/to/Korp6Dv1/korp6Dv1.bin
    pytest tests/physics/forces/test_korp_reference_parity.py

The expected energies below were produced by the bundled ``korpe_gcc`` against
``korp6Dv1.bin`` (sha256 8c586500...fbf971). They are recorded rather than
recomputed so the numbers stay meaningful if the reference binary is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HELPERS = Path(__file__).resolve().parents[1] / "helpers"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from korp_pdb import parse_backbone  # noqa: E402

from pymcpu.forcefields.korp_map import load_korp_map, score_structure  # noqa: E402

#: Structure (relative to the KORP distribution root) -> reference energy.
#: The three T0860D1 entries are a CASP target and two of its decoys, so they
#: span a wide energy range on identical sequence and topology; 1CEO is a
#: different, larger protein.
REFERENCE_ENERGIES = {
    "CASP12DCsel20/T0860D1.pdb": -3693.586739,
    "CASP12DCsel20/T0860D1_s026m1.pdb": -1004.099531,
    "CASP12DCsel20/T0860D1_s119m1.pdb": -2444.948077,
    "rcd6/1CEO.pdb": -11463.957486,
}

#: float32 table values accumulated in float64 against upstream's own float64
#: accumulation. Observed spread is ~5e-9; 1e-7 leaves headroom without being
#: loose enough to hide a genuine convention error, which would be off by
#: percent or more.
RELATIVE_TOLERANCE = 1e-7


def _map_path():
    path = os.environ.get("KORP_MAP_PATH")
    if not path:
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")
    return Path(path)


def _structure(rel):
    root = _map_path().parent
    pdb = root / rel
    if not pdb.is_file():
        pytest.skip(f"{rel} not found next to the map (needs the KORP bundle)")
    return pdb


@pytest.fixture(scope="module")
def korp_map():
    return load_korp_map(_map_path())


@pytest.mark.parametrize("rel,expected", sorted(REFERENCE_ENERGIES.items()))
def test_matches_reference_korpe(korp_map, rel, expected):
    coords, names, res_seq, chain_ids = parse_backbone(_structure(rel))
    energy = score_structure(korp_map, coords, names,
                             res_seq=res_seq, chain_ids=chain_ids)
    assert energy == pytest.approx(expected, rel=RELATIVE_TOLERANCE)


def test_the_map_is_the_one_these_numbers_came_from(korp_map):
    """Guards the fixture: other KORP maps score differently and legitimately."""
    assert korp_map.cutoff == pytest.approx(16.0)
    assert (korp_map.frame_model, korp_map.dimensions) == (10, 6)
    assert (korp_map.nonbonding, korp_map.nonbonding2) == (1, 4)
    assert korp_map.bonding_factor == pytest.approx(1.8, abs=1e-6)
    assert korp_map.nr == 10
    assert [s.ncells for s in korp_map.shells] == [36] * 10
    assert [s.nchi for s in korp_map.shells] == [8] * 10


def test_decoys_score_worse_than_the_native(korp_map):
    """KORP's actual job. A sign error would pass the parity tests and fail here."""
    def score(rel):
        coords, names, seq, chains = parse_backbone(_structure(rel))
        return score_structure(korp_map, coords, names,
                               res_seq=seq, chain_ids=chains)

    native = score("CASP12DCsel20/T0860D1.pdb")
    for decoy in ("CASP12DCsel20/T0860D1_s026m1.pdb",
                  "CASP12DCsel20/T0860D1_s119m1.pdb"):
        assert native < score(decoy)
