"""The KORP map reader, exercised against synthetic maps.

Deliberately needs no ``korp6Dv1.bin``: that file is ~316 MiB and cannot be
distributed, so if the only coverage required it there would effectively be
none. A map is written here byte by byte in the upstream layout, which also
makes this file the executable statement of what that layout *is*.

The one test that does want the real map is skipped without it.
"""

from __future__ import annotations

import math
import os
import struct
import sys

import numpy as np
import pytest

from pymcpu.forcefields.korp_map import (
    KORP_RESIDUE_ORDER,
    KorpMapError,
    load_korp_map,
)

# The released map, for the checks that can only be made against it.
REAL_MAP_SIZE = 331_777_205
REAL_MAP_SHA256 = "8c586500f80ad31f297652e050d391702a01927ee58d598017650ef2e0fbf971"


def _write_map(path, *, nr=2, nring=3, cells_per_ring=(1, 2, 1), nchi=2,
               nonbonding=1, nonbonding2=4, bonding_factor=1.8,
               each_bonding=False, dimensions=6, frame_model=10, ngauss=0,
               cutoff=None, fill=None, truncate_bytes=0):
    """Write a synthetic map in the exact upstream layout.

    Mirrors ``readMapHeader`` + ``readMeshes`` + ``getMeshes`` + ``getMap``.
    """
    ncells = sum(cells_per_ring)
    assert len(cells_per_ring) == nring
    br = [3.0] + [3.0 + (i + 1) * 2.0 for i in range(nr)]
    if cutoff is None:
        cutoff = br[-1]
    dchi = 2.0 * math.pi / nchi

    use_bonding = nonbonding2 >= 0
    if each_bonding:
        nslices = 1 + nonbonding2 - nonbonding
    elif use_bonding:
        nslices = 2
    else:
        nslices = 1

    buf = bytearray()
    buf += struct.pack("<i f b i i f i b b b b i i",
                       dimensions, cutoff, frame_model, nonbonding,
                       nonbonding2, bonding_factor, ngauss,
                       0, 0, int(each_bonding), 0, 0, 0)
    buf += struct.pack("<i", nr)
    buf += np.asarray(br, dtype="<f4").tobytes()
    buf += np.asarray([ncells] * nr, dtype="<i4").tobytes()

    icell = np.cumsum([0] + list(cells_per_ring[:-1])).astype("<i4")
    theta = np.asarray(
        [math.pi * (i + 1) / nring for i in range(nring)], dtype="<f4")
    dpsi = np.asarray(
        [2.0 * math.pi / c for c in cells_per_ring], dtype="<f4")
    for _ in range(nr):
        buf += struct.pack("<ii", ncells, nring)
        buf += np.asarray(cells_per_ring, dtype="<i4").tobytes()
        buf += icell.tobytes()
        buf += theta.tobytes()
        buf += dpsi.tobytes()
        buf += struct.pack("<f", dchi)

    block = nr * ncells * ncells * nchi
    n_floats = nslices * 20 * 20 * block
    values = (np.arange(n_floats, dtype=np.float64) * 0.001
              if fill is None else np.full(n_floats, fill, dtype=np.float64))
    buf += values.astype("<f4").tobytes()

    data = bytes(buf) if truncate_bytes == 0 else bytes(buf)[:-truncate_bytes]
    path.write_bytes(data)
    return {"nr": nr, "ncells": ncells, "nchi": nchi, "nslices": nslices,
            "block": block, "n_floats": n_floats, "cutoff": cutoff, "br": br}


def test_round_trips_every_header_and_mesh_field(tmp_path):
    p = tmp_path / "synthetic.bin"
    meta = _write_map(p)
    m = load_korp_map(p, mmap=False)

    assert (m.dimensions, m.frame_model, m.ngauss) == (6, 10, 0)
    assert m.nonbonding == 1 and m.nonbonding2 == 4
    assert m.bonding_factor == pytest.approx(1.8)
    assert m.use_bonding is True and m.each_bonding is False
    assert m.nr == meta["nr"]
    assert m.min_r == pytest.approx(3.0)
    assert m.cutoff == pytest.approx(meta["cutoff"])
    assert [s.ncells for s in m.shells] == [meta["ncells"]] * meta["nr"]
    assert [s.nchi for s in m.shells] == [meta["nchi"]] * meta["nr"]
    assert m.block_stride == meta["block"]
    assert m.table.shape == (meta["n_floats"],)


def test_table_values_land_where_the_index_says(tmp_path):
    p = tmp_path / "synthetic.bin"
    _write_map(p)
    m = load_korp_map(p, mmap=False)
    # The synthetic payload is value == flat_index * 0.001, so a correct
    # index arithmetic reproduces the index itself.
    for (s, a, b, ir, ca, cb, ic) in [
        (0, 0, 0, 0, 0, 0, 0),
        (1, 3, 7, 1, 2, 1, 1),
        (1, 19, 19, 1, 3, 3, 1),
    ]:
        sh = m.shells[ir]
        expect = (m.block_base(s, a, b) + int(m.shell_off[ir])
                  + (ca * sh.ncells + cb) * sh.nchi + ic)
        assert m.lookup(s, a, b, ir, ca, cb, ic) == pytest.approx(
            expect * 0.001, rel=1e-6)


def test_slice_mapping_matches_upstream_derivation(tmp_path):
    p = tmp_path / "synthetic.bin"
    _write_map(p, nonbonding=1, nonbonding2=4, each_bonding=False)
    m = load_korp_map(p, mmap=False)
    assert m.nslices == 2
    # |i-j| <= nonbonding is excluded; 2..4 are local; >=5 non-local.
    assert m.slice_for_separation(1) == -1
    assert [m.slice_for_separation(k) for k in (2, 3, 4)] == [1, 1, 1]
    assert [m.slice_for_separation(k) for k in (5, 9)] == [0, 0]
    # Above BONDING_THR everything folds back to non-bonding.
    assert m.slice_for_separation(40) == 0
    assert m.fmapping[0] == pytest.approx(1.0)
    assert m.fmapping[1] == pytest.approx(1.8)


def test_each_bonding_gives_one_slice_per_separation(tmp_path):
    p = tmp_path / "synthetic.bin"
    _write_map(p, nonbonding=1, nonbonding2=4, each_bonding=True)
    m = load_korp_map(p, mmap=False)
    assert m.nslices == 4
    assert [m.slice_for_separation(k) for k in (2, 3, 4)] == [1, 2, 3]


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"dimensions": 3}, "6D"),
        ({"frame_model": 11}, "frame_model"),
        ({"ngauss": 8}, "Gaussian"),
        ({"cutoff": 99.0}, "cutoff"),
    ],
)
def test_unsupported_maps_are_rejected(tmp_path, kwargs, needle):
    p = tmp_path / "bad.bin"
    _write_map(p, **kwargs)
    with pytest.raises(KorpMapError, match=needle):
        load_korp_map(p, mmap=False)


def test_truncation_is_caught_by_the_size_check(tmp_path):
    p = tmp_path / "short.bin"
    _write_map(p, truncate_bytes=4)
    with pytest.raises(KorpMapError, match="size"):
        load_korp_map(p, mmap=False)


def test_trailing_garbage_is_caught_too(tmp_path):
    p = tmp_path / "long.bin"
    _write_map(p)
    with open(p, "ab") as fh:
        fh.write(b"\0\0\0\0")
    with pytest.raises(KorpMapError, match="size"):
        load_korp_map(p, mmap=False)


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(KorpMapError, match="not found"):
        load_korp_map(tmp_path / "nope.bin")


def test_mmap_and_read_paths_agree(tmp_path):
    p = tmp_path / "synthetic.bin"
    _write_map(p)
    a = load_korp_map(p, mmap=True)
    b = load_korp_map(p, mmap=False)
    assert isinstance(a.table, np.memmap)
    assert not isinstance(b.table, np.memmap)
    assert np.asarray(a.table).tobytes() == np.asarray(b.table).tobytes()


def test_in_memory_table_is_read_only_and_owns_its_buffer(tmp_path):
    p = tmp_path / "synthetic.bin"
    meta = _write_map(p)
    table = load_korp_map(p).table   # default: in-memory
    assert not table.flags.writeable
    assert table.flags.c_contiguous and table.dtype == np.dtype("<f4")
    assert table.shape == (meta["n_floats"],)
    with pytest.raises(ValueError):
        table[0] = 1.0
    # Only the array is left; its buffer must stay valid after the map goes.
    import gc
    gc.collect()
    payload = p.read_bytes()[-4 * meta["n_floats"]:]
    assert table.tobytes() == payload


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="the huge-page path is Linux-only")
def test_in_memory_table_starts_on_a_huge_page_boundary(tmp_path):
    # Only the MADV_HUGEPAGE path aligns the table; the np.empty fallback
    # would not, so this fails if the advice ever goes missing again.
    p = tmp_path / "synthetic.bin"
    _write_map(p)
    table = load_korp_map(p).table
    assert table.ctypes.data % (2 << 20) == 0
    assert table.base is not None


def test_a_reload_reuses_a_table_that_outlived_its_map(tmp_path):
    # A C++ map keeps only the table alive; loading the file again while it
    # exists must not read a second private copy.
    import gc
    p = tmp_path / "synthetic.bin"
    _write_map(p)
    table = load_korp_map(p).table
    gc.collect()
    assert load_korp_map(p).table is table


def test_loads_share_one_map_per_file_and_mode(tmp_path):
    p = tmp_path / "synthetic.bin"
    _write_map(p)
    a = load_korp_map(p)
    assert load_korp_map(str(p)) is a
    assert load_korp_map(tmp_path / "." / "synthetic.bin") is a
    m = load_korp_map(p, mmap=True)
    assert m is not a and load_korp_map(p, mmap=True) is m
    # A changed file is a new map.
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert load_korp_map(p) is not a


@pytest.mark.parametrize("mmap", [False, True])
def test_sha256_is_the_file_digest_in_both_modes(tmp_path, mmap):
    import hashlib
    p = tmp_path / "synthetic.bin"
    _write_map(p)
    want = hashlib.sha256(p.read_bytes()).hexdigest()
    m = load_korp_map(p, mmap=mmap)
    assert m.sha256 is None
    # Asking later for the digest fills it in on the shared map.
    assert load_korp_map(p, mmap=mmap, sha256=True) is m
    assert m.sha256 == want


def test_korp_residue_order_is_one_letter_alphabetical():
    # KORP indexes by one-letter code, pyMCPU's AMINO_INDEX by three-letter.
    # Conflating them is silent, so pin the ordering itself.
    one_letter = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    codes = [one_letter[r] for r in KORP_RESIDUE_ORDER]
    assert codes == sorted(codes)
    assert len(KORP_RESIDUE_ORDER) == 20


@pytest.mark.skipif(
    not os.environ.get("KORP_MAP_PATH"),
    reason="needs the real korp6Dv1.bin via KORP_MAP_PATH",
)
def test_real_map_matches_published_identity():
    m = load_korp_map(os.environ["KORP_MAP_PATH"], sha256=True)
    assert m.path.stat().st_size == REAL_MAP_SIZE
    assert m.sha256 == REAL_MAP_SHA256
    assert (m.dimensions, m.frame_model) == (6, 10)
    assert m.cutoff == pytest.approx(16.0)
