"""Tests for the compact parameter codec (``pymcpu.paramcodec``).

The codec is what lets 678 MiB of float32 tables ship inside a ~2 MiB wheel.
Its one hard requirement is that decoding is **bit-exact**: the decoded array
must be byte-identical to ``np.fromfile(raw, np.float32)``, because the C++
engine consumes it as a plain float32 buffer and the suite pins exact
accept-bit streams.

Two anti-drift locks are asserted here:

* every table's stored digest is re-computed independently from the decoded
  bytes, so re-encoding without updating the digest fails, and corrupting the
  data without updating the digest also fails;
* shapes are checked against the **builders' own** constants rather than
  literals, and via the ``.npy`` headers so nothing is decompressed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymcpu.paramcodec import (
    FORMAT_ID,
    ParamCodecError,
    decode_table,
    encode_set,
    read_header,
    table_info,
)
from pymcpu.params import table_layout

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE = _REPO_ROOT / "pymcpu" / "data" / "params" / "mcpu08" / "tables.npz"

# Element counts derived from the builders that consume each table, so a
# reshape change in a builder surfaces here instead of at a user's runtime.
def _expected_element_counts() -> dict[str, int]:
    from pymcpu.forcefields.builders import mu_builder
    from pymcpu.forcefields.builders.aromatic_builder import AromaticPotentialBuilder
    from pymcpu.forcefields.builders.hbond_builder import HydrogenBondBuilder
    from pymcpu.forcefields.builders.triplet_builder import (
        SidechainTripletBuilder,
        TripletPotentialBuilder,
    )

    return {
        "mu_potentials": mu_builder.N_ATOM_TYPES**2,
        "triplet_potentials": TripletPotentialBuilder.BB_DIM_RES**3
        * TripletPotentialBuilder.BLOCK_SIZE,
        "sidechain_triplet_potentials": SidechainTripletBuilder.SC_DIM_RES**3
        * SidechainTripletBuilder.BLOCK_SIZE,
        "hydrogen_bond_potentials": HydrogenBondBuilder.N_SS_TYPES
        ** 1
        * 3 ** (HydrogenBondBuilder.HBOND_DIM + 3),  # 3**13
        "hbond_seq_dep": HydrogenBondBuilder.N_SS_TYPES * 20 * 20,
        "aromatic_potentials": AromaticPotentialBuilder.NUM_ANGLE_BINS,
    }


requires_archive = pytest.mark.skipif(
    not _ARCHIVE.is_file(), reason="shipped archive absent; run scripts/encode_params.py"
)


@requires_archive
def test_header_is_valid() -> None:
    header = read_header(_ARCHIVE)
    assert header["format"] == FORMAT_ID
    assert header["endianness"] == "little"
    assert header["set"] == "mcpu08"
    assert set(header["tables"]) == set(table_layout("mcpu08"))


@requires_archive
def test_shapes_match_the_builders_without_decompressing() -> None:
    """``table_info`` reads only .npy headers -- microseconds, no decode."""
    info = table_info(_ARCHIVE)
    expected = _expected_element_counts()
    assert set(info) == set(expected)
    for name, n_expected in expected.items():
        n_actual, dtype, relpath, nbytes = info[name]
        assert n_actual == n_expected, f"{name}: {n_actual} != {n_expected}"
        assert nbytes == n_expected * 4, f"{name}: decoded nbytes mismatch"
        assert dtype in (np.dtype(np.int16), np.dtype(np.float32))
        assert relpath.startswith("mcpu_params/")


@requires_archive
@pytest.mark.parametrize(
    "name",
    ["mu_potentials", "hbond_seq_dep", "aromatic_potentials",
     "hydrogen_bond_potentials", "triplet_potentials"],
)
def test_decode_is_bit_exact_against_the_stored_digest(name: str) -> None:
    """Independently recompute the digest -- this is the anti-drift lock."""
    import hashlib

    decoded = decode_table(_ARCHIVE, name, verify=True)
    assert decoded.dtype == np.dtype(np.float32)
    assert decoded.flags.c_contiguous
    with np.load(_ARCHIVE, allow_pickle=False) as archive:
        stored_digest = bytes(archive[f"{name}__sha256"])
    assert hashlib.sha256(decoded.tobytes()).digest() == stored_digest


@requires_archive
@pytest.mark.slow
def test_decode_is_bit_exact_for_the_large_table() -> None:
    """The 633 MiB sidechain table, split out because it costs ~1.7 s to hash."""
    import hashlib

    decoded = decode_table(_ARCHIVE, "sidechain_triplet_potentials", verify=True)
    with np.load(_ARCHIVE, allow_pickle=False) as archive:
        stored = bytes(archive["sidechain_triplet_potentials__sha256"])
    assert hashlib.sha256(decoded.tobytes()).digest() == stored


def test_int16_widening_is_exact_for_every_value() -> None:
    """Why the codec needs no value palette: int16 -> float32 is lossless for
    all 65,536 values, so narrowing is free and reversible."""
    everything = np.arange(-32768, 32768, dtype=np.int16)
    widened = everything.astype(np.float32)
    assert np.array_equal(widened.astype(np.int16), everything)


def _encode_one(tmp_path: Path, values: np.ndarray) -> Path:
    raw_dir = tmp_path / "raw" / "mcpu_params"
    raw_dir.mkdir(parents=True)
    (raw_dir / "t.bin").write_bytes(values.tobytes())
    out = tmp_path / "out.npz"
    encode_set(
        tmp_path / "raw", out, set_name="test", version="0",
        tables={"t": "mcpu_params/t.bin"},
    )
    return out


def test_encoder_keeps_float32_when_narrowing_would_be_lossy(tmp_path: Path) -> None:
    values = np.array([0.5, 1.0, -2.25], dtype=np.float32)
    out = _encode_one(tmp_path, values)
    assert table_info(out)["t"][1] == np.dtype(np.float32)
    assert np.array_equal(decode_table(out, "t").view(np.uint32), values.view(np.uint32))


def test_encoder_narrows_integral_values_to_int16(tmp_path: Path) -> None:
    values = np.array([-1000, 0, 836, 1000], dtype=np.float32)
    out = _encode_one(tmp_path, values)
    assert table_info(out)["t"][1] == np.dtype(np.int16)
    assert np.array_equal(decode_table(out, "t").view(np.uint32), values.view(np.uint32))


def test_signed_zero_survives_encoding(tmp_path: Path) -> None:
    """The trap this codec is built around.

    ``hbond_seq_dep.bin`` holds 64 genuine ``-0.0`` and no ``+0.0``. A
    value-based narrowing check (``==`` or ``np.allclose``) treats -0.0 as
    equal to +0.0, so a table holding BOTH would silently lose bit-exactness
    while every tolerance test stayed green. The codec compares bit patterns.
    """
    values = np.array([-0.0, 0.0, 1.0], dtype=np.float32)
    out = _encode_one(tmp_path, values)
    decoded = decode_table(out, "t")
    # bitwise, not ==: -0.0 == 0.0 is True in float comparison
    assert np.array_equal(decoded.view(np.uint32), values.view(np.uint32))
    assert decoded.view(np.uint32)[0] == np.uint32(0x80000000)  # still negative zero


def test_rejects_an_unknown_format_id(tmp_path: Path) -> None:
    out = tmp_path / "bad.npz"
    np.savez_compressed(
        out,
        __format__=np.asarray("something-else/9"),
        __set__=np.asarray("x"),
        __source_version__=np.asarray("0"),
        __endianness__=np.asarray("little"),
        __tables__=np.asarray([], dtype=str),
    )
    with pytest.raises(ParamCodecError, match="unsupported parameter archive format"):
        read_header(out)


def test_rejects_a_non_archive(tmp_path: Path) -> None:
    out = tmp_path / "plain.npz"
    np.savez_compressed(out, something=np.zeros(3, dtype=np.float32))
    with pytest.raises(ParamCodecError, match="not a pymcpu parameter archive"):
        read_header(out)


@requires_archive
def test_decode_detects_a_corrupted_digest(tmp_path: Path) -> None:
    """Flipping the stored digest must make verify=True fail."""
    with np.load(_ARCHIVE, allow_pickle=False) as archive:
        members = {k: archive[k] for k in archive.files}
    members["aromatic_potentials__sha256"] = np.zeros(32, dtype=np.uint8)
    doctored = tmp_path / "doctored.npz"
    np.savez_compressed(doctored, **members)
    with pytest.raises(ParamCodecError, match="does not match the digest"):
        decode_table(doctored, "aromatic_potentials", verify=True)
