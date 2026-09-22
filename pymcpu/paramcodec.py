"""Compact, lossless encoding of the pretrained potential tables.

The tables ship inside the wheel. Stored raw they are 678 MiB of float32;
encoded they are ~2 MiB, which is why ``pip install pymcpu`` can work offline
with no download step at all.

Why it compresses so well
-------------------------
The tables are not sparse, but they are extremely low-cardinality. The
sidechain-triplet table holds 165,888,000 values drawn from just 1,723
distinct ones, and ``1000.0`` -- the "no observation for this bin" sentinel --
accounts for 99.806% of them. Four of the six tables are exact integers in
``int16`` range (the legacy quasi-chemical fits are scaled by 1000). Keeping
them as exact integers is a precision *win*, not a compromise: the scaled
values are representable exactly, so the narrowing is bitwise lossless and
round-trips to the same ``float32`` bits. Do not "simplify" it away.

So the encoder does two things and lets ``zlib`` do the rest:

1. narrow each table to ``int16`` when that is **bitwise** lossless, else keep
   ``float32`` verbatim;
2. hand the array to ``numpy.savez_compressed``.

A value palette was measured and deliberately rejected: for the three large
tables ``int16 + zlib`` is 1.910 MiB versus 1.917 MiB for a
``uint16``-palette, i.e. the codebook costs 6.7 KiB and buys nothing, because
zlib's Huffman stage already discovers the same symbol distribution. A sparse
COO format was also rejected -- measured *larger* than dense-compressed.

The bitwise narrowing check matters
-----------------------------------
``narrowed.astype(float32).view(uint32) == values.view(uint32)`` is compared on
the **bit patterns**, not with ``==`` or ``np.allclose``. ``hbond_seq_dep.bin``
contains 64 genuine ``-0.0`` values and no ``+0.0``; a float-value comparison
treats those as equal to ``+0.0``, so the day a table holds both, a value-based
check would silently lose bit-exactness while every tolerance test stayed
green. Do not relax this comparison.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_ID = "pymcpu-params-npz/1"

# Archive-level members.
_KEY_FORMAT = "__format__"
_KEY_SET = "__set__"
_KEY_SOURCE_VERSION = "__source_version__"
_KEY_ENDIANNESS = "__endianness__"
_KEY_TABLES = "__tables__"

# Per-table member suffixes.
_SUFFIX_RELPATH = "__relpath"
_SUFFIX_SHA256 = "__sha256"
_SUFFIX_NBYTES = "__nbytes"


class ParamCodecError(RuntimeError):
    """Raised when a compact parameter archive is malformed or inconsistent."""


def _as_str(value: Any) -> str:
    """Read a 0-d numpy string member back as a plain ``str``."""
    return str(np.asarray(value).item())


def _narrow(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as ``int16`` if that is bitwise lossless, else as-is.

    The comparison is on raw bit patterns so that signed zeros (and any future
    NaN) cannot be silently folded together. See the module docstring.
    """
    if values.dtype != np.float32:
        raise ParamCodecError(f"expected float32, got {values.dtype}")
    lo, hi = float(values.min()), float(values.max())
    if not (np.iinfo(np.int16).min <= lo and hi <= np.iinfo(np.int16).max):
        return values
    narrowed = values.astype(np.int16)
    widened = narrowed.astype(np.float32)
    if np.array_equal(widened.view(np.uint32), values.view(np.uint32)):
        return narrowed
    return values


def _decoded_digest(values: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).digest()


def read_header(npz_path: Path) -> dict[str, Any]:
    """Return the archive-level metadata, validating format and endianness."""
    with np.load(npz_path, allow_pickle=False) as archive:
        if _KEY_FORMAT not in archive:
            raise ParamCodecError(f"{npz_path} is not a pymcpu parameter archive")
        fmt = _as_str(archive[_KEY_FORMAT])
        if fmt != FORMAT_ID:
            raise ParamCodecError(
                f"unsupported parameter archive format {fmt!r} "
                f"(this build understands {FORMAT_ID!r})"
            )
        endianness = _as_str(archive[_KEY_ENDIANNESS])
        if endianness != "little":
            raise ParamCodecError(
                f"{npz_path} was written on a {endianness}-endian host; "
                "the decoded .bin layout would not match"
            )
        return {
            "format": fmt,
            "set": _as_str(archive[_KEY_SET]),
            "source_version": _as_str(archive[_KEY_SOURCE_VERSION]),
            "endianness": endianness,
            "tables": [str(t) for t in archive[_KEY_TABLES]],
        }


def table_info(npz_path: Path) -> dict[str, tuple[int, np.dtype, str, int]]:
    """Per-table ``(n_elements, stored dtype, relpath, decoded nbytes)``.

    Reads only the members' ``.npy`` headers, so this is ~100 bytes of I/O per
    table and does **not** decompress anything -- shape assertions in tests can
    use it instead of decoding 633 MiB.
    """
    info: dict[str, tuple[int, np.dtype, str, int]] = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        for name in (str(t) for t in archive[_KEY_TABLES]):
            member = archive[name]
            info[name] = (
                int(member.size),
                member.dtype,
                _as_str(archive[f"{name}{_SUFFIX_RELPATH}"]),
                int(np.asarray(archive[f"{name}{_SUFFIX_NBYTES}"]).item()),
            )
    return info


def decode_table(npz_path: Path, name: str, *, verify: bool = True) -> np.ndarray:
    """Decode one table to a C-contiguous 1-D ``float32`` array.

    The result is byte-identical to ``np.fromfile(<relpath>, np.float32)`` on
    the raw tree. ``verify`` re-hashes the decoded bytes against the digest
    stored at encode time; it costs ~1.7 s on the 633 MiB table, so callers
    that decode on a hot path should pass ``verify=False`` and rely on the
    one-time check performed at materialization.
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        if name not in archive:
            raise ParamCodecError(f"{npz_path} has no table {name!r}")
        stored = archive[name]
        expected = bytes(archive[f"{name}{_SUFFIX_SHA256}"])
    decoded = np.ascontiguousarray(stored, dtype=np.float32)
    if verify and _decoded_digest(decoded) != expected:
        raise ParamCodecError(
            f"decoded table {name!r} does not match the digest recorded at "
            f"encode time -- {npz_path} is corrupt"
        )
    return decoded


def encode_set(
    src_root: Path,
    out_path: Path,
    *,
    set_name: str,
    version: str,
    tables: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Encode the raw ``.bin`` tables under ``src_root`` into ``out_path``.

    ``tables`` maps table name -> path relative to ``src_root`` (this is the
    registry's ``layout.tables``). Returns a per-table report.
    """
    members: dict[str, Any] = {
        _KEY_FORMAT: np.asarray(FORMAT_ID),
        _KEY_SET: np.asarray(set_name),
        _KEY_SOURCE_VERSION: np.asarray(version),
        _KEY_ENDIANNESS: np.asarray("little"),
        _KEY_TABLES: np.asarray(list(tables), dtype=object).astype(str),
    }
    report: dict[str, dict[str, Any]] = {}

    for name, relpath in tables.items():
        raw = src_root / relpath
        if not raw.is_file():
            raise ParamCodecError(f"missing source table for {name!r}: {raw}")
        values = np.fromfile(raw, dtype=np.float32)
        stored = _narrow(values)
        members[name] = stored
        members[f"{name}{_SUFFIX_RELPATH}"] = np.asarray(relpath)
        members[f"{name}{_SUFFIX_SHA256}"] = np.frombuffer(
            _decoded_digest(values), dtype=np.uint8
        )
        members[f"{name}{_SUFFIX_NBYTES}"] = np.asarray(values.nbytes, dtype=np.int64)
        report[name] = {
            "relpath": relpath,
            "elements": int(values.size),
            "raw_bytes": int(values.nbytes),
            "stored_dtype": str(stored.dtype),
            "distinct": int(np.unique(values).size),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **members)
    # np.savez_compressed appends .npz when the name lacks it
    if not out_path.exists() and out_path.with_suffix(out_path.suffix + ".npz").exists():
        out_path.with_suffix(out_path.suffix + ".npz").replace(out_path)
    return report
