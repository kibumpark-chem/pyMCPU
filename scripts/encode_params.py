"""Encode the raw pretrained ``.bin`` tables into the compact shipped archive.

The raw tables are 678 MiB of float32 and are gitignored; the encoded archive
is ~2 MiB and ships inside the wheel, which is what lets ``pip install pymcpu``
work with no download step. See ``pymcpu/paramcodec.py`` for why it compresses
(low cardinality, not sparsity) and why the x1000 fixed-point tables must stay
integers.

Usage::

    python scripts/encode_params.py \\
        --src src/pymcpu/parameters/pretrained/mcpu08 \\
        --out pymcpu/data/params/mcpu_v1 \\
        --set mcpu_v1

    # release-prep gate: fail if the committed archive is stale
    python scripts/encode_params.py --check

The table list and the constants list both come from the registry's
``layout``, never from a local copy.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pymcpu.paramcodec import decode_table, encode_set, read_header  # noqa: E402
from pymcpu.params import constants_files, table_layout  # noqa: E402

_DEFAULT_SRC = REPO_ROOT / "src" / "pymcpu" / "parameters" / "pretrained" / "mcpu08"
_ARCHIVE_NAME = "tables.npz"


def _mib(n: int) -> float:
    return n / (1 << 20)


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy preserving content, via a temp file in the destination directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.")
    tmp = Path(tmp_name)
    try:
        with open(fd, "wb") as out, src.open("rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            import os

            os.fsync(out.fileno())
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _encode_into(src: Path, out_dir: Path, set_name: str, version: str, level: int,
                 verify: bool) -> dict:
    tables = table_layout(set_name)
    if not tables:
        raise SystemExit(f"registry layout for {set_name!r} declares no tables")

    archive = out_dir / _ARCHIVE_NAME
    report = encode_set(src, archive, set_name=set_name, version=version, tables=tables)

    # The constants (atom types, amino-acid template, rotamer library, rama
    # mixture) are small and shipped verbatim -- they are not tables.
    copied = []
    for rel in constants_files(set_name):
        source = src / rel
        if not source.is_file():
            continue          # rama_mixture.json is optional
        _atomic_copy(source, out_dir / rel)
        copied.append(rel)

    total_raw = sum(r["raw_bytes"] for r in report.values())
    encoded = archive.stat().st_size
    print(f"{'table':32s} {'elements':>14s} {'raw':>12s} {'dtype':>8s} {'distinct':>9s}")
    for name, r in report.items():
        print(f"{name:32s} {r['elements']:>14,} {_mib(r['raw_bytes']):>9.2f} MiB "
              f"{r['stored_dtype']:>8s} {r['distinct']:>9,}")
    print(f"\n  raw total      : {_mib(total_raw):>9.2f} MiB")
    print(f"  archive        : {_mib(encoded):>9.2f} MiB  ({total_raw / max(encoded,1):.0f}x)")
    print(f"  constants      : {len(copied)} files copied verbatim")

    if verify:
        bad = []
        for name, r in report.items():
            decoded = decode_table(archive, name, verify=True)
            reference = np.fromfile(src / r["relpath"], dtype=np.float32)
            if not np.array_equal(decoded.view(np.uint32), reference.view(np.uint32)):
                bad.append(name)
        if bad:
            raise SystemExit(f"ROUND-TRIP FAILED (not bit-exact): {', '.join(bad)}")
        print(f"  round-trip     : BIT-EXACT for all {len(report)} tables")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=_DEFAULT_SRC,
                   help="raw params root containing constants/ and mcpu_params/")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "pymcpu" / "data" / "params" / "mcpu_v1",
                   help="destination for tables.npz + constants/")
    p.add_argument("--set", dest="set_name", default="mcpu_v1")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--compresslevel", type=int, default=9)
    p.add_argument("--no-verify", action="store_true",
                   help="skip the post-encode bit-exactness check (not recommended)")
    p.add_argument("--check", action="store_true",
                   help="encode to a temp dir and diff against --out; exit 1 on drift")
    args = p.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "mcpu_v1"
            _encode_into(src, staging, args.set_name, args.version,
                         args.compresslevel, not args.no_verify)
            # Compare decoded content, not archive bytes: zlib output is not
            # required to be reproducible across versions.
            committed = args.out / _ARCHIVE_NAME
            if not committed.is_file():
                raise SystemExit(f"no committed archive at {committed}")
            fresh = staging / _ARCHIVE_NAME
            if read_header(fresh)["tables"] != read_header(committed)["tables"]:
                raise SystemExit("DRIFT: table list differs")
            for name in read_header(fresh)["tables"]:
                a = decode_table(fresh, name, verify=False)
                b = decode_table(committed, name, verify=False)
                if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
                    raise SystemExit(f"DRIFT: table {name} differs from the committed archive")
            for rel in constants_files(args.set_name):
                s, d = staging / rel, args.out / rel
                if s.is_file() and not (d.is_file() and filecmp.cmp(s, d, shallow=False)):
                    raise SystemExit(f"DRIFT: constant {rel} differs")
            print("\n  --check: committed archive is up to date")
        return 0

    _encode_into(src, args.out.resolve(), args.set_name, args.version,
                 args.compresslevel, not args.no_verify)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
