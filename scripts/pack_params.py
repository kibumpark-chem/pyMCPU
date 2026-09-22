#!/usr/bin/env python3
"""Pack local pretrained params into release tarballs + print sha256.

Developer / Release helper (not required at runtime)::

    python scripts/pack_params.py --out dist/params

Produces:
  mcpu_v1-1.0.0-core.tar.gz

Paste printed sha256 values into pymcpu/data/params_registry.json and
attach archives to a GitHub Release (e.g. params-v1.0.0).
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

from pymcpu import PACKAGE_ROOT
from pymcpu.params import required_files


# The file list is derived from the registry `layout`, never hand-listed. The
# previous local CORE_FILES omitted "constants/bbind02.May.lib", so every
# archive this script produced was rejected by pymcpu.params._ensure_files.


def _default_src() -> Path:
    """The raw params tree, from the registry's ``local_source``.

    Resolved through PACKAGE_ROOT the same way ``params._resolve_local_source``
    does. Previously hardcoded as ``PACKAGE_ROOT / "parameters" / ...``, which
    went through a ``pymcpu/parameters`` symlink that has since been removed
    (scikit-build-core dereferenced it into the wheel).
    """
    import json

    registry = json.loads(
        (Path(PACKAGE_ROOT) / "data" / "params_registry.json").read_text()
    )
    rel = registry["sets"]["mcpu_v1"]["local_source"]
    return (Path(PACKAGE_ROOT) / rel).resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pack(src_root: Path, files: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        for rel in files:
            full = src_root / rel
            if not full.is_file():
                raise FileNotFoundError(full)
            tf.add(full, arcname=f"mcpu_v1/{rel}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=_default_src(),
        help="Source params root (constants/ + mcpu_params/)",
    )
    p.add_argument("--out", type=Path, default=Path("dist/params"))
    p.add_argument("--version", default="1.0.0")
    args = p.parse_args()

    src = args.src.resolve()
    # One archive for the complete set. The old core/SC split existed only
    # because the SC-triplet table was 633 MB; it is ~1 MB in the compact
    # format, and MCPUForceField requires it unconditionally anyway, so a
    # "core-only" archive was never actually usable.
    out = args.out / f"mcpu_v1-{args.version}.tar.gz"
    _pack(src, list(required_files().values()), out)
    print(f"wrote {out}")
    print(f"  sha256: {_sha256(out)}")
    print("  paste url + sha256 into pymcpu/data/params_registry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
