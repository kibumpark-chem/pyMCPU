"""Pretrained parameter resolution, download, checksum, and cache.

Resolution order for ``ensure_params(set_name)``:

1. ``MCPU_PARAMS_DIR`` — use as-is (must contain ``constants/`` + ``mcpu_params/``).
2. Dev tree ``local_source`` under the package (repo checkout).
3. A cache directory that is already populated.
4. ``MCPU_PARAMS_BUNDLE`` — path to a ``.tar.gz`` to unpack into the cache.
5. The compact archive shipped **inside the wheel**, decoded into the cache.
   This is what makes a plain ``pip install pymcpu`` work offline. It sits
   deliberately below steps 1-4 so a pre-staged HPC root or a locally refitted
   table still wins.
6. Registry URL via ``pooch`` (requires a published Release + sha256).

Environment:

- ``MCPU_PARAMS_DIR``: offline / HPC pre-placed params root
- ``MCPU_CACHE_DIR``: override cache parent (default ``~/.cache/pymcpu``)
- ``MCPU_PARAMS_BUNDLE``: local archive override for the requested set
- ``MCPU_NO_DOWNLOAD=1``: refuse network / missing URL (fail clearly)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import tarfile
import tempfile
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymcpu import PACKAGE_ROOT

_REGISTRY_PATH = Path(PACKAGE_ROOT) / "data" / "params_registry.json"
_DEFAULT_SET = "mcpu_v1"

logger = logging.getLogger(__name__)


class ParamsError(RuntimeError):
    """Raised when pretrained parameters cannot be resolved."""


def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.is_file():
        raise ParamsError(f"Missing params registry: {_REGISTRY_PATH}")
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_cache_dir() -> Path:
    """Return the cache root for downloaded / unpacked parameter sets."""
    override = os.environ.get("MCPU_CACHE_DIR", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
    else:
        try:
            import pooch

            root = Path(pooch.os_cache("pymcpu"))
        except Exception:
            root = Path.home() / ".cache" / "pymcpu"
    params = root / "params"
    params.mkdir(parents=True, exist_ok=True)
    return params


def _set_entry(set_name: str) -> dict[str, Any]:
    reg = _load_registry()
    sets = reg.get("sets") or {}
    if set_name not in sets:
        known = ", ".join(sorted(sets)) or "(none)"
        raise ParamsError(f"Unknown parameter set {set_name!r}. Known: {known}")
    return sets[set_name]


def _warn_if_include_sc(include_sc: bool | None) -> None:
    """``include_sc`` is accepted and ignored; warn once per call site.

    The core/optional split existed only because
    ``sidechain_triplet_potentials.bin`` was 633 MB. In the shipped compact
    format it is ~1 MB, so the split bought nothing -- and it never actually
    worked: ``MCPUForceField._load_parameters`` lists the SC table as
    unconditionally required, so a core-only root was always rejected
    downstream.
    """
    if include_sc is None:
        return
    warnings.warn(
        "ensure_params(include_sc=...) is deprecated and ignored: the "
        "sidechain-triplet table is part of every complete parameter set. "
        "The argument will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


def _layout(set_name: str) -> dict[str, Any]:
    """Return a set's ``layout`` block, which declares what the set contains."""
    entry = _set_entry(set_name)
    layout = entry.get("layout")
    if not layout:
        raise ParamsError(
            f"Parameter set {set_name!r} has no 'layout' block in "
            f"{_REGISTRY_PATH}.\n"
            "The layout is the single source of truth for a set's contents; "
            "see the mcpu_v1 entry for the expected shape."
        )
    return layout


def required_files(set_name: str = _DEFAULT_SET) -> dict[str, str]:
    """Role -> relative path for every file a *complete* set must contain.

    This is the single source of truth. It is consumed by
    ``ensure_params``'s completeness check, by
    ``MCPUForceField._load_parameters``, by ``scripts/pack_params.py``, and by
    the test stubs -- so those four can no longer disagree about what
    "complete" means. Two bugs came from them disagreeing: the packer omitted
    ``constants/bbind02.May.lib`` (so the archive it built was rejected by
    ``_ensure_files``), and the registry omitted
    ``mcpu_params/hbond_seq_dep.bin`` (so ``ensure_params`` could succeed and
    the force field then raise ``FileNotFoundError``).

    Insertion order is preserved: it is the order in which
    ``_load_parameters`` reports a missing file.
    """
    return dict(_layout(set_name)["required"])


def optional_files(set_name: str = _DEFAULT_SET) -> dict[str, str]:
    """Role -> relative path for files a set *may* contain.

    Currently just ``constants/rama_mixture.json``: the move it serves defaults
    to ``pivot_rama_probability = 0.0``, so a set without it is valid.
    """
    return dict(_layout(set_name).get("optional") or {})


def table_layout(set_name: str = _DEFAULT_SET) -> dict[str, str]:
    """Compact-encoder table name -> relative ``.bin`` path."""
    return dict(_layout(set_name).get("tables") or {})


def constants_files(set_name: str = _DEFAULT_SET) -> list[str]:
    """Relative paths of the non-table (``constants/``) files in a set."""
    return list(_layout(set_name).get("constants") or [])


def _looks_like_params_root(path: Path, files: list[str]) -> bool:
    if not path.is_dir():
        return False
    return all((path / rel).is_file() for rel in files)


def _complete_error(set_name: str, detail: str) -> ParamsError:
    return ParamsError(
        f"{detail}\n\n"
        f"Tried to resolve pretrained set {set_name!r}. Options:\n"
        f"  1. Set MCPU_PARAMS_DIR to a directory containing constants/ and mcpu_params/\n"
        f"  2. Set MCPU_PARAMS_BUNDLE to a local .tar.gz of the set\n"
        f"  3. Place a GitHub Release URL + sha256 in pymcpu/data/params_registry.json\n"
        f"  4. Run: mcpu download-params --set {set_name}\n"
        f"  5. For HPC offline nodes, pre-stage params and export MCPU_PARAMS_DIR\n"
        f"Cache directory: {get_cache_dir()}"
    )


def _resolve_local_source(entry: dict[str, Any]) -> Path | None:
    rel = entry.get("local_source")
    if not rel:
        return None
    candidate = (Path(PACKAGE_ROOT) / rel).resolve()
    files = list((entry.get("layout") or {}).get("required", {}).values())
    if files and _looks_like_params_root(candidate, files):
        return candidate
    return None


def _unpack_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        # Python 3.12+ supports filter=; keep portable extraction.
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)


def _download_with_pooch(url: str, sha256: str, fname: str, path: Path) -> Path:
    import pooch

    if not sha256:
        raise ParamsError(
            f"Registry URL is set for {fname} but sha256 is missing. "
            "Refuse to download without a checksum."
        )
    path.mkdir(parents=True, exist_ok=True)
    known_hash = sha256 if sha256.startswith("sha256:") else f"sha256:{sha256}"
    return Path(
        pooch.retrieve(
            url=url,
            known_hash=known_hash,
            fname=fname,
            path=str(path),
            progressbar=True,
        )
    )


def bundled_tables_path(set_name: str = _DEFAULT_SET) -> Path | None:
    """Path to the compact archive shipped inside the package, or ``None``.

    Works for both wheels and editable installs because it is resolved from
    ``PACKAGE_ROOT``. ``importlib.resources`` would buy nothing here: pymcpu
    ships a compiled extension, so it can never be imported from a zip.
    """
    candidate = Path(PACKAGE_ROOT) / "data" / "params" / set_name / "tables.npz"
    return candidate if candidate.is_file() else None


def _bundled_root(set_name: str) -> Path:
    return Path(PACKAGE_ROOT) / "data" / "params" / set_name


def _digest12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _atomic_write_bytes(dst: Path, payload: bytes) -> None:
    """Write ``payload`` to ``dst`` atomically (same idiom as checkpointing)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fsync_dir(path: Path) -> None:
    """Ensure renames into ``path`` are durable before the directory rename."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _materialize_into(staging: Path, set_name: str, archive: Path, verify: bool) -> None:
    """Decode every table and copy every constant into ``staging``."""
    from pymcpu.paramcodec import decode_table  # local: keeps numpy off this import path

    for name, relpath in table_layout(set_name).items():
        values = decode_table(archive, name, verify=verify)
        _atomic_write_bytes(staging / relpath, values.tobytes())
        del values
    src_root = _bundled_root(set_name)
    for relpath in constants_files(set_name):
        source = src_root / relpath
        if source.is_file():
            _atomic_write_bytes(staging / relpath, source.read_bytes())
    _fsync_dir(staging)


def materialize_from_wheel(
    set_name: str = _DEFAULT_SET,
    *,
    timeout: float = 900.0,
    verify: bool = True,
) -> Path:
    """Decode the in-wheel compact archive into a complete params root.

    Returns a directory holding ``constants/`` + ``mcpu_params/*.bin``, i.e.
    exactly the shape every other resolution step returns, so the loader has a
    single code path.

    The cache directory is **content-addressed** (``<set>-<sha256(npz)[:12]>``),
    which removes staleness logic entirely: a wheel upgrade that changes the
    tables produces a different directory, so there is nothing to invalidate.

    Concurrency: ``scripts/job_template.slurm`` starts N MPI ranks per node
    against a shared ``$HOME``, so N processes can race here. ``os.mkdir`` is
    the mutex -- a single atomic operation that returns ``EEXIST`` to losers --
    rather than ``open(O_CREAT|O_EXCL)``, which is unreliable on NFSv3. The
    winner builds a private staging directory and renames it into place; losers
    poll. Deliberately MPI-agnostic (no ``mpi4py`` import) so it also covers
    WESTPA workers, job arrays, and unrelated concurrent runs.

    In production, prefer avoiding the race altogether::

        export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu_v1)"

    before ``mpirun``, which turns an N-way race into one serial call.
    """
    archive = bundled_tables_path(set_name)
    if archive is None:
        raise _complete_error(
            set_name,
            "No compact parameter archive ships with this install "
            f"(looked for {_bundled_root(set_name) / 'tables.npz'}).",
        )

    required = list(required_files(set_name).values())
    base = get_cache_dir() / "materialized"
    base.mkdir(parents=True, exist_ok=True)
    tag = f"{set_name}-{_digest12(archive)}"
    final = base / tag
    lock = base / f".{tag}.lock"

    deadline = time.monotonic() + timeout
    delay = 0.25
    while True:
        if _looks_like_params_root(final, required):
            return final

        try:
            os.mkdir(lock)
        except FileExistsError:
            # --- loser: wait for the winner, with escapes so a dead winner
            #     cannot wedge an entire MPI job.
            if time.monotonic() > deadline:
                owner = "unknown"
                try:
                    owner = (lock / "owner").read_text().strip()
                except OSError:
                    pass
                raise _complete_error(
                    set_name,
                    f"Timed out after {timeout:.0f}s waiting for another process "
                    f"to materialize parameters.\n"
                    f"  target: {final}\n  lock:   {lock}\n  owner:  {owner}\n"
                    "If that process is gone, remove the lock directory. To avoid "
                    "the race entirely, pre-stage once and export MCPU_PARAMS_DIR.",
                ) from None
            if not lock.exists() and not final.exists():
                continue                      # winner vanished; try to take over
            try:                              # steal a stale lock
                age = time.time() - (lock / "owner").stat().st_mtime
                if age > timeout:
                    logger.warning(
                        "stealing a stale params lock (%.0fs old) at %s", age, lock
                    )
                    shutil.rmtree(lock, ignore_errors=True)
                    continue
            except OSError:
                pass
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)
            continue

        # --- winner
        staging = base / f".{tag}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            (lock / "owner").write_text(
                f"{socket.gethostname()}:{os.getpid()}:"
                f"{datetime.now(timezone.utc).isoformat()}\n"
            )
            if _looks_like_params_root(final, required):
                return final          # a crashed-but-complete predecessor
            staging.mkdir(parents=True)
            _materialize_into(staging, set_name, archive, verify)
            _ensure_files(staging, required, set_name)
            _fsync_dir(base)
            try:
                os.rename(staging, final)
            except OSError:
                if _looks_like_params_root(final, required):
                    shutil.rmtree(staging, ignore_errors=True)
                else:
                    raise
            logger.info("materialized %r parameters into %s", set_name, final)
            return final
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(lock, ignore_errors=True)


def _ensure_files(root: Path, files: list[str], set_name: str) -> None:
    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        raise _complete_error(
            set_name,
            "Parameter directory is incomplete. Missing:\n  - "
            + "\n  - ".join(missing)
            + f"\nUnder: {root}",
        )


def ensure_params(
    set_name: str = _DEFAULT_SET,
    *,
    include_sc: bool | None = None,
) -> Path:
    """Return a directory with pretrained parameters for ``set_name``.

    The returned path contains ``constants/`` and ``mcpu_params/``.
    """
    _warn_if_include_sc(include_sc)
    entry = _set_entry(set_name)
    required = list(required_files(set_name).values())

    # 1) Explicit offline / HPC root
    env_dir = os.environ.get("MCPU_PARAMS_DIR", "").strip()
    if env_dir:
        root = Path(env_dir).expanduser().resolve()
        _ensure_files(root, required, set_name)
        return root

    # 2) In-tree / editable install source (developer & shared lab checkouts)
    local = _resolve_local_source(entry)
    if local is not None and _looks_like_params_root(local, required):
        return local

    # 3) Cache already populated
    cache_root = get_cache_dir() / (entry.get("unpack_root") or set_name)
    if _looks_like_params_root(cache_root, required):
        return cache_root

    no_download = os.environ.get("MCPU_NO_DOWNLOAD", "").strip() in {
        "1",
        "true",
        "True",
        "YES",
        "yes",
    }

    # 4) Local bundle override
    bundle = os.environ.get("MCPU_PARAMS_BUNDLE", "").strip()
    if bundle:
        archive = Path(bundle).expanduser().resolve()
        if not archive.is_file():
            raise _complete_error(set_name, f"MCPU_PARAMS_BUNDLE is not a file: {archive}")
        _unpack_archive(archive, cache_root)
        # Archives may unpack with or without a top-level mcpu_v1/ directory.
        if not _looks_like_params_root(cache_root, required):
            nested = cache_root / (entry.get("unpack_root") or set_name)
            if _looks_like_params_root(nested, required):
                return nested
            # Flatten one nested directory if present
            kids = [p for p in cache_root.iterdir() if p.is_dir()]
            if len(kids) == 1 and _looks_like_params_root(kids[0], required):
                return kids[0]
        _ensure_files(cache_root, required, set_name)
        return cache_root

    # 5) Compact archive shipped inside the package.
    #
    # Deliberately BELOW the four steps above: anyone whose parameters resolve
    # today keeps resolving the same way, byte for byte. In particular a lab
    # member who refits a table into src/pymcpu/parameters/ still sees their
    # edit (step 2) rather than the shipped one, and an HPC user with
    # MCPU_PARAMS_DIR pre-staged is untouched (step 1).
    #
    # It is above the pooch download so a fresh `pip install` works offline
    # with no network and no environment variables at all.
    if bundled_tables_path(set_name) is not None:
        root = materialize_from_wheel(set_name)
        logger.info("pretrained params %r resolved via in-wheel archive -> %s",
                    set_name, root)
        return root

    if no_download:
        raise _complete_error(
            set_name,
            "Parameters not found locally and MCPU_NO_DOWNLOAD=1 forbids download.",
        )

    # 5) Remote download via pooch
    url = entry.get("url")
    sha = entry.get("sha256")
    archive_name = entry.get("archive") or f"{set_name}.tar.gz"
    if not url:
        raise _complete_error(
            set_name,
            "No local parameters and registry URL is null "
            "(GitHub Release not published yet).",
        )

    archived = _download_with_pooch(
        url=str(url),
        sha256=str(sha or ""),
        fname=str(archive_name),
        path=get_cache_dir() / "downloads",
    )
    if cache_root.exists():
        shutil.rmtree(cache_root)
    _unpack_archive(archived, cache_root)
    if not _looks_like_params_root(cache_root, required):
        nested = cache_root / (entry.get("unpack_root") or set_name)
        if _looks_like_params_root(nested, required):
            cache_root = nested
    _ensure_files(cache_root, required, set_name)

    return cache_root


def params_path(
    set_name: str, filename: str, *, include_sc: bool | None = None
) -> Path:
    """Convenience: ``ensure_params(set_name) / filename``."""
    return ensure_params(set_name, include_sc=include_sc) / filename


def download_params(
    set_name: str = _DEFAULT_SET,
    *,
    dest: Path | str | None = None,
    include_sc: bool = True,
) -> Path:
    """Download/unpack (or resolve) params; optionally copy into ``dest``."""
    root = ensure_params(set_name, include_sc=include_sc)
    if dest is None:
        return root
    dest_path = Path(dest).expanduser().resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    # Copy tree contents into dest for explicit staging directories.
    for item in root.iterdir():
        target = dest_path / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return dest_path
