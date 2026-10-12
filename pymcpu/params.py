"""Pretrained parameter resolution and cache.

Resolution order for ``ensure_params(set_name)``:

1. ``MCPU_PARAMS_DIR`` — use as-is (must contain ``constants/`` + ``mcpu_params/``).
2. Dev tree ``local_source`` under the package (repo checkout).
3. The compact archive shipped **inside the wheel**, decoded into the cache.
   This is what makes a plain ``pip install pymcpu`` work offline. It sits
   deliberately below steps 1-2 so a pre-staged HPC root or a locally refitted
   table still wins.

Environment:

- ``MCPU_PARAMS_DIR``: offline / HPC pre-placed params root
- ``MCPU_CACHE_DIR``: override cache parent (default ``~/.cache/pymcpu``)

:func:`param_signature` identifies the files of a resolved folder, wherever it
is; checkpoints record it, so a run cannot resume with other tables.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymcpu import PACKAGE_ROOT

_REGISTRY_PATH = Path(PACKAGE_ROOT) / "data" / "params_registry.json"
_DEFAULT_SET = "mcpu08"

#: A parameter set's own description file. A folder need not have one; when
#: it does, :func:`param_signature` covers it as it covers the tables.
PARAMSET_MANIFEST = "constants/paramset.json"

logger = logging.getLogger(__name__)


class ParamsError(RuntimeError):
    """Raised when pretrained parameters cannot be resolved."""


def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.is_file():
        raise ParamsError(f"Missing params registry: {_REGISTRY_PATH}")
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def cache_root() -> Path:
    """Return the pyMCPU cache root: ``MCPU_CACHE_DIR`` or ``~/.cache/pymcpu``."""
    override = os.environ.get("MCPU_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "pymcpu"


def get_cache_dir() -> Path:
    """Return the cache directory for materialized parameter sets."""
    params = cache_root() / "params"
    params.mkdir(parents=True, exist_ok=True)
    return params


def _set_entry(set_name: str) -> dict[str, Any]:
    reg = _load_registry()
    sets = reg.get("sets") or {}
    if set_name not in sets:
        known = ", ".join(sorted(sets)) or "(none)"
        raise ParamsError(f"Unknown parameter set {set_name!r}. Known: {known}")
    return sets[set_name]


def _layout(set_name: str) -> dict[str, Any]:
    """Return a set's ``layout`` block, which declares what the set contains."""
    entry = _set_entry(set_name)
    layout = entry.get("layout")
    if not layout:
        raise ParamsError(
            f"Parameter set {set_name!r} has no 'layout' block in "
            f"{_REGISTRY_PATH}.\n"
            "The layout is the single source of truth for a set's contents; "
            "see the mcpu08 entry for the expected shape."
        )
    return layout


def required_files(set_name: str = _DEFAULT_SET) -> dict[str, str]:
    """Role -> relative path for every file a *complete* set must contain.

    This is the single source of truth. It is consumed by
    ``ensure_params``'s completeness check, by
    ``MCPUForceField._load_parameters``, and by the test stubs -- so those can
    no longer disagree about what "complete" means. Two bugs came from them
    disagreeing: a release packer omitted
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


def signed_files(set_name: str = _DEFAULT_SET) -> list[str]:
    """Relative paths :func:`param_signature` covers, sorted: every file the
    set's layout names (required, optional, tables and constants), which are
    the files ``MCPUForceField`` reads, and :data:`PARAMSET_MANIFEST`."""
    layout = _layout(set_name)
    paths = set(layout.get("required", {}).values())
    paths.update((layout.get("optional") or {}).values())
    paths.update((layout.get("tables") or {}).values())
    paths.update(layout.get("constants") or [])
    paths.add(PARAMSET_MANIFEST)
    return sorted(paths)


#: sha256 of each file read so far, by (real path, device, inode, size,
#: modification time): see _file_sha256.
_FILE_SHA256: dict[tuple[str, int, int, int, int], str] = {}


def _file_sha256(path: Path) -> str:
    """sha256 (hex) of a file's bytes, kept for the life of the process.

    Kept by the file's real path, device, inode, size and modification time,
    so a process that builds many runs from one folder reads each file once
    (the mcpu08 tables are 0.7 GB, about 2 s to hash), and a file that has
    been replaced or rewritten since is read again.
    """
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    digest = _FILE_SHA256.get(key)
    if digest is None:
        with open(real, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        _FILE_SHA256[key] = digest
    return digest


def param_signature(param_dir: str | Path, set_name: str = _DEFAULT_SET) -> str:
    """sha256 (hex) identifying the parameter files a force field reads from
    ``param_dir``, a folder laid out as ``set_name``.

    It covers each of :func:`signed_files` by its relative path and the
    sha256 of its bytes; a file the folder lacks counts as absent. So two
    folders that hold the same files have the same signature wherever they
    are, and changing, adding or removing one of those files changes it.
    Other files in the folder do not count. Links are followed. Each file is
    read once per process while it stays unchanged.
    """
    root = Path(param_dir)
    digest = hashlib.sha256()
    for rel in signed_files(set_name):
        path = root / rel
        entry = _file_sha256(path) if path.is_file() else "absent"
        digest.update(f"{rel}\0{entry}\n".encode())
    return digest.hexdigest()


def _looks_like_params_root(path: Path, files: list[str]) -> bool:
    if not path.is_dir():
        return False
    return all((path / rel).is_file() for rel in files)


def _complete_error(set_name: str, detail: str) -> ParamsError:
    return ParamsError(
        f"{detail}\n\n"
        f"Tried to resolve pretrained set {set_name!r}. Options:\n"
        f"  1. Set MCPU_PARAMS_DIR to a directory containing constants/ and mcpu_params/\n"
        f"  2. Install a pyMCPU wheel, which ships the parameters, and run:\n"
        f"     mcpu materialize-params --set {set_name}\n"
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


def _cache_digest12(set_name: str, archive: Path) -> str:
    """First 12 hex digits of a sha256 over everything a materialized set holds.

    That is the table archive and every constants file copied next to it,
    each by its relative path (sorted) and contents, so a release that changes
    only a constants file gets a new cache directory too. A constants file
    the package lacks is hashed as absent, as ``_materialize_into`` skips it.
    """
    digest = hashlib.sha256()

    def _add(label: str, path: Path) -> None:
        digest.update(label.encode() + b"\0")
        if not path.is_file():
            digest.update(b"absent\0")
            return
        digest.update(str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)

    _add("tables.npz", archive)
    src_root = _bundled_root(set_name)
    for relpath in sorted(constants_files(set_name)):
        _add(relpath, src_root / relpath)
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

    The cache directory is **content-addressed** (``<set>-<sha256[:12]>``, the
    hash covering the table archive and every constants file), which removes
    staleness logic entirely: a wheel upgrade that changes the tables or a
    constants file produces a different directory, so there is nothing to
    invalidate.

    Concurrency: ``scripts/job_template.slurm`` starts N MPI ranks per node
    against a shared ``$HOME``, so N processes can race here. ``os.mkdir`` is
    the mutex -- a single atomic operation that returns ``EEXIST`` to losers --
    rather than ``open(O_CREAT|O_EXCL)``, which is unreliable on NFSv3. The
    winner builds a private staging directory and renames it into place; losers
    poll. Deliberately MPI-agnostic (no ``mpi4py`` import) so it also covers
    sampler worker pools, job arrays, and unrelated concurrent runs.

    In production, prefer avoiding the race altogether::

        export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"

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
    tag = f"{set_name}-{_cache_digest12(set_name, archive)}"
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


def ensure_params(set_name: str = _DEFAULT_SET) -> Path:
    """Return a directory with pretrained parameters for ``set_name``.

    The returned path contains ``constants/`` and ``mcpu_params/``.
    """
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
    if local is not None:
        return local

    # 3) Compact archive shipped inside the package.
    #
    # Deliberately below the two steps above: a lab member who refits a table
    # into src/pymcpu/parameters/ still sees their edit (step 2) rather than
    # the shipped one, and an HPC user with MCPU_PARAMS_DIR pre-staged is
    # untouched (step 1).
    if bundled_tables_path(set_name) is not None:
        root = materialize_from_wheel(set_name)
        logger.info("pretrained params %r resolved via in-wheel archive -> %s",
                    set_name, root)
        return root

    raise _complete_error(
        set_name,
        "No parameters found: MCPU_PARAMS_DIR is unset, there is no in-tree "
        "parameter source, and no compact archive ships with this install.",
    )


def params_path(set_name: str, filename: str) -> Path:
    """Convenience: ``ensure_params(set_name) / filename``."""
    return ensure_params(set_name) / filename
