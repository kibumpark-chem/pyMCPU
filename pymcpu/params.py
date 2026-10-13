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
:func:`load_paramset_manifest` reads the folder's own description of its set,
``constants/paramset.json``, when it has one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import numbers
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from pymcpu import PACKAGE_ROOT

_REGISTRY_PATH = Path(PACKAGE_ROOT) / "data" / "params_registry.json"
_DEFAULT_SET = "mcpu08"

#: A parameter set's own description file. A folder need not have one; when
#: it does, :func:`param_signature` covers it as it covers the tables, and
#: :func:`load_paramset_manifest` reads it.
PARAMSET_MANIFEST = "constants/paramset.json"

#: The ``format`` a :data:`PARAMSET_MANIFEST` declares.
PARAMSET_FORMAT = "pymcpu-paramset/1"

#: The terms a manifest's ``folded_weights`` may name: energy groups 1 to 5.
FOLDED_WEIGHT_TERMS = ("mu", "bb_torsion", "sc_torsion", "hbond", "aromatic")

_PARAMSET_KEYS = (
    "format", "name", "description", "hbond_beta_favor", "folded_weights", "provenance",
)

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


def shipped_constants(set_name: str = _DEFAULT_SET) -> list[str]:
    """Relative paths of the files a set ships as they are, next to its
    table archive: :func:`constants_files` and :data:`PARAMSET_MANIFEST`,
    once each.

    ``scripts/encode_params.py`` copies them, the wheel materialiser copies
    them into the cache and the cache key covers them, so a set's own
    defaults reach a pip install as they reach a checkout or
    ``MCPU_PARAMS_DIR``. Each step skips a file the set does not have, as
    mcpu08 has no manifest.
    """
    files = constants_files(set_name)
    if PARAMSET_MANIFEST not in files:
        files.append(PARAMSET_MANIFEST)
    return files


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


#: The digests a parameter file gets, from one read: sha256 for
#: param_signature, md5 for a manifest's provenance.
_DIGEST_ALGORITHMS = ("md5", "sha256")
_DIGEST_LENGTHS = {"md5": 32, "sha256": 64}

#: Digests of the files read so far in this process, by file identity
#: (_file_identity).
_FILE_DIGESTS: dict[tuple[str, int, int, int, int], dict[str, str]] = {}

#: The file, under the cache root's params/ folder, that keeps the digests
#: of parameter files between processes (see _files_digests), and its format.
DIGEST_STORE = "digests.json"
_DIGEST_STORE_FORMAT = "pymcpu-digests/1"
#: Files the store keeps at most; the oldest entries go first.
_DIGEST_STORE_LIMIT = 4096


def _file_identity(path: str | Path) -> tuple[str, int, int, int, int]:
    """What the digest caches trust a file's contents by, without reading
    it: its real path, inode, size, and modification and change times (ns).

    Not the device: one file on a network filesystem has another st_dev on
    each node. The change time cannot be set back by hand, so a file
    rewritten with its size and modification time restored is read again.
    """
    real = os.path.realpath(path)
    st = os.stat(real)
    return (real, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _digest_store_path() -> Path:
    return cache_root() / "params" / DIGEST_STORE


def _read_digest_store() -> dict[str, Any]:
    """The digest store's entries by real path; empty when there is no store
    or it cannot be read."""
    try:
        data = json.loads(_digest_store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("format") != _DIGEST_STORE_FORMAT:
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def _stored_digests(store: dict[str, Any], identity: tuple) -> dict[str, str] | None:
    """The digests the store holds for this identity of a file, or None."""
    entry = store.get(identity[0])
    if not isinstance(entry, dict) or [
        entry.get(key) for key in ("inode", "size", "mtime_ns", "ctime_ns")
    ] != list(identity[1:]):
        return None
    digests = {algorithm: entry.get(algorithm) for algorithm in _DIGEST_ALGORITHMS}
    for algorithm, digest in digests.items():
        if not (isinstance(digest, str) and len(digest) == _DIGEST_LENGTHS[algorithm]
                and re.fullmatch(r"[0-9a-f]+", digest)):
            return None
    return digests


def _keep_digests(taken: dict[tuple, dict[str, str]]) -> None:
    """Add digests just taken to the store, merged with what it holds now
    and written atomically. A store that cannot be written is left alone:
    the digests then last for this process only."""
    if not taken:
        return
    path = _digest_store_path()
    try:
        store = _read_digest_store()
        for identity, digests in taken.items():
            real, inode, size, mtime_ns, ctime_ns = identity
            store.pop(real, None)  # one entry per path: the file as it is now
            store[real] = {"inode": inode, "size": size, "mtime_ns": mtime_ns,
                           "ctime_ns": ctime_ns, **digests}
        for real in list(store)[: max(0, len(store) - _DIGEST_STORE_LIMIT)]:
            del store[real]
        payload = {"format": _DIGEST_STORE_FORMAT, "files": store}
        _atomic_write_bytes(path, json.dumps(payload, indent=1).encode())
    except OSError as exc:
        logger.debug("could not keep parameter digests in %s: %s", path, exc)


def _hash_file(path: str) -> dict[str, str]:
    """Every digest of :data:`_DIGEST_ALGORITHMS` of a file's bytes, from one
    read."""
    hashers = {algorithm: hashlib.new(algorithm) for algorithm in _DIGEST_ALGORITHMS}
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {algorithm: hasher.hexdigest() for algorithm, hasher in hashers.items()}


def _files_digests(paths: list[Path]) -> list[dict[str, str]]:
    """The md5 and sha256 (hex) of each file's bytes.

    Both come from one read, and are kept by the file's identity
    (:func:`_file_identity`) for the life of the process and in the digest
    store under the cache root (``MCPU_CACHE_DIR``, default
    ``~/.cache/pymcpu``): a parameter file is read once per cache root while
    it stays unchanged, not once per process (the mcpu08 tables are 0.7 GB:
    about 3 s from the page cache, up to 45 s from a home directory when
    they are not cached). A file replaced, rewritten or touched since has
    another identity and is read again; so is one that changed while it was
    read, whose digests are not kept.
    """
    identities = [_file_identity(path) for path in paths]
    missing = [identity for identity in dict.fromkeys(identities) if identity not in _FILE_DIGESTS]
    if missing:
        store = _read_digest_store()
        taken: dict[tuple, dict[str, str]] = {}
        for identity in missing:
            digests = _stored_digests(store, identity)
            if digests is None:
                digests = _hash_file(identity[0])
                if _file_identity(identity[0]) != identity:
                    _FILE_DIGESTS.pop(identity, None)
                    raise ParamsError(
                        f"{identity[0]} changed while it was read; read it again when it "
                        "is no longer being written"
                    )
                taken[identity] = digests
            _FILE_DIGESTS[identity] = digests
        _keep_digests(taken)
    return [_FILE_DIGESTS[identity] for identity in identities]


def _file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """The ``algorithm`` (md5 or sha256) digest of one file: see
    :func:`_files_digests`."""
    return _files_digests([Path(path)])[0][algorithm]


def param_file_identities(
    param_dir: str | Path, set_name: str = _DEFAULT_SET
) -> dict[str, tuple[str, int, int, int] | None]:
    """For each of :func:`signed_files` under ``param_dir``: its real path,
    inode, size and modification time (ns), or None when the folder lacks it.

    A few stat calls, no reads. A file rewritten or replaced since, or a link
    pointed elsewhere, gives another value, so comparing two of these tells
    whether a folder still holds the files a force field read.
    """
    root = Path(param_dir)
    identities: dict[str, tuple[str, int, int, int] | None] = {}
    for rel in signed_files(set_name):
        real = os.path.realpath(root / rel)
        try:
            st = os.stat(real)
        except FileNotFoundError:
            identities[rel] = None
            continue
        identities[rel] = (real, st.st_ino, st.st_size, st.st_mtime_ns) if os.path.isfile(real) else None
    return identities


def param_signature(param_dir: str | Path, set_name: str = _DEFAULT_SET) -> str:
    """sha256 (hex) identifying the parameter files a force field reads from
    ``param_dir``, a folder laid out as ``set_name``.

    It covers each of :func:`signed_files` by its relative path and the
    sha256 of its bytes; a file the folder lacks counts as absent. So two
    folders that hold the same files have the same signature wherever they
    are, and changing, adding or removing one of those files changes it.
    Other files in the folder do not count. Links are followed. Each file is
    read once per cache root (``MCPU_CACHE_DIR``) while it stays unchanged:
    its digests are kept there, in ``params/digests.json``.
    """
    root = Path(param_dir)
    files = signed_files(set_name)
    present = [rel for rel in files if (root / rel).is_file()]
    digests = dict(zip(present, _files_digests([root / rel for rel in present])))
    digest = hashlib.sha256()
    for rel in files:
        entry = digests[rel]["sha256"] if rel in digests else "absent"
        digest.update(f"{rel}\0{entry}\n".encode())
    return digest.hexdigest()


def load_paramset_manifest(
    param_dir: str | Path, *, check_md5: bool = True
) -> dict[str, Any] | None:
    """Read and check the :data:`PARAMSET_MANIFEST` of a parameter folder, or
    return None when the folder has none.

    The file lets a parameter set carry its own defaults. It is a JSON
    object::

        {
          "format": "pymcpu-paramset/1",
          "name": "mcpu26",
          "description": "PISCES 2026 tables, fitted weights",
          "hbond_beta_favor": 3.0,
          "folded_weights": {"mu": 1.25, "hbond": 0.9},
          "provenance": {"md5": {"mcpu_params/mu_potentials.bin": "<32 hex digits>"}}
        }

    ``format`` (exactly :data:`PARAMSET_FORMAT`) and ``name`` (a non-empty
    string) are required; ``description`` is a string. ``hbond_beta_favor``
    is the set's hydrogen-bond sheet factor, which
    :class:`~pymcpu.MCPUForceField` uses when it is not given one; it must be
    a number that stays finite and above 0 in float32, the rule the force
    field applies to its own argument. ``folded_weights`` maps any of
    :data:`FOLDED_WEIGHT_TERMS` to the factor w_new / w_legacy that was
    already multiplied into that term's table, a number that stays finite in
    float32 and is 0 or above (0 for a term the fit switched off). It is a
    record only: the engine applies nothing, and its group weights stay the
    legacy ones. ``provenance`` is an object of any content, except that
    an ``md5`` entry maps paths relative to the folder to the md5 each file
    had when the set was written, and every listed file must still have it.

    Unknown keys, a key given twice anywhere in the file and a value of the
    wrong kind are refused with ValueError naming the file, as is a listed
    file that is missing or has changed. Returns the checked manifest, with
    numbers as floats. Comparing a listed file's md5 reads it the first time
    (0.7 GB for mcpu08's tables, about 3 s with the sha256 the signature
    takes from the same read), and then once per cache root
    (``MCPU_CACHE_DIR``) while it stays unchanged. With ``check_md5=False`` the
    listed files are not read, only checked to exist;
    :func:`check_paramset_provenance` compares them later.
    """
    root = Path(param_dir)
    path = root / PARAMSET_MANIFEST
    if not path.is_file():
        if os.path.lexists(path):
            raise ValueError(f"{path} exists but is not a file that can be read")
        return None

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(f"{path}: keys given more than once: {repeated}")
        return dict(pairs)

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must hold a JSON object, not {type(manifest).__name__}")
    unknown = sorted(set(manifest) - set(_PARAMSET_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; allowed {list(_PARAMSET_KEYS)}")
    missing = [key for key in ("format", "name") if key not in manifest]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    if manifest["format"] != PARAMSET_FORMAT:
        raise ValueError(
            f"{path}: 'format' must be {PARAMSET_FORMAT!r}, got {manifest['format']!r}"
        )
    name = manifest["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: 'name' must be a non-empty string, got {name!r}")
    checked: dict[str, Any] = {"format": PARAMSET_FORMAT, "name": name}
    if "description" in manifest:
        if not isinstance(manifest["description"], str):
            raise ValueError(
                f"{path}: 'description' must be a string, got {manifest['description']!r}"
            )
        checked["description"] = manifest["description"]
    if "hbond_beta_favor" in manifest:
        checked["hbond_beta_favor"] = _positive_factor(
            manifest["hbond_beta_favor"], "hbond_beta_favor", path
        )
    if "folded_weights" in manifest:
        weights = manifest["folded_weights"]
        if not isinstance(weights, dict):
            raise ValueError(
                f"{path}: 'folded_weights' must be an object of factors, got {weights!r}"
            )
        unknown = sorted(set(weights) - set(FOLDED_WEIGHT_TERMS))
        if unknown:
            raise ValueError(
                f"{path}: unknown 'folded_weights' terms {unknown}; allowed "
                f"{list(FOLDED_WEIGHT_TERMS)}"
            )
        checked["folded_weights"] = {
            term: _folded_weight(value, f"folded_weights.{term}", path)
            for term, value in weights.items()
        }
    if "provenance" in manifest:
        provenance = manifest["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError(f"{path}: 'provenance' must be an object, got {provenance!r}")
        if "md5" in provenance:
            _check_provenance_md5(root, path, provenance["md5"], compare=check_md5)
        checked["provenance"] = provenance
    return checked


def _positive_factor(value: Any, key: str, path: Path) -> float:
    """``value`` as a float, or ValueError unless it is a number (not a bool)
    that stays finite and above 0 in float32, the engine's precision: the rule
    MCPUForceField applies to ``hbond_beta_favor``."""
    import numpy as np  # local: keeps numpy off this import path

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            factor = float(value)
        except OverflowError:  # an integer too large for a float
            factor = float("inf")
        if 0.0 < factor <= float(np.finfo(np.float32).max) and np.float32(factor) > 0.0:
            return factor
    raise ValueError(f"{path}: '{key}' must be a finite number above 0, got {value!r}")


def _folded_weight(value: Any, key: str, path: Path) -> float:
    """``value`` as a float, or ValueError unless it is a number (not a bool)
    that stays finite in float32 and is 0 or above. A folded weight is a
    record of what was multiplied into a table, and a fit with weights of 0
    or above can switch a term off with an exact 0; the engine applies none
    of them, so they need not be above 0 as ``hbond_beta_favor`` must."""
    import numpy as np  # local: keeps numpy off this import path

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            weight = float(value)
        except OverflowError:  # an integer too large for a float
            weight = float("inf")
        if 0.0 <= weight <= float(np.finfo(np.float32).max):
            return weight
    raise ValueError(f"{path}: '{key}' must be a finite number, 0 or above, got {value!r}")


def _check_provenance_md5(root: Path, path: Path, md5s: Any, *, compare: bool = True) -> None:
    """Raise ValueError unless ``md5s`` maps paths inside ``root`` to md5
    digests that the files there still have. With ``compare=False``, only
    what it lists is checked, and no file is read."""
    if not isinstance(md5s, dict) or not all(
        isinstance(rel, str) and isinstance(md5, str) for rel, md5 in md5s.items()
    ):
        raise ValueError(
            f"{path}: 'provenance.md5' must map file paths, relative to the folder, to md5 "
            f"strings, got {md5s!r}"
        )
    for rel, expected in md5s.items():
        pure = PurePosixPath(rel)
        if not rel or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(
                f"{path}: 'provenance.md5' lists {rel!r}; a path must be relative to the "
                "folder and stay inside it"
            )
        if pure.as_posix() == PARAMSET_MANIFEST:
            raise ValueError(f"{path}: 'provenance.md5' cannot list the manifest itself")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", expected):
            raise ValueError(
                f"{path}: 'provenance.md5' gives {rel} {expected!r}, which is not an md5 "
                "(32 hexadecimal digits)"
            )
        target = root / pure
        if not target.is_file():
            raise ValueError(
                f"{path}: 'provenance.md5' lists {rel}, which is not a file in {root}"
            )
    if not compare:
        return
    targets = [root / PurePosixPath(rel) for rel in md5s]
    for target, expected, digests in zip(targets, md5s.values(), _files_digests(targets)):
        found = digests["md5"]
        if found != expected.lower():
            raise ValueError(
                f"{target} has md5 {found}, but {path} records {expected.lower()}: the "
                "file changed after the parameter set was written; restore it, or write "
                "the set again"
            )


def check_paramset_provenance(param_dir: str | Path, manifest: dict[str, Any] | None) -> None:
    """Raise ValueError unless every file whose md5 a checked manifest
    (:func:`load_paramset_manifest`) lists under ``provenance.md5`` still has
    it. Nothing to do for None or a manifest without md5s.

    For a manifest read with ``check_md5=False``: MPI replica exchange reads
    it so on every rank, and compares the md5s on rank 0 alone.
    """
    md5s = ((manifest or {}).get("provenance") or {}).get("md5")
    if md5s:
        root = Path(param_dir)
        _check_provenance_md5(root, root / PARAMSET_MANIFEST, md5s)


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

    That is the table archive and every file copied next to it
    (:func:`shipped_constants`), each by its relative path (sorted) and
    contents, so a release that changes only a constants file, or only the
    set's manifest, gets a new cache directory too. A constants file the
    package lacks is hashed as absent, as ``_materialize_into`` skips it. An
    absent manifest that the layout does not list is left out, so a set
    without one (mcpu08) keeps the directory it had before manifests shipped.
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
    listed = set(constants_files(set_name))
    for relpath in sorted(shipped_constants(set_name)):
        if relpath not in listed and not (src_root / relpath).is_file():
            continue
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
    """Decode every table, and copy every constant and the set's manifest
    (:func:`shipped_constants`), into ``staging``."""
    from pymcpu.paramcodec import decode_table  # local: keeps numpy off this import path

    for name, relpath in table_layout(set_name).items():
        values = decode_table(archive, name, verify=verify)
        _atomic_write_bytes(staging / relpath, values.tobytes())
        del values
    src_root = _bundled_root(set_name)
    for relpath in shipped_constants(set_name):
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
    hash covering the table archive and every constants file, the set's
    manifest included), which removes staleness logic entirely: a wheel
    upgrade that changes the tables or a constants file produces a different
    directory, so there is nothing to invalidate.

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
