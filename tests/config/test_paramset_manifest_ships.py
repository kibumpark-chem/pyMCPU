"""A parameter set's own manifest, ``constants/paramset.json``, reaches a
wheel install as it reaches a checkout or ``MCPU_PARAMS_DIR``.

``scripts/encode_params.py`` copies it next to the table archive, and
``--check`` compares it; the wheel materialiser copies it into the cache, and
the cache folder's name covers it. So a set unpacked from the wheel resolves
the same sheet factor and signs the same as its raw tree. Without this, a
set's factor was its manifest's from a checkout but 2.0 from a wheel, and
the two signatures differed, so a checkpoint written with one was refused by
the other.

A stand-in set of two small tables and one constants file is registered for
each test, so nothing large is encoded or decoded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pymcpu import params
from pymcpu.params import PARAMSET_FORMAT, PARAMSET_MANIFEST

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SET = "tiny"
_LAYOUT = {
    "required": {"template": "constants/a.json", "table": "tiny_params/t.bin"},
    "tables": {"t": "tiny_params/t.bin", "u": "tiny_params/u.bin"},
    "constants": ["constants/a.json"],
}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Register the stand-in set and write its raw tree, with a manifest.

    Returns the raw tree (``raw``), the encoder's output folder, which plays
    the package's ``pymcpu/data/params/tiny`` (``out``), and the cache.
    """
    registry = {"schema_version": 1, "sets": {_SET: {"version": "1.0.0", "layout": _LAYOUT}}}
    monkeypatch.setattr(params, "_load_registry", lambda: registry)
    raw, out, cache = tmp_path / "raw", tmp_path / "package" / _SET, tmp_path / "cache"
    (raw / "constants").mkdir(parents=True)
    (raw / "tiny_params").mkdir()
    (raw / "constants" / "a.json").write_text('{"a": 1}')
    (np.arange(12, dtype=np.float32) * 1000).tofile(raw / "tiny_params" / "t.bin")
    np.array([0.5, -1.25], dtype=np.float32).tofile(raw / "tiny_params" / "u.bin")
    _write_manifest(raw, 3.0)
    monkeypatch.setenv("MCPU_CACHE_DIR", str(cache))
    monkeypatch.setattr(params, "_bundled_root", lambda set_name: out)
    monkeypatch.setattr(params, "bundled_tables_path", lambda set_name: out / "tables.npz")
    return {"raw": raw, "out": out, "cache": cache}


def _write_manifest(root: Path, factor: float) -> None:
    path = root / PARAMSET_MANIFEST
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(
        {"format": PARAMSET_FORMAT, "name": _SET, "hbond_beta_favor": factor}))


def _encoder():
    """``scripts/encode_params.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "encode_params_under_test", _REPO_ROOT / "scripts" / "encode_params.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode(encoder, src: Path, out: Path) -> None:
    encoder._encode_into(src, out, _SET, "1.0.0", 9, True)


def _check(encoder, src: Path, out: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr("sys.argv", [
        "encode_params.py", "--src", str(src), "--out", str(out), "--set", _SET, "--check"])
    return encoder.main()


def test_a_wheel_install_reads_the_manifest_as_the_raw_tree_does(tiny) -> None:
    raw, out = tiny["raw"], tiny["out"]
    _encode(_encoder(), raw, out)
    assert (out / PARAMSET_MANIFEST).read_bytes() == (raw / PARAMSET_MANIFEST).read_bytes()

    unpacked = params.materialize_from_wheel(_SET)
    assert unpacked.parent.parent.parent == tiny["cache"]
    assert params.load_paramset_manifest(unpacked)["hbond_beta_favor"] == 3.0
    assert params.param_signature(unpacked, _SET) == params.param_signature(raw, _SET)


def test_a_changed_manifest_gets_a_cache_folder_of_its_own(tiny) -> None:
    out = tiny["out"]
    _encode(_encoder(), tiny["raw"], out)
    first = params.materialize_from_wheel(_SET)

    _write_manifest(out, 2.5)
    second = params.materialize_from_wheel(_SET)
    assert second != first
    assert params.load_paramset_manifest(second)["hbond_beta_favor"] == 2.5
    assert params.load_paramset_manifest(first)["hbond_beta_favor"] == 3.0


def test_check_compares_the_manifest_and_an_encode_mirrors_it(tiny, monkeypatch) -> None:
    encoder = _encoder()
    raw, out = tiny["raw"], tiny["out"]
    _encode(encoder, raw, out)
    assert _check(encoder, raw, out, monkeypatch) == 0

    _write_manifest(raw, 2.5)
    with pytest.raises(SystemExit, match=f"DRIFT: constant {PARAMSET_MANIFEST} differs"):
        _check(encoder, raw, out, monkeypatch)

    (raw / PARAMSET_MANIFEST).unlink()
    with pytest.raises(SystemExit, match=f"DRIFT: constant {PARAMSET_MANIFEST} is only in --out"):
        _check(encoder, raw, out, monkeypatch)
    # Encoding again removes the manifest the raw tree no longer has.
    _encode(encoder, raw, out)
    assert not (out / PARAMSET_MANIFEST).exists()
    assert _check(encoder, raw, out, monkeypatch) == 0


def test_a_set_without_a_manifest_keeps_its_cache_folder(tiny) -> None:
    """The cache folder's name is the one it had before manifests shipped:
    a set without one (mcpu08) is not unpacked again after an upgrade."""
    raw, out = tiny["raw"], tiny["out"]
    (raw / PARAMSET_MANIFEST).unlink()
    _encode(_encoder(), raw, out)
    archive = out / "tables.npz"

    # The name as it was computed before: the archive, then each listed
    # constants file, by relative path, size and contents.
    digest = hashlib.sha256()
    for label, path in [("tables.npz", archive), ("constants/a.json", out / "constants/a.json")]:
        digest.update(label.encode() + b"\0" + str(path.stat().st_size).encode() + b"\0")
        digest.update(path.read_bytes())
    assert params._cache_digest12(_SET, archive) == digest.hexdigest()[:12]
    assert params.materialize_from_wheel(_SET).name == f"{_SET}-{digest.hexdigest()[:12]}"
