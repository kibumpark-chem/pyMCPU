"""The parameter cache directory changes when a constants file changes.

The directory name used to hash only the table archive, so a release that
changed only a constants file (``rama_mixture.json``, ``bbind02.May.lib``,
...) kept serving the old copy from the cache. The package's own files are
swapped for small stand-ins here, and the table decode is skipped, so the
test unpacks nothing of the real tables. The last test hashes the files the
package really ships for mcpu08 and mcpu26, and unpacks nothing either.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pymcpu import params

SETS = params.available_sets()


@pytest.fixture
def fake_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "package_set"
    (root / "constants").mkdir(parents=True)
    (root / "tables.npz").write_bytes(b"tables v1")
    (root / "constants" / "a.json").write_text('{"v": 1}')
    (root / "constants" / "b.lib").write_text("b v1")

    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(params, "bundled_tables_path", lambda set_name: root / "tables.npz")
    monkeypatch.setattr(params, "_bundled_root", lambda set_name: root)
    monkeypatch.setattr(params, "table_layout", lambda set_name: {})
    monkeypatch.setattr(
        params, "constants_files", lambda set_name: ["constants/b.lib", "constants/a.json"]
    )
    monkeypatch.setattr(
        params, "required_files",
        lambda set_name: {"a": "constants/a.json", "b": "constants/b.lib"},
    )
    return root


@pytest.mark.parametrize("set_name", SETS)
def test_same_files_reuse_the_directory(fake_set: Path, set_name: str) -> None:
    first = params.materialize_from_wheel(set_name, verify=False)
    assert params.materialize_from_wheel(set_name, verify=False) == first
    assert first.name.startswith(f"{set_name}-")


@pytest.mark.parametrize("set_name", SETS)
@pytest.mark.parametrize("changed", ["constants/a.json", "constants/b.lib", "tables.npz"])
def test_any_changed_file_gets_a_new_directory(fake_set: Path, changed: str, set_name: str) -> None:
    first = params.materialize_from_wheel(set_name, verify=False)

    (fake_set / changed).write_bytes(b"version 2")
    second = params.materialize_from_wheel(set_name, verify=False)

    assert second != first
    if changed.startswith("constants/"):
        assert (second / changed).read_bytes() == b"version 2"
        assert (first / changed).read_bytes() != b"version 2"


def test_mcpu26s_real_key_is_its_own_and_covers_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key of the files mcpu26 ships, with the registry's real layout
    (nothing stubbed): it differs from mcpu08's, it is the same for a byte
    copy of the files elsewhere, and it changes when only mcpu26's manifest
    changes. mcpu08 has no manifest; mcpu26's gives its defaults, so a
    release that changes only those must get a new directory too."""
    keys = {name: params._cache_digest12(name, params.bundled_tables_path(name))
            for name in ("mcpu08", "mcpu26")}
    assert keys["mcpu26"] != keys["mcpu08"]
    real_root = params._bundled_root
    assert params.PARAMSET_MANIFEST in params.shipped_constants("mcpu26")
    assert (real_root("mcpu26") / params.PARAMSET_MANIFEST).is_file()
    assert not (real_root("mcpu08") / params.PARAMSET_MANIFEST).exists()

    copied = tmp_path / "mcpu26"
    shutil.copytree(real_root("mcpu26"), copied)
    monkeypatch.setattr(
        params, "_bundled_root", lambda name: copied if name == "mcpu26" else real_root(name))
    assert params._cache_digest12("mcpu26", copied / "tables.npz") == keys["mcpu26"]
    # Only the sheet factor changes; the tables and constants stay.
    manifest = copied / params.PARAMSET_MANIFEST
    text = manifest.read_text(encoding="utf-8")
    factor = f'"hbond_beta_favor": {json.loads(text)["hbond_beta_favor"]!r}'
    assert text.count(factor) == 1
    manifest.write_text(text.replace(factor, '"hbond_beta_favor": 3.0'), encoding="utf-8")
    assert params._cache_digest12("mcpu26", copied / "tables.npz") != keys["mcpu26"]
    assert params._cache_digest12("mcpu08", params.bundled_tables_path("mcpu08")) == keys["mcpu08"]
