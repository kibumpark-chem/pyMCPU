"""The parameter cache directory changes when a constants file changes.

The directory name used to hash only the table archive, so a release that
changed only a constants file (``rama_mixture.json``, ``bbind02.May.lib``,
...) kept serving the old copy from the cache. The package's own files are
swapped for small stand-ins here, and the table decode is skipped, so the
test unpacks nothing of the real tables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu import params


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


def test_same_files_reuse_the_directory(fake_set: Path) -> None:
    first = params.materialize_from_wheel("mcpu08", verify=False)
    assert params.materialize_from_wheel("mcpu08", verify=False) == first
    assert first.name.startswith("mcpu08-")


@pytest.mark.parametrize("changed", ["constants/a.json", "constants/b.lib", "tables.npz"])
def test_any_changed_file_gets_a_new_directory(fake_set: Path, changed: str) -> None:
    first = params.materialize_from_wheel("mcpu08", verify=False)

    (fake_set / changed).write_bytes(b"version 2")
    second = params.materialize_from_wheel("mcpu08", verify=False)

    assert second != first
    if changed.startswith("constants/"):
        assert (second / changed).read_bytes() == b"version 2"
        assert (first / changed).read_bytes() != b"version 2"
