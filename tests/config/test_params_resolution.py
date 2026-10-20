"""Tests for ``pymcpu.params``'s parameter-directory resolution logic:
``ensure_params``, ``get_cache_dir``.

Env vars and monkeypatches force each resolution branch here: the cache-dir
override, the in-wheel archive and the nothing-available failure. The
``MCPU_PARAMS_DIR`` override and the incomplete-directory failure are checked
in ``test_params_manifest_single_source.py``, on a tree built from the
registry.

The required filename list is read from the registry via
``pymcpu.params.required_files()`` rather than mirrored by hand, for every
registered set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu.params import (
    ParamsError,
    available_sets,
    ensure_params,
    get_cache_dir,
    required_files,
    table_layout,
)

SETS = available_sets()


def test_mcpu_cache_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    cache = get_cache_dir()
    # get_cache_dir() appends a "params" subdirectory under the override root
    assert cache == (tmp_path / "cache" / "params").resolve()
    assert cache.is_dir()  # get_cache_dir() creates the directory eagerly


@pytest.mark.parametrize("set_name", SETS)
def test_resolves_via_the_in_wheel_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, set_name: str
) -> None:
    """A fresh install resolves parameters offline with no environment variables.

    This is the point of shipping the compact archive.
    """
    import pymcpu.params as params_mod

    if params_mod.bundled_tables_path(set_name) is None:
        pytest.skip("no in-wheel archive (run scripts/encode_params.py)")

    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    # Force a miss on the dev-tree step so step 3 is what answers.
    monkeypatch.setattr(params_mod, "_resolve_local_source", lambda entry: None)

    root = ensure_params(set_name)
    assert root.name.startswith(f"{set_name}-")
    # Registry-derived, so this can never disagree with what the loader demands.
    for rel in required_files(set_name).values():
        assert (root / rel).is_file(), f"materialized root missing {rel}"


@pytest.mark.parametrize("set_name", SETS)
def test_fails_clearly_when_nothing_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, set_name: str
) -> None:
    """With no env override, no dev tree AND no in-wheel archive, the failure
    must still name the ways out."""
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "empty_cache"))
    import pymcpu.params as params_mod

    monkeypatch.setattr(params_mod, "_resolve_local_source", lambda entry: None)
    # ...and no shipped archive either, which is the only remaining source.
    monkeypatch.setattr(params_mod, "bundled_tables_path", lambda set_name=None: None)
    with pytest.raises(ParamsError) as excinfo:
        ensure_params(set_name)
    msg = str(excinfo.value)
    assert "MCPU_PARAMS_DIR" in msg  # _complete_error's boilerplate lists this as an option
    assert f"materialize-params --set {set_name}" in msg
    # The folder it asks for is laid out as this set: its own table folder.
    (table_folder,) = {rel.split("/")[0] for rel in table_layout(set_name).values()}
    assert f"laid out as {set_name!r}, containing constants/ and {table_folder}/" in msg
