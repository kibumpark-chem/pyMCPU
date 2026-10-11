"""Tests for ``pymcpu.params``'s parameter-directory resolution logic:
``ensure_params``, ``get_cache_dir``.

Env vars and monkeypatches force each resolution branch here: the cache-dir
override, the in-wheel archive and the nothing-available failure. The
``MCPU_PARAMS_DIR`` override and the incomplete-directory failure are checked
in ``test_params_manifest_single_source.py``, on a tree built from the
registry.

The required filename list is read from the registry via
``pymcpu.params.required_files()`` rather than mirrored by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu.params import ParamsError, ensure_params, get_cache_dir, required_files

# Registry-derived, so this can never disagree with what the loader demands.
_REQUIRED_PARAM_FILES = tuple(required_files("mcpu08").values())


def test_mcpu_cache_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    cache = get_cache_dir()
    # get_cache_dir() appends a "params" subdirectory under the override root
    assert cache == (tmp_path / "cache" / "params").resolve()
    assert cache.is_dir()  # get_cache_dir() creates the directory eagerly


def test_resolves_via_the_in_wheel_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh install resolves parameters offline with no environment variables.

    This is the point of shipping the compact archive.
    """
    import pymcpu.params as params_mod

    if params_mod.bundled_tables_path("mcpu08") is None:
        pytest.skip("no in-wheel archive (run scripts/encode_params.py)")

    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    # Force a miss on the dev-tree step so step 3 is what answers.
    monkeypatch.setattr(params_mod, "_resolve_local_source", lambda entry: None)

    root = ensure_params("mcpu08")
    for rel in _REQUIRED_PARAM_FILES:
        assert (root / rel).is_file(), f"materialized root missing {rel}"


def test_fails_clearly_when_nothing_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        ensure_params("mcpu08")
    msg = str(excinfo.value)
    assert "MCPU_PARAMS_DIR" in msg  # _complete_error's boilerplate lists this as an option
    assert "materialize-params" in msg
