"""Tests for ``pymcpu.params``'s parameter-directory resolution logic:
``ensure_params``, ``get_cache_dir``.

No network: a local stub params tree is built per-test and env vars are used
to force each resolution branch (``MCPU_PARAMS_DIR`` override, cache-dir
override, no-download failure, incomplete directory failure).

The required filename list is READ FROM the registry via
``pymcpu.params.required_files()`` rather than mirrored by hand. The previous
hand-copied tuple had already drifted -- it omitted
``mcpu_params/hbond_seq_dep.bin``, which the force field requires -- so the
stub described a "complete" tree that the real loader would reject.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu.params import ParamsError, ensure_params, get_cache_dir, required_files

_SC_TRIPLET = "mcpu_params/sidechain_triplet_potentials.bin"
# Registry-derived, so this can never disagree with what the loader demands.
_REQUIRED_PARAM_FILES = tuple(required_files("mcpu08").values())

# A couple of files need content that parses; the rest only need to exist.
_STUB_CONTENT = {
    "constants/atom_types.csv": b"residue,atom,type,radius\n",
    "constants/standard_amino_acids.json": b"{}",
}


def _stub_params_download_tree(root: Path, *, with_sc: bool = True) -> None:
    """Build a minimal on-disk tree satisfying ``_looks_like_params_root``.

    ``with_sc=False`` deliberately omits the large SC-triplet table to produce
    an *incomplete* tree for the failure-path tests.
    """
    for rel in _REQUIRED_PARAM_FILES:
        if not with_sc and rel == _SC_TRIPLET:
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(_STUB_CONTENT.get(rel, b"x"))


def test_mcpu_params_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_params_download_tree(tmp_path)
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(tmp_path))
    monkeypatch.delenv("MCPU_PARAMS_BUNDLE", raising=False)
    root = ensure_params("mcpu08")
    assert root == tmp_path.resolve()  # MCPU_PARAMS_DIR is used as-is (resolution step 1)
    assert (root / "constants" / "atom_types.csv").is_file()


def test_mcpu_cache_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    cache = get_cache_dir()
    # get_cache_dir() appends a "params" subdirectory under the override root
    assert cache == (tmp_path / "cache" / "params").resolve()
    assert cache.is_dir()  # get_cache_dir() creates the directory eagerly


def test_no_download_succeeds_via_the_in_wheel_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``MCPU_NO_DOWNLOAD=1`` must now SUCCEED, because nothing needs downloading.

    This is the point of shipping the compact archive: a fresh install resolves
    parameters offline with no environment variables set. Before the archive
    existed this same configuration raised ``ParamsError``.
    """
    import pymcpu.params as params_mod

    if params_mod.bundled_tables_path("mcpu08") is None:
        pytest.skip("no in-wheel archive (run scripts/encode_params.py)")

    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    monkeypatch.delenv("MCPU_PARAMS_BUNDLE", raising=False)
    monkeypatch.setenv("MCPU_NO_DOWNLOAD", "1")
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    # Force a miss on the dev-tree step so step 5 is what answers.
    monkeypatch.setattr(params_mod, "_resolve_local_source", lambda entry: None)

    root = ensure_params("mcpu08")
    for rel in _REQUIRED_PARAM_FILES:
        assert (root / rel).is_file(), f"materialized root missing {rel}"


def test_no_download_fails_clearly_when_nothing_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no env override, no dev tree, no cache AND no in-wheel archive, the
    failure must still name the ways out."""
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    monkeypatch.delenv("MCPU_PARAMS_BUNDLE", raising=False)
    monkeypatch.setenv("MCPU_NO_DOWNLOAD", "1")
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "empty_cache"))
    import pymcpu.params as params_mod

    monkeypatch.setattr(params_mod, "_resolve_local_source", lambda entry: None)
    # ...and no shipped archive either, which is the only remaining source.
    monkeypatch.setattr(params_mod, "bundled_tables_path", lambda set_name=None: None)
    with pytest.raises(ParamsError) as excinfo:
        ensure_params("mcpu08")
    msg = str(excinfo.value)
    assert "MCPU_PARAMS_DIR" in msg  # _complete_error's boilerplate lists this as an option
    assert "MCPU_NO_DOWNLOAD" in msg or "forbids download" in msg


def test_incomplete_params_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "constants").mkdir()  # present but empty -- missing all required files
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(tmp_path))
    with pytest.raises(ParamsError) as excinfo:
        ensure_params("mcpu08")
    msg = str(excinfo.value)
    assert "Missing" in msg or "incomplete" in msg.lower()
