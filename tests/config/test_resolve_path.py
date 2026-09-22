"""Regression test for ``resolve_path``'s repo-root fallback.

``resolve_path``'s ``repo_root`` parameter used to shadow the module-level
``repo_root()`` function of the same name, so its "resolve relative to CWD
first, then the repo root" fallback branch called ``None()`` and crashed
with ``TypeError`` any time a relative path wasn't found under CWD --
i.e. this fallback had never worked. Every existing call site in the repo
happens to use absolute paths or CWD-resolvable ones, so it went unnoticed.
"""

from __future__ import annotations

from pathlib import Path

from pymcpu.config import repo_root, resolve_path


def test_relative_path_not_under_cwd_falls_back_to_repo_root() -> None:
    result = resolve_path("this/does/not/exist/anywhere.pdb")
    assert result == repo_root() / "this/does/not/exist/anywhere.pdb"


def test_explicit_repo_root_override_still_honored() -> None:
    result = resolve_path("foo.pdb", repo_root="/tmp/custom_root")
    assert result == Path("/tmp/custom_root/foo.pdb")


def test_absolute_path_returned_unchanged() -> None:
    result = resolve_path("/already/absolute/path.pdb")
    assert str(result) == "/already/absolute/path.pdb"
