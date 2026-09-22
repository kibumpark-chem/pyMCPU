"""Installed metadata must agree with the source and with the conda recipe.

The version used to live in three places -- ``pyproject.toml``,
``pymcpu/__init__.py``, and ``conda-recipe/meta.yaml``. Two are now one:
``pyproject.toml`` declares ``dynamic = ["version"]`` and reads
``pymcpu/__init__.py`` through scikit-build-core's regex provider. The recipe
stays a separate copy on purpose -- a conda recipe is built from a released
sdist and cannot import the package -- so it is checked here instead.
"""

from __future__ import annotations

import re
from importlib.metadata import metadata, version
from pathlib import Path

import pytest

import pymcpu

REPO = Path(__file__).resolve().parent.parent.parent


def test_installed_version_matches_source() -> None:
    assert version("pymcpu") == pymcpu.__version__


def test_pyproject_does_not_pin_a_second_version() -> None:
    """A static ``version`` alongside the dynamic provider would shadow it."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    assert not re.search(r"^version\s*=", project, re.MULTILINE), (
        "pyproject.toml [project] sets a static `version`; it should declare "
        'dynamic = ["version"] and let the regex provider read '
        "pymcpu/__init__.py"
    )


def test_conda_recipe_version_matches() -> None:
    recipe = REPO / "conda-recipe" / "meta.yaml"
    if not recipe.exists():
        pytest.skip("no conda recipe in this tree")
    match = re.search(r'{%\s*set\s+version\s*=\s*"([^"]+)"\s*%}', recipe.read_text())
    assert match, "could not find the version jinja variable in meta.yaml"
    assert match.group(1) == pymcpu.__version__


def test_changelog_documents_this_version() -> None:
    """A release with no changelog entry is a release nobody can read."""
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(
        rf"^##\s*\[{re.escape(pymcpu.__version__)}\]", text, re.MULTILINE
    ), f"CHANGELOG.md has no `## [{pymcpu.__version__}]` section"


@pytest.mark.parametrize(
    "field",
    ["Name", "Version", "Summary", "Requires-Python", "Author-Email"],
)
def test_core_metadata_field_is_populated(field: str) -> None:
    value = metadata("pymcpu").get(field)
    assert value and value.strip(), f"{field} is empty in the installed metadata"


def test_license_is_declared() -> None:
    """PEP 639 emits ``License-Expression``, not the legacy ``License``.

    Worth asserting explicitly: ``pip show`` reads the legacy field and
    displays a blank License for a perfectly well-formed PEP 639 package, which
    looks like a bug and is not one.
    """
    md = metadata("pymcpu")
    expression = md.get("License-Expression")
    legacy = md.get("License")
    assert expression or legacy, "neither License-Expression nor License is set"
    if expression:
        assert expression.strip() == "MIT"
        assert not [
            c for c in md.get_all("Classifier") or [] if c.startswith("License ::")
        ], (
            "a License:: classifier alongside a PEP 639 license expression is a "
            "hard error in pyproject_metadata"
        )


def test_project_urls_are_present() -> None:
    urls = metadata("pymcpu").get_all("Project-URL") or []
    labels = {u.split(",", 1)[0].strip() for u in urls}
    missing = {"Homepage", "Documentation", "Repository", "Issues"} - labels
    assert not missing, f"missing Project-URL entries: {sorted(missing)}"


def test_declared_runtime_dependencies_match_pyproject() -> None:
    """Catches a dependency added to the code but not to the metadata."""
    installed = {
        re.split(r"[<>=!;\[ ]", d)[0].lower()
        for d in metadata("pymcpu").get_all("Requires-Dist") or []
        if "extra ==" not in d
    }
    assert {"numpy", "mdtraj", "pandas", "pyyaml"} <= installed, (
        f"core runtime dependencies missing from metadata: {installed}"
    )


def test_citation_cff_exists_and_matches_the_version() -> None:
    """README promised a file that did not exist.

    ``README.md`` has stated "``CITATION.cff`` is included in the repository"
    since before v0.1.0 while no such file was present -- a false claim in the
    document every new user reads first. The file now exists; this test keeps
    the claim true and keeps its version from drifting away from
    ``pymcpu.__version__`` the way an unpinned duplicate always does.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    cff_path = root / "CITATION.cff"
    assert cff_path.is_file(), (
        "CITATION.cff is missing, but README.md tells readers it is included. "
        "Either add the file or remove the claim."
    )
    cff = yaml.safe_load(cff_path.read_text())
    for key in ("cff-version", "message", "title", "authors"):
        assert key in cff, f"CITATION.cff is missing the required key {key!r}"
    assert str(cff["version"]) == pymcpu.__version__, (
        f"CITATION.cff version {cff['version']!r} != "
        f"pymcpu.__version__ {pymcpu.__version__!r}"
    )
    assert "CITATION.cff" in (root / "README.md").read_text()


def test_citation_cff_has_no_placeholder_doi() -> None:
    """A placeholder DOI is worse than none -- it looks resolvable.

    The Zenodo deposition must precede the first tag or the first archived DOI
    belongs to 0.1.1 rather than to the release people cite. Until then the
    key is deliberately absent; if it appears it must be real.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    cff = yaml.safe_load((root / "CITATION.cff").read_text())
    doi = cff.get("doi")
    if doi is not None:
        assert doi.startswith("10."), f"not a DOI: {doi!r}"
        assert "xxxx" not in doi.lower() and "placeholder" not in doi.lower()
