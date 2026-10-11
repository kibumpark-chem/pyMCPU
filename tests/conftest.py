"""Root pytest configuration: the import-path fix and fixture re-exports.

The actual fixture/builder implementations live in
``tests/fixtures/context_builders.py`` (physics context construction) and
``tests/integration/conftest.py`` (MPI/checkpoint-only fixtures, scoped to
the integration subtree that needs them). This file exists because pytest
only auto-discovers fixtures declared in a ``conftest.py`` -- importing them
here makes them available to every test under ``tests/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scikit-build-core's editable install registers meta_path finders that
# redirect `import pymcpu` into the CMake build tree, which can hold a stale
# or mismatched build. Strip them and put the repo root first, so tests
# import the extension that sits next to the sources. conftest.py is always
# collected before any test module, so this runs before the first
# `import pymcpu`.
sys.meta_path[:] = [
    f
    for f in sys.meta_path
    if type(f).__name__ not in ("ScikitBuildRedirectingFinder", "ScikitBuildInplaceFinder")
]
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.fixtures.engine_specs import engine_spec_factory  # noqa: F401,E402
from tests.fixtures.context_builders import (  # noqa: F401,E402
    chignolin_context,
    chignolin_pdb_path,
    chignolin_with_qbias,
    minimal_pdb_path,
)
