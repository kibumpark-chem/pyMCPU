"""Root pytest configuration: fixture re-exports only.

The actual fixture/builder implementations live in
``tests/fixtures/context_builders.py`` (physics context construction) and
``tests/integration/conftest.py`` (MPI/checkpoint-only fixtures, scoped to
the integration subtree that needs them). This file exists because pytest
only auto-discovers fixtures declared in a ``conftest.py`` -- importing them
here makes them available to every test under ``tests/``.
"""

from __future__ import annotations

from tests.helpers.import_fixup import ensure_repo_importable

# Must run before any test module's own `import pymcpu` -- conftest.py is
# always collected first, so this is the one place this needs to happen.
ensure_repo_importable()

from tests.fixtures.engine_specs import engine_spec_factory  # noqa: F401,E402
from tests.fixtures.context_builders import (  # noqa: F401,E402
    ATOL,
    chignolin_context,
    chignolin_pdb_path,
    chignolin_with_qbias,
    minimal_pdb_path,
    require_safe_math_for_accept_determinism,
)
