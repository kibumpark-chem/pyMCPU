"""Import-path workaround shared by tests that import the compiled pymcpu extension.

scikit-build-core's editable install registers `ScikitBuildRedirectingFinder` /
`ScikitBuildInplaceFinder` meta_path finders that intercept `import pymcpu` and
redirect it into the CMake build tree instead of the already-built extension
sitting next to the repo. When tests are collected/run directly (e.g. by path,
outside of the normal package entry point), that redirect can resolve to a
stale or mismatched build directory rather than the extension the test actually
wants. Stripping the finders and inserting the repo root at the front of
sys.path forces plain filesystem-based imports of pymcpu instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_importable() -> Path:
    """Strip scikit-build's redirecting finders and make the repo root importable.

    Returns the resolved repo root (mirrors the `ROOT` local previously defined
    at each call site).
    """
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if type(f).__name__
        not in ("ScikitBuildRedirectingFinder", "ScikitBuildInplaceFinder")
    ]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    return _REPO_ROOT
