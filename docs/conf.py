import os
import pathlib
import sys

# Anchor to this file, NOT to the CWD: sphinx-build may be invoked from
# anywhere, and os.path.abspath("..") would then resolve somewhere else
# entirely. (The previous value, "../src", was doubly wrong -- src/ is the
# C++ tree; the Python package lives at the repo root.)
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

project = "pyMCPU"
copyright = "2026, Kibum Park"
author = "Kibum Park"

# Single-source the version: prefer installed metadata, fall back to parsing
# pymcpu/__init__.py so the docs build without installing the package (which
# would require compiling the C++ extension -- see autodoc_mock_imports).
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("pymcpu")
except Exception:  # not installed
    import re

    _init = _REPO_ROOT / "pymcpu" / "__init__.py"
    _m = re.search(r'__version__ = "([^"]+)"', _init.read_text(encoding="utf-8"))
    release = _m.group(1) if _m else "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "mdtraj": ("https://mdtraj.org/1.9.7/", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/kibumpark-chem/pyMCPU",
    "navbar_end": ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
}
html_static_path = ["_static"]



exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "**/01_single_trajectory_executed*",
    "requirements.txt",
    "Makefile",
    "make.bat",
]

# nbsphinx: never re-execute notebooks at build time
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# Mock the compiled extension and its heavy runtime deps so the docs build
# without a compiler. Unconditional on purpose: a local `make html` should
# behave the same way Read the Docs does, and RTD cannot compile the C++20
# extension. `pymcpu/__init__.py` loads mcpu_core AT IMPORT TIME and then
# imports MCPUForceField (which pulls mdtraj + pandas), so mocking mcpu_core
# alone is not sufficient.
#
# Set PYMCPU_DOCS_NO_MOCK=1 to build against a real installed extension, which
# renders the pybind11 docstrings that mocking necessarily hides.
if os.environ.get("PYMCPU_DOCS_NO_MOCK") != "1":
    autodoc_mock_imports = [
        "pymcpu.mcpu_core",
        "mdtraj",
        "pandas",
        "yaml",
        "h5py",
        "mpi4py",
        "scipy",
        "westpa",
        "matplotlib",
    ]
