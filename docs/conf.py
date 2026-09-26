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

html_theme = "furo"
# Furo has no navbar and renders the page TOC in the right sidebar by default,
# so the pydata-era "navbar_end"/"secondary_sidebar_items" knobs have no
# equivalent and are simply dropped. The source_* trio is what turns on Furo's
# "Edit this page" link; footer_icons replaces the navbar GitHub icon.
html_theme_options = {
    "source_repository": "https://github.com/kibumpark-chem/pyMCPU",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/kibumpark-chem/pyMCPU",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
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
