"""Every pyMCPU name used in a documentation code block must exist.

Why this is a separate test from ``test_autodoc_targets_exist``: Sphinx does
not validate the *contents* of a ``code-block``. ``docs/quickstart.rst`` passed
``sphinx-build -W`` with a clean build while every single line of its main
example was fictional -- ``mc.System.from_pdb``, ``mc.MuContactForce()``,
``mc.MCIntegrator(...)``, ``mc.DCDReporter(...)``, ``ctx.minimize_energy(...)``,
``mc.get_native_contacts(...)``, ``mc.compute_q_value(...)``. None of those
ever existed. A reader copying that block got an ``AttributeError`` on line
one, and no tool in the repo objected.

So this test reads the code blocks, parses them, and resolves every attribute
accessed on a pyMCPU alias against the real package.

Scope and deliberate limits:

- It checks NAMES, not behaviour. Snippets are often illustrative (a path that
  does not exist, a variable defined in an earlier block), so executing them is
  not practical. Names are the part that goes stale on a rename.
- It only follows aliases bound to pyMCPU itself (``import pymcpu as mc``,
  ``from pymcpu import ...``, ``pymcpu.mcpu_core``). Calls on instances
  (``sim.step``, ``ctx.set_positions``) are not resolved, because the static
  type of a local is not knowable here -- ``test_autodoc_targets_exist`` and
  the suite cover the class surfaces instead.
"""

from __future__ import annotations

import ast
import importlib
import re
import textwrap
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent.parent / "docs"

#: ``.. code-block:: python`` followed by an indented body.
_RST_BLOCK = re.compile(
    r"\.\.\s+code-block::\s*(?:python|py|python3)\s*\n"
    r"(?:\s+:[a-z-]+:.*\n)*"          # directive options, e.g. :linenos:
    r"\n"
    r"((?:(?:[ \t]+.*)?\n)+)"
)
#: MyST / Markdown fenced python blocks.
_MD_BLOCK = re.compile(r"```(?:python|py|python3)\n(.*?)```", re.DOTALL)

#: Names that are pyMCPU itself when used as an attribute base.
_PYMCPU_ALIASES = {"pymcpu", "mc", "mcpu_core"}


def _iter_blocks() -> list[tuple[str, int, str]]:
    """Yield ``(relpath, block_index, source)`` for every python block."""
    out: list[tuple[str, int, str]] = []
    for path in sorted(DOCS.rglob("*.rst")) + sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(DOCS.parent))
        blocks = [m.group(1) for m in _RST_BLOCK.finditer(text)]
        blocks += [m.group(1) for m in _MD_BLOCK.finditer(text)]
        for i, body in enumerate(blocks):
            out.append((rel, i, textwrap.dedent(body)))
    return out


def _pymcpu_attribute_chains(source: str) -> set[str]:
    """Dotted chains rooted at a pyMCPU alias, e.g. ``mc.Integrator``.

    Also follows ``from pymcpu import X`` / ``from pymcpu.sampling import Y``
    so bare uses of an imported name are checked against its real module.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Illustrative fragments (``...``, elided bodies) are not our problem.
        return set()

    found: set[str] = set()

    for node in ast.walk(tree):
        # from pymcpu[.sub] import A, B
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "pymcpu" or node.module.startswith("pymcpu."):
                for alias in node.names:
                    if alias.name != "*":
                        found.add(f"{node.module}.{alias.name}")
            continue

        # mc.Integrator / pymcpu.mcpu_core.System / mc.analysis.open_writer
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            cur: ast.expr = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id in _PYMCPU_ALIASES:
                root = "pymcpu" if cur.id in {"pymcpu", "mc"} else "pymcpu.mcpu_core"
                found.add(".".join([root, *reversed(parts)]))

    return found


def _resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001 -- not a module prefix; try shorter
            continue
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


BLOCKS = _iter_blocks()

#: (relpath, dotted_name) for every pyMCPU name used in a doc code block.
REFERENCES = sorted(
    {
        (rel, name)
        for rel, _idx, src in BLOCKS
        for name in _pymcpu_attribute_chains(src)
    }
)


def test_docs_contain_python_code_blocks() -> None:
    """Guard the guard: an empty parametrization would pass silently."""
    assert BLOCKS, f"no python code blocks found under {DOCS} -- regex stale?"
    assert REFERENCES, "no pyMCPU API references found in any code block"


@pytest.mark.parametrize(("relpath", "name"), REFERENCES, ids=[f"{r[1]}" for r in REFERENCES])
def test_doc_code_block_name_exists(relpath: str, name: str) -> None:
    assert _resolves(name), (
        f"{relpath} contains a code block using `{name}`, which does not "
        f"exist. Sphinx does not check code-block contents, so a reader "
        f"copying that block gets an AttributeError. Fix the snippet."
    )
