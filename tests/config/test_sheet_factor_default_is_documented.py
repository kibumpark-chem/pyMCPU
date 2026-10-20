"""Every page that states the default hydrogen-bond sheet factor says it is
the parameter set's own first.

``MCPUForceField(hbond_beta_favor=None)``, the default, takes the factor from
the parameter set's ``constants/paramset.json``, and 2.0 only when the set
gives none, as for mcpu08. Pages that said "2.0 by default" were right for
mcpu08 alone, and mislead for any set whose manifest gives a factor. A page
about mcpu08 alone (``docs/hbond_legacy_parity.md``) is not checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

#: The pages that state the force field's default sheet factor.
PAGES = [
    "README.md",
    "docs/physics_notes/hbond_directional.md",
    "docs/api/forcefield.rst",
    "docs/checkpointing.md",
]


def _stated_defaults(text: str) -> list[str]:
    """The paragraphs that speak of the sheet factor and give a default of 2."""
    return [
        paragraph
        for paragraph in re.split(r"\n\s*\n", text)
        if ("sheet factor" in paragraph or "hbond_beta_favor" in paragraph)
        and re.search(r"\bdefault", paragraph, re.IGNORECASE)
        and re.search(r"\b2(\.0)?\b", paragraph)
    ]


@pytest.mark.parametrize("page", PAGES)
def test_a_stated_default_names_the_sets_own_first(page: str) -> None:
    paragraphs = _stated_defaults((REPO / page).read_text(encoding="utf-8"))
    assert paragraphs, f"{page} no longer states the default sheet factor; drop it from PAGES"
    unqualified = [p for p in paragraphs if "paramset.json" not in p and "set's own" not in p]
    assert not unqualified, (
        f"{page} gives 2.0 as the default sheet factor without saying that a parameter "
        "set's own (its constants/paramset.json) comes first:\n\n" + "\n\n".join(unqualified)
    )
