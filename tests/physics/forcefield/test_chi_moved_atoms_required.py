"""A residue template that names chi atoms must name the atoms they move.

``chi_atoms`` holds the four atoms that define each chi angle, and
``chi_moved_atoms`` the atoms that turn with it, one row per chi. The
side-chain moves turn only the atoms in those rows. A template without them
used to load: every chi got the moved range [-1, -1], so no side chain ever
moved, and the energies, which do not read the rows, looked normal. A
research parameter set did this in September 2026. ``MCPUForceField`` now
refuses such a template when it reads it, and names the file and the
residues.

Each case edits a copy of mcpu08's template in a folder that links mcpu08's
other files. ``test_chi_moved_atoms_topology.py`` checks what the rows hold;
this file checks that a template without them cannot be used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import mdtraj as md
import pytest

from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import resolve_test_pdb
from tests.helpers.param_folders import linked_param_folder

TEMPLATE = "constants/standard_amino_acids.json"

Edit = Callable[[dict], None]


def _heavy(pdb: Path) -> md.Trajectory:
    traj = md.load(str(pdb))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def chignolin() -> md.Trajectory:
    return _heavy(default_example_pdb())


@pytest.fixture(scope="module")
def actin() -> md.Trajectory:
    """Actin, which holds all 20 residues (chignolin if it is not there)."""
    return _heavy(resolve_test_pdb())


@pytest.fixture(scope="module")
def shipped(actin: md.Trajectory) -> MCPUForceField:
    return MCPUForceField(actin)


def _edited_folder(root: Path, edit: Edit) -> Path:
    """mcpu08's files linked into ``root``, with the template replaced by a
    copy that ``edit`` has changed."""
    linked_param_folder(root)
    path = root / TEMPLATE
    template = json.loads(path.read_text())
    edit(template)
    path.unlink()
    path.write_text(json.dumps(template))
    return root


def _drop_key(*residues: str) -> Edit:
    """Remove ``chi_moved_atoms`` from these residues, or from all of them."""
    def edit(template: dict) -> None:
        for name in residues or tuple(template):
            template[name].pop("chi_moved_atoms", None)
    return edit


def _set_rows(residue: str, change: Callable[[list], object]) -> Edit:
    """Replace one residue's ``chi_moved_atoms`` with ``change(rows)``."""
    def edit(template: dict) -> None:
        template[residue]["chi_moved_atoms"] = change(template[residue]["chi_moved_atoms"])
    return edit


def test_a_template_without_the_key_is_refused(tmp_path: Path, chignolin: md.Trajectory) -> None:
    """The September 2026 case: the key gone from every residue. All 18
    residues with chi angles are named; GLY and ALA have none, so not them."""
    folder = _edited_folder(tmp_path / "params", _drop_key())
    with pytest.raises(ValueError) as caught:
        MCPUForceField(chignolin, param_dir=str(folder))
    message = str(caught.value)
    assert message.startswith(f"{folder / TEMPLATE}: ")
    assert "'chi_moved_atoms'" in message
    named = re.findall(r"\b([A-Z]{3}) has none\b", message)
    template = json.loads((folder / TEMPLATE).read_text())
    assert named == sorted(name for name, entry in template.items() if entry.get("chi_atoms"))
    assert len(named) == 18 and "GLY" not in named and "ALA" not in named
    assert "18 do not" in message


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        # Chignolin has no lysine: the template is checked, not the structure.
        (_drop_key("LYS"), "LYS has none"),
        (_set_rows("ARG", lambda rows: rows[:2]), "ARG has 2 rows for 4 chi"),
        (_set_rows("SER", lambda rows: rows + rows), "SER has 2 rows for 1 chi"),
        (_set_rows("SER", lambda rows: [[]]), "SER has no atoms for chi1"),
        (_set_rows("LYS", lambda rows: [rows[0], [], rows[2], []]),
         "LYS has no atoms for chi2, chi4"),
        (_set_rows("MET", lambda rows: "CG SD CE"), "MET has a str, not a list of rows"),
    ],
    ids=["key-missing", "too-few-rows", "too-many-rows", "empty-row", "two-empty-rows",
         "not-a-list"],
)
def test_one_bad_residue_is_refused_and_named(
    tmp_path: Path, chignolin: md.Trajectory, edit: Edit, expected: str
) -> None:
    folder = _edited_folder(tmp_path / "params", edit)
    with pytest.raises(ValueError) as caught:
        MCPUForceField(chignolin, param_dir=str(folder))
    message = str(caught.value)
    assert message.startswith(f"{folder / TEMPLATE}: ")
    assert f"1 do not: {expected}. " in message


def test_an_entry_that_is_not_a_residue_is_left_alone(
    tmp_path: Path, chignolin: md.Trajectory
) -> None:
    """A note in the template, which loaded before the check, still loads."""
    folder = _edited_folder(tmp_path / "params", lambda t: t.update(_comment="written by hand"))
    MCPUForceField(chignolin, param_dir=str(folder))


def test_mcpu08_loads_and_every_chi_moves(shipped: MCPUForceField) -> None:
    """mcpu08's template loads as it is, and every chi angle of the test
    structure (actin: every residue in the template) has atoms to turn."""
    on_disk = json.loads((shipped.param_dir / TEMPLATE).read_text())
    assert shipped.ff_template == on_disk
    resolved = [[row[0] >= 0 for row in rows] for rows in shipped.chi_atom_indices]
    live = [[lo >= 0 for lo, _hi in ranges] for ranges in shipped.chi_moved_atom_ranges]
    assert sum(map(sum, resolved)) > 0
    assert live == resolved


def test_an_unedited_copy_loads_the_same(
    tmp_path: Path, actin: md.Trajectory, shipped: MCPUForceField
) -> None:
    """The copy the refusal cases edit, left unedited, loads with the same
    chi tables as mcpu08's own folder: the refusals come from the edits."""
    folder = _edited_folder(tmp_path / "params", lambda t: None)
    copied = MCPUForceField(actin, param_dir=str(folder))
    assert copied.chi_atom_indices == shipped.chi_atom_indices
    assert copied.chi_moved_atom_ranges == shipped.chi_moved_atom_ranges
