"""A parameter folder's signature identifies the files a force field reads.

``pymcpu.params.param_signature`` is a sha256 over the files the set's layout
names (and the set's own description file, ``constants/paramset.json``, when
there is one), each by its relative path and contents. Checkpoints record it,
so a run cannot resume with other tables; the resume checks are in
``tests/integration/checkpointing/test_param_set_resume.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mdtraj as md
import pytest

from pymcpu import params
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.params import (
    PARAMSET_MANIFEST,
    constants_files,
    ensure_params,
    optional_files,
    param_signature,
    required_files,
    signed_files,
    table_layout,
)
from tests.helpers.param_folders import linked_param_folder


def _replace(path: Path, data: bytes) -> None:
    """Put a file of its own at ``path``, which may be a link."""
    path.unlink()
    path.write_bytes(data)


def test_the_signature_covers_the_layout_and_the_set_description() -> None:
    expected = (
        set(required_files("mcpu08").values())
        | set(optional_files("mcpu08").values())
        | set(table_layout("mcpu08").values())
        | set(constants_files("mcpu08"))
        | {PARAMSET_MANIFEST}
    )
    assert signed_files("mcpu08") == sorted(expected)
    assert PARAMSET_MANIFEST == "constants/paramset.json"


def test_the_same_files_elsewhere_have_the_same_signature(tmp_path: Path) -> None:
    shipped = param_signature(ensure_params("mcpu08"))
    assert len(shipped) == 64 and int(shipped, 16) >= 0
    assert param_signature(linked_param_folder(tmp_path / "linked")) == shipped
    # Files the layout does not name do not count.
    (tmp_path / "linked" / "README.txt").write_text("notes")
    (tmp_path / "linked" / "mcpu_params" / "old_table.bin").write_bytes(b"\0" * 8)
    assert param_signature(tmp_path / "linked") == shipped


@pytest.mark.parametrize("change", ["table", "added description", "removed optional file"])
def test_changing_adding_or_removing_a_file_changes_it(tmp_path: Path, change: str) -> None:
    folder = linked_param_folder(tmp_path / "f")
    before = param_signature(folder)
    if change == "table":
        aromatic = folder / "mcpu_params" / "aromatic_potentials.bin"
        data = bytearray(aromatic.read_bytes())
        data[0] ^= 1
        _replace(aromatic, bytes(data))
    elif change == "added description":
        (folder / PARAMSET_MANIFEST).write_text('{"name": "test"}')
    else:
        (folder / "constants" / "rama_mixture.json").unlink()
    assert param_signature(folder) != before


def test_each_file_is_read_once_until_it_changes(tmp_path: Path, monkeypatch) -> None:
    """A process that builds many runs from one folder hashes it once; a file
    rewritten since is read again."""
    reads: list[str] = []
    real_digest = hashlib.file_digest

    def counting(handle, name):
        reads.append(Path(handle.name).name)
        return real_digest(handle, name)

    monkeypatch.setattr(params.hashlib, "file_digest", counting)
    folder = tmp_path / "small"
    (folder / "constants").mkdir(parents=True)
    (folder / "constants" / "atom_types.csv").write_text("a,b\n")
    (folder / "constants" / "standard_amino_acids.json").write_text("{}")

    first = param_signature(folder)
    assert sorted(reads) == ["atom_types.csv", "standard_amino_acids.json"]
    assert param_signature(folder) == first
    assert len(reads) == 2
    (folder / "constants" / "atom_types.csv").write_text("a,b,c\n")
    assert param_signature(folder) != first
    assert reads[2:] == ["atom_types.csv"]


def test_the_force_field_takes_its_signature_when_asked(chignolin_pdb_path: str) -> None:
    traj = md.load(chignolin_pdb_path)
    ff = MCPUForceField(traj.atom_slice(traj.topology.select("not element H")))
    assert ff._param_signature is None  # building it does not read the files again
    assert ff.param_signature == param_signature(ff.param_dir, "mcpu08")
    assert ff._param_signature == ff.param_signature
