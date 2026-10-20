"""A parameter folder's signature identifies the files a force field reads.

``pymcpu.params.param_signature`` is a sha256 over the files the set's layout
names (and the set's own description file, ``constants/paramset.json``, when
there is one), each by its relative path and contents. Checkpoints record it,
so a run cannot resume with other tables; the resume checks are in
``tests/integration/checkpointing/test_param_set_resume.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import mdtraj as md
import pytest

from pymcpu import params
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.params import (
    PARAMSET_FORMAT,
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


def test_changing_the_set_description_changes_it(tmp_path: Path, chignolin_pdb_path: str) -> None:
    """The force field reads constants/paramset.json (its hbond_beta_favor,
    for one), so the signature follows its contents."""
    traj = md.load(chignolin_pdb_path)
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    folder = linked_param_folder(tmp_path / "f")
    manifest = folder / PARAMSET_MANIFEST
    signatures = []
    # Each with a size of its own: the digest cache keys on size and mtime,
    # and two writes can fall within one clock tick.
    for factor in (3.0, 2.25):
        if manifest.exists():
            manifest.unlink()
        manifest.write_text(json.dumps(
            {"format": PARAMSET_FORMAT, "name": "mcpu08", "hbond_beta_favor": factor}))
        ff = MCPUForceField(heavy, param_dir=str(folder))
        assert ff.hbond_beta_favor == factor
        assert ff.param_signature == param_signature(folder)
        signatures.append(ff.param_signature)
    assert signatures[0] != signatures[1]
    assert param_signature(ensure_params("mcpu08")) not in signatures


def _count_reads(monkeypatch, tmp_path: Path) -> list[str]:
    """The name of each parameter file read in full from now on, with a
    digest store of the test's own (``MCPU_CACHE_DIR``)."""
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    reads: list[str] = []
    hash_file = params._hash_file
    monkeypatch.setattr(params, "_hash_file",
                        lambda path: reads.append(Path(path).name) or hash_file(path))
    return reads


def _small_folder(root: Path) -> Path:
    (root / "constants").mkdir(parents=True)
    (root / "constants" / "atom_types.csv").write_text("a,b\n")
    (root / "constants" / "standard_amino_acids.json").write_text("{}")
    return root


def test_each_file_is_read_once_until_it_changes(tmp_path: Path, monkeypatch) -> None:
    """A process that builds many runs from one folder hashes it once; a file
    rewritten since is read again."""
    reads = _count_reads(monkeypatch, tmp_path)
    folder = _small_folder(tmp_path / "small")

    first = param_signature(folder)
    assert sorted(reads) == ["atom_types.csv", "standard_amino_acids.json"]
    assert param_signature(folder) == first
    assert len(reads) == 2
    (folder / "constants" / "atom_types.csv").write_text("a,b,c\n")
    assert param_signature(folder) != first
    assert reads[2:] == ["atom_types.csv"]


def test_a_file_rewritten_with_its_size_and_mtime_restored_is_read_again(
    tmp_path: Path, monkeypatch
) -> None:
    """``cp -p`` or ``rsync --inplace -t`` onto a file of the same size keeps
    its inode, size and modification time, but not its change time."""
    reads = _count_reads(monkeypatch, tmp_path)
    folder = _small_folder(tmp_path / "small")
    table = folder / "constants" / "atom_types.csv"
    before = table.stat()
    first = param_signature(folder)
    time.sleep(0.05)  # past the filesystem's timestamp tick
    with table.open("r+b") as handle:  # the same inode
        handle.write(b"x,y\n")
    os.utime(table, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = table.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns)
    assert param_signature(folder) != first
    assert reads.count("atom_types.csv") == 2


def test_another_process_takes_the_digests_from_the_store(tmp_path: Path, monkeypatch) -> None:
    """The digests of each file read are kept under the cache root, by its
    real path, inode, size and times, so another process (here: this one
    with its own memory cleared) does not read the file again while it is
    unchanged. Another cache root, a changed file and a damaged store mean
    a new read."""
    reads = _count_reads(monkeypatch, tmp_path)
    folder = _small_folder(tmp_path / "small")
    first = param_signature(folder)
    assert len(reads) == 2
    store = json.loads((tmp_path / "cache" / "params" / params.DIGEST_STORE).read_text())
    table = os.path.realpath(folder / "constants" / "atom_types.csv")
    entry = store["files"][table]
    assert entry["md5"] == hashlib.md5(b"a,b\n").hexdigest()
    assert entry["sha256"] == hashlib.sha256(b"a,b\n").hexdigest()
    assert entry["size"] == 4 and entry["inode"] == os.stat(table).st_ino

    monkeypatch.setattr(params, "_FILE_DIGESTS", {})  # a new process
    assert param_signature(folder) == first
    assert len(reads) == 2
    (folder / "constants" / "atom_types.csv").write_text("a,b,c\n")
    monkeypatch.setattr(params, "_FILE_DIGESTS", {})
    second = param_signature(folder)
    assert second != first and reads[2:] == ["atom_types.csv"]

    monkeypatch.setattr(params, "_FILE_DIGESTS", {})
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "other_cache"))
    assert param_signature(folder) == second
    assert len(reads) == 5
    (tmp_path / "other_cache" / "params" / params.DIGEST_STORE).write_text("{not json")
    monkeypatch.setattr(params, "_FILE_DIGESTS", {})
    assert param_signature(folder) == second
    assert len(reads) == 7


def test_one_read_gives_the_md5_and_the_sha256(tmp_path: Path, monkeypatch) -> None:
    """Checking a manifest's provenance md5 and signing the folder read each
    listed file once between them."""
    reads = _count_reads(monkeypatch, tmp_path)
    folder = _small_folder(tmp_path / "small")
    md5 = hashlib.md5(b"a,b\n").hexdigest()
    (folder / PARAMSET_MANIFEST).write_text(json.dumps({
        "format": PARAMSET_FORMAT, "name": "test",
        "provenance": {"md5": {"constants/atom_types.csv": md5}}}))
    params.load_paramset_manifest(folder)
    assert reads == ["atom_types.csv"]
    param_signature(folder)
    assert sorted(reads) == ["atom_types.csv", "paramset.json", "standard_amino_acids.json"]


@pytest.mark.parametrize("change", ["table rewritten", "description added", "link moved"])
def test_files_changed_after_the_force_field_read_them_are_not_signed(
    tmp_path: Path, chignolin_pdb_path: str, change: str
) -> None:
    """The signature is taken when first asked for, not when the force field
    reads the files, so it must not describe files that changed in between:
    the force field refuses, naming them. One built afterwards signs them."""
    traj = md.load(chignolin_pdb_path)
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    folder = linked_param_folder(tmp_path / "f")
    ff = MCPUForceField(heavy, param_dir=str(folder))
    aromatic = folder / "mcpu_params" / "aromatic_potentials.bin"
    if change == "table rewritten":
        data = bytearray(aromatic.read_bytes())
        data[0] ^= 1
        _replace(aromatic, bytes(data))
        named = "mcpu_params/aromatic_potentials.bin"
    elif change == "description added":
        (folder / PARAMSET_MANIFEST).write_text(
            json.dumps({"format": PARAMSET_FORMAT, "name": "mcpu08"}))
        named = PARAMSET_MANIFEST
    else:  # the same contents, another file
        copy = tmp_path / "aromatic_copy.bin"
        copy.write_bytes(aromatic.read_bytes())
        aromatic.unlink()
        aromatic.symlink_to(copy)
        named = "mcpu_params/aromatic_potentials.bin"
    with pytest.raises(params.ParamsError, match=re.escape(
        f"parameter files in {folder} have changed since this force field read them: {named}. "
    )):
        ff.param_signature
    assert MCPUForceField(heavy, param_dir=str(folder)).param_signature == param_signature(folder)


def test_the_force_field_takes_its_signature_when_asked(chignolin_pdb_path: str) -> None:
    traj = md.load(chignolin_pdb_path)
    ff = MCPUForceField(traj.atom_slice(traj.topology.select("not element H")))
    assert ff._param_signature is None  # building it does not read the files again
    assert ff.param_signature == param_signature(ff.param_dir, "mcpu08")
    assert ff._param_signature == ff.param_signature
