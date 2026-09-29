"""``MCPUForceField`` parameter-directory resolution: control-flow only.

Checks which code path ``MCPUForceField.__init__`` takes -- use the
explicitly passed ``param_dir`` as-is, or fall back to ``ensure_params()``
(the network/default-lookup helper) when none is given -- not any real
physics or parameter values. All heavy potential/topology loading is
stubbed out via monkeypatch, so no network access or real binary parameter
files are needed.

Named ``..._param_dir_resolution`` (not ``..._param_resolution``) to be
specific about what's covered: this is directory selection, not parameter
*value* resolution/parsing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pymcpu.params import required_files


def _stub_forcefield_param_tree(root: Path) -> None:
    """Write a plausible-looking but inert param directory tree.

    Content is never actually parsed here (all loaders below are
    monkeypatched) -- only paths need to exist and look structurally sane.
    """
    (root / "constants").mkdir(parents=True)
    (root / "mcpu_params").mkdir(parents=True)
    (root / "constants" / "atom_types.csv").write_text(
        "residue,atom,type,radius\nALA,CA,1,1.5\n"
    )
    (root / "constants" / "standard_amino_acids.json").write_text(
        '{"ALA": {"atoms": ["N", "CA", "C", "O"]}}'
    )
    (root / "constants" / "bbind02.May.lib").write_text(
        "ALA\t1\t0\t0\t0\t1\t1\t100\t0\t100\t0\t0\t10\t0\t10\t0\t10\t0\t10\n"
    )
    # Registry-derived rather than hand-listed, so this stub cannot drift away
    # from what MCPUForceField._load_parameters actually opens.
    for rel in required_files("mcpu08").values():
        dst = root / rel
        if dst.exists():          # the three constants above have real content
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"\x00" * 16)


def _patch_forcefield_loaders(monkeypatch, mcpu_ff, MCPUForceField, atom_types=None) -> None:
    """Stub the heavy potential/topology loaders shared by both tests below.

    Doesn't touch ``ensure_params`` -- callers patch that themselves since
    its behavior differs between the explicit-param_dir and
    default-lookup cases (that's the actual thing under test).
    """
    monkeypatch.setattr(
        mcpu_ff.MuPotentialBuilder,
        "load_atom_types",
        classmethod(lambda cls, p: dict(atom_types or {})),
    )
    monkeypatch.setattr(
        mcpu_ff.MuPotentialBuilder, "load_parameters", classmethod(lambda cls, p: MagicMock())
    )
    monkeypatch.setattr(
        mcpu_ff.TripletPotentialBuilder,
        "load_parameters",
        classmethod(lambda cls, p: MagicMock()),
    )
    monkeypatch.setattr(
        mcpu_ff.SidechainTripletBuilder,
        "load_parameters",
        classmethod(lambda cls, p: MagicMock()),
    )
    monkeypatch.setattr(
        mcpu_ff.HydrogenBondBuilder,
        "load_parameters",
        classmethod(lambda cls, p: MagicMock()),
    )
    monkeypatch.setattr(
        mcpu_ff.HydrogenBondBuilder,
        "load_seq_dep_parameters",
        classmethod(lambda cls, p: MagicMock()),
    )
    monkeypatch.setattr(
        mcpu_ff.AromaticPotentialBuilder,
        "load_parameters",
        classmethod(lambda cls, p: MagicMock()),
    )
    monkeypatch.setattr(
        mcpu_ff.RotamerLibraryBuilder,
        "load_parameters",
        classmethod(lambda cls, p: MagicMock()),
    )
    monkeypatch.setattr(MCPUForceField, "_validate_topology", lambda self, top: None)
    monkeypatch.setattr(
        MCPUForceField,
        "_order_atoms",
        lambda self, top: setattr(self, "ordered_atom_list", []),
    )
    monkeypatch.setattr(MCPUForceField, "_infer_hydrogens", lambda self, traj: MagicMock())
    monkeypatch.setattr(MCPUForceField, "_initialize_attributes", lambda self, top: None)


def test_explicit_param_dir_is_used_without_calling_ensure_params(tmp_path, monkeypatch) -> None:
    _stub_forcefield_param_tree(tmp_path)

    import pymcpu.forcefields.mcpu as mcpu_ff
    from pymcpu.forcefields.mcpu import MCPUForceField

    _patch_forcefield_loaders(
        monkeypatch, mcpu_ff, MCPUForceField, atom_types={("ALA", "CA"): (1, 1.5)}
    )

    def _no_ensure(*_a, **_k):
        raise AssertionError("ensure_params must not be called when param_dir is set")

    monkeypatch.setattr(mcpu_ff, "ensure_params", _no_ensure)

    traj = MagicMock()
    traj.topology = MagicMock()
    ff = MCPUForceField(traj, param_dir=str(tmp_path))
    assert Path(ff.param_dir) == tmp_path


def test_omitted_param_dir_falls_back_to_ensure_params(tmp_path, monkeypatch) -> None:
    _stub_forcefield_param_tree(tmp_path)

    import pymcpu.forcefields.mcpu as mcpu_ff
    from pymcpu.forcefields.mcpu import MCPUForceField

    monkeypatch.setattr(mcpu_ff, "ensure_params", lambda *a, **k: tmp_path)
    _patch_forcefield_loaders(monkeypatch, mcpu_ff, MCPUForceField)

    traj = MagicMock()
    traj.topology = MagicMock()
    ff = MCPUForceField(traj)
    assert Path(ff.param_dir) == tmp_path
