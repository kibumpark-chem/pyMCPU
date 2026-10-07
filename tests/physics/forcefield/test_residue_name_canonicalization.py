"""Residue-name canonicalization regression test.

MCPU's parameter tables are keyed by the 20 standard three-letter residue
names. Structures from CHARMM/AMBER pipelines, or from protonation tools like
PDB2PQR and PROPKA, routinely carry protonation-state spellings instead --
``HSD``/``HSE``/``HSP`` (CHARMM histidine), ``HID``/``HIE``/``HIP`` (AMBER
histidine), ``CYX`` (disulfide-bonded cysteine), ``CYM``, ``ASH``, ``GLH``,
``LYN``, ``ARN``. These are the *same molecule in heavy atoms*: pyMCPU strips
hydrogens, so the atom names, chi atoms, and MCPU atom types are all
unchanged. Only the residue label differs.

``MCPUForceField._canonicalize_residue_names`` folds those labels to the
standard name in place, before any of the ~15 downstream consumers (atom
typing, the PRO/CYS contact rules in ``mu_builder``, chi-atom resolution,
per-residue torsion counts) reads a residue name -- so none of them needs its
own alias table -- and warns that it did so.

Two properties matter and are both locked in here:

1. The rename is *physically inert*. Renaming every HIS to HSD (etc.) must
   leave the total energy bit-identical and the per-residue torsion counts
   unchanged. If the alias table ever grew an entry whose heavy atoms are NOT
   identical, this catches it.
2. Residues that are genuinely unrepresentable still raise, with a message
   that says why. ``MSE`` (selenomethionine, SD replaced by SE) and the
   phosphorylated residues (extra P/O atoms) are deliberately not aliased --
   turning them into standard residues is a modelling decision, not a
   spelling fix, so silently accepting them would hide it.

Internal-consistency test: every assertion compares pyMCPU against its own
unmodified-input result on a real actin PDB. No legacy MCPU binary, log, or
hardcoded numeric output is read.
"""

from __future__ import annotations

import logging

import mdtraj as md
import numpy as np
import pytest

from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu import mcpu_core
from tests.fixtures.context_builders import resolve_test_pdb


def _load_heavy() -> md.Trajectory:
    traj = md.load(str(resolve_test_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _rename_all(traj: md.Trajectory, old: str, new: str) -> int:
    """Relabel every ``old`` residue as ``new`` in place; return the count.

    Done on the loaded topology rather than by rewriting the PDB text because
    mdtraj's PDB reader already folds some of these names itself (the
    histidine set and CYX), which would make a text-level test vacuous -- the
    point here is to exercise pyMCPU's own table.
    """
    n = 0
    for res in traj.topology.residues:
        if res.name == old:
            res.name = new
            n += 1
    return n


def _energy_and_torsions(traj: md.Trajectory) -> tuple[float, tuple[int, ...]]:
    ff = MCPUForceField(traj, param_set="mcpu08")
    system = ff.create_system(traj.topology)
    context = mcpu_core.Context(system)
    context.set_positions(np.ascontiguousarray((ff.coords[0] * 10.0).T, dtype=np.float32))
    return context.calculate_total_energy(), tuple(system.get_torsions_per_residue())


@pytest.fixture(scope="module")
def reference() -> tuple[float, tuple[int, ...]]:
    return _energy_and_torsions(_load_heavy())


# (standard name, variant spelling) -- every entry of the alias table whose
# standard residue actually occurs in the test structure.
ALIASES = [
    ("HIS", "HSD"),
    ("HIS", "HSE"),
    ("HIS", "HSP"),
    ("HIS", "HID"),
    ("HIS", "HIE"),
    ("HIS", "HIP"),
    ("CYS", "CYX"),
    ("CYS", "CYM"),
    ("ASP", "ASH"),
    ("GLU", "GLH"),
    ("LYS", "LYN"),
    ("ARG", "ARN"),
]


@pytest.mark.parametrize("standard,variant", ALIASES, ids=lambda v: v)
def test_alias_is_physically_inert(standard, variant, reference):
    """A protonation-variant spelling must not change any number."""
    traj = _load_heavy()
    n = _rename_all(traj, standard, variant)
    assert n > 0, f"test structure has no {standard} residues to relabel"

    energy, torsions = _energy_and_torsions(traj)
    ref_energy, ref_torsions = reference

    assert energy == ref_energy, (
        f"relabelling {n} {standard} residues as {variant} changed the total "
        f"energy ({energy} vs {ref_energy}); the alias is not heavy-atom "
        f"identical and does not belong in _RESIDUE_ALIASES"
    )
    assert torsions == ref_torsions


def test_alias_emits_a_warning(caplog):
    """Silently rewriting the user's residue names would be worse than loud."""
    traj = _load_heavy()
    _rename_all(traj, "HIS", "HSD")
    with caplog.at_level(logging.WARNING, logger="pymcpu.forcefields.base"):
        MCPUForceField(traj, param_set="mcpu08")
    assert any("HSD->HIS" in rec.getMessage() for rec in caplog.records), (
        "canonicalization must report what it renamed"
    )


def test_unmodified_input_warns_about_nothing(caplog):
    traj = _load_heavy()
    with caplog.at_level(logging.WARNING, logger="pymcpu.forcefields.base"):
        MCPUForceField(traj, param_set="mcpu08")
    assert not [r for r in caplog.records if "Renamed" in r.getMessage()], (
        "a topology with only standard residue names must be left alone"
    )


@pytest.mark.parametrize(
    "standard,variant,hint",
    [
        ("MET", "MSE", "SE"),
        ("SER", "SEP", "Phosphorylated"),
        ("THR", "TPO", "Phosphorylated"),
        ("ALA", "UNK", "no parameters for"),
    ],
    ids=lambda v: v,
)
def test_unrepresentable_residue_raises_with_a_hint(standard, variant, hint):
    """Not aliasable -- must fail loudly and explain what to do instead."""
    traj = _load_heavy()
    assert _rename_all(traj, standard, variant) > 0

    with pytest.raises(ValueError) as excinfo:
        MCPUForceField(traj, param_set="mcpu08")
    message = str(excinfo.value)
    assert variant in message
    assert hint in message
