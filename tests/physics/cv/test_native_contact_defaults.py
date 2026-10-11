"""One native-contact definition everywhere: CA atoms, 6 Å, ``|i - j| >= 4``.

The defaults used to differ by entry point: 6 Å and 4 in replica exchange
and the configs, 8 Å and 4 in ``FoldingRunner`` and ``build_cv``, 8 Å and 3
in ``NativeContactsCV``. Chignolin has 7 native contacts by the first and 11
by the second, so the same Q meant different things. These tests check every
entry point's defaults, then that the CV objects they build agree on the
contact pairs of one structure.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pymcpu import runners
from pymcpu.config import (
    DEFAULT_CONTACT_ATOM_MODE,
    DEFAULT_CONTACT_CUTOFF,
    DEFAULT_MIN_SEQ_SEP,
    FoldingConfig,
    ReplicaExchangeConfig,
    yaml_dict_to_config,
)
from pymcpu.sampling import (
    FoldingRunner,
    NativeContactsCV,
    ReplicaExchange,
    build_contact_atom_index,
    build_cv,
    reference_contact_from_pdb,
)
from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange

DEFAULTS = (DEFAULT_CONTACT_CUTOFF, DEFAULT_MIN_SEQ_SEP, DEFAULT_CONTACT_ATOM_MODE)


def test_the_shared_default_is_ca_6_angstrom_4_apart() -> None:
    assert DEFAULTS == (6.0, 4, "ca")


@pytest.mark.parametrize(
    "func, cutoff_name",
    [
        (NativeContactsCV.__init__, "contact_cutoff"),
        (FoldingRunner.__init__, "contact_cutoff_ang"),
        (ReplicaExchange.__init__, "contact_cutoff"),
        (MPIReplicaExchange.__init__, "contact_cutoff"),
        (runners.run_folding, "contact_cutoff"),
        (runners.run_replica_exchange_2d, "contact_cutoff"),
        (runners.run_mpi_replica_exchange_2d, "contact_cutoff"),
    ],
    ids=lambda v: getattr(v, "__qualname__", v),
)
def test_signature_defaults(func, cutoff_name: str) -> None:
    params = inspect.signature(func).parameters
    assert (
        params[cutoff_name].default,
        params["min_seq_sep"].default,
        params["contact_atom_mode"].default,
    ) == DEFAULTS


@pytest.mark.parametrize("make", [ReplicaExchangeConfig, FoldingConfig])
def test_config_defaults(make) -> None:
    cfg = make()
    assert (cfg.contact_cutoff, cfg.min_seq_sep, cfg.contact_atom_mode) == DEFAULTS


@pytest.mark.parametrize("temperatures", [[0.5], [0.5, 0.6]], ids=["folding", "remd"])
def test_yaml_defaults(temperatures: list[float]) -> None:
    cfg = yaml_dict_to_config({"pdb": "x.pdb", "temperatures": temperatures})
    block = cfg.folding if cfg.mode == "folding" else cfg.replica_exchange
    assert (block.contact_cutoff, block.min_seq_sep, block.contact_atom_mode) == DEFAULTS


def _pairs(cv: NativeContactsCV) -> list[tuple[int, int]]:
    i, j = cv.atom_pair_indices()
    return sorted(zip(i.tolist(), j.tolist()))


def test_the_entry_points_agree_on_the_contact_pairs() -> None:
    """The CV builder, a bare NativeContactsCV and FoldingRunner, each at its
    defaults, find the same 7 contacts in chignolin."""
    from pymcpu.forcefields import load_forcefield

    pdb = str(runners.default_example_pdb())
    forcefield = load_forcefield(pdb, "mcpu08", None)
    ca = build_contact_atom_index(forcefield)
    bare = NativeContactsCV(ca, reference_contact_from_pdb(pdb))
    built = build_cv([{"type": "native_contacts_n", "reference_pdb": pdb}], forcefield).cv

    runner = FoldingRunner.__new__(FoldingRunner)
    for name, value in dict(
        reference_pdb=pdb, forcefield=forcefield, linker_residues=[], native_contact_pairs=None,
        _q_cv=None, contact_atom_mode=DEFAULT_CONTACT_ATOM_MODE,
        contact_cutoff_ang=inspect.signature(FoldingRunner.__init__)
        .parameters["contact_cutoff_ang"].default,
        min_seq_sep=inspect.signature(FoldingRunner.__init__).parameters["min_seq_sep"].default,
    ).items():
        setattr(runner, name, value)
    folding = runner._ensure_q_cv()

    assert bare.n_contacts == 7
    assert _pairs(built) == _pairs(bare) == _pairs(folding)
    assert built.q_cutoff == bare.q_cutoff == folding.q_cutoff == 6.0
    native = np.asarray(forcefield.coords[0] * 10.0).T
    assert bare.compute_Q(native) == 1.0
