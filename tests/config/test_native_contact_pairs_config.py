"""YAML round-trip and runner-wiring checks for ``native_contact_pairs``.

Pure config-plumbing: verifies ``yaml_dict_to_config`` parses/validates the
new ``native_contact_pairs`` top-level YAML key the way ``pymcpu/config.py``'s
``ReplicaExchangeConfig`` dataclass defines it, and that the YAML schema
knows the key (so a valid config isn't rejected as having an unknown key). No
physics computation is involved here -- deep semantic validation (index
range, self-pairs, duplicates, energy-ignored residues) needs ``n_res`` and
is deliberately deferred to ``NativeContactsCV`` at construction time (see
``tests/physics/cv/test_native_contacts_cv.py``); the YAML-round-trip tests
only cover what ``yaml_dict_to_config`` itself can and does check (list-of-2
arity) plus the plain dataclass plumbing. Mirrors
``test_contact_atom_mode_config.py``'s style/fixtures.

The runner-wiring tests below close a real gap an adversarial review found:
``ReplicaExchangeConfig.native_contact_pairs`` parsed correctly, but
``pymcpu/runners.py``'s ``run_replica_exchange_2d``/``run_mpi_replica_exchange_2d``/
``run_from_config`` silently dropped the field before it ever reached
``ReplicaExchange``/``MPIReplicaExchange`` -- i.e. every YAML-driven production
entry point (``mcpu run`` / ``scripts/run_mcpu_replica_exchange.py -c``) would
discard an explicit ``native_contact_pairs`` with no error. These use
``monkeypatch`` to capture the kwargs actually passed to the next layer down,
proving the value crosses each boundary rather than merely appearing in a
signature.
"""

from __future__ import annotations

import warnings

import pytest

from pymcpu.config import (
    IntegratorConfig,
    OutputsConfig,
    ReplicaExchangeConfig,
    SimulationConfig,
    check_yaml_keys,
    yaml_dict_to_config,
)
from pymcpu.utils.yaml_parser import load_yaml

# Minimal valid base kwargs shared by every case below; each test overrides
# only the field(s) it's actually exercising.
_BASE_YAML = {
    "pdb": "examples/data/1uao.pdb",
    "temperatures": [0.4, 0.5],
    "num_cycles": 1,
    "mc_replica_steps": 10,
}


def test_native_contact_pairs_roundtrips_into_config() -> None:
    cfg = yaml_dict_to_config(
        {**_BASE_YAML, "native_contact_pairs": [[0, 5], [1, 6], [2, 8]]}
    )
    assert cfg.replica_exchange is not None
    assert cfg.replica_exchange.native_contact_pairs == [[0, 5], [1, 6], [2, 8]]


def test_native_contact_pairs_defaults_to_none() -> None:
    cfg = yaml_dict_to_config(dict(_BASE_YAML))
    assert cfg.replica_exchange is not None
    # Matches ReplicaExchangeConfig.native_contact_pairs's dataclass default.
    assert cfg.replica_exchange.native_contact_pairs is None


def test_native_contact_pairs_wrong_arity_raises_at_parse_time() -> None:
    """A malformed entry (e.g. a 3-element tuple instead of a pair) must be
    rejected by ``yaml_dict_to_config`` itself, before it ever reaches
    ``NativeContactsCV`` -- this is arity checking only, not the deeper
    range/self-pair/duplicate semantic validation that needs ``n_res`` and
    lives in ``NativeContactsCV`` instead."""
    with pytest.raises(ValueError, match="exactly 2"):
        yaml_dict_to_config(
            {**_BASE_YAML, "native_contact_pairs": [[1, 2, 3]]}
        )


def test_native_contact_pairs_is_a_known_field() -> None:
    """A valid ``native_contact_pairs`` key must not be rejected as unknown."""
    check_yaml_keys({**_BASE_YAML, "native_contact_pairs": [[0, 5], [1, 6]]})


def test_native_contact_pairs_key_does_not_warn(tmp_path) -> None:
    f = tmp_path / "native_contact_pairs.yaml"
    f.write_text(
        "pdb: examples/data/1uao.pdb\n"
        "temperatures: [0.4, 0.5]\n"
        "num_cycles: 1\n"
        "mc_replica_steps: 10\n"
        "native_contact_pairs: [[0, 5], [1, 6]]\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_yaml(f)


def test_native_contact_pairs_reaches_replica_exchange_constructor(monkeypatch) -> None:
    """Regression for a real bug found by adversarial review:
    ``run_replica_exchange_2d`` accepted ``native_contact_pairs`` as a
    parameter but never passed it into ``ReplicaExchange(...)``."""
    import pymcpu.runners as runners_mod

    captured: dict = {}

    class _StubReplicaExchange:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        def describe(self) -> str:
            return "stub"

        def run(self, *args, **kwargs):
            return None

    monkeypatch.setattr(runners_mod, "ReplicaExchange", _StubReplicaExchange)

    runners_mod.run_replica_exchange_2d(
        pdb="dummy.pdb",
        temperatures=[0.4, 0.5],
        cycles=1,
        steps_per_cycle=1,
        verbose=False,
        native_contact_pairs=[[0, 5], [1, 6]],
    )
    assert captured.get("native_contact_pairs") == [[0, 5], [1, 6]]


def test_native_contact_pairs_reaches_mpi_replica_exchange_constructor(monkeypatch) -> None:
    """Same regression as above, for the MPI runner's ``rex_kwargs`` dict."""
    import pymcpu.runners as runners_mod

    captured: dict = {}

    class _StubMPIReplicaExchange:
        def __init__(self, comm, pdb_path, **kwargs) -> None:
            captured.update(kwargs)

        def describe(self) -> str:
            return "stub"

        def run(self, *args, **kwargs):
            return None

    class _StubComm:
        def Get_rank(self) -> int:
            return 0

        def Barrier(self) -> None:
            return None

    monkeypatch.setattr(
        "pymcpu.sampling.mpi_replica_exchange.MPIReplicaExchange", _StubMPIReplicaExchange
    )

    runners_mod.run_mpi_replica_exchange_2d(
        comm=_StubComm(),
        pdb="dummy.pdb",
        temperatures=[0.4, 0.5],
        cycles=1,
        steps_per_cycle=1,
        verbose=False,
        native_contact_pairs=[[0, 5], [1, 6]],
    )
    assert captured.get("native_contact_pairs") == [[0, 5], [1, 6]]


def test_native_contact_pairs_reaches_run_replica_exchange_2d_via_run_from_config(
    monkeypatch,
) -> None:
    """Regression: ``run_from_config``'s ``shared_kwargs`` must include
    ``native_contact_pairs=rex.native_contact_pairs``, not silently drop it
    -- this is the exact path every YAML-driven production run takes."""
    import pymcpu.runners as runners_mod

    captured: dict = {}

    def _stub_run_replica_exchange_2d(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(runners_mod, "run_replica_exchange_2d", _stub_run_replica_exchange_2d)

    cfg = SimulationConfig(
        mode="replica_exchange_2d",
        pdb="dummy.pdb",
        integrator=IntegratorConfig(),
        outputs=OutputsConfig(),
        replica_exchange=ReplicaExchangeConfig(
            temperatures=[0.4, 0.5],
            native_contact_pairs=[[0, 5], [1, 6]],
        ),
    )
    runners_mod.run_from_config(cfg, verbose=False)
    assert captured.get("native_contact_pairs") == [[0, 5], [1, 6]]
