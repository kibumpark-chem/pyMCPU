"""Config/CLI-facing plumbing for the fixed-residue feature.

Pure software correctness -- no physics. The engine-level behavior (that
fixed residues actually stay put under MC) is covered instead in
``tests/physics/test_fixed_residues.py``; this file only checks that the
constraint reaches the engine correctly from config/CLI input:

* ``SimulationConfig.constraints.fixed_residues`` schema parsing and its
  backwards-compatible default
* the ``parse_int_list`` CLI-argument parsing utility
* the ``Simulation``-level set/get/clear convenience wrapper around the
  ``Integrator`` API
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from pymcpu.config import config_from_dict, parse_int_list
from pymcpu.simulation import Simulation
from tests.fixtures.context_builders import build_test_context


class TestConfigFixedResidues:
    def test_fixed_residues_parsed_from_constraints_block(self) -> None:
        data = {
            "mode": "folding",
            "pdb": "test.pdb",
            "constraints": {"fixed_residues": [0, 1, 2]},
        }
        cfg = config_from_dict(data)
        assert cfg.constraints.fixed_residues == [0, 1, 2]

    def test_missing_constraints_block_defaults_to_empty(self) -> None:
        """Backwards compat: configs written before this feature existed
        must still parse, with no residues fixed."""
        data = {"mode": "folding", "pdb": "test.pdb"}
        cfg = config_from_dict(data)
        assert cfg.constraints.fixed_residues == []

    def test_empty_constraints_block_defaults_to_empty(self) -> None:
        data = {"mode": "folding", "pdb": "test.pdb", "constraints": {}}
        cfg = config_from_dict(data)
        assert cfg.constraints.fixed_residues == []


class TestParseIntList:
    def test_basic_comma_separated_list(self) -> None:
        assert parse_int_list("0,1,2") == [0, 1, 2]

    def test_surrounding_and_interior_whitespace_is_stripped(self) -> None:
        assert parse_int_list(" 3 , 5 , 7 ") == [3, 5, 7]

    def test_single_value(self) -> None:
        assert parse_int_list("42") == [42]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_int_list("")


class TestSimulationFixedResidues:
    """``Simulation.set_fixed_residues``/``get_fixed_residues``/``clear_fixed_residues``
    are a thin convenience wrapper delegating to the Integrator -- checked
    here as wiring, not as a re-test of the Integrator's own enforcement."""

    def test_set_get_and_clear_round_trip(self) -> None:
        context, _ = build_test_context(with_qbias=False)
        integrator = mcpu_core.Integrator(0.6)
        sim = Simulation(None, context.get_system(), integrator)

        sim.set_fixed_residues([0, 1])
        assert sim.get_fixed_residues() == [0, 1]
        sim.clear_fixed_residues()
        assert sim.get_fixed_residues() == []
