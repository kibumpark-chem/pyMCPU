"""``Integrator`` has no default temperature.

The constructor used to default to ``temperature=300.0``, a physical-units
value about 500x the top of the useful reduced range (0.3-0.6), so a call
that left it out silently ran a near-random walk. There is no reduced
temperature that suits every protein, so it must now be given.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core


@pytest.mark.parametrize("kwargs", [{}, {"step_size_rad": 0.1}], ids=["no-args", "step-only"])
def test_leaving_out_the_temperature_raises(kwargs: dict) -> None:
    with pytest.raises(TypeError):
        mcpu_core.Integrator(**kwargs)


def test_an_explicit_temperature_still_works() -> None:
    by_keyword = mcpu_core.Integrator(temperature=0.5)
    positional = mcpu_core.Integrator(0.5, 0.1)
    assert by_keyword.backbone_step_size_rad() == positional.backbone_step_size_rad()
