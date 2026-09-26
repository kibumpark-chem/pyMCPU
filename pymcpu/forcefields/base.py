"""Abstract contract shared by all pyMCPU force fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mdtraj as md

    from pymcpu import mcpu_core


class BaseForceField(ABC):
    """Abstract base class defining the contract for all ForceFields."""

    @classmethod
    def prepare_trajectory(cls, trajectory):
        """Reduce a structure to the atoms this force field simulates.

        The default keeps everything, which is right for a force field that
        does its own selection. :class:`~pymcpu.forcefields.mcpu.MCPUForceField`
        overrides it to drop hydrogens.

        This exists so a factory can build either force field from the same
        loaded trajectory without knowing which one needs what -- MCPU wants
        heavy atoms, KORP wants the backbone and slices itself.
        """
        return trajectory

    def apply_energy_weights(self, context) -> None:
        """Set the per-group outer weights this force field expects.

        A no-op by default, so MCPU keeps the legacy weights it has always
        used. Weights live on ``Context``, which does not exist yet when
        :meth:`create_system` runs, so this is a separate step.
        """
        return None

    @abstractmethod
    def create_system(self, topology: "md.Topology") -> "mcpu_core.System":
        """Take an MDTraj topology and return a fully initialized C++ System.

        Args:
            topology: The molecular topology to map.

        Returns:
            The C++ engine system instance.
        """
        pass