"""Abstract contract shared by all pyMCPU force fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mdtraj as md

    from pymcpu import mcpu_core


class BaseForceField(ABC):
    """Abstract base class defining the contract for all ForceFields."""

    @abstractmethod
    def create_system(self, topology: "md.Topology") -> "mcpu_core.System":
        """Take an MDTraj topology and return a fully initialized C++ System.

        Args:
            topology: The molecular topology to map.

        Returns:
            The C++ engine system instance.
        """
        pass