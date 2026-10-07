"""Abstract contract shared by all pyMCPU force fields."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mdtraj as md

    from pymcpu import mcpu_core

logger = logging.getLogger(__name__)


class BaseForceField(ABC):
    """Abstract base class defining the contract for all ForceFields."""

    #: Residue names that are the SAME MOLECULE in heavy atoms as a standard
    #: residue, differing only in protonation. pyMCPU is heavy-atom only (H are
    #: stripped or virtual), so for these the rename is exact -- same atom names,
    #: same chi atoms, same types. mdtraj already folds the histidine set and CYX
    #: for PDB input, but not these, and other readers fold nothing, so
    #: canonicalise here rather than depend on the reader.
    #:
    #: Deliberately NOT included, because they are not renames:
    #:   MSE  selenomethionine -- SD is replaced by SE, so the atom NAMES differ
    #:        and Se is not S (different radius and contact type). Supporting it
    #:        means renaming SE->SD and accepting sulfur parameters for selenium,
    #:        which is a modelling decision, not a spelling one.
    #:   SEP/TPO/PTR  phosphorylated residues -- extra P/O atoms with no MCPU type.
    #:   UNK, ligands, nucleotides -- no sidechain-torsion or contact definition.
    _RESIDUE_ALIASES = {
        # histidine protonation states (CHARMM / AMBER)
        "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        # cysteine: disulfide-bonded and deprotonated
        "CYX": "CYS", "CYM": "CYS",
        # AMBER neutral/protonated forms
        "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG",
        # occasional alternate spellings
        "HISD": "HIS", "HISE": "HIS",
    }

    def _canonicalize_residue_names(self, topology: "md.Topology") -> None:
        """Rename protonation-state variants to their standard equivalents.

        Done in place on the topology, before anything reads a residue name, so
        every downstream consumer (atom typing, PRO/CYS contact rules, chi-atom
        resolution, torsion counts, the KORP residue order) sees the canonical
        name without each needing its own alias table.
        """
        renamed: dict[str, str] = {}
        for res in topology.residues:
            canon = self._RESIDUE_ALIASES.get(res.name)
            if canon is not None and canon != res.name:
                renamed[res.name] = canon
                res.name = canon
        if renamed:
            logger.warning(
                "Renamed protonation-state residue variants to their standard "
                "equivalents: %s. These are identical in heavy atoms, so the "
                "energy is unaffected; pyMCPU has no separate parameters for "
                "the protonation states.",
                ", ".join(f"{k}->{v}" for k, v in sorted(renamed.items())),
            )

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