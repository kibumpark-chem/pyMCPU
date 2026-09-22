Physics Background
==================

pyMCPU implements a knowledge-based statistical potential
for protein folding Monte Carlo simulation. This section
describes each force term and the MC move types.

Python force objects are documented in :doc:`/api/forces`.
Acceptance and temperature are documented in
:doc:`/api/integrator`.

Overview of the energy function
-------------------------------

The total potential energy is a sum of five terms:

.. math::

   E = w_\mu E_\mu + w_{tor} E_{tor} + w_{sct} E_{sct}
     + w_{hb} E_{hb} + w_{aro} E_{aro}

where each of the five core table terms is a knowledge-based statistical
potential derived from protein structure databases. The optional
native-contact umbrella bias (:class:`~pymcpu.NativeContactsBiasPotential`,
energy group 6) is documented in :doc:`/api/forces` and is not part of the
default five-term sum. All energies are
**unitless** — they are direct table lookups, not in
physical energy units.

Energy terms
------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Potential
     - Default weight
     - Description
   * - :doc:`mu_potential`
     - 1.0
     - Pairwise contact energy between atom types (μ-potential)
   * - :doc:`triplet_torsion` (backbone)
     - 1.35
     - 4D backbone torsion statistics (φ, ψ, pCA, bCA)
   * - :doc:`triplet_torsion` (sidechain)
     - 2.50
     - χ₁–χ₄ sidechain torsion statistics
   * - :doc:`hbond_directional`
     - 1.35
     - Directional 7D hydrogen bond potential
   * - :doc:`aromatic_stacking`
     - 5.0
     - Ring–ring aromatic stacking (PHE, TRP only)

MC move types
-------------

Three move types are implemented:

**Pivot move:** Rotates a contiguous backbone segment
around a randomly chosen φ or ψ bond axis. Produces
large conformational changes.

**KIC move (kinematic loop closure):** Solves the
tripeptide closure problem analytically, guaranteeing
exact chain geometry. See :doc:`kic_jacobian` for the
Jacobian correction needed for detailed balance.

**Sidechain move:** Rotates χ₁ of a randomly chosen
residue. Small, local perturbation.

Temperature convention
----------------------

.. important::
   Temperature in pyMCPU is a **dimensionless** reduced
   parameter. Typical production values: T = 0.3–0.6.
   This is not a physical temperature unit.
   The same convention is used in legacy MCPU.

See also :doc:`mc_acceptance` for the Metropolis criterion.
