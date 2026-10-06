Physics background
==================

pyMCPU samples protein conformations by Monte Carlo, scoring them with
knowledge-based potentials: energy tables derived from the statistics of
known protein structures. It has two force fields, used one at a time:

* **mcpu08** (:class:`~pymcpu.MCPUForceField`): all-atom, with the five MCPU
  potentials described on this page and the pages after it.
* **KORP** (:class:`~pymcpu.KORPForceField`): backbone-only, with one
  orientation-dependent residue-pair potential; see :doc:`korp_6d`.

The potential objects are documented in :doc:`/api/forces`, and the move
settings in :doc:`/api/integrator`.

The mcpu08 energy
-----------------

The energy is a weighted sum of five terms:

.. math::

   E = w_\mu E_\mu + w_{bb} E_{bb} + w_{sc} E_{sc}
     + w_{hb} E_{hb} + w_{aro} E_{aro}

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Term
     - Weight
     - What it scores
   * - :doc:`Mu contact <mu_potential>`
     - 0.4
     - Contacts between atom pairs, by atom type. A pair closer than its
       hard-core distance is a clash, and a move that makes one is rejected.
   * - :doc:`Backbone torsion <triplet_torsion>`
     - 1.35
     - The φ and ψ of each residue and the orientation of its two
       neighbours, with a table for each three-residue sequence.
   * - :doc:`Sidechain torsion <triplet_torsion>`
     - 2.5
     - The χ angles of each residue, with a table for each three-residue
       sequence.
   * - :doc:`Hydrogen bond <hbond_directional>`
     - 2.7
     - Backbone N–H···O=C hydrogen bonds, by their geometry and by whether
       they are helix-like, parallel or antiparallel.
   * - :doc:`Aromatic <aromatic_stacking>`
     - 5.0
     - The angle between the rings of two PHE or TRP residues whose rings are
       within 7 Å.

Energies are unitless: each term is a sum of table values, and the weights
are dimensionless. The weights shown are the ones an mcpu08 run uses; the
hydrogen-bond weight of 2.7 is 1.35 times a fixed factor of 2.0 carried over
from legacy MCPU. :doc:`/api/forces` shows how to read and change them, and
the name that labels each term in ``energy_breakdown()`` and the energy CSV.

Replica exchange adds a sixth term, the native-contacts umbrella bias; see
:doc:`/api/forces`.

Monte Carlo moves
-----------------

Each step tries one move and accepts or rejects it with the Metropolis
criterion; see :doc:`mc_acceptance`. There are three kinds of move. By default
a step is a pivot a quarter of the time, a KIC move a quarter of the time and
a sidechain move half of the time; ``Integrator.set_move_weights`` changes
this.

**Pivot.** Picks a residue and one of its backbone torsions, φ or ψ (never
the φ of a proline), and rotates the shorter end of the chain about that bond
by a random angle, drawn from a Gaussian of width ``step_size_rad``. A small
rotation near the middle of the chain moves the far end a long way.
Optionally, some pivot moves set (φ, ψ) to a pair drawn from a Ramachandran
library instead; this is off by default
(``Integrator.set_pivot_rama_probability``).

**KIC (kinematic closure).** Rotates a backbone torsion next to a window of
three residues, then rebuilds the backbone of the window so that the rest of
the chain does not move. The window keeps the bond lengths and angles of the
starting structure. It is a local backbone move; :doc:`kic_jacobian`
describes the Jacobian correction it needs.

**Sidechain.** Changes the χ angles of one residue. By default it draws a
rotamer from a rotamer library; in ``'continuous'`` mode it perturbs each χ
by a Gaussian angle instead. Glycine and alanine have no χ angles. A KORP run
sets this move's weight to 0, because its residues have no sidechains.

Temperature
-----------

Temperature is a dimensionless reduced parameter, as in legacy MCPU, not a
temperature in Kelvin. Where a protein unfolds depends on the protein:
chignolin melts at about 0.65 to 0.7 (see
:doc:`/tutorials/02_replica_exchange`).

References
----------

The mcpu08 potentials are those of the MCPU program from the Shakhnovich lab.
Each term first appears in one of these papers:

* Contact, backbone torsion and hydrogen-bond terms: J. S. Yang, W. W. Chen,
  J. Skolnick and E. I. Shakhnovich, "All-atom ab initio folding of a diverse
  set of proteins", *Structure* 15, 53–63 (2007),
  https://doi.org/10.1016/j.str.2006.11.010.
* Sidechain torsion term: J. S. Yang, S. Wallin and E. I. Shakhnovich,
  "Universality and diversity of folding mechanics for three-helix bundle
  proteins", *PNAS* 105, 895–900 (2008),
  https://doi.org/10.1073/pnas.0707284105.
* Aromatic term: J. Tian, J. C. Woodard, A. Whitney and E. I. Shakhnovich,
  "Thermal stabilization of dihydrofolate reductase using Monte Carlo
  unfolding simulations and its functional consequences", *PLoS
  Computational Biology* 11, e1004207 (2015),
  https://doi.org/10.1371/journal.pcbi.1004207.
