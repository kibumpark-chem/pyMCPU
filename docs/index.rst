pyMCPU documentation
====================

**pyMCPU** is a Monte Carlo protein folding engine with a Python API
inspired by OpenMM. It wraps a modern C++20 compute backend behind a
simple, composable Python interface.

**Start here:** :doc:`installation` → :doc:`quickstart`.

.. note::
   **Temperature** in pyMCPU is a dimensionless reduced
   parameter, not in physical units. Typical values:
   0.3 (cold/folded) to 0.6 (hot/unfolded).
   **Energy** values are unitless sums of knowledge-based
   potential table entries.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   cli
   running_remd
   checkpointing
   integrating_pymcpu

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_single_trajectory

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/forcefield
   api/simulation
   api/system
   api/context
   api/forces
   api/integrator
   api/analysis
   api/reporters
   api/sampling

.. toctree::
   :maxdepth: 2
   :caption: Physics Background

   physics_notes/background
   physics_notes/mu_potential
   physics_notes/hbond_directional
   physics_notes/triplet_torsion
   physics_notes/mc_acceptance
   physics_notes/kic_jacobian
   physics_notes/aromatic_stacking
   hbond_legacy_parity

.. toctree::
   :maxdepth: 1
   :caption: Limitations

   known_issues
   arch_baseline_decision

.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   architecture/topology_bridge
