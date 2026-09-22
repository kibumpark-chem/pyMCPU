# Topology bridge (Python to C++)

pyMCPU draws an OpenMM-style boundary: Python owns topology
interpretation and parameter assembly, C++ owns the Monte Carlo hot path.
There is no serialized blob crossing the boundary. Python builds a
{doc}`System </api/system>` object by calling its pybind11 setters one at
a time, then registers the five energy terms on it, and a
{doc}`Context </api/context>` is constructed from the finished `System`.

Everything below is in `pymcpu/forcefields/mcpu.py` (`MCPUForceField`),
`pymcpu/forcefields/builders/` (one builder per energy term), and the
`System` / `Context` bindings in `src/bindings/bindings.cpp`.

## The engine atom layout

The C++ engine does not read structure files and does not reorder atoms.
It requires the atoms to arrive already grouped into contiguous
segments, and `MCPUForceField._order_atoms()` is what produces that
order:

```
[ BB: N CA C per residue ][ O: O/OXT/OCT ][ SC: sidechains ][ H: amide H ]
```

The H segment exists only when `virtual_amide_h=False`. With the default
`virtual_amide_h=True`, `get_total_h_atoms()` is 0 and the hydrogen-bond
term computes virtual amide-H positions from backbone geometry at
evaluation time instead.

The `BlockIndices` record per residue (`include/pymcpu/System.h`) indexes
into that layout. Python fills `bb_start`, `c_start`, `o_start`,
`sc_start`, `h_start`, `sc_count` and `amide_donor`, and `-1` means
"absent". As `MCPUForceField` builds it, `bb_start`, `c_start`,
`o_start` and `sc_start` are always assigned — every residue gets a
sidechain start, either its CB or, for GLY, the duplicated CA described
below — so with the default virtual hydrogens the only `-1` field is
`h_start`. `res_begin` and `res_end` are **not** set by Python — they
stay `-1` unless the optional
`Context.set_atom_reorder_mode("init_only")` residue-contiguous
permutation is applied in C++, which is off by default.

`DownstreamCache` (`first_sc_of_residue`, `first_o_of_residue`,
`first_h_of_residue`) is precomputed in Python so the hot path never has
to scan forward for the next residue that has a given segment.

### The engine index space is not the topology index space

Two things make the engine's atom count differ from the input topology's:

- GLY has no CB, so `_order_atoms()` duplicates its CA into the
  sidechain segment. This gives the sidechain move a valid index range
  for every residue. The duplicate carries `to_write=False`.
- Explicit amide hydrogens, when requested, have no counterpart in a
  heavy-atom input topology at all (`original_index=-1`).

`MCPUForceField.inverse_mapping` maps engine index to topology index and
returns `-1` for any slot with no counterpart, which is what the XTC
reporter uses to skip those slots. For 1uao (chignolin, 10 residues, 3
of them GLY) the input heavy-atom topology has 77 atoms while the engine
layout has 80 slots, 3 of which map to `-1`; for `examples/actin`
(377 residues, 28 GLY) it is 2943 atoms against 2971 slots.

The GLY CA duplicate is a storage-layout artifact, not a chemical claim:
`MuPotentialBuilder` treats backbone-named atoms as backbone for
clash/contact eligibility regardless of which segment they sit in, so
only the sidechain-segment copy contributes Mu energy.

## Build sequence

`MCPUForceField.__init__(trajectory, ...)` does the topology work once:

1. `_load_parameters()` — resolve the parameter set (default `mcpu_v1`)
   and read the tables.
2. `_canonicalize_residue_names(topology)` — fold protonation-state
   spellings (`HSD`, `CYX`, `ASH`, …) onto standard names, in place.
3. `_validate_topology(topology)` — reject any residue the parameter
   template does not define.
4. `_compute_secondary_structure(trajectory)` — a DSSP-derived H/E/C
   string, or `""` when `compute_dssp=False` (the engine reads `""` as
   all-coil).
5. `_order_atoms(topology)` — build the BB/O/SC order above.
6. `_infer_hydrogens(trajectory)` — reorder the coordinates to match,
   appending explicit amide H only when `virtual_amide_h=False`.
7. `_initialize_attributes(topology)` — derive `BlockIndices`,
   `DownstreamCache`, per-residue chi atom indices and chi moved-atom
   ranges, proline flags and amino indices.

`create_system(topology)` then constructs the C++ object and pushes each
table across explicitly:

```python
system = mcpu_core.System(self.n_atoms, self.n_res)
system.set_block_indices(self.blocks)
system.atom_to_residue = self.atom_to_res
system.set_torsions_per_residue(ntorsions)
system.set_chi_atom_indices(self.chi_atom_indices)
system.set_chi_moved_atom_ranges(self.chi_moved_atom_ranges)
system.set_rotamer_library(...)
system.set_rama_mixture_library(...)        # only if that table loaded
system.set_atom_counts(bb, o, sc, h)
system.set_virtual_amide_h(self.virtual_amide_h)
system.set_downstream_cache(self.downstream)
system.set_is_proline(...)
system.set_amino_index(...)
system.set_secondary_structure(self.secondary_structure)
```

and finally registers five potentials, each built by its own delegate in
`pymcpu/forcefields/builders/` and tagged with its legacy energy group:

| Energy group | Potential | Builder |
|---|---|---|
| 1 | `MuPotential` | `mu_builder.MuPotentialBuilder` |
| 2 | `TripletPotential` (backbone) | `triplet_builder.TripletPotentialBuilder` |
| 3 | `SidechainTripletPotential` | `triplet_builder.SidechainTripletBuilder` |
| 4 | `HBondPotential` | `hbond_builder.HydrogenBondBuilder` |
| 5 | `AromaticPotential` | `aromatic_builder.AromaticPotentialBuilder` |

The group numbers are load-bearing: the legacy outer weights are keyed
on them (see {doc}`/physics_notes/mc_acceptance`).

## Where coordinates enter

`System` carries no coordinates. `Context` is constructed from the
finished `System` alone — it sizes its `State` from
`get_num_atoms()`/`get_num_residues()` and initializes the neighbour
system — and coordinates arrive afterwards:

```python
system = ff.create_system(traj.topology)
ctx = mcpu_core.Context(system)
ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
ctx.calculate_total_energy(-1)   # seeds the running total
```

`set_positions` takes a `(3, n_atoms)` float32 array in **Angstroms**, in
the engine's atom order; MDTraj's `xyz` is nanometres, hence the
`* 10.0`. It does not seed the running total energy, so a raw-`Context`
caller must follow it with `calculate_total_energy(-1)`;
`pymcpu.Simulation` does that for you. `examples/actin/bench.py` is a
short working example of the whole sequence.

## What each side owns

| Python | C++ |
|---|---|
| Structure-file parsing (via MDTraj), residue-name canonicalization, validation | Nothing; the engine never parses a structure file |
| The BB/O/SC/H atom order and all per-residue block bookkeeping | Trusts the order it is given; does not reorder unless `init_only` reorder is enabled |
| Reading parameter files, shaping tables, Mu eligibility masks | Table lookup and geometry in the hot path |
| Which potentials exist and their energy groups | Polymorphic evaluation through `Potential` |
| Coordinate units (nm to Å) and the external atom order | Coordinates, neighbour grids and the live contact list during MC |

## Extending it

**A new Mu contact or clash rule.** The coordinate-independent
eligibility masks come from `MuPotentialBuilder.build_topology_masks()`,
and the per-atom role and residue-class metadata from
`MuPotentialBuilder.layer1_atom_meta()`. `create_system` passes them
through `MuPotential.set_topology_atom_meta()` and
`MuPotential.cache_necessary_data()`, which bakes them into a per-pair
flag table the hot path reads. So for the default configuration a rule
change is a Python change.

It is not a Python-only change in general: `MuPotential`
(`include/pymcpu/forces/knowledge_based/MuPotential.h`) keeps a second,
hand-mirrored copy of the same rules in `topology_pair_flags()`, used
when the precomputed table is disabled with `MCPU_TOPO_FLAGS=0`. The two
copies have drifted apart before. Change both, or accept that the
environment variable changes trajectories rather than just the code
path.

**A new per-residue table.** Add the field to `System`
(`include/pymcpu/System.h`), bind a setter in
`src/bindings/bindings.cpp`, compute it in
`MCPUForceField._initialize_attributes()` and push it in
`create_system()`. There is no schema to update and no validation pass
to extend — the setters are the contract.

**A new energy term.** Subclass `Potential`
(`include/pymcpu/Potential.h`), bind the class, add a builder under
`pymcpu/forcefields/builders/`, and register it in `create_system()`
with an energy group. See {doc}`/api/forces`.
