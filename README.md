# pyMCPU

[![PyPI](https://img.shields.io/pypi/v/pymcpu.svg)](https://pypi.org/project/pymcpu/)
[![Python](https://img.shields.io/pypi/pyversions/pymcpu.svg)](https://pypi.org/project/pymcpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kibumpark-chem/pyMCPU/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/pymcpu/badge/?version=latest)](https://pymcpu.readthedocs.io)
[![Lint](https://github.com/kibumpark-chem/pyMCPU/actions/workflows/lint.yml/badge.svg)](https://github.com/kibumpark-chem/pyMCPU/actions/workflows/lint.yml)

Monte Carlo protein folding and unfolding with knowledge-based statistical
potentials — a C++20 compute core behind an OpenMM-style Python API.

Two force fields are available. **MCPU** is all-atom: five energy terms
(contact/solvation "Mu", backbone torsion, sidechain torsion, directional
hydrogen bonding, aromatic stacking) read from potentials fitted to the PDB.
**KORP** is backbone-only — a 6D orientation-dependent residue-pair potential
that reads just N, CA and C, paired with a CA–CA excluded-volume filter. It
needs an energy map that is [downloaded separately](#korp-force-field). Sampling uses Metropolis Monte Carlo with pivot,
continuous sidechain, rotamer-library and kinematic-closure loop moves, plus
temperature/umbrella replica exchange and WESTPA weighted-ensemble support.

**Platform:** Linux x86-64. The published wheel targets the `x86-64-v3`
baseline — AVX2, FMA and BMI2, so Intel Haswell (2013) and AMD Zen (2017) and
newer. See [Building from source](#building-from-source) to target an older or
a newer CPU.

Full documentation: **<https://pymcpu.readthedocs.io>**

## Install

```bash
pip install pymcpu
```

or

```bash
conda install -c conda-forge pymcpu
```

The fitted potentials ship inside the package, so this works offline with no
extra download and no environment variables.

Optional extras: `analysis` (HDF5 output), `mpi` (MPI replica exchange),
`westpa` (weighted ensemble), `training` (refitting potentials), `docs`, `dev`.

```bash
pip install "pymcpu[analysis,mpi]"
```

### Building from source

```bash
git clone https://github.com/kibumpark-chem/pyMCPU.git
cd pyMCPU
conda env create -f environment.yml && conda activate mcpu
pip install --no-build-isolation -e .
python scripts/install_check.py
```

`--no-build-isolation` is required: the build uses the `pybind11` and `numpy`
already present in the environment.

To target a different CPU baseline — a machine without AVX-512, or `native` for
the machine you will actually run on:

```bash
MCPU_ARCH=x86-64-v3 pip install --no-build-isolation -e .
```

**One toolchain rule, and it is the only one that matters:** the `libstdc++`
your compiler targets must be *no newer* than the `libstdc++` present at run
time. Break that and the extension fails to import with
`version 'CXXABI_x.y.z' not found` — it will build fine, so the error only
appears at `import pymcpu`.

Two ways to satisfy it:

- **Let conda supply both** (recommended, and what `environment.yml` does):
  `gxx_linux-64` provides the compiler and `libstdcxx-ng` the matching runtime,
  so they cannot drift apart. Note that a conda interpreter carries an RPATH of
  `$ORIGIN/../lib`, so the environment's own `lib/libstdc++.so.6` wins over
  `LD_LIBRARY_PATH` — you cannot fix a mismatch by setting that variable.
- **On a module-based HPC system**, load at run time the same compiler you
  built with, and make sure it is not newer than the interpreter's runtime. A
  stock Miniforge/Mambaforge base environment ships a `libstdc++` capped at
  `CXXABI_1.3.14`, which GCC 14 exceeds; GCC 13 and older do not.

C++20 is required. GCC 8.5 is the oldest version verified to build and pass the
full test suite.

## Quickstart

```python
import numpy as np
import mdtraj as md
import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

# Any PDB works; the bundled 1UAO chignolin model is used here
traj = md.load(str(default_example_pdb()))
heavy = traj.atom_slice(traj.topology.select("not element H"))

# Build the force field and the system
forcefield = MCPUForceField(heavy, param_set="mcpu08")
system = forcefield.create_system(heavy.topology)

# Temperature is a DIMENSIONLESS reduced parameter (~0.3 cold .. 0.6 hot)
integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
integrator.set_seed(42)

sim = mc.Simulation(heavy.topology, system, integrator)
sim.context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))

# Reporters write files; the library does not print to stdout
sim.add_energy_reporter("energies.csv", interval=100)

sim.step(10_000)

print(sim.context.energy_breakdown(weighted=True))
```

Note that `mc.Integrator` defaults to `temperature=300.0`, which is **not** a
reduced temperature — always pass one explicitly.

## Energy components

| Component | Energy group | Default weight | Description |
|---|---|---|---|
| `mu` | 1 | 0.4 | Pairwise contact/solvation potential between atom types |
| `backbone_torsion` | 2 | 1.35 | 4D backbone virtual-torsion statistics |
| `sidechain_torsion` | 3 | 2.5 | χ₁–χ₄ sidechain torsion statistics |
| `hydrogen_bond` | 4 | 1.35 (**effective 2.7**) | Directional 7D hydrogen bond potential |
| `aromatic` | 5 | 5.0 | Ring–ring aromatic stacking (PHE, TRP) |
| `native_contacts_bias` | 6 | 1.0 | Umbrella bias on the native-contact count |
| `korp_6d` | 7 | 1.0 | KORP 6D orientational residue-pair potential |
| `calpha_excluded_volume` | 8 | 1.0 | CA–CA steric filter (0 in any accepted state) |

Groups 1–5 are MCPU's; 7–8 are KORP's. The two force fields are alternatives,
not layers — nothing installs both.

Energies are **unitless**: sums of knowledge-based table entries scaled by a
dimensionless per-group weight. There is no Boltzmann constant, no Kelvin and
no kT conversion anywhere in the engine.

The hydrogen-bond row is the one subtlety — 1.35 is the configured group
weight, but the effective multiplier applied to the raw H-bond energy is
**2.7** (`1.35 × RDTHREE_CON`, with `RDTHREE_CON = 2.0`).

`energy_breakdown()` returns both raw and weighted values.

## Physics defaults

| Setting | Default |
|---|---|
| Legacy outer energy weights | on |
| Virtual amide H (H-bond-only hydrogens) | on |
| Periodic boundary conditions | none (open boundary) |
| Mu neighbour skin | 0 (cell-grid dense list; Verlet engages only for skin > 0) |
| Floating-point math | IEEE-ish release FP, no fast-math |

## Examples

```bash
# OpenMM-style Python API
python examples/openmm_style/run_example.py
python examples/openmm_style/run_folding.py --steps 1000 --output-dir ./out_folding

# GROMACS-style YAML input
python examples/gromacs_style/run.py --input examples/gromacs_style/example_input.yaml
```

Replica exchange, weighted ensemble and production MPI runs are covered in the
documentation:
[running REMD](https://pymcpu.readthedocs.io/en/latest/running_remd.html) ·
[Integrating pyMCPU](https://pymcpu.readthedocs.io/en/latest/integrating_pymcpu.html) ·
[checkpointing](https://pymcpu.readthedocs.io/en/latest/checkpointing.html)

## Command line

```bash
mcpu version
mcpu run config.yaml                   # run a JSON/YAML simulation config
mcpu validate config.yaml              # check a config without running
mcpu download-params --set mcpu08     # resolve parameters, print the path
mcpu materialize-params --set mcpu08  # decode the bundled tables once
```

## Pretrained parameters

`pymcpu.params.ensure_params("mcpu08")` resolves the fitted potentials in this
order; the first hit wins:

| Priority | Source |
|---|---|
| 1 | `MCPU_PARAMS_DIR` — a pre-staged root (offline / HPC) |
| 2 | Editable-checkout tree (`src/pymcpu/parameters/pretrained/mcpu08`) |
| 3 | An already-populated cache (`MCPU_CACHE_DIR`, default `~/.cache/pymcpu`) |
| 4 | `MCPU_PARAMS_BUNDLE` — a local `.tar.gz` |
| 5 | **The compact archive shipped inside the package**, decoded into the cache |
| 6 | Registry URL via `pooch` (needs a published release + checksum) |

Step 5 is what makes a plain `pip install` work with no network. It sits below
steps 1–4 deliberately: a pre-staged HPC root, or a table you refitted locally,
still takes precedence over the shipped one.

For multi-rank MPI jobs, decode once before launching so N ranks do not race
against a shared `$HOME`:

```bash
export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"
mpirun -n 32 python my_remd_run.py
```

## KORP force field

KORP ([López-Blanco & Chacón, *Bioinformatics* 2019](https://doi.org/10.1093/bioinformatics/btz026))
scores residue pairs from a local frame built on each residue's N, CA and C,
with the pair coordinate being CA–CA. It reads no sidechain atom, so
`KORPForceField` drops sidechains rather than carrying them unused.

### Getting the energy map

The `korp6Dv1.bin` map is **not shipped**: at 316 MiB it is well over PyPI's
per-file limit. Download it once from the Chacón lab:

```bash
# https://chaconlab.org/modeling/korp -> "KORP Linux64" -> Korp6Dv1.txz
tar xJf Korp6Dv1.txz
export KORP_MAP_PATH="$PWD/Korp6Dv1/korp6Dv1.bin"
```

Check you have the map these results were validated against — a different
map scores differently, and legitimately:

```bash
sha256sum "$KORP_MAP_PATH"
# 8c586500f80ad31f297652e050d391702a01927ee58d598017650ef2e0fbf971
```

On a cluster, pre-stage it once and point `MCPU_PARAMS_DIR` or `KORP_MAP_PATH`
at the shared copy, exactly as for the MCPU parameter sets.

### Running it

```python
import mdtraj as md, numpy as np, pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.korp import KORPForceField

traj = md.load("protein.pdb")
forcefield = KORPForceField(traj)          # or KORPForceField(traj, map_path=...)
system = forcefield.create_system(traj.topology)

integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
integrator.set_move_weights(0.5, 0.5, 0.0)   # pivot + KIC only; see below

sim = mc.Simulation(forcefield.output_topology, system, integrator)
sim.context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))
forcefield.apply_energy_weights(sim.context)
sim.step(10_000)
```

Two things differ from an MCPU run:

- **Sidechain moves must be off.** These residues have no chi angles, so every
  sidechain proposal would return without proposing anything. `set_move_weights`
  with a zero third argument says so; `Integrator.run` raises rather than
  silently discarding that share of the budget.
- **The trajectory is backbone-only.** Load it against
  `forcefield.output_topology`, not against your input PDB's topology.

pyMCPU reproduces the reference `korpe` binary to within 4e-8 relative; see
[the physics note](https://pymcpu.readthedocs.io/en/latest/physics_notes/korp_6d.html)
for the frame convention, the binning, and a discrepancy between the KORP paper
and the code that built the released map.

If you use KORP, cite López-Blanco JR & Chacón P, *Bioinformatics* 2019,
35(17):3013–3019, alongside pyMCPU.

## Development

```bash
python -m pytest -q          # test suite
ruff check .                 # lint
bash scripts/dev_install.sh  # clean rebuild + verify + test
```

## Citation

If you use pyMCPU in published work, please cite it. `CITATION.cff` is included
in the repository; the BibTeX form is:

```bibtex
@software{pymcpu,
  author  = {Park, Kibum},
  title   = {pyMCPU: Monte Carlo protein folding with knowledge-based potentials},
  url     = {https://github.com/kibumpark-chem/pyMCPU},
  version = {0.1.0},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](https://github.com/kibumpark-chem/pyMCPU/blob/main/LICENSE).
