# pyMCPU

[![PyPI](https://img.shields.io/pypi/v/pymcpu.svg)](https://pypi.org/project/pymcpu/)
[![Python](https://img.shields.io/pypi/pyversions/pymcpu.svg)](https://pypi.org/project/pymcpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kibumpark-chem/pyMCPU/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/pymcpu/badge/?version=latest)](https://pymcpu.readthedocs.io)
[![Lint](https://github.com/kibumpark-chem/pyMCPU/actions/workflows/lint.yml/badge.svg)](https://github.com/kibumpark-chem/pyMCPU/actions/workflows/lint.yml)

Monte Carlo protein folding and unfolding with knowledge-based statistical
potentials — a C++20 compute core behind an OpenMM-style Python API.

The force field, **MCPU**, is all-atom: five energy terms
(contact/solvation "Mu", backbone torsion, sidechain torsion, directional
hydrogen bonding, aromatic stacking) read from potentials fitted to the PDB.
Sampling uses Metropolis Monte Carlo with pivot, continuous sidechain,
rotamer-library and kinematic-closure loop moves, plus temperature/umbrella
replica exchange and WESTPA weighted-ensemble support.

mcpu26 is planned for 0.2.0; KORP may return then if its licence is cleared.

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
- **On a RHEL 8 or Rocky 8 cluster** (FASRC included), run
  `source /opt/rh/gcc-toolset-15/enable` before `pip install`. The toolset
  compiles the parts of the C++ runtime newer than the system's into the
  extension, so the result imports under any Python. Do not `module load gcc`
  instead: a stock Miniforge/Mambaforge base environment ships a `libstdc++`
  capped at `CXXABI_1.3.14`, which a module GCC 14 or newer exceeds.

C++20 is required. GCC 15 is the default compiler and the one the wheels are
built with. GCC 8.5 is the oldest version verified to build and pass the full
test suite; its builds run about 3-10% slower, and CMake warns about it.

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

## Energy components

| Component | Energy group | Default weight | Description |
|---|---|---|---|
| `mu` | 1 | 0.4 | Pairwise contact/solvation potential between atom types |
| `backbone_torsion` | 2 | 1.35 | 4D backbone virtual-torsion statistics |
| `sidechain_torsion` | 3 | 2.5 | χ₁–χ₄ sidechain torsion statistics |
| `hydrogen_bond` | 4 | 1.35 (**effective 2.7**) | Directional 7D hydrogen bond potential |
| `aromatic` | 5 | 5.0 | Ring–ring aromatic stacking (PHE, TRP) |
| `native_contacts_bias` | 6 | 1.0 | Umbrella bias on the native-contact count |

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
| Floating-point math | IEEE-ish release FP, no fast-math |

## Examples

```bash
# OpenMM-style Python API
python examples/openmm_style/run_example.py
python examples/openmm_style/run_folding.py --steps 1000 --output-dir ./out_folding

# YAML input
mcpu run examples/configs/template.yaml
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
mcpu materialize-params --set mcpu08  # decode the bundled tables once
```

## Pretrained parameters

`pymcpu.params.ensure_params("mcpu08")` resolves the fitted potentials in this
order; the first hit wins:

| Priority | Source |
|---|---|
| 1 | `MCPU_PARAMS_DIR` — a pre-staged root (offline / HPC) |
| 2 | Editable-checkout tree (`src/pymcpu/parameters/pretrained/mcpu08`) |
| 3 | **The compact archive shipped inside the package**, decoded into the cache (`MCPU_CACHE_DIR`, default `~/.cache/pymcpu`) |

Step 3 is what makes a plain `pip install` work with no network. It sits below
steps 1–2 deliberately: a pre-staged HPC root, or a table you refitted locally,
still takes precedence over the shipped one.

For multi-rank MPI jobs, decode once before launching so N ranks do not race
against a shared `$HOME`:

```bash
export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"
mpirun -n 32 python my_remd_run.py
```

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
