# pyMCPU

[![PyPI](https://img.shields.io/pypi/v/pymcpu.svg)](https://pypi.org/project/pymcpu/)
[![Python](https://img.shields.io/pypi/pyversions/pymcpu.svg)](https://pypi.org/project/pymcpu/)
[![License: MIT + third-party](https://img.shields.io/badge/License-MIT%20%2B%20third--party-blue.svg)](https://github.com/kibumpark-chem/pyMCPU#license)
[![Docs](https://readthedocs.org/projects/pymcpu/badge/?version=latest)](https://pymcpu.readthedocs.io)
[![Lint](https://github.com/kibumpark-chem/pyMCPU/actions/workflows/lint.yml/badge.svg)](https://github.com/kibumpark-chem/pyMCPU/actions/workflows/lint.yml)

Monte Carlo protein folding and unfolding with knowledge-based statistical
potentials — a C++20 compute core behind an OpenMM-style Python API.

The force field, **MCPU**, is all-atom: five energy terms
(contact/solvation "Mu", backbone torsion, sidechain torsion, directional
hydrogen bonding, aromatic stacking) read from potentials fitted to the PDB.
Sampling uses Metropolis Monte Carlo with pivot, continuous sidechain,
rotamer-library and kinematic-closure loop moves, plus temperature/umbrella
replica exchange in one process or over MPI.

This release ships the `mcpu08` force field (legacy MCPU's energy function
and tables, with one default changed; see Physics defaults below); a second
force field, `mcpu26`, is planned for 0.2.0.

Full documentation: **<https://pymcpu.readthedocs.io>**

## Install

```bash
pip install pymcpu
```

- **Linux x86-64 and Python 3.12 or newer.** Wheels are published for
  CPython 3.12, 3.13 and 3.14.
- **A CPU with AVX2.** The wheels are built for the `x86-64-v3` baseline
  (AVX2, FMA and BMI2): Intel Haswell (2013), AMD Zen (2017) and newer. On an
  older CPU, `import pymcpu` stops with an `ImportError` that names the
  missing instructions, instead of crashing with "Illegal instruction".
  [Build from source](https://github.com/kibumpark-chem/pyMCPU#building-from-source)
  with `MCPU_ARCH=v2` for such a CPU.
- **About 0.7 GB of disk for the parameters.** The fitted potentials ship
  inside the package as a 2 MB archive, so nothing is downloaded. The first
  run unpacks them into `~/.cache/pymcpu` (about 0.7 GB), and later runs reuse
  them. To keep them somewhere else, set `MCPU_CACHE_DIR` before the first
  run:

  ```bash
  export MCPU_CACHE_DIR=/scratch/$USER/pymcpu-cache
  ```

MPI replica exchange also needs mpi4py, built for the MPI library that your
`mpirun` comes from. `pip install "pymcpu[mpi]"` adds mpi4py from PyPI; the
[installation guide](https://pymcpu.readthedocs.io/en/latest/installation.html#mpi-optional)
shows how to make it match your MPI.

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

`MCPU_ARCH` sets the CPU baseline: `v3` (the default, as in the wheels), `v2`
for a CPU without AVX2, `v4` for AVX-512, or `native` for the machine you build
on:

```bash
MCPU_ARCH=v2 pip install --no-build-isolation -e .
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

# Temperature is reduced (dimensionless); with the default settings,
# chignolin melts at about 0.6
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

The hydrogen-bond row needs two notes. 1.35 is the configured group
weight, but the effective multiplier applied to the raw H-bond energy is
**2.7** (`1.35 × RDTHREE_CON`, with `RDTHREE_CON = 2.0`). And a bond between
residues more than four apart, as in a β-sheet, is also multiplied by the
sheet factor `hbond_beta_favor`: by default the parameter set's own (from its
`constants/paramset.json`), else 2, as for mcpu08; legacy MCPU used 3.

`energy_breakdown()` returns both raw and weighted values.

## Physics defaults

| Setting | Default |
|---|---|
| Legacy outer energy weights | on |
| H-bond sheet factor (`hbond_beta_favor`) | the set's `paramset.json`, else 2.0 (mcpu08; legacy MCPU: 3.0) |
| Hard core (`clash_alpha`, `clash_bond_exclusion`) | the set's `paramset.json` (mcpu26: 0.85 and 3), else 0.75 and 0 (mcpu08) |
| Rotamer proposal (`rotamer_library`) | `dunbrack2002_sym`, both images of symmetric χ (legacy MCPU's draw: `dunbrack2002`) |
| Virtual amide H (H-bond-only hydrogens) | on |
| Periodic boundary conditions | none (open boundary) |
| Floating-point math | IEEE-ish release FP, no fast-math |

## Examples

A folding run from Python, on the chignolin structure that ships with the
package:

```python
from pymcpu.runners import default_example_pdb, run_folding

run_folding(pdb=default_example_pdb(), temperature=0.6, steps=20_000,
            output_dir="out_folding")
```

The same run from a YAML config, with the `mcpu` command:

```bash
python -c "import shutil, pymcpu.runners as r; shutil.copy(r.default_example_pdb(), '1uao.pdb')"
cat > fold.yaml <<'EOF'
pdb: 1uao.pdb
temperatures: [0.6]     # one temperature and no umbrella targets: folding
steps: 20000            # total MC steps (in replica exchange, steps per cycle)
output_prefix: out/fold
EOF
mcpu validate fold.yaml
mcpu run fold.yaml
```

The repository has more examples; clone it to run them:

```bash
git clone https://github.com/kibumpark-chem/pyMCPU.git && cd pyMCPU
python examples/openmm_style/run_example.py
python examples/openmm_style/run_folding.py --steps 1000 --output-dir ./out_folding
mcpu run examples/configs/template.yaml   # 2D replica exchange, 44 replicas
```

Replica exchange, production MPI runs and checkpointing are covered in the
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

Two sets ship: `mcpu08`, the default, and `mcpu26` (`param_set="mcpu26"`),
mcpu08's energy function with tables re-estimated from PISCES 2026 and fitted
weights. `pymcpu.params.ensure_params("mcpu08")` resolves a set's fitted
potentials in this order; the first hit wins:

| Priority | Source |
|---|---|
| 1 | `MCPU_PARAMS_DIR` — a pre-staged root of that set (offline / HPC) |
| 2 | Editable-checkout tree (`src/pymcpu/parameters/pretrained/<set>`) |
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

From a clone of the repository:

```bash
python -m pytest -q          # test suite
ruff check .                 # lint
bash scripts/dev_install.sh  # clean rebuild + verify + test
```

See [CONTRIBUTING.md](https://github.com/kibumpark-chem/pyMCPU/blob/main/CONTRIBUTING.md).

## Citation

If you use pyMCPU in published work, please cite it.
[`CITATION.cff`](https://github.com/kibumpark-chem/pyMCPU/blob/main/CITATION.cff)
is included in the repository; the BibTeX form is:

```bibtex
@software{pymcpu,
  author  = {Park, Kibum},
  title   = {pyMCPU: Monte Carlo protein folding with knowledge-based potentials},
  url     = {https://github.com/kibumpark-chem/pyMCPU},
  version = {0.1.0},
  year    = {2026}
}
```

The rotamer moves use the backbone-independent rotamer library (2002) of
Roland L. Dunbrack, Jr., Fox Chase Cancer Center. The 2002 file states no
licence. pyMCPU redistributes it following, as a guide, the terms the Dunbrack
lab gives for its 2010 backbone-dependent library:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), which asks
that the source be acknowledged (see
[THIRD_PARTY_NOTICES](https://github.com/kibumpark-chem/pyMCPU/blob/main/THIRD_PARTY_NOTICES)).
Please cite R. L. Dunbrack Jr. and F. E. Cohen, "Bayesian statistical analysis
of protein side-chain rotamer preferences", *Protein Science* **6**, 1661–1681
(1997), <https://doi.org/10.1002/pro.5560060807>.

## License

pyMCPU's own code is MIT-licensed — see
[LICENSE](https://github.com/kibumpark-chem/pyMCPU/blob/main/LICENSE). The
compiled extension also contains code from xdrfile (BSD-2-Clause), pybind11
(BSD-3-Clause) and Eigen (MPL-2.0), and the KIC move's loop closure, a
modified port of the tripeptide loop closure code of Seok, Coutsias, Jacobson
and Dill (LGPL-2.1-or-later, with Graphics-Gems for its Sturm solver). The
package as a whole is therefore under `MIT AND BSD-2-Clause AND BSD-3-Clause
AND MPL-2.0 AND LGPL-2.1-or-later AND Graphics-Gems`.
Their licences, the terms of the bundled rotamer library and the credit to
legacy MCPU are in
[THIRD_PARTY_NOTICES](https://github.com/kibumpark-chem/pyMCPU/blob/main/THIRD_PARTY_NOTICES).
