# Running replica exchange

Replica exchange (REMD) runs several copies of the protein, called replicas,
side by side. Each has its own temperature and, optionally, its own umbrella
window on N, the number of native contacts. After every cycle of MC steps,
neighbouring replicas try to swap places. This page covers running it from a
YAML config, on one process or with MPI. The config holds the simulation
settings; the number of processes, the nodes and the run time belong to the
launch command. To set up replica exchange in Python instead, see
[Sampling](api/sampling.rst).

## The replica grid

A YAML config runs replica exchange when it lists more than one temperature.
With a single temperature it runs folding instead, and any targets are
ignored.

```text
n_replicas = number of temperatures × number of windows
```

- **Temperatures:** a `temperatures:` list, or `temp_min`, `temp_step`
  (default 0.025) and `n_temps`.
- **Windows:** one per entry of `n_targets`, the umbrella centres in native
  contacts N, or of `q_targets`, the same centres as a fraction Q of the
  native contacts (N = Q × the number of native contacts). Set one of the
  two, not both. `k_bias` (default 1.0) is the strength of the harmonic
  umbrella on N.
- **No targets:** there is one window, centred at N = 0, and its umbrella
  still applies with `k_bias`, which pulls every replica toward unfolded
  structures. For plain temperature REMD, set `k_bias: 0`.

Native contacts are the residue pairs, at least `min_seq_sep` (default 4)
apart in sequence, whose contact atoms (`contact_atom_mode`, default `ca`)
lie within `contact_cutoff` (default 6 Å) of each other in `reference_pdb`
(default: the starting `pdb`).

For example, 11 temperatures and 4 Q targets give 44 replicas. A small
config with 4 × 3 = 12 replicas:

```yaml
pdb: protein.pdb
temperatures: [0.40, 0.45, 0.50, 0.55]
n_targets: [0, 20, 40]
k_bias: 0.5
num_cycles: 1000          # each cycle: MC steps, then one round of swaps
mc_replica_steps: 1000    # MC steps per replica per cycle
output_prefix: out/run1
```

`examples/configs/template.yaml` shows the common keys with their defaults.

## Force field

`forcefield:` picks the force field, `mcpu08` (all-atom, the default) or
`korp` (backbone-only); see [Force fields](api/forcefield.rst). KORP's own
options go in `forcefield_options:`. KORP has no sidechains, so a KORP config
must set the sidechain move weight to zero, or the run stops before it
starts; `param_set` and `param_dir` belong to mcpu08 and are refused with
KORP. Native-contact windows, fixed residues and `linker_residues` work with
either, but contacts must use `contact_atom_mode: ca`.

```yaml
pdb: protein.pdb
forcefield: korp
forcefield_options:
  map_path: /path/to/korp6Dv1.bin   # or set KORP_MAP_PATH
  map_mmap: true                    # ranks on a node share one copy of the map
move_weights: [0.5, 0.5, 0.0]       # pivot, KIC, sidechain
temperatures: [0.40, 0.45, 0.50, 0.55]
k_bias: 0
output_prefix: out/korp1
```

## Running on one process

```bash
mcpu run config.yaml
```

All replicas run in one process, one after another. Use it to try a config
before a large run; it needs no MPI. In a source checkout,
`python scripts/run_mcpu_replica_exchange.py -c config.yaml` does the same.

## Running with MPI

Set up MPI and mpi4py as described in [Installation](installation.rst), then
start a short script under `mpirun`:

```python
# my_remd_run.py
from mpi4py import MPI

from pymcpu.config import load_config_auto
from pymcpu.runners import run_from_config

run_from_config(load_config_auto("config.yaml"), comm=MPI.COMM_WORLD)
```

```bash
mpirun -n 12 python my_remd_run.py
```

In a source checkout, `scripts/run_mcpu_replica_exchange.py --mpi -c
config.yaml` does the same. The script takes its settings only from the
config file; besides `-c` and `--mpi` it accepts `--hdf5` and the checkpoint
options of `mcpu run` (`--resume` among them).

Each process runs a block of replicas. Fewer processes than replicas is
fine: a process with several replicas runs them one after another, which
trades wall time for nodes. More processes than replicas is an error. An
`mpi:` key in the config is accepted and ignored; only the launch decides.

### On a Slurm cluster

A source checkout has a submission helper. `scripts/submit.sh` reads the
replica count from the config and submits `scripts/job_template.slurm` with
that many tasks:

```bash
bash scripts/submit.sh config.yaml --dry-run   # print the sbatch command only
bash scripts/submit.sh config.yaml --partition=shared --time=2-00:00:00
```

It passes any other options on to `sbatch`, and it checks the config's keys
first, so a typo fails before the job waits in the queue. Before the first
submission, edit `job_template.slurm` for your cluster: the lines that load
modules and activate the environment, and the `#SBATCH` defaults.

## Output

Files go to the directory of `output_prefix` and are named after its last
part, `run1` for `output_prefix: out/run1`:

- `run1_tXX_qYY.xtc` and `run1_tXX_qYY_data.csv` for each temperature XX and
  window YY. A slot keeps its temperature and window while replicas move
  between slots; the `walker_id` column says which replica was there.
- `run1_rex_stats.json`: exchange attempts and acceptance rates.
- With `forcefield: korp`, `run1_topology.pdb`: the backbone-only topology
  (and starting structure) the XTC files hold. Load them against this file,
  not against `pdb`.
- With `exchange_log: all`, `run1_exchange.csv` records every attempt. With
  `state_log_interval: K`, `run1_state.csv` records each slot's N, Q and
  energy every K cycles. With `hdf5: FILE`, the per-cycle samples go to FILE
  (relative to the output directory) for MBAR reweighting.

Checkpoints go to `checkpoints/` every 50 cycles and at the end of the run,
by default. To continue a stopped run, use `mcpu run config.yaml --resume`;
with the MPI script, set `resume: true` in the config's `checkpointing`
block. See [Checkpointing](checkpointing.md).
