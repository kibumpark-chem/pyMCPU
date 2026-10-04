# Running REMD with pyMCPU

## Design principle

YAML input files describe **physics** (temperatures, move schedule,
force fields). Launch scripts describe **infrastructure** (nodes,
MPI ranks, walltime). Do not mix them.

## How replica count is determined

The number of replicas is always:

```text
n_replicas = len(temperatures) × len(q_targets | n_targets)
```

Temperatures may be an explicit `temperatures:` list or generated from
`temp_min` / `temp_step` / `n_temps`. If neither `q_targets` nor
`n_targets` / `native_contact_targets` is set, there is one window,
centred at N = 0, and its umbrella still applies with `k_bias` (default
1.0), which pulls every replica toward unfolded structures. For plain
temperature REMD, set `k_bias: 0`.

For a YAML with 11 temperatures and 4 Q targets: **44 replicas**.

## Serial run (debugging, single-process REMD)

```bash
python scripts/run_mcpu_replica_exchange.py -c inputs/template.yaml
```

Or via the config runner (serial backend):

```bash
mcpu run inputs/template.yaml
```

No `mpirun` needed. Same YAML works unchanged.

## MPI REMD run (production)

```bash
# Step 1: compute replica count from YAML and submit
bash scripts/submit.sh inputs/template.yaml

# Dry-run (print replica count, do not sbatch):
bash scripts/submit.sh inputs/template.yaml --dry-run

# Or manually:
mpirun -n 44 python scripts/run_mcpu_replica_exchange.py \
    --mpi -c inputs/template.yaml
```

MPI ranks do **not** need to match the replica count exactly:
`partition_replicas()` supports fewer ranks than replicas, distributing
multiple replicas per rank (each rank steps its local replicas
sequentially before exchange attempts). Launching with more ranks than
replicas is rejected with a `ValueError`. `scripts/submit.sh` sizes
`--ntasks` to the full replica count so each rank gets exactly one
replica, which is the common case, but running with fewer ranks is a
supported way to trade wall-time for node count.

## The `mpi: true` YAML key

`mpi` is accepted in a YAML config but not used. MPI is controlled
entirely by how you launch the script (`--mpi` plus `mpirun` /
`srun -n N_RANKS`), not by anything in the YAML.

## Environment variables

| Variable | Effect |
|----------|--------|
| `MCPU_USE_CELL_PAIR` | `0` to disable cell-pair inversion |
| `MCPU_TOPO_FLAGS` | `0` for layered v1 on-the-fly Layer 1 (debug; slower) |
| `MCPU_MU_SKIN` | `1` to enable partial Verlet skin (default off) |
