# Checkpointing and resume

Long pyMCPU runs — production replica exchange in particular — are expected to
outlive a single scheduler allocation. Every runner can therefore snapshot its
full state and resume from it.

## What a checkpoint is

Checkpoints are `.chk` files written into `checkpoint_dir`:

| File | Meaning |
|---|---|
| `checkpoint_cycle_000050.chk` | Versioned snapshot taken at cycle 50 |
| `last.chk` | Always the most recent save |
| `best.chk` | Best-metric snapshot, when a metric is being tracked |

Older versioned files are pruned automatically; `keep_last_n` (default `3`)
sets how many to retain. `last.chk` and `best.chk` are never pruned.

### Crash safety

A checkpoint is written to a temporary file in the destination directory,
`fsync`'d, and then moved into place with `os.replace`. Because that final move
is atomic on POSIX filesystems, a crash part-way through a write cannot leave a
truncated or partially-written `.chk` behind: either the old snapshot survives
intact or the new one does.

### Payload

`CheckpointState` carries 26 fields — enough to reconstruct a run bit-exactly
rather than merely approximately. Grouped by purpose:

- **Position in the run** — `cycle`, `global_step`, `epoch`, `current_steps`
- **Configuration needed to rebuild the system** — `pdb_path`,
  `reference_pdb`, `temperatures`, `n_targets`, `k_bias`, `contact_cutoff`,
  `min_seq_sep`, `contact_atom_mode`, `native_contact_pairs`,
  `fixed_residues`, `linker_residues`, `linker_energy_mode`, `n_replicas`
- **Coordinates and replica bookkeeping** — `replica_coords`,
  `walker_at_state`
- **Random state** — `seed`, `exchange_rng`, `exchange_rng_state` (the NumPy
  BitGenerator state driving exchange decisions) and `integrator_rng_states`
  (one serialized `std::mt19937` state string per replica)
- **Output alignment** — `traj_frame_indices`, `format_version`, `kind`

Capturing both RNG streams is what makes a resumed trajectory a continuation
of the original rather than a new sample from the same ensemble.

## Resuming

On resume the run continues from the cycle *after* the saved checkpoint, and
trajectory files are first truncated back to the frame count recorded in
`traj_frame_indices`. Without that truncation, any frames written after the
last checkpoint but before the crash would be duplicated when appending
resumes.

Truncation is implemented for XTC, CSV, HDF5 and NPZ outputs.

```{warning}
**DCD is the one exception.** `truncate_all_trajectories_on_resume` emits a
warning and skips DCD files, because `truncate_dcd_to_frame()` is not
implemented. If you resume a run that wrote DCD, delete the DCD manually first
or you will get duplicated frames. Prefer XTC.
```

### From the command line

```bash
mcpu run config.yaml
```

with `resume: true` in the config's `checkpointing` block.

### From YAML

```yaml
checkpointing:
  enabled: true
  checkpoint_dir: "checkpoints"
  checkpoint_interval: 50        # save every 50 cycles
  keep_last_n: 3                 # retain the last 3 versioned files
  resume: false
  cloud_sync: false
  cloud_bucket: ""               # e.g. s3://my-bucket/run-name/
  cloud_sync_cmd: "aws s3 cp"    # or "gsutil cp" / "rclone copy"
```

## MPI runs

Under MPI replica exchange, only rank 0 writes checkpoints. The state it saves
covers every replica, so a resumed job reconstructs the whole ladder from that
one file — there is no per-rank checkpoint to keep consistent.

## Cloud sync

With `cloud_sync: true`, `last.chk` is uploaded to `cloud_bucket` after every
save. The upload runs in a background subprocess and does not block sampling,
so a slow or failing upload costs throughput but not correctness. The
corresponding CLI tool (`aws`, `gsutil` or `rclone`) must be installed and
authenticated.

## Reproducibility caveat

Exact MC RNG restore requires resuming with an `mcpu_core` whose
`Integrator::getRngState` / `setRngState` serialization is compatible with the
binary that wrote the checkpoint — that is, the same build, or one built
against a compatible `libstdc++`. The RNG state is serialized through
`std::mt19937`'s stream operators, so it is a text format produced by the C++
standard library rather than a pyMCPU-defined one.

Coordinates, cycle counters and exchange state restore correctly regardless;
only the *continuation* of the exact random stream depends on the binary.
