# Checkpointing and resume

Long runs, replica exchange in particular, often outlast one job allocation.
Every runner can save its state to a checkpoint and continue from it later.
Checkpointing is on by default.

## Settings

A YAML config takes the settings in a `checkpointing` block, or at its top
level; a JSON config takes them in a `checkpoint` block:

```yaml
checkpointing:
  checkpoint_dir: checkpoints   # where the files go
  checkpoint_interval: 50       # save every 50 cycles
  keep_last_n: 3                # numbered checkpoints to keep
  resume: false                 # true to continue from the last checkpoint
```

A cycle is one round of MC steps and swaps in replica exchange, and one report
interval in a folding run (`log_interval` in YAML, `report_interval` in JSON).
To turn checkpointing off, set `checkpoint_dir: null`; `enabled: false` is
accepted but has no effect under `mcpu run`. The `mcpu run` options
`--checkpoint-dir`, `--checkpoint-interval`, `--keep-last-n` and `--resume`
override these settings; see [Command-line interface](cli.rst).

## When a run saves

- Every `checkpoint_interval` cycles.
- At the end of the run, except under MPI, where only the interval applies.
- Replica exchange also saves when it receives SIGTERM or SIGINT, as from
  `scancel`, a job time limit or Ctrl-C: it finishes the current cycle, saves
  and stops. A folding run does not, and resumes from its last regular save.

## Files

| File | Contents |
|---|---|
| `checkpoint_cycle_000050.chk` | Saved at cycle 50 |
| `last.chk` | The most recent save |

Only the newest `keep_last_n` numbered files are kept; `last.chk` always is.
Each file is written under a temporary name and then renamed, so a crash
during a save leaves the previous checkpoint intact.

A checkpoint holds what the run needs to continue exactly: the coordinates of
every replica, its position in the run, the random number streams of the MC
moves and of the exchanges, the move counters, and how much of each output
file had been written.

## Resuming

```bash
mcpu run config.yaml --resume
```

or set `resume: true` in the config. The run continues after the state in
`last.chk` in `checkpoint_dir`. Output files are first cut back to what had
been written at that checkpoint and then appended to, so nothing written after
the last save is duplicated. This covers the XTC, CSV, HDF5 and NPZ files
pyMCPU writes.

If there is no checkpoint yet, folding and MPI replica exchange start from the
beginning, while single-process replica exchange stops with an error.

Under MPI, rank 0 writes and reads the checkpoint, which covers every replica.

### Checkpoints from other versions

Every checkpoint records its format version, and a file newer than the
installed pyMCPU understands is refused. It also records the force field, and
a run resumed with a different `forcefield:` stops with an error; checkpoints
from before this was recorded count as mcpu08. A checkpoint whose atom count does
not match the system stops with an error that names both counts. Format
version 2 stores each glycine CA once, so a version 1 checkpoint of a protein
with glycine cannot be resumed; start that run again from its input
structure. Version 1 checkpoints of glycine-free proteins, and of KORP runs,
still resume.

### Exact continuation

The MC random state is saved as text written by the C++ standard library, so
the same random stream continues only with the same pyMCPU build, or one built
against a compatible libstdc++. With another build, the coordinates, cycle
count and exchange state still restore, but the run continues with a
different random stream.
