#!/bin/bash
#SBATCH --job-name=pymcpu_remd
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# --ntasks is set by scripts/submit.sh from the YAML replica grid.
# Do not hardcode --ntasks here.
#
# Design: YAML = physics; sbatch/mpirun = infrastructure.
# See docs/running_remd.md

set -euo pipefail

CONFIG="${1:?Usage: sbatch --ntasks=N job_template.slurm CONFIG.yaml}"

# Resolve the repo root.
#
# Do NOT derive this from BASH_SOURCE. sbatch does not execute this file in
# place -- SLURM copies the batch script to the compute node and runs the copy
# out of /var/slurmd/spool/slurmd/scripts/. BASH_SOURCE[0] therefore points into
# the spool directory, and because that directory is itself named "scripts",
# "${SCRIPT_DIR}/.." silently yielded /var/slurmd/spool/slurmd and the job died
# on a near-plausible path:
#     python: can't open file
#     '/var/slurmd/spool/slurmd/scripts/run_mcpu_replica_exchange.py'
# scripts/submit.sh runs un-spooled, so its own BASH_SOURCE resolution IS valid;
# it exports the answer to us as PYMCPU_REPO. The fallback below covers running
# this file directly (bash scripts/job_template.slurm ...), nothing spooled.
if [[ -n "${PYMCPU_REPO:-}" ]]; then
  REPO_ROOT="${PYMCPU_REPO}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

RUNNER="${REPO_ROOT}/scripts/run_mcpu_replica_exchange.py"
if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: repo root does not contain the REMD runner." >&2
  echo "       REPO_ROOT = ${REPO_ROOT}" >&2
  echo "       expected  = ${RUNNER}" >&2
  echo "  Submit via 'bash scripts/submit.sh CONFIG.yaml', or export PYMCPU_REPO." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p logs

# ADDED: activate the environment explicitly. This script previously ran a bare
# `python`, inheriting whatever the submitting shell happened to have on PATH.
# Under sbatch that resolves to the base Mambaforge python, whose pymcpu comes
# from the ~/.local py3.10 user-site install -- a *different* build from the one
# in mcpu_dev, and one that cannot be rebuilt with the current toolchain
# (GCC 14.2 emits CXXABI_1.3.15; Mambaforge's libstdc++ provides only 1.3.14).
# Production REMD was therefore silently running whatever binary that install
# last held.
#
# The unset block matters because sbatch
# inherits the submitting shell's environment, and a pre-activated conda env
# there can make "module load Mambaforge" + "mamba activate" silently leave
# PATH pointing at the wrong python.
unset CONDA_SHLVL CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER \
      CONDA_EXE CONDA_PYTHON_EXE _CE_M _CE_CONDA MAMBA_EXE MAMBA_ROOT_PREFIX

module load Mambaforge
mamba activate mcpu_dev
echo "python: $(python -V)  $(which python)"
echo "mpirun: ${CONDA_PREFIX}/bin/mpirun"
python -c "import pymcpu, pymcpu.mcpu_core as m; print('mcpu_core:', m.__file__)"

# MPI intent lives here, not in the YAML.
# SLURM_NTASKS is set from submit.sh (--ntasks derived from temperatures × Q/N).
# Launch with the SAME MPI install mpi4py is linked against. mpi4py in mcpu_dev
# resolves libmpi.so.40 out of its own package dir to the conda Open MPI 5.0.10;
# a foreign mpirun hands the ranks PMIx wire-up its linked library cannot read,
# and MPI_Init segfaults -- the failure recorded in p18.8.X/submission.sh
# ("FASRC mpirun caused segfaults"). Deliberately NO fallback to a PATH mpirun:
# silently launching with a foreign one is exactly the bug, so this must be fatal.
MPIRUN="${CONDA_PREFIX}/bin/mpirun"
if [[ ! -x "${MPIRUN}" ]]; then
  echo "ERROR: no mpirun in the active env: ${MPIRUN}" >&2
  exit 1
fi

"${MPIRUN}" -n "${SLURM_NTASKS}" \
  python "${RUNNER}" \
  --mpi \
  -c "${CONFIG}"
