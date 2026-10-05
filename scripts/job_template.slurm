#!/bin/bash
#SBATCH --job-name=pymcpu_remd
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Run MPI replica exchange for one config. Submit it with scripts/submit.sh,
# which sets --ntasks to the config's replica count; do not set --ntasks here.
# Edit the #SBATCH defaults above and the environment block below for your
# cluster. See docs/running_remd.md.

set -euo pipefail

CONFIG="${1:?Usage: sbatch --ntasks=N job_template.slurm CONFIG.yaml}"

# SLURM runs a spooled copy of this file, so the file cannot find the
# repository from its own path; scripts/submit.sh passes it in PYMCPU_REPO.
# The fallback covers running this file directly with bash.
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

# ---- Environment: edit for your cluster ------------------------------------
# The job inherits the environment you submit from (submit.sh exports it).
# To set it up here instead, load the MPI library that mpi4py was built
# against and activate the Python environment pyMCPU is installed in, e.g.
#
#   module load openmpi
#   source ~/venvs/pymcpu/bin/activate
#
# or, with conda:
#
#   module load Mambaforge
#   mamba activate pymcpu
# -----------------------------------------------------------------------------

echo "python: $(python -V)  $(command -v python)"
echo "mpirun: $(command -v mpirun)"
python -c "import pymcpu, pymcpu.mcpu_core as m; print('mcpu_core:', m.__file__)"

# The mpirun must belong to the MPI that mpi4py was built against; a different
# one can crash MPI_Init (see docs/installation.rst).
mpirun -n "${SLURM_NTASKS}" python "${RUNNER}" --mpi -c "${CONFIG}"
