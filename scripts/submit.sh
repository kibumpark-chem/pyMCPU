#!/bin/bash
# Submit an MPI REMD job with --ntasks derived from the YAML physics grid.
#
# Design: YAML describes physics (temperatures, Q/N targets). This script
# describes infrastructure (nodes, walltime, MPI ranks). Do not put mpi: in YAML.
#
# Usage:
#   bash scripts/submit.sh inputs/template.yaml
#   bash scripts/submit.sh inputs/template.yaml --dry-run
#
# See docs/running_remd.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-inputs/template.yaml}"
shift || true

DRY_RUN=0
EXTRA_SBATCH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      EXTRA_SBATCH+=("$1")
      shift
      ;;
  esac
done

# Resolve config relative to CWD, then repo root
if [[ ! -f "${CONFIG}" && -f "${REPO_ROOT}/${CONFIG}" ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

# Derive replica count from YAML so --ntasks stays in sync automatically.
# Formula: len(temperatures) * len(q_targets|n_targets)  [or * 1 if neither]
N_REPLICAS=$(python -c "
import yaml, sys
sys.path.insert(0, '${REPO_ROOT}')
from pymcpu.config import replica_grid_dims
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
print(replica_grid_dims(c)[0])
")

echo "Submitting: ${N_REPLICAS} replicas for ${CONFIG}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "(dry-run) would run:"
  echo "  sbatch --ntasks=${N_REPLICAS} --export=ALL,PYMCPU_REPO=${REPO_ROOT} ${EXTRA_SBATCH[*]:-} ${SCRIPT_DIR}/job_template.slurm ${CONFIG}"
  exit 0
fi

# SLURM writes --output/--error relative to the submit dir and creates them
# BEFORE the job body runs, so logs/ must exist here, not inside the job.
mkdir -p logs

# #SBATCH cannot use shell variables — pass --ntasks on the sbatch command line.
# PYMCPU_REPO is exported because the batch script cannot locate the repo
# itself: SLURM spools it, so its BASH_SOURCE points into /var/slurmd/spool.
sbatch \
  --ntasks="${N_REPLICAS}" \
  --export="ALL,PYMCPU_REPO=${REPO_ROOT}" \
  "${EXTRA_SBATCH[@]}" \
  "${SCRIPT_DIR}/job_template.slurm" \
  "${CONFIG}"
