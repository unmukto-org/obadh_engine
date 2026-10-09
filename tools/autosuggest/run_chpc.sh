#!/usr/bin/env bash
# Invoke only through the user's chpc-gpu run/start wrapper.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
module load deeplearning/24.12.torch
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
exec apptainer exec --nv "$CONTAINERFILE" python "$@"
