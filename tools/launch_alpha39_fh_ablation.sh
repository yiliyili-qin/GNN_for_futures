#!/usr/bin/env bash
# Run with an existing environment and verified private inputs; no deployment.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${ALPHA39_PYTHON:-python}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec "$PYTHON" -I -B "$ROOT/tools/run_alpha39_fh_ablation.py" "$@"
