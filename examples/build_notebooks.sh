#!/usr/bin/env bash
# Regenerate the executed notebooks from the .py sources (the .py files are the source of truth).
# Needs: pip install jupytext nbconvert ipykernel, plus requirements-dev.txt
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
for f in 0*.py; do
  jupytext --quiet --to ipynb "$f"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "${f%.py}.ipynb"
done
