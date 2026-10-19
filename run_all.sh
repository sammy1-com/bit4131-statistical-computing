#!/usr/bin/env bash
# Re-run every topic in Python and R. Usage (from repo root): bash run_all.sh
set -u
for d in topics/*/; do
  echo "== $d"
  python "$d/python/analysis.py" > /dev/null || echo "  Python FAILED"
  Rscript "$d/r/analysis.R"      > /dev/null || echo "  R FAILED"
done
