#!/bin/bash
set -e
CSV="C:\Users\acost\repos\csvs"

for arch in arch37 arch89 rand032; do
  echo "=== results_basediff_${arch} (arch=$arch) ==="
  python train_loo_basediff.py --csv_dir "$CSV" --architecture "$arch" \
      --h_hidden 32 --h_layers 1 --output_dir "results_basediff_${arch}" --workers 6
done
