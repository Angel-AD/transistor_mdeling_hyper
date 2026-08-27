#!/bin/bash
set -e
CSV="C:\Users\acost\repos\csvs"

for arch in arch37 arch89 rand032; do
  echo "=== results_deltaweights_noreg_${arch} (arch=$arch, no reg) ==="
  python train_loo_deltaweights_noreg.py --csv_dir "$CSV" --architecture "$arch" \
      --h_hidden 32 --h_layers 1 --output_dir "results_deltaweights_noreg_${arch}" --workers 6

  echo "=== results_deltaweights_reg_${arch} (arch=$arch, reg=0.01) ==="
  python train_loo_deltaweights_reg.py --csv_dir "$CSV" --architecture "$arch" \
      --delta_reg_weight 0.01 --h_hidden 32 --h_layers 1 \
      --output_dir "results_deltaweights_reg_${arch}" --workers 6
done
