#!/bin/bash
set -e
CSV="C:\Users\acost\repos\csvs"

for arch in arch37 arch89 rand032; do
  echo "=== results_basediff_joint_${arch} (arch=$arch) ==="
  python train_loo_basediff_joint.py --csv_dir "$CSV" --architecture "$arch" \
      --h_hidden 32 --h_layers 1 --output_dir "results_basediff_joint_${arch}" --workers 7
done
