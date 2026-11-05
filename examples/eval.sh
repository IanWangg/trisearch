#!/usr/bin/env bash
# Evaluate TriSearch and the baselines on the held-out 3D/4D polytopes (500 visits).
set -euo pipefail
ITER=${ITER:-2000}
declare -A SETS=(
  [3d_d13]="ywang3/bistellar_dataset_chunk_7 50"
  [3d_d14]="ywang3/bistellar_dataset_chunk_10 50"
  [4d_d13]="ywang3/bistellar_dataset_4D_eval_chunk_0 20"
  [4d_d14]="ywang3/bistellar_dataset_4D_eval_chunk_1 20"
)
for name in "${!SETS[@]}"; do
  read -r dataset count <<< "${SETS[$name]}"
  dim=${name%%_*}
  for objective in min_tri min_edge_length min_diameter; do
    trisearch-eval --dataset "$dataset" --objective "$objective" --method policy \
      --checkpoint "runs/${dim}_${objective}/${ITER}.pt" --budget 500 --max_polytopes "$count" \
      --num_workers 4 --output "results/${name}_${objective}_policy.npz"
    for method in greedy dfs best_first; do
      trisearch-eval --dataset "$dataset" --objective "$objective" --method "$method" --budget 500 \
        --max_polytopes "$count" --num_workers 8 --output "results/${name}_${objective}_${method}.npz"
    done
  done
done
trisearch-summarize results/*.npz --budgets 500
