#!/usr/bin/env bash
# Train one 4D policy per objective with the paper settings.
set -euo pipefail
for objective in min_tri min_edge_length min_diameter; do
  trisearch-train --objective "$objective" \
    --datasets ywang3/polytope_triangulation_4d_chunk_large_0 \
               ywang3/polytope_triangulation_4d_chunk_large_1 \
               ywang3/polytope_triangulation_4d_chunk_large_2 \
    --num_states 512 --save_interval 500 --eval_interval 100 \
    --num_workers 32 --output_dir "runs/4d_${objective}"
done
