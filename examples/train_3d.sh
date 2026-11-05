#!/usr/bin/env bash
# Train one 3D policy per objective with the paper settings.
set -euo pipefail
for objective in min_tri min_edge_length min_diameter; do
  trisearch-train --objective "$objective" \
    --datasets ywang3/bistellar_dataset_chunk_0 ywang3/bistellar_dataset_chunk_1 \
    --num_workers 16 --output_dir "runs/3d_${objective}"
done
