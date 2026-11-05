# TriSearch

Learning to navigate triangulation spaces of simplicial polytopes with bistellar flips.

TriSearch trains one policy per objective that optimizes a triangulation of an
unseen 3D or 4D polytope by a sequence of bistellar flips. The objectives are:

| `--objective`      | minimized quantity |
|--------------------|--------------------|
| `min_tri`          | number of top-dimensional simplices |
| `min_edge_length`  | total edge length (minimum-weight triangulation) |
| `min_diameter`     | diameter of the dual graph (simplices adjacent iff they share a facet) |

The model has three parts:

- **Encoder:** an EGNN on the 1-skeleton of the current triangulation.
- **Actor:** a simplicial neural network over the top simplices. It lifts vertex embeddings to simplices, propagates along the normalized down Laplacian, and max-pools over `T+(Z)` to score each flippable circuit `Z`.
- **Critic:** pools the same simplex features.

Training uses PPO with GAE and a count-based exploration bonus over the discovered part of the flip graph.

## Installation

```bash
pip install -e .            # numpy, torch, torch_geometric, datasets, tqdm
pip install -e ".[dev]"     # + pytest
pip install wandb           # optional logging
```

Install PyTorch and PyTorch Geometric for your CUDA version first if needed.
Training and evaluation are pure Python plus PyTorch. Flips are enumerated exactly with rational arithmetic; no SageMath or TOPCOM is needed. Only the optional data generation needs Sage (see below).

## Datasets

The datasets are on the Hugging Face Hub. Each row is one triangulation (`tri_simplices`, `tri_edges`, ...) of a polytope (`vertices`, `point_config_index`).

| role | dimension | polytopes | Hugging Face dataset |
|---|---|---|---|
| train | 3D | 140 (6–11 vertices), ≤ 2048 triangulations each | `ywang3/bistellar_dataset_chunk_0`, `ywang3/bistellar_dataset_chunk_1` |
| eval, 13 vertices | 3D | first 50 of 63 | `ywang3/bistellar_dataset_chunk_7` |
| eval, 14 vertices | 3D | first 50 of 63 | `ywang3/bistellar_dataset_chunk_10` |
| train | 4D | 150 (9–11 vertices), ≤ 10000 triangulations each | `ywang3/polytope_triangulation_4d_chunk_large_{0,1,2}` |
| eval, 13 vertices | 4D | 20 | `ywang3/bistellar_dataset_4D_eval_chunk_0` |
| eval, 14 vertices | 4D | 20 | `ywang3/bistellar_dataset_4D_eval_chunk_1` |

Any `datasets.save_to_disk` directory with the same columns works in place of a hub id.

On first use, the flip lookup of every polytope is computed and cached under `cache/flip_lookup/` (`--flip_cache_dir`). This takes a few minutes for the 4D sets.

## Training

```bash
# 3D (defaults are the paper settings: 2001 iterations, 128 rollouts x 50 flips, beta = 1e-3)
trisearch-train --objective min_tri \
    --datasets ywang3/bistellar_dataset_chunk_0 ywang3/bistellar_dataset_chunk_1 \
    --num_workers 16 --output_dir runs/3d_min_tri

# 4D
trisearch-train --objective min_tri \
    --datasets ywang3/polytope_triangulation_4d_chunk_large_0 \
               ywang3/polytope_triangulation_4d_chunk_large_1 \
               ywang3/polytope_triangulation_4d_chunk_large_2 \
    --num_states 512 --save_interval 500 --eval_interval 100 \
    --num_workers 32 --output_dir runs/4d_min_tri
```

- **Held-out polytopes:** `min(20, #polytopes // 10)` polytopes are held out and used for the periodic evaluation return.
- **Checkpoints:** written to `<output_dir>/<iteration>.pt` as `{"config", "state_dict"}`, next to `config.json`.
- **Workers:** `--num_workers` parallelizes flip transitions over processes.
- **More options:** `trisearch-train --help` lists all of them (PPO coefficients, augmentation, model width, wandb).

## Evaluation

```bash
# TriSearch: argmax over the policy among unvisited neighbours, 500 visits
trisearch-eval --dataset ywang3/bistellar_dataset_chunk_7 --objective min_tri \
    --method policy --checkpoint runs/3d_min_tri/2000.pt \
    --budget 500 --max_polytopes 50 --num_workers 4 --output results/3d_d7_min_tri_policy.npz

# Baselines: greedy | dfs | best_first
trisearch-eval --dataset ywang3/bistellar_dataset_chunk_7 --objective min_tri \
    --method best_first --budget 500 --max_polytopes 50 --num_workers 8 \
    --output results/3d_d7_min_tri_best_first.npz
```

How each method searches and how it is charged against the budget:

| method | search | budget charge |
|---|---|---|
| `policy` | one flip per step to the unvisited neighbour with the highest logit (`--stochastic --temperature T` samples instead); every generated neighbour is marked visited, and the search stops if none is left | 1 visit per flip |
| `greedy` | moves to the best neighbour, even when it is worse | one visit per neighbour evaluated |
| `dfs` | depth-first, best neighbour first, no revisits | one visit per neighbour evaluated |
| `best_first` | best-first frontier ordered by the objective, no revisits | one visit per neighbour evaluated |

Each polytope is searched from its first triangulation in the dataset, seeded with `seed + 1000003 · point_config_index`.

The output `.npz` maps `point_config_index` to a `4 × K` array of rows `visits, expansions, best_value, wall_time_s`. A `.json` sidecar stores the settings.

```bash
trisearch-summarize results/*.npz --budgets 500 5000 [--reference optima.csv]
```

This reports the mean ± standard error of the best value within each budget.

With a reference file (CSV with columns `point_config_index,value`, holding an optimum or best known value per polytope) it also reports:

- the relative gap `(y − opt) / |y|`
- the hit ratio

## Generating new data (optional)

This needs SageMath with the TOPCOM engine, for example `conda install -c conda-forge sage topcom`.

```bash
trisearch-datagen --dim 4 --vertices 9 10 11 --per_vertex_count 50 --num_points 24 \
    --max_triangulations 10000 --out data/my_4d_train
```

The pipeline:

1. **Sample a candidate:** each candidate polytope is the convex hull of `--num_points` random points.
2. **Pick the distribution:** for each candidate, one of `--distributions` is chosen at random:
   - `uniform` on `[-bound, bound]^d`
   - `normal` with standard deviation `bound/2`

   The default `bound` is 10, which is how the released datasets were sampled.
3. **Filter:** only simplicial polytopes that are not combinatorially isomorphic to an already kept one are kept.
4. **Enumerate:** TOPCOM enumerates up to `--max_triangulations` triangulations per polytope.

## Layout

```
trisearch/
  flips.py        exact circuit / bistellar-flip enumeration, per-polytope lookup + disk cache
  state.py        TriangulationState (actions = flip supports), transition pool
  objectives.py   min_tri, min_edge_length, min_diameter
  topology.py     down Laplacian and candidate-simplex memberships for the SNN
  data.py         dataset loading, featurization, similarity augmentation
  models/         EGNN encoder, SNN actor, TriSearchPolicy
  train.py        PPO training         (trisearch-train)
  search.py       policy + greedy/dfs/best_first search
  evaluate.py     batch evaluation     (trisearch-eval)
  summarize.py    budget tables        (trisearch-summarize)
  datagen/        optional Sage/TOPCOM data generation (trisearch-datagen)
```

Run the tests with `pytest`. The Sage-dependent test is skipped when Sage is not installed.

## Citation

```bibtex
@article{trisearch2026,
  title   = {TriSearch: Learning to Navigate Triangulation Spaces},
  journal = {arXiv preprint arXiv:2605.30220},
  year    = {2026}
}
```

## License

MIT
