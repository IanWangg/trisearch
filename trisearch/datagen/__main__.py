"""Generate a triangulation dataset.

Example (4D training set as used for TriSearch)::

    trisearch-datagen --dim 4 --vertices 9 10 11 --per_vertex_count 50 \\
        --num_points 24 --max_triangulations 10000 --out data/4d_train
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from tqdm import tqdm


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--vertices", type=int, nargs="+", required=True, help="Vertex counts to generate.")
    p.add_argument("--per_vertex_count", type=int, required=True)
    p.add_argument("--num_points", type=int, default=24, help="Points sampled per candidate hull.")
    p.add_argument("--bound", type=float, default=10.0)
    p.add_argument("--distributions", nargs="+", default=["uniform", "normal"], choices=["uniform", "normal"])
    p.add_argument("--max_triangulations", type=int, default=2000, help="Per polytope (<= 0 for all).")
    p.add_argument("--index_offset", type=int, default=0, help="First point_config_index.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output directory (datasets.save_to_disk).")
    p.add_argument("--push_to_hub", default=None, help="Optionally upload to this Hugging Face dataset repo.")
    args = p.parse_args(argv)

    import datasets

    from .polytopes import sample_polytopes
    from .triangulations import enumerate_triangulations, triangulation_row

    polytopes = sample_polytopes(
        args.dim, args.vertices, args.per_vertex_count, num_points=args.num_points,
        bound=args.bound, distributions=args.distributions, seed=args.seed,
    )
    limit = args.max_triangulations if args.max_triangulations > 0 else None
    rows, index = [], args.index_offset
    for n in sorted(polytopes):
        for vertices in tqdm(polytopes[n], desc=f"{n} vertices"):
            float_vertices = [[float(v) for v in vertex] for vertex in vertices]
            for simplices in enumerate_triangulations(vertices, limit):
                rows.append(triangulation_row(float_vertices, simplices, index))
            index += 1
    dataset = datasets.Dataset.from_list(rows)
    dataset.save_to_disk(args.out)
    print(f"Saved {len(dataset)} triangulations of {index - args.index_offset} polytopes to {args.out}")
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
