"""Random simplicial polytopes, deduplicated up to combinatorial isomorphism."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np

from . import require_sage


def sample_polytopes(
    dim: int,
    vertex_counts: Sequence[int],
    per_vertex_count: int,
    *,
    num_points: int,
    bound: float = 10.0,
    distributions: Sequence[str] = ("uniform", "normal"),
    seed: int = 0,
    max_duplicates: int = 32,
) -> Dict[int, List[list]]:
    """Sample ``per_vertex_count`` polytopes for each vertex count.

    Each candidate is the convex hull of ``num_points`` i.i.d. points drawn from
    one of ``distributions`` (chosen at random per candidate): ``uniform`` on
    ``[-bound, bound]^d`` or ``normal`` with standard deviation ``bound / 2``.
    Points are converted to exact rationals.  A bucket stops early after
    ``max_duplicates`` isomorphic duplicates.  Returns ``n -> [vertices]``
    (each a list of Sage rationals, in Sage's vertex order).
    """
    require_sage()
    from sage.all import Polyhedron, Rational

    np.random.seed(seed)
    random.seed(seed)
    kept: Dict[int, list] = {n: [] for n in vertex_counts}
    duplicates = {n: 0 for n in vertex_counts}

    def open_buckets():
        return [n for n in kept if len(kept[n]) < per_vertex_count and duplicates[n] < max_duplicates]

    while open_buckets():
        distribution = random.choice(tuple(distributions))
        if distribution == "uniform":
            points = np.random.uniform(-bound, bound, size=(num_points, dim))
        elif distribution == "normal":
            points = np.random.normal(0.0, bound / 2, size=(num_points, dim))
        else:
            raise ValueError(f"Unknown distribution {distribution!r}")
        polytope = Polyhedron(vertices=[[Rational(v) for v in p] for p in points])
        if not polytope.is_simplicial():
            continue
        n = polytope.n_vertices()
        if n not in kept or len(kept[n]) >= per_vertex_count:
            continue
        if any(polytope.is_combinatorially_isomorphic(other) for other in kept[n]):
            duplicates[n] += 1
            continue
        kept[n].append(polytope)
        print(f"  {n} vertices: {len(kept[n])}/{per_vertex_count}")
    return {n: [p.vertices_list() for p in polys] for n, polys in kept.items()}
