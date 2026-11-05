"""Triangulation enumeration (TOPCOM via Sage) and dataset rows."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from . import require_sage

Simplex = Tuple[int, ...]


def enumerate_triangulations(vertices: Sequence[Sequence], max_triangulations: Optional[int]) -> List[Tuple[Simplex, ...]]:
    """Up to ``max_triangulations`` triangulations of the point configuration, in TOPCOM's order."""
    require_sage()
    from sage.all import Rational
    from sage.geometry.triangulation.point_configuration import PointConfiguration

    PointConfiguration.set_engine("topcom")
    configuration = PointConfiguration([[Rational(v) for v in vertex] for vertex in vertices])
    out = []
    for triangulation in configuration.triangulations():
        out.append(tuple(sorted(tuple(sorted(int(v) for v in s)) for s in triangulation)))
        if max_triangulations is not None and len(out) >= max_triangulations:
            break
    return out


def triangulation_row(vertices: Sequence[Sequence[float]], simplices: Sequence[Simplex], point_config_index: int) -> Dict:
    """Dataset row with the faces of a triangulation (boundary facets lie in exactly one simplex)."""
    simplices = sorted(tuple(sorted(s)) for s in simplices)
    facet_count: Dict[Simplex, int] = {}
    edges = set()
    for simplex in simplices:
        edges.update(combinations(simplex, 2))
        for facet in combinations(simplex, len(simplex) - 1):
            facet_count[facet] = facet_count.get(facet, 0) + 1
    facets = set(facet_count)
    boundary_facets = {f for f, c in facet_count.items() if c == 1}
    boundary_edges = {e for f in boundary_facets for e in combinations(f, 2)}

    def listed(faces):
        return [list(face) for face in sorted(faces)]

    return {
        "tri_simplices": listed(simplices),
        "tri_edges": listed(edges),
        "tri_facets": listed(facets),
        "tri_boundary_edges": listed(boundary_edges),
        "tri_boundary_facets": listed(boundary_facets),
        "tri_internal_edges": listed(edges - boundary_edges),
        "tri_internal_facets": listed(facets - boundary_facets),
        "vertices": [[float(v) for v in vertex] for vertex in vertices],
        "point_config_index": int(point_config_index),
    }
