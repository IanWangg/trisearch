import pytest

from conftest import BIPYRAMID, BIPYRAMID_TWO
from trisearch.datagen.triangulations import triangulation_row


def test_triangulation_row_faces():
    row = triangulation_row(BIPYRAMID, BIPYRAMID_TWO, point_config_index=3)
    assert row["point_config_index"] == 3
    assert len(row["tri_facets"]) == 7 and len(row["tri_boundary_facets"]) == 6
    assert row["tri_internal_facets"] == [[0, 1, 2]]
    assert len(row["tri_edges"]) == 9 and row["tri_internal_edges"] == []


def test_sage_pipeline_small():
    pytest.importorskip("sage.all")
    from trisearch.datagen.polytopes import sample_polytopes
    from trisearch.datagen.triangulations import enumerate_triangulations
    from trisearch.flips import build_flip_lookup
    from trisearch.state import TriangulationState

    polytopes = sample_polytopes(3, [6], 1, num_points=8, seed=0)
    (vertices,) = polytopes[6]
    triangulations = enumerate_triangulations(vertices, max_triangulations=None)
    float_vertices = [[float(v) for v in p] for p in vertices]
    lookup = build_flip_lookup(float_vertices)
    keys = {TriangulationState(float_vertices, 0, t).key for t in triangulations}
    # The flip graph of a 3D point set in convex position is connected: flips from one triangulation reach all.
    start = TriangulationState(float_vertices, 0, triangulations[0])
    seen, frontier = {start.key}, [start]
    while frontier:
        state = frontier.pop()
        state.find_available_actions(lookup)
        for action in state.subcomplex_actions():
            simplices, edges, key = state.transition(action)
            if key not in seen:
                seen.add(key)
                frontier.append(state.successor(simplices, edges))
    assert seen == keys
