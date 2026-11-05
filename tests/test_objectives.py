import math

import pytest

from conftest import BIPYRAMID, BIPYRAMID_THREE, BIPYRAMID_TWO, make_state
from trisearch.objectives import Objective, dual_graph_diameter, num_simplices, total_edge_length
from trisearch.state import TriangulationState

TWO = TriangulationState(BIPYRAMID, 0, BIPYRAMID_TWO)
THREE = TriangulationState(BIPYRAMID, 0, BIPYRAMID_THREE)


def test_num_simplices():
    assert num_simplices(TWO) == 2.0 and num_simplices(THREE) == 3.0


def test_total_edge_length_adds_apex_edge():
    # The 2 -> 3 flip adds exactly the apex-apex edge of length 2.
    assert total_edge_length(THREE) - total_edge_length(TWO) == pytest.approx(2.0)
    expected = 2 ** 0.5 + 2 * 5 ** 0.5  # equator triangle
    expected += 2 * (2 * 2 ** 0.5 + 3 ** 0.5)  # spokes from both apexes
    assert total_edge_length(TWO) == pytest.approx(expected)


def test_dual_graph_diameter():
    assert dual_graph_diameter(TWO) == 1.0
    assert dual_graph_diameter(THREE) == 1.0
    # A path of three triangles in a strip has diameter 2.
    strip = TriangulationState([(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)], 0, [(0, 1, 2), (1, 2, 3), (1, 3, 4)])
    assert dual_graph_diameter(strip) == 2.0
    single = TriangulationState([(0, 0), (1, 0), (0, 1)], 0, [(0, 1, 2)])
    assert dual_graph_diameter(single) == 0.0


def test_objective_reward_and_cache():
    objective = Objective("min_tri")
    assert objective.reward(THREE, TWO) == 1.0
    assert objective.reward(TWO, THREE) == -1.0
    with pytest.raises(ValueError):
        Objective("max_tri")


@pytest.mark.parametrize("name", ["min_tri", "min_edge_length", "min_diameter"])
def test_objectives_on_fixtures(name, rows, lookups_for):
    lookups = lookups_for(rows)
    objective = Objective(name)
    for row in rows:
        value = objective(make_state(row, lookups))
        assert math.isfinite(value) and value > 0
