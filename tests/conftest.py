import json
from pathlib import Path

import pytest

from trisearch.data import polytope_vertices
from trisearch.flips import build_flip_lookups
from trisearch.state import TriangulationState

FIXTURES = Path(__file__).parent / "fixtures"

# Triangular bipyramid: a 2 <-> 3 flip exchanges the two tetrahedra glued along
# the equator with three tetrahedra around the apex-apex edge.
BIPYRAMID = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
BIPYRAMID_TWO = [(0, 1, 2, 3), (0, 1, 2, 4)]
BIPYRAMID_THREE = [(0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4)]


def load_rows(name):
    return json.loads((FIXTURES / f"tiny_{name}.json").read_text())


@pytest.fixture(params=["3d", "4d"])
def rows(request):
    return load_rows(request.param)


@pytest.fixture
def rows_3d():
    return load_rows("3d")


@pytest.fixture
def lookups_for():
    def build(rows):
        return build_flip_lookups(polytope_vertices(rows), num_workers=1)

    return build


def make_state(row, lookups):
    state = TriangulationState.from_row(row)
    state.find_available_actions(lookups[state.point_config_index])
    return state


@pytest.fixture
def local_dataset(tmp_path):
    """Save fixture rows as a ``datasets.save_to_disk`` directory."""
    import datasets

    def save(name):
        path = tmp_path / f"ds_{name}"
        datasets.Dataset.from_list(load_rows(name)).save_to_disk(str(path))
        return str(path)

    return save
