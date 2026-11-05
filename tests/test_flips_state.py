import pickle

import pytest

from conftest import BIPYRAMID, BIPYRAMID_THREE, BIPYRAMID_TWO, make_state
from trisearch import flips
from trisearch.flips import bistellar_flips, build_flip_lookup, circuits, load_or_build_flip_lookups
from trisearch.state import TriangulationState, batch_transitions


def test_square_has_one_circuit():
    square = [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert circuits(square) == (((0, 3), (1, 2)),)
    (left, right), = bistellar_flips(square)
    assert left == ((0, 1, 3), (0, 2, 3))
    assert right == ((0, 1, 2), (1, 2, 3))


def test_bipyramid_two_three_flip():
    (left, right), = bistellar_flips(BIPYRAMID)
    assert {len(left), len(right)} == {2, 3}
    assert set(left) | set(right) == set(BIPYRAMID_TWO) | set(BIPYRAMID_THREE)


def test_collinear_points_use_small_circuits():
    # Non-uniform configuration: a 3-point circuit on a line inside a triangle.
    points = [(0, 0), (2, 0), (1, 0), (1, 2)]
    assert ((0, 1), (2,)) in circuits(points)


def test_state_transition_round_trip():
    lookup = build_flip_lookup(BIPYRAMID)
    state = TriangulationState(BIPYRAMID, 0, BIPYRAMID_TWO)
    state.find_available_actions(lookup)
    assert state.subcomplex_actions() == ((0, 1, 2, 3, 4),)
    simplices, edges, key = state.transition((0, 1, 2, 3, 4))
    assert simplices == frozenset(BIPYRAMID_THREE)
    assert (3, 4) in edges and len(edges) == 10
    flipped = state.successor(simplices, edges)
    assert flipped.key == key and flipped == key
    flipped.find_available_actions(lookup)
    assert flipped.transition((0, 1, 2, 3, 4))[2] == state.key
    with pytest.raises(ValueError):
        state.transition((0, 1, 2))


def test_fixture_flips_are_reversible(rows, lookups_for):
    lookups = lookups_for(rows)
    for row in rows:
        state = make_state(row, lookups)
        assert state.subcomplex_actions(), "every seed triangulation has a flip"
        for action in state.subcomplex_actions():
            simplices, edges, key = state.transition(action)
            assert key != state.key
            nxt = state.successor(simplices, edges)
            nxt.find_available_actions(lookups[state.point_config_index])
            assert action in nxt.subcomplex_actions()
            assert nxt.transition(action)[2] == state.key


def test_state_hashing_and_row_round_trip(rows_3d, lookups_for):
    row = rows_3d[0]
    state = TriangulationState.from_row(row)
    again = TriangulationState.from_row(state.to_row())
    assert state == again and hash(state) == hash(again)
    table = {state: state}
    assert table.get(state.key) is state
    assert pickle.loads(pickle.dumps(state)) == state


def test_batch_transitions_with_pool(rows_3d, lookups_for):
    from trisearch.state import TransitionPool

    lookups = lookups_for(rows_3d)
    states = [make_state(r, lookups) for r in rows_3d[:4]]
    actions = [s.subcomplex_actions()[0] for s in states]
    pool = TransitionPool(num_workers=2)
    try:
        assert batch_transitions(states, actions, pool) == batch_transitions(states, actions)
    finally:
        pool.shutdown()


def test_flip_cache_round_trip(tmp_path, monkeypatch):
    vertices = {7: tuple(BIPYRAMID)}
    first = load_or_build_flip_lookups("owner/name", vertices, cache_dir=str(tmp_path))
    assert (tmp_path / "owner" / "name.pkl").exists()

    def fail(*args, **kwargs):
        raise AssertionError("cache should have been used")

    monkeypatch.setattr(flips, "build_flip_lookups", fail)
    assert load_or_build_flip_lookups("owner/name", vertices, cache_dir=str(tmp_path)) == first
    with pytest.raises(ValueError):
        load_or_build_flip_lookups("../escape", vertices, cache_dir=str(tmp_path))
