"""Triangulation states of the flip-graph MDP.

A state is a triangulation of a fixed point configuration (one polytope).  Its
actions are the bistellar flips available in it.  Each action is identified
by its *subcomplex*: the sorted vertex set of the flip's circuit (at most
``d + 2`` vertices), which is what the policy scores.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .flips import FlipLookup

Simplex = Tuple[int, ...]
Subcomplex = Tuple[int, ...]
SimplexSet = FrozenSet[Simplex]
Transition = Tuple[SimplexSet, FrozenSet[Tuple[int, int]], str]


def state_key(point_config_index: int, simplices: Iterable[Simplex]) -> str:
    canonical = tuple(sorted(tuple(sorted(simplex)) for simplex in simplices))
    return f"{point_config_index}:{canonical}"


def edges_from_simplices(simplices: Iterable[Simplex]) -> FrozenSet[Tuple[int, int]]:
    edges = set()
    for simplex in simplices:
        edges.update(combinations(tuple(sorted(simplex)), 2))
    return frozenset(sorted(edges))


class TriangulationState:
    """A triangulation, its key, and (lazily) its available flips.

    States compare and hash by ``key``, and also compare equal to their key
    string, so a ``dict`` keyed by states can be queried with a key.
    """

    # Available subcomplex actions per state key, shared by all instances so
    # that revisited states do not recompute them.
    _ACTION_CACHE: Dict[str, Tuple[Tuple[Subcomplex, ...], Dict[Subcomplex, Tuple[SimplexSet, SimplexSet]], FrozenSet[Subcomplex]]] = {}

    def __init__(
        self,
        vertices: Sequence[Sequence[float]],
        point_config_index: int,
        simplices: Iterable[Simplex],
        edges: Optional[Iterable[Tuple[int, int]]] = None,
    ):
        self.vertices = vertices
        self.point_config_index = int(point_config_index)
        self.simplices: SimplexSet = frozenset(sorted(tuple(sorted(s)) for s in simplices))
        self.edges = (
            edges_from_simplices(self.simplices)
            if edges is None
            else frozenset(sorted(tuple(sorted(e)) for e in edges))
        )
        self.key = state_key(self.point_config_index, self.simplices)
        self.visitation = 0
        self.available_flip_pairs: Tuple[Tuple[SimplexSet, SimplexSet], ...] = ()
        self.actions_ready = False
        self._subcomplex_actions: Tuple[Subcomplex, ...] = ()
        self._subcomplex_to_flips: Dict[Subcomplex, Tuple[SimplexSet, SimplexSet]] = {}
        self._ambiguous: FrozenSet[Subcomplex] = frozenset()
        self._subcomplex_ready = self._load_cached_actions()

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TriangulationState":
        """Build a state from a dataset row (``vertices``, ``tri_simplices``, ``tri_edges``, ``point_config_index``)."""
        return cls(
            vertices=row["vertices"],
            point_config_index=row["point_config_index"],
            simplices=row["tri_simplices"],
            edges=row["tri_edges"],
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "vertices": self.vertices,
            "tri_simplices": list(self.simplices),
            "tri_edges": list(self.edges),
            "point_config_index": self.point_config_index,
        }

    @property
    def dim(self) -> int:
        return len(next(iter(self.simplices))) - 1

    # -- actions ------------------------------------------------------------

    def find_available_actions(self, lookup: FlipLookup) -> None:
        """Collect the flips whose one side is a subcomplex of this triangulation."""
        if self.actions_ready:
            self._ensure_subcomplex_actions()
            return
        candidates = set()
        for simplex in self.simplices:
            candidates.update(lookup.simplex_to_flip_indices.get(simplex, ()))
        pairs = []
        for index in sorted(candidates):
            left_raw, right_raw = lookup.flip_pairs[index]
            left, right = frozenset(left_raw), frozenset(right_raw)
            if left.issubset(self.simplices):
                pairs.append((left, right))
            elif right.issubset(self.simplices):
                pairs.append((right, left))
        self.available_flip_pairs = tuple(pairs)
        self.actions_ready = True
        self._ensure_subcomplex_actions()

    def _load_cached_actions(self) -> bool:
        cached = self._ACTION_CACHE.get(self.key)
        if cached is None:
            return False
        self._subcomplex_actions, to_flips, self._ambiguous = cached
        self._subcomplex_to_flips = dict(to_flips)
        return True

    def _ensure_subcomplex_actions(self) -> None:
        if self._subcomplex_ready or self._load_cached_actions():
            self._subcomplex_ready = True
            return
        actions: List[Subcomplex] = []
        to_flips: Dict[Subcomplex, Tuple[SimplexSet, SimplexSet]] = {}
        ambiguous = set()
        for flip_from, flip_to in self.available_flip_pairs:
            subcomplex = tuple(sorted({v for s in flip_from for v in s} | {v for s in flip_to for v in s}))
            if subcomplex in to_flips:
                ambiguous.add(subcomplex)
            else:
                actions.append(subcomplex)
                to_flips[subcomplex] = (flip_from, flip_to)
        self._subcomplex_actions = tuple(actions)
        self._subcomplex_to_flips = to_flips
        self._ambiguous = frozenset(ambiguous)
        self._subcomplex_ready = True
        self._ACTION_CACHE[self.key] = (self._subcomplex_actions, dict(to_flips), self._ambiguous)

    def subcomplex_actions(self) -> Tuple[Subcomplex, ...]:
        """Available actions, in a deterministic order (the policy's logit order)."""
        if not self._subcomplex_ready:
            if not self.actions_ready:
                raise RuntimeError("Call find_available_actions(lookup) first.")
            self._ensure_subcomplex_actions()
        return self._subcomplex_actions

    # -- transitions --------------------------------------------------------

    def transition(self, subcomplex: Iterable[int]) -> Transition:
        """``(simplices, edges, key)`` of the triangulation reached by flipping ``subcomplex``."""
        target = tuple(sorted({int(v) for v in subcomplex if int(v) >= 0}))
        self.subcomplex_actions()
        if target in self._ambiguous:
            raise ValueError(f"Subcomplex {target} matches several flips in state {self.key}.")
        if target not in self._subcomplex_to_flips:
            raise ValueError(f"Subcomplex {target} is not an available flip in state {self.key}.")
        flip_from, flip_to = self._subcomplex_to_flips[target]
        simplices = (self.simplices - flip_from).union(flip_to)
        return simplices, edges_from_simplices(simplices), state_key(self.point_config_index, simplices)

    def successor(self, simplices: SimplexSet, edges) -> "TriangulationState":
        return TriangulationState(self.vertices, self.point_config_index, simplices, edges)

    # -- identity -----------------------------------------------------------

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TriangulationState):
            return self.key == other.key
        if isinstance(other, str):
            return self.key == other
        return False

    def __repr__(self) -> str:
        return f"TriangulationState({self.key})"


# ---------------------------------------------------------------------------
# Optional process pool for batched transitions
# ---------------------------------------------------------------------------

def _transition_worker(payload: Tuple[TriangulationState, Subcomplex]) -> Transition:
    state, subcomplex = payload
    return state.transition(subcomplex)


class TransitionPool:
    """Spawn-based process pool that applies flips to many states in parallel."""

    def __init__(self, num_workers: int = 0, chunksize: int = 4):
        workers = num_workers if num_workers > 0 else (os.cpu_count() or 1)
        self._executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        self.chunksize = max(1, int(chunksize))

    def transitions(self, states: Sequence[TriangulationState], actions: Sequence[Subcomplex]) -> List[Transition]:
        return list(self._executor.map(_transition_worker, zip(states, actions), chunksize=self.chunksize))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def batch_transitions(
    states: Sequence[TriangulationState],
    actions: Sequence[Iterable[int]],
    pool: Optional[TransitionPool] = None,
) -> List[Transition]:
    actions = [tuple(int(v) for v in action) for action in actions]
    if pool is None:
        return [state.transition(action) for state, action in zip(states, actions)]
    return pool.transitions(states, actions)
