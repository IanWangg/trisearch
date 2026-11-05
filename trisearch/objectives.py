"""Triangulation objectives (all minimized).

The RL reward of a flip ``s -> s'`` is the one-step improvement
``f(s) - f(s')``, so positive rewards mean a better triangulation.

* ``min_tri``          number of top-dimensional simplices.
* ``min_edge_length``  total Euclidean length of the 1-skeleton (minimum-weight triangulation).
* ``min_diameter``     diameter of the dual graph (top simplices adjacent iff they share a facet).
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Callable, Dict

import numpy as np

from .state import TriangulationState


def num_simplices(state: TriangulationState) -> float:
    return float(len(state.simplices))


def total_edge_length(state: TriangulationState) -> float:
    if not state.edges:
        return 0.0
    vertices = np.asarray(state.vertices, dtype=float)
    # Iterate the frozenset directly: summation order then matches the reference implementation bit for bit.
    edges = np.asarray(list(state.edges), dtype=int)
    return float(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).sum())


def dual_graph_diameter(state: TriangulationState) -> float:
    if not state.simplices:
        return 0.0
    size = max(len(s) for s in state.simplices)
    simplices = [s for s in state.simplices if len(s) == size]
    if len(simplices) <= 1:
        return 0.0
    face_to_simplices: Dict[tuple, list] = {}
    for index, simplex in enumerate(simplices):
        for face in combinations(simplex, size - 1):
            face_to_simplices.setdefault(face, []).append(index)
    adjacency = [set() for _ in simplices]
    for owners in face_to_simplices.values():
        if len(owners) == 2:
            a, b = owners
            adjacency[a].add(b)
            adjacency[b].add(a)

    def eccentricity(start: int) -> int:
        distance = [-1] * len(simplices)
        distance[start] = 0
        queue = deque([start])
        farthest = 0
        while queue:
            node = queue.popleft()
            farthest = max(farthest, distance[node])
            for neighbor in adjacency[node]:
                if distance[neighbor] == -1:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        return farthest

    return float(max(eccentricity(i) for i in range(len(simplices))))


_OBJECTIVES: Dict[str, Callable[[TriangulationState], float]] = {
    "min_tri": num_simplices,
    "min_edge_length": total_edge_length,
    "min_diameter": dual_graph_diameter,
}
OBJECTIVES = tuple(_OBJECTIVES)


class Objective:
    """Memoized objective ``f`` (keyed by state key) with ``reward(s, s') = f(s) - f(s')``."""

    def __init__(self, name: str):
        if name not in _OBJECTIVES:
            raise ValueError(f"Unknown objective {name!r}; choose from {', '.join(OBJECTIVES)}.")
        self.name = name
        self._fn = _OBJECTIVES[name]
        self._cache: Dict[str, float] = {}

    def __call__(self, state: TriangulationState) -> float:
        value = self._cache.get(state.key)
        if value is None:
            value = self._fn(state)
            self._cache[state.key] = value
        return value

    def reward(self, state: TriangulationState, next_state: TriangulationState) -> float:
        return self(state) - self(next_state)
