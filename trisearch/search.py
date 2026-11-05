"""Flip-graph search: the learned policy and the classical baselines.

All methods start from one triangulation and stop after ``budget`` *visits*:

* ``policy``      one move per step: flip to the unvisited neighbour with the
                  highest policy logit (or sample with ``temperature``); every
                  generated neighbour is marked visited.  Ends early when all
                  neighbours were already seen.  Costs 1 visit per move.
* ``greedy``      move to the best neighbour, even if it is worse (revisits allowed).
* ``dfs``         depth-first search, best neighbours explored first, no revisits.
* ``best_first``  best-first search on a heap keyed by the objective, no revisits.

The baselines evaluate the objective on every neighbour of an expanded state
and are charged one visit per neighbour.

Returns ``[visits, expansions, best_value, wall_time_s]`` (4 × K array), one
column per recorded step.
"""

from __future__ import annotations

import heapq
import random
import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from .data import featurize
from .objectives import Objective
from .state import TriangulationState

METHODS = ("policy", "greedy", "dfs", "best_first")


def _policy_logits(policy, state: TriangulationState, device) -> List[float]:
    with torch.no_grad():
        _, logits = policy.get_value_and_logits(Batch.from_data_list([featurize(state)]).to(device))
    return [float(v) for v in logits[0, : len(state.subcomplex_actions())].cpu().tolist()]


def _select(logits: List[float], temperature: float, deterministic: bool) -> int:
    if deterministic or temperature <= 0.0:
        return int(np.argmax(np.asarray(logits, dtype=float)))
    scaled = torch.tensor(logits, dtype=torch.float) / float(temperature)
    return int(torch.distributions.Categorical(logits=scaled).sample().item())


def search(
    init_state: TriangulationState,
    objective: Objective,
    lookups,
    *,
    method: str,
    budget: int,
    deterministic: bool = True,
    temperature: float = 1.0,
    policy=None,
    device="cpu",
) -> np.ndarray:
    """Minimize ``objective`` from ``init_state``; see module docstring.

    With ``deterministic=False`` the baselines break ties between equally good
    neighbours at random, and ``policy`` samples from ``softmax(logits / temperature)``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}.")
    if method == "policy" and policy is None:
        raise ValueError("method='policy' requires a policy.")

    best = objective(init_state)
    visited = {init_state.key}
    frontier: deque = deque([init_state])
    heap: list = []
    pushes = 0

    def push(state: TriangulationState, value: float) -> None:
        nonlocal pushes
        tie = float(pushes) if deterministic else random.random()
        heapq.heappush(heap, (value, tie, pushes, state))
        pushes += 1

    if method == "best_first":
        frontier.clear()
        push(init_state, best)

    visits = expansions = 0
    started = time.perf_counter()
    record = [[0], [0], [best], [0.0]]

    def log() -> None:
        for row, value in zip(record, (visits, expansions, best, time.perf_counter() - started)):
            row.append(value)

    while (heap if method == "best_first" else frontier) and visits < budget:
        current = heapq.heappop(heap)[-1] if method == "best_first" else frontier.pop()
        expansions += 1
        if not current.actions_ready:
            current.find_available_actions(lookups[current.point_config_index])
        actions = current.subcomplex_actions()
        logits = _policy_logits(policy, current, device) if method == "policy" else None
        if method != "policy":
            visits += len(actions)

        neighbours, neighbour_logits = [], []
        for i, action in enumerate(actions):
            simplices, edges, key = current.transition(action)
            if key in visited and method != "greedy":
                continue
            if method != "best_first":
                visited.add(key)
            neighbours.append(current.successor(simplices, edges))
            if logits is not None:
                neighbour_logits.append(logits[i])
        if not neighbours:
            continue

        if method == "policy":
            chosen = neighbours[_select(neighbour_logits, temperature, deterministic)]
            best = min(best, objective(chosen))
            visits += 1
            log()
            frontier = deque([chosen])
            continue

        values = [objective(s) for s in neighbours]
        step_best = min(values)
        best = min(best, step_best)
        log()

        if method == "dfs":
            # Stack: the best neighbour is pushed last and expanded next.
            ordered = sorted(neighbours, key=objective, reverse=True)
            if not deterministic:
                ties = [i for i, v in enumerate(values) if v == step_best]
                if len(ties) > 1:
                    pick = neighbours[random.choice(ties)]
                    ordered = [s for s in ordered if s is not pick] + [pick]
            frontier.extend(ordered)
        elif method == "best_first":
            for state, value in zip(neighbours, values):
                visited.add(state.key)
                push(state, value)
        else:  # greedy
            ties = [i for i, v in enumerate(values) if v == step_best]
            pick = ties[0] if deterministic or len(ties) == 1 else random.choice(ties)
            frontier = deque([neighbours[pick]])

    return np.vstack(record)


def seed_everything(seed: int, point_config_index: int) -> None:
    """Per-polytope seed, so results do not depend on worker scheduling."""
    local = (int(seed) + int(point_config_index) * 1000003) % (2**32)
    random.seed(local)
    np.random.seed(local)
    torch.manual_seed(local)
