"""Datasets, state featurization and coordinate augmentation.

Dataset rows (Hugging Face ``datasets`` format) carry ``vertices``
(``[n, d]`` floats), ``tri_simplices``, ``tri_edges`` and
``point_config_index`` (the polytope id).  Each row is one seed triangulation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import Data

from .flips import FlipLookup, load_or_build_flip_lookups
from .state import TriangulationState
from .topology import build_simplex_topology, simplex_tensor


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(dataset_id: str):
    """Load a dataset from the Hugging Face hub (``owner/name``) or a ``save_to_disk`` directory."""
    import datasets

    if os.path.isdir(dataset_id):
        loaded = datasets.load_from_disk(dataset_id)
        return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded
    return datasets.load_dataset(dataset_id, split="train")


def polytope_vertices(rows: Iterable[dict]) -> Dict[int, Tuple[Tuple[float, ...], ...]]:
    """``point_config_index -> vertices`` (first occurrence)."""
    out: Dict[int, Tuple[Tuple[float, ...], ...]] = {}
    for row in rows:
        index = int(row["point_config_index"])
        if index not in out:
            out[index] = tuple(tuple(float(v) for v in vertex) for vertex in row["vertices"])
    return out


def load_datasets(
    dataset_ids: Sequence[str],
    *,
    flip_cache_dir: Optional[str],
    num_workers: int = 0,
):
    """Load and concatenate datasets, plus the flip lookups of every polytope in them."""
    import datasets

    if not dataset_ids:
        raise ValueError("At least one dataset is required.")
    parts, lookups = [], {}
    for dataset_id in dataset_ids:
        print(f"Loading dataset {dataset_id}")
        rows = load_rows(dataset_id)
        parts.append(rows)
        for index, lookup in load_or_build_flip_lookups(
            dataset_id,
            lambda rows=rows: polytope_vertices(rows),
            cache_dir=flip_cache_dir,
            num_workers=num_workers,
        ).items():
            if index in lookups and lookups[index] != lookup:
                raise ValueError(f"Datasets disagree on polytope point_config_index={index}.")
            lookups[index] = lookup
    rows = parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)
    return rows, lookups


def coordinate_dim(rows) -> int:
    return len(rows[0]["vertices"][0])


def split_by_polytope(rows, *, num_test: int, seed: int = 0):
    """Hold out ``num_test`` whole polytopes (all their seed triangulations)."""
    all_indices = list(set(rows["point_config_index"]))
    if num_test >= len(all_indices):
        raise ValueError("num_test must be smaller than the number of polytopes.")
    np.random.seed(seed)
    train_indices = np.random.choice(all_indices, size=len(all_indices) - num_test, replace=False)
    test_indices = list(set(all_indices) - set(train_indices))
    num_proc = min(8, os.cpu_count() or 1) if len(rows) > 10_000 else None
    train = rows.filter(lambda r: r["point_config_index"] in train_indices, num_proc=num_proc)
    test = rows.filter(lambda r: r["point_config_index"] in test_indices, num_proc=num_proc)
    return train, test


def actionable_states(
    rows: Iterable[dict],
    lookups: Dict[int, FlipLookup],
    *,
    one_per_polytope: bool = False,
    max_polytopes: Optional[int] = None,
    skip_empty: bool = True,
) -> List[TriangulationState]:
    """States (with actions computed) for the given rows.

    With ``one_per_polytope`` only the first usable row of each polytope is
    kept, for at most ``max_polytopes`` polytopes.  With ``skip_empty``,
    states without any available flip are dropped.
    """
    states: List[TriangulationState] = []
    seen = set()
    for row in rows:
        index = int(row["point_config_index"])
        if one_per_polytope:
            if max_polytopes is not None and len(seen) >= max_polytopes:
                break
            if index in seen:
                continue
        state = TriangulationState.from_row(row)
        state.find_available_actions(lookups[index])
        if skip_empty and not state.subcomplex_actions():
            continue
        seen.add(index)
        states.append(state)
    return states


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

_FEATURE_CACHE: Dict[str, Data] = {}


def clear_feature_cache() -> None:
    _FEATURE_CACHE.clear()


def _build_features(state: TriangulationState) -> Data:
    edges = list(state.edges)
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    )
    actions = state.subcomplex_actions()
    width = state.dim + 2
    if any(len(a) > width for a in actions):
        raise ValueError("Found a flip support wider than d + 2.")
    subcomplex = torch.full((len(actions), width), -1, dtype=torch.long)
    for i, action in enumerate(actions):
        subcomplex[i, : len(action)] = torch.tensor(action, dtype=torch.long)

    simplices = simplex_tensor(state.simplices)
    topology = build_simplex_topology(simplices, subcomplex)
    return Data(
        x=torch.tensor(state.vertices, dtype=torch.float),
        edge_index=pyg.utils.to_undirected(edge_index),
        subcomplex_vertices=subcomplex,
        num_actions=len(actions),
        simplex_vertices=simplices,
        num_simplices_top=int(simplices.size(0)),
        laplacian_row=topology.laplacian_row,
        laplacian_col=topology.laplacian_col,
        laplacian_value=topology.laplacian_value,
        num_laplacian_entries=int(topology.laplacian_row.numel()),
        membership_candidate=topology.candidate,
        membership_simplex=topology.simplex,
        num_memberships=int(topology.candidate.numel()),
    )


def featurize(state: TriangulationState, transform: Optional["SimilarityTransform"] = None) -> Data:
    """PyG ``Data`` for the policy.

    Tensors are cached per state key and shared; ``transform`` replaces ``x`` on
    a shallow copy, so the cache is never modified.
    """
    cached = _FEATURE_CACHE.get(state.key)
    if cached is None:
        cached = _build_features(state)
        _FEATURE_CACHE[state.key] = cached
    data = Data(**cached.to_dict())
    if transform is not None:
        data.x = transform.apply(data.x)
    return data


# ---------------------------------------------------------------------------
# Similarity augmentation (rotation / reflection, log-uniform scale, shift)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimilarityTransform:
    matrix: torch.Tensor  # [d, d]
    bias: torch.Tensor  # [1, d]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.t() + self.bias


def sample_similarity_transform(
    x: torch.Tensor,
    *,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_std: float = 0.05,
    reflect_prob: float = 0.1,
) -> SimilarityTransform:
    """Random similarity about the vertex centroid.

    The shift is scaled by the mean distance of the vertices to their centroid.
    """
    num_nodes, dim = x.shape
    if num_nodes == 0:
        return SimilarityTransform(torch.eye(dim), torch.zeros((1, dim)))
    centroid = x.mean(dim=0, keepdim=True)
    radius = (x - centroid).norm(dim=1).mean().clamp_min(1e-6)

    q, r = torch.linalg.qr(torch.randn((dim, dim), dtype=x.dtype))
    diag = torch.diagonal(r)
    rotation = q * torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    if reflect_prob > 0.0 and float(torch.rand((), dtype=x.dtype)) < reflect_prob:
        axis = int(torch.randint(0, dim, (1,)).item())
        rotation = rotation.clone()
        rotation[:, axis] = -rotation[:, axis]

    scale_min, scale_max = scale_range
    if scale_min == scale_max:
        scale = torch.tensor(scale_min, dtype=x.dtype)
    else:
        u = torch.rand((), dtype=x.dtype)
        scale = torch.exp(math.log(scale_min) + u * (math.log(scale_max) - math.log(scale_min)))
    matrix = scale * rotation
    shift = torch.randn((1, dim), dtype=x.dtype) * (shift_std * radius)
    return SimilarityTransform(matrix=matrix, bias=centroid - centroid @ matrix.t() + shift)
