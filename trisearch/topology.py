"""Simplex-level topology consumed by the SNN actor.

For a triangulation with top simplices ``σ_1..σ_M`` we precompute

* the normalized top-degree down Laplacian ``L_d = B_dᵀ B_d / max_row_abs_sum``
  (two top simplices interact iff they share a facet), as COO entries;
* the membership pairs ``(candidate, simplex)`` with ``σ ⊆ Z`` for every
  candidate flip ``Z``, i.e. the simplices of ``T+(Z)`` the actor pools over.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class SimplexTopology:
    laplacian_row: torch.Tensor
    laplacian_col: torch.Tensor
    laplacian_value: torch.Tensor
    candidate: torch.Tensor
    simplex: torch.Tensor


def simplex_tensor(simplices: Iterable[Iterable[int]]) -> torch.Tensor:
    """``[M, d+1]`` tensor of top simplices in canonical (sorted) order."""
    canonical = tuple(sorted(tuple(sorted(int(v) for v in s)) for s in simplices))
    if not canonical:
        return torch.empty((0, 0), dtype=torch.long)
    return torch.tensor(canonical, dtype=torch.long)


def down_laplacian(simplex_vertices: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized sparse ``B_dᵀ B_d`` for the given top simplices."""
    device = simplex_vertices.device
    num_simplices, width = simplex_vertices.shape
    if num_simplices == 0 or width <= 1:
        size = (num_simplices, num_simplices)
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            size=size,
        ).coalesce()

    facets, signs = [], []
    for omitted in range(width):
        keep = torch.ones(width, dtype=torch.bool, device=device)
        keep[omitted] = False
        facets.append(simplex_vertices[:, keep])
        signs.append(torch.full((num_simplices,), 1.0 if omitted % 2 == 0 else -1.0, dtype=dtype, device=device))
    unique_facets, inverse = torch.unique(torch.cat(facets, dim=0), dim=0, return_inverse=True)
    simplex_ids = torch.arange(num_simplices, dtype=torch.long, device=device).repeat(width)
    boundary = torch.sparse_coo_tensor(
        torch.stack([inverse.long(), simplex_ids]),
        torch.cat(signs),
        size=(int(unique_facets.size(0)), num_simplices),
    ).coalesce()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        laplacian = torch.sparse.mm(boundary.transpose(0, 1), boundary).coalesce()

    indices, values = laplacian.indices(), laplacian.values()
    if values.numel() == 0:
        return laplacian
    row_abs_sums = torch.zeros(num_simplices, dtype=dtype, device=device).scatter_add_(0, indices[0], values.abs())
    scale = row_abs_sums.max().clamp_min(1e-12)
    return torch.sparse_coo_tensor(indices, values / scale, size=laplacian.shape).coalesce()


def candidate_memberships(simplex_vertices: torch.Tensor, subcomplex_vertices: torch.Tensor):
    """Pairs ``(candidate, simplex)`` such that every vertex of the simplex is in the candidate."""
    candidates, simplices = [], []
    simplex_list = simplex_vertices.tolist()
    for candidate_id, row in enumerate(subcomplex_vertices.tolist()):
        vertex_set = {v for v in row if v >= 0}
        found = False
        for simplex_id, simplex in enumerate(simplex_list):
            if all(v in vertex_set for v in simplex):
                candidates.append(candidate_id)
                simplices.append(simplex_id)
                found = True
        if not found:
            raise ValueError("A candidate flip contains no top simplex of the current triangulation.")
    return torch.tensor(candidates, dtype=torch.long), torch.tensor(simplices, dtype=torch.long)


def build_simplex_topology(simplex_vertices: torch.Tensor, subcomplex_vertices: torch.Tensor) -> SimplexTopology:
    laplacian = down_laplacian(simplex_vertices, dtype=torch.float32)
    candidate, simplex = candidate_memberships(simplex_vertices, subcomplex_vertices)
    return SimplexTopology(
        laplacian_row=laplacian.indices()[0].contiguous(),
        laplacian_col=laplacian.indices()[1].contiguous(),
        laplacian_value=laplacian.values().contiguous(),
        candidate=candidate,
        simplex=simplex,
    )
