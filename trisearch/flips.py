"""Exact bistellar-flip enumeration for point configurations.

A bistellar flip of a point configuration is supported on a *circuit*: a
minimally affinely dependent subset ``Z = Z+ ⊔ Z-``.  The two triangulations of
``conv(Z)`` are ``T+ = {Z \\ {z} : z in Z-}`` and ``T- = {Z \\ {z} : z in Z+}``.
A flip replaces whichever of the two is a subcomplex of the current
triangulation by the other one.

Everything here is pure Python with exact rational arithmetic (no SageMath or
TOPCOM).  ``build_flip_lookup`` precomputes, per polytope, every candidate flip
and an index from top simplex to the flips touching it, so that finding the
flips available at a triangulation is a set lookup.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import numbers
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import gcd, lcm
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Simplex = Tuple[int, ...]
FlipPair = Tuple[Tuple[Simplex, ...], Tuple[Simplex, ...]]
Circuit = Tuple[Tuple[int, ...], Tuple[int, ...]]  # (Z+, Z-)
Vertices = Tuple[Tuple[float, ...], ...]


# ---------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------

def _to_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value), 1)
    try:
        numerator = getattr(value, "numerator")
        denominator = getattr(value, "denominator")
        numerator = numerator() if callable(numerator) else numerator
        denominator = denominator() if callable(denominator) else denominator
        return Fraction(int(numerator), int(denominator))
    except Exception:
        pass
    try:
        # Floats go through their shortest decimal representation, so a
        # coordinate stored as 0.1 is treated as exactly 1/10.
        return Fraction(str(value))
    except Exception:
        return Fraction(float(value)).limit_denominator()


def _rref(matrix: Sequence[Sequence[Fraction]]) -> Tuple[List[List[Fraction]], Tuple[int, ...]]:
    rows = [list(row) for row in matrix]
    if not rows:
        return [], ()
    num_rows, num_cols = len(rows), len(rows[0])
    pivot_row = 0
    pivots: List[int] = []
    for col in range(num_cols):
        pivot_idx = next((r for r in range(pivot_row, num_rows) if rows[r][col] != 0), None)
        if pivot_idx is None:
            continue
        rows[pivot_row], rows[pivot_idx] = rows[pivot_idx], rows[pivot_row]
        pivot_value = rows[pivot_row][col]
        rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
        for r in range(num_rows):
            factor = rows[r][col]
            if r != pivot_row and factor != 0:
                rows[r] = [cur - factor * piv for cur, piv in zip(rows[r], rows[pivot_row])]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == num_rows:
            break
    return rows, tuple(pivots)


def _primitive_kernel_vector(matrix: Sequence[Sequence[Fraction]]) -> Tuple[int, ...]:
    """Primitive integer generator of a one-dimensional kernel."""
    rref_matrix, pivots = _rref(matrix)
    num_cols = len(rref_matrix[0])
    free_cols = [col for col in range(num_cols) if col not in pivots]
    if len(free_cols) != 1:
        raise ValueError("Expected nullity one for a circuit support.")
    solution = [Fraction(0)] * num_cols
    solution[free_cols[0]] = Fraction(1)
    for row_idx in range(len(pivots) - 1, -1, -1):
        solution[pivots[row_idx]] = -sum(
            (rref_matrix[row_idx][col] * solution[col] for col in free_cols), Fraction(0)
        )
    denominator = 1
    for value in solution:
        denominator = lcm(denominator, value.denominator)
    integers = [int(value * denominator) for value in solution]
    divisor = 0
    for value in integers:
        divisor = gcd(divisor, abs(value))
    if divisor > 1:
        integers = [value // divisor for value in integers]
    first_nonzero = next(value for value in integers if value != 0)
    if first_nonzero < 0:
        integers = [-value for value in integers]
    return tuple(integers)


class _CircuitEnumerator:
    def __init__(self, points: Iterable[Iterable[Any]]):
        self.points = tuple(tuple(_to_fraction(c) for c in point) for point in points)
        if not self.points:
            raise ValueError("Expected at least one point.")
        self.dim = len(self.points[0])
        if self.dim == 0 or any(len(p) != self.dim for p in self.points):
            raise ValueError("All points must share a positive coordinate dimension.")
        # Homogenize: affine dependencies of points = linear dependencies of (p, 1).
        self.projective = tuple(point + (Fraction(1),) for point in self.points)
        self.num_points = len(self.points)
        self._rank_cache: Dict[Tuple[int, ...], int] = {}

    def _support_matrix(self, support: Tuple[int, ...]) -> List[List[Fraction]]:
        columns = [self.projective[i] for i in support]
        return [[columns[c][r] for c in range(len(columns))] for r in range(len(columns[0]))]

    def rank(self, support: Tuple[int, ...]) -> int:
        if support not in self._rank_cache:
            self._rank_cache[support] = len(_rref(self._support_matrix(support))[1])
        return self._rank_cache[support]

    def is_independent(self, support: Tuple[int, ...]) -> bool:
        return self.rank(support) == len(support)

    def is_uniform(self) -> bool:
        """True if every (d+1)-subset is affinely independent (general position)."""
        subset_size = min(self.num_points, self.dim + 1)
        return all(self.is_independent(s) for s in combinations(range(self.num_points), subset_size))

    def is_circuit(self, support: Tuple[int, ...]) -> bool:
        if self.rank(support) != len(support) - 1:
            return False
        return all(self.is_independent(s) for s in combinations(support, len(support) - 1))

    def circuits(self) -> Tuple[Circuit, ...]:
        if self.num_points <= self.dim + 1:
            return ()
        max_size = min(self.num_points, self.dim + 2)
        sizes: Iterable[int] = (max_size,) if self.is_uniform() else range(2, max_size + 1)
        out: List[Circuit] = []
        for size in sizes:
            for support in combinations(range(self.num_points), size):
                if not self.is_circuit(support):
                    continue
                kernel = _primitive_kernel_vector(self._support_matrix(support))
                z_plus = tuple(support[i] for i, c in enumerate(kernel) if c > 0)
                z_minus = tuple(support[i] for i, c in enumerate(kernel) if c < 0)
                out.append((z_plus, z_minus))
        return tuple(out)


def circuits(points: Iterable[Iterable[Any]]) -> Tuple[Circuit, ...]:
    """All circuits ``(Z+, Z-)`` of a point configuration."""
    return _CircuitEnumerator(points).circuits()


def _circuit_triangulation(keep: Tuple[int, ...], drop_one_of: Tuple[int, ...]) -> Tuple[Simplex, ...]:
    simplices = [
        tuple(sorted(keep + drop_one_of[:i] + drop_one_of[i + 1:])) for i in range(len(drop_one_of))
    ]
    return tuple(sorted(simplices))


def bistellar_flips(points: Iterable[Iterable[Any]]) -> Tuple[FlipPair, ...]:
    """``(T+, T-)`` for every circuit of the point configuration."""
    return tuple(
        (_circuit_triangulation(z_plus, z_minus), _circuit_triangulation(z_minus, z_plus))
        for z_plus, z_minus in circuits(points)
    )


# ---------------------------------------------------------------------------
# Per-polytope flip lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlipLookup:
    """Every flip of one point configuration, indexed by the simplices it uses."""

    coordinate_dim: int
    simplex_size: int
    flip_pairs: Tuple[FlipPair, ...]
    simplex_to_flip_indices: Dict[Simplex, Tuple[int, ...]]


def _canonical_collection(raw: Iterable[Iterable[int]]) -> Tuple[Simplex, ...]:
    return tuple(sorted(tuple(sorted(int(v) for v in simplex)) for simplex in raw))


def _canonical_flip_pair(raw_left, raw_right) -> FlipPair:
    left, right = _canonical_collection(raw_left), _canonical_collection(raw_right)
    if (len(left), left) > (len(right), right):
        left, right = right, left
    return left, right


def lookup_from_flips(raw_flips: Iterable[Tuple[Any, Any]], *, coordinate_dim: int) -> FlipLookup:
    flip_pairs: List[FlipPair] = []
    simplex_to_flips: Dict[Simplex, List[int]] = defaultdict(list)
    simplex_size: Optional[int] = None
    for raw_left, raw_right in raw_flips:
        pair = _canonical_flip_pair(raw_left, raw_right)
        if simplex_size is None:
            reference = pair[0] or pair[1]
            if not reference:
                continue
            simplex_size = len(reference[0])
        index = len(flip_pairs)
        flip_pairs.append(pair)
        for simplex in pair[0] + pair[1]:
            simplex_to_flips[simplex].append(index)
    return FlipLookup(
        coordinate_dim=int(coordinate_dim),
        simplex_size=int(simplex_size or 0),
        flip_pairs=tuple(flip_pairs),
        simplex_to_flip_indices={s: tuple(ids) for s, ids in simplex_to_flips.items()},
    )


def build_flip_lookup(vertices: Sequence[Sequence[float]]) -> FlipLookup:
    canonical = tuple(tuple(float(v) for v in vertex) for vertex in vertices)
    return lookup_from_flips(bistellar_flips(canonical), coordinate_dim=len(canonical[0]))


def _build_worker(payload: Tuple[int, Vertices]) -> Tuple[int, FlipLookup]:
    point_config_index, vertices = payload
    return int(point_config_index), build_flip_lookup(vertices)


def build_flip_lookups(
    vertices_by_polytope: Mapping[int, Vertices],
    *,
    num_workers: int = 0,
    chunksize: int = 1,
) -> Dict[int, FlipLookup]:
    """Build lookups for many polytopes in a spawn process pool.

    ``num_workers <= 0`` uses all CPUs.
    """
    payload = sorted(vertices_by_polytope.items(), key=lambda item: int(item[0]))
    if not payload:
        return {}
    workers = int(num_workers) if int(num_workers) > 0 else (os.cpu_count() or 1)
    workers = min(len(payload), workers)
    if workers == 1:
        return dict(_build_worker(item) for item in payload)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return dict(pool.map(_build_worker, payload, chunksize=max(1, int(chunksize))))


# ---------------------------------------------------------------------------
# On-disk cache, one pickle per dataset
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = os.path.join("cache", "flip_lookup")


def _cache_file(dataset_id: str, cache_dir: str) -> Path:
    if os.path.isdir(dataset_id):
        digest = hashlib.sha1(os.path.abspath(dataset_id).encode()).hexdigest()[:10]
        relative = Path("local") / f"{Path(dataset_id).name}-{digest}"
    else:
        relative = Path(dataset_id)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError(f"Invalid dataset id for a cache path: {dataset_id!r}")
    return Path(os.path.expanduser(cache_dir)) / f"{relative}.pkl"


def load_or_build_flip_lookups(
    dataset_id: str,
    vertices_by_polytope: Union[Mapping[int, Vertices], Callable[[], Mapping[int, Vertices]]],
    *,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    num_workers: int = 0,
) -> Dict[int, FlipLookup]:
    """Load the lookups of one dataset from the cache, building them if missing.

    Concurrent jobs coordinate through an exclusive lock file, so only one of
    them builds a missing cache.  ``cache_dir=None`` disables caching.
    """

    def build() -> Dict[int, FlipLookup]:
        source = vertices_by_polytope() if callable(vertices_by_polytope) else vertices_by_polytope
        return build_flip_lookups(source, num_workers=num_workers)

    if cache_dir is None:
        return build()

    path = _cache_file(dataset_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{path}.lock")
    while True:
        if path.exists():
            print(f"Loading cached flip lookups for {dataset_id} from {path}")
            with open(path, "rb") as handle:
                return pickle.load(handle)
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(f"Waiting for another process to build {path} ...")
            time.sleep(1.0)
            continue
        try:
            os.write(lock_fd, f"{os.getpid()}\n".encode())
            os.close(lock_fd)
            print(f"Building flip lookups for {dataset_id} (cached at {path})")
            lookups = build()
            tmp_path = f"{path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as handle:
                pickle.dump(lookups, handle)
            os.replace(tmp_path, path)
            return lookups
        finally:
            if lock_path.exists():
                lock_path.unlink()
