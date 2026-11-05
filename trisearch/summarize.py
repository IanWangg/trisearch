"""Summarize ``trisearch-eval`` outputs at fixed visit budgets.

For each result file and budget ``b`` the per-polytope value is the best
objective found within the first ``b`` visits.  The table reports the mean
± standard error (population std / sqrt(N)) over polytopes.

With ``--reference ref.csv`` (columns ``point_config_index,value``: the optimum
or best known value of each polytope) it also reports the relative gap
``(y - opt) / |y|`` and the hit ratio ``y <= opt + tol``.

Example::

    trisearch-summarize results/*.npz --budgets 500 5000 --reference optima_d7.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Optional, Sequence

import numpy as np


def value_at_budget(trajectory: np.ndarray, budget: float) -> float:
    """Best value among records with ``visits <= budget`` (the search carries its best forward)."""
    visits, best = trajectory[0], trajectory[2]
    mask = visits <= budget
    return float(best[mask][-1]) if mask.any() else float(best[0])


def relative_gap(value: float, optimum: float) -> float:
    if value == 0.0:
        return 0.0 if optimum == 0.0 else float("inf")
    return (value - optimum) / abs(value)


def read_reference(path: str) -> Dict[str, float]:
    with open(path, newline="") as handle:
        return {str(int(row["point_config_index"])): float(row["value"]) for row in csv.DictReader(handle)}


def _mean_se(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4g} ± {arr.std() / np.sqrt(arr.size):.2g}"


def summarize(path: str, budgets: Sequence[float], reference: Optional[Dict[str, float]], tol: float) -> list:
    results = np.load(path)
    keys = sorted(results.files, key=int)
    if reference is not None:
        missing = [k for k in keys if k not in reference]
        if missing:
            raise KeyError(f"{path}: no reference value for polytopes {missing[:5]}...")
    rows = []
    for budget in budgets:
        values = [value_at_budget(results[k], budget) for k in keys]
        row = {"file": os.path.basename(path), "budget": budget, "n": len(keys), "best": _mean_se(values)}
        if reference is not None:
            optima = [reference[k] for k in keys]
            row["gap_%"] = _mean_se([100.0 * relative_gap(v, o) for v, o in zip(values, optima)])
            row["hit_%"] = f"{100.0 * np.mean([v <= o + tol for v, o in zip(values, optima)]):.1f}"
        rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="NPZ files written by trisearch-eval.")
    p.add_argument("--budgets", type=float, nargs="+", default=[500])
    p.add_argument("--reference", default=None, help="CSV with point_config_index,value.")
    p.add_argument("--tol", type=float, default=1e-6, help="Tolerance for the hit ratio.")
    args = p.parse_args(argv)

    reference = read_reference(args.reference) if args.reference else None
    rows = [row for path in args.results for row in summarize(path, args.budgets, reference, args.tol)]
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in columns))


if __name__ == "__main__":
    main()
