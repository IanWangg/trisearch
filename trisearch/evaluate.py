"""Evaluate a search method on held-out polytopes.

For every polytope in the dataset (first seed triangulation, in dataset order,
at most ``--max_polytopes``), run one search and store its trajectory.

Output: ``<output>.npz`` with one ``4 × K`` array per polytope, keyed by
``point_config_index``.  Rows are ``visits, expansions, best_value, wall_time_s``.
A ``<output>.json`` sidecar records the settings.

Example::

    trisearch-eval --dataset ywang3/bistellar_dataset_chunk_7 --objective min_tri \\
        --method policy --checkpoint runs/min_tri_3d_seed0/2000.pt --budget 500
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from . import __version__
from .data import actionable_states, load_datasets
from .flips import DEFAULT_CACHE_DIR
from .models import load_policy
from .objectives import OBJECTIVES, Objective
from .search import METHODS, search, seed_everything
from .state import TriangulationState


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", required=True, help="Hugging Face dataset id(s) or save_to_disk directories.")
    p.add_argument("--objective", choices=OBJECTIVES, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--checkpoint", default=None, help="Policy checkpoint (required for --method policy).")
    p.add_argument("--budget", type=int, default=500, help="Maximum number of visits per polytope.")
    p.add_argument("--max_polytopes", type=int, default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample policy actions (softmax / temperature) and break baseline ties at random.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="Policy device (default: cuda if available).")
    p.add_argument("--num_workers", type=int, default=1, help="Parallel polytopes (spawned processes).")
    p.add_argument("--flip_cache_dir", default=DEFAULT_CACHE_DIR, help="Flip-lookup cache ('none' to disable).")
    p.add_argument("--flip_workers", type=int, default=0)
    p.add_argument("--output", required=True, help="Output path (.npz).")
    return p.parse_args(argv)


_WORKER: Dict[str, object] = {}


def _init_worker(lookups, args_dict) -> None:
    torch.set_num_threads(1)
    _WORKER["lookups"] = lookups
    _WORKER["args"] = args_dict
    _WORKER["objective"] = Objective(args_dict["objective"])
    _WORKER["policy"] = None
    if args_dict["method"] == "policy":
        _WORKER["policy"] = load_policy(args_dict["checkpoint"], device=args_dict["device"])


def _run_one(row: dict):
    args = _WORKER["args"]
    state = TriangulationState.from_row(row)
    seed_everything(args["seed"], state.point_config_index)
    trajectory = search(
        state,
        _WORKER["objective"],
        _WORKER["lookups"],
        method=args["method"],
        budget=args["budget"],
        deterministic=not args["stochastic"],
        temperature=args["temperature"],
        policy=_WORKER["policy"],
        device=args["device"],
    )
    return state.point_config_index, trajectory


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.method == "policy" and not args.checkpoint:
        raise SystemExit("--method policy requires --checkpoint")
    if args.device is None:
        args.device = "cuda" if args.method == "policy" and torch.cuda.is_available() else "cpu"
    started = time.perf_counter()

    cache_dir = None if str(args.flip_cache_dir).lower() == "none" else args.flip_cache_dir
    rows, lookups = load_datasets(args.dataset, flip_cache_dir=cache_dir, num_workers=args.flip_workers)
    states = actionable_states(rows, lookups, one_per_polytope=True, max_polytopes=args.max_polytopes, skip_empty=False)
    tasks = [s.to_row() for s in sorted(states, key=lambda s: s.point_config_index)]
    lookups = {s.point_config_index: lookups[s.point_config_index] for s in states}
    print(f"Evaluating {args.method} on {len(tasks)} polytopes (budget {args.budget})")

    args_dict = vars(args)
    results = {}
    if args.num_workers <= 1:
        _init_worker(lookups, args_dict)
        for task in tqdm(tasks):
            key, trajectory = _run_one(task)
            results[str(key)] = trajectory
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.num_workers, initializer=_init_worker, initargs=(lookups, args_dict)) as pool:
            for key, trajectory in tqdm(pool.imap_unordered(_run_one, tasks), total=len(tasks)):
                results[str(key)] = trajectory

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output = args.output if args.output.endswith(".npz") else f"{args.output}.npz"
    np.savez(output, **dict(sorted(results.items(), key=lambda kv: int(kv[0]))))
    final = [traj[2, -1] for traj in results.values()]
    meta = {
        "args": args_dict,
        "trisearch_version": __version__,
        "num_polytopes": len(results),
        "mean_final_best": float(np.mean(final)) if final else None,
        "rows": ["visits", "expansions", "best_value", "wall_time_s"],
        "elapsed_s": time.perf_counter() - started,
    }
    with open(output[: -len(".npz")] + ".json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Saved {output}; mean final best {args.objective} = {meta['mean_final_best']}")


if __name__ == "__main__":
    main()
