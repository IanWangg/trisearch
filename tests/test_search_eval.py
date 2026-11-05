import csv
import json
import subprocess
import sys
from collections import deque

import numpy as np
import pytest
import torch

from conftest import load_rows, make_state
from trisearch.evaluate import main as eval_main
from trisearch.models import TriSearchPolicy
from trisearch.objectives import Objective
from trisearch.search import METHODS, search
from trisearch.summarize import main as summarize_main, value_at_budget
from trisearch.train import save_checkpoint


def _flip_graph_optimum(state, lookups, objective):
    """Exhaustive BFS over the (small) flip graph."""
    seen, queue, best = {state.key}, deque([state]), objective(state)
    while queue:
        cur = queue.popleft()
        cur.find_available_actions(lookups[cur.point_config_index])
        for action in cur.subcomplex_actions():
            simplices, edges, key = cur.transition(action)
            if key not in seen:
                seen.add(key)
                nxt = cur.successor(simplices, edges)
                best = min(best, objective(nxt))
                queue.append(nxt)
    return best


@pytest.mark.parametrize("method", METHODS)
def test_search_trajectory_invariants(method, rows_3d, lookups_for):
    lookups = lookups_for(rows_3d)
    objective = Objective("min_tri")
    torch.manual_seed(0)
    policy = TriSearchPolicy(in_channels=3, hidden_channels=16, out_channels=16, num_layers=2).eval()
    init = make_state(rows_3d[-1], lookups)
    traj = search(init, objective, lookups, method=method, budget=40, policy=policy)
    visits, expansions, best = traj[0], traj[1], traj[2]
    assert traj.shape[0] == 4 and best[0] == objective(init)
    assert np.all(np.diff(visits) > 0) and np.all(np.diff(expansions) > 0)
    assert np.all(np.diff(best) <= 0)
    if method == "policy":
        assert np.all(np.diff(visits) == 1)
    again = search(make_state(rows_3d[-1], lookups), objective, lookups, method=method, budget=40, policy=policy)
    assert np.array_equal(traj[:3], again[:3])


@pytest.mark.parametrize("method", ["dfs", "best_first"])
def test_exhaustive_search_finds_optimum(method, rows_3d, lookups_for):
    lookups = lookups_for(rows_3d)
    for name in ("min_tri", "min_edge_length", "min_diameter"):
        objective = Objective(name)
        init = make_state(rows_3d[0], lookups)
        optimum = _flip_graph_optimum(init, lookups, objective)
        traj = search(init, objective, lookups, method=method, budget=10**6)
        assert traj[2, -1] == optimum


def test_value_at_budget():
    traj = np.array([[0, 3, 7, 12], [0, 1, 2, 3], [9.0, 8.0, 6.0, 5.0], [0, 0, 0, 0]])
    assert value_at_budget(traj, 0) == 9.0
    assert value_at_budget(traj, 6) == 8.0
    assert value_at_budget(traj, 100) == 5.0


def test_evaluate_and_summarize_end_to_end(tmp_path, local_dataset, capsys):
    dataset = local_dataset("3d")
    ckpt = str(tmp_path / "policy.pt")
    torch.manual_seed(0)
    save_checkpoint(TriSearchPolicy(in_channels=3, hidden_channels=16, out_channels=16, num_layers=2), ckpt)
    outputs = {}
    for method in ("policy", "greedy"):
        out = str(tmp_path / f"{method}.npz")
        eval_main([
            "--dataset", dataset, "--objective", "min_tri", "--method", method, "--budget", "30",
            "--checkpoint", ckpt, "--device", "cpu", "--flip_cache_dir", "none", "--output", out,
            "--num_workers", "2" if method == "greedy" else "1",
        ])
        outputs[method] = np.load(out)
        meta = json.loads((tmp_path / f"{method}.json").read_text())
        assert meta["num_polytopes"] == 3
    assert sorted(outputs["policy"].files, key=int) == ["0", "1", "2"]

    ref = tmp_path / "ref.csv"
    with open(ref, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_config_index", "value"])
        for key in outputs["greedy"].files:
            writer.writerow([key, outputs["greedy"][key][2, -1]])
    capsys.readouterr()
    summarize_main([str(tmp_path / "greedy.npz"), "--budgets", "30", "--reference", str(ref)])
    printed = capsys.readouterr().out
    assert "gap_%" in printed and "100.0" in printed  # greedy hits its own final values


@pytest.mark.parametrize("module", ["train", "evaluate", "summarize", "datagen"])
def test_cli_help(module):
    result = subprocess.run([sys.executable, "-m", f"trisearch.{module}", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "usage" in result.stdout
