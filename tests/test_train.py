import os

import numpy as np
import torch

from trisearch.models import load_policy
from trisearch.state import TriangulationState
from trisearch.train import compute_gae, count_bonus, main, record_visits, sample_start_states

from conftest import BIPYRAMID, BIPYRAMID_THREE, BIPYRAMID_TWO


def test_gae_matches_reference_loop():
    torch.manual_seed(0)
    rewards, values, bootstrap = torch.randn(5, 3), torch.randn(5, 3), torch.randn(3)
    gamma, lam = 0.9, 0.8
    adv, targets = compute_gae(rewards, values, bootstrap, gamma, lam)
    r, v, b = rewards.numpy(), values.numpy(), bootstrap.numpy()
    for j in range(3):
        expected, nxt = [], b[j]
        acc = 0.0
        for t in reversed(range(5)):
            acc = r[t, j] + gamma * nxt - v[t, j] + gamma * lam * acc
            expected.append(acc)
            nxt = v[t, j]
        assert np.allclose(adv[:, j].numpy(), expected[::-1], atol=1e-6)
    assert torch.allclose(targets, adv + values)


def test_visit_counts_and_bonus():
    seed = TriangulationState(BIPYRAMID, 0, BIPYRAMID_TWO)
    reached = TriangulationState(BIPYRAMID, 0, BIPYRAMID_THREE)
    table, runtime = {seed: seed}, {}
    record_visits([seed, seed, reached], table, runtime)
    assert seed.visitation == 2 and runtime == {reached.key: 1}
    bonus = count_bonus([seed, reached], table, runtime, coef=1.0, exponent=0.5)
    assert np.allclose(bonus, [3 ** -0.5, 2 ** -0.5])
    assert count_bonus([seed], table, runtime, coef=0.0, exponent=0.5) == [0.0]


def test_start_states_prefer_rarely_visited():
    a = TriangulationState(BIPYRAMID, 0, BIPYRAMID_TWO)
    b = TriangulationState(BIPYRAMID, 0, BIPYRAMID_THREE)
    a.visitation = 10_000
    np.random.seed(0)
    picks = sample_start_states({a: a, b: b}, 1000)
    assert sum(p is b for p in picks) > 900


def test_train_end_to_end(tmp_path, local_dataset):
    out = tmp_path / "run"
    main([
        "--datasets", local_dataset("3d"), "--objective", "min_edge_length",
        "--num_iterations", "3", "--num_states", "4", "--rollout_length", "3", "--batch_size", "5",
        "--hidden_channels", "16", "--out_channels", "16", "--num_layers", "2",
        "--device", "cpu", "--flip_cache_dir", str(tmp_path / "cache"),
        "--save_interval", "2", "--eval_interval", "1", "--output_dir", str(out),
    ])
    assert sorted(os.listdir(out)) == ["0.pt", "2.pt", "config.json"]
    policy = load_policy(str(out / "2.pt"))
    assert policy.config["in_channels"] == 3
