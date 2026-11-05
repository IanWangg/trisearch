"""PPO training of the TriSearch policy.

Each iteration samples ``num_states`` start triangulations from the discovered
part of the flip graph (inverse-sqrt visit counts), rolls the policy out for
``rollout_length`` flips, and updates actor and critic with clipped PPO and
GAE.  The per-step reward is the objective improvement plus a count-based
exploration bonus ``β · (N(s') + 1)^(-exponent)``.

Example (3D, minimum number of simplices)::

    trisearch-train --objective min_tri \\
        --datasets ywang3/bistellar_dataset_chunk_0 ywang3/bistellar_dataset_chunk_1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from tqdm import tqdm

from . import __version__
from .data import (
    SimilarityTransform,
    actionable_states,
    coordinate_dim,
    featurize,
    load_datasets,
    sample_similarity_transform,
    split_by_polytope,
)
from .flips import DEFAULT_CACHE_DIR
from .models import TriSearchPolicy
from .objectives import OBJECTIVES, Objective
from .state import TransitionPool, TriangulationState, batch_transitions

StateTable = Dict[TriangulationState, TriangulationState]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--datasets", nargs="+", required=True, help="Hugging Face dataset ids or save_to_disk directories.")
    g.add_argument("--objective", choices=OBJECTIVES, required=True)
    g.add_argument("--flip_cache_dir", default=DEFAULT_CACHE_DIR, help="Flip-lookup cache ('none' to disable).")

    g = p.add_argument_group("PPO")
    g.add_argument("--num_iterations", type=int, default=2001)
    g.add_argument("--num_states", type=int, default=128, help="Parallel rollouts per iteration.")
    g.add_argument("--rollout_length", type=int, default=50)
    g.add_argument("--num_epochs", type=int, default=1)
    g.add_argument("--batch_size", type=int, default=128)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--gae_lambda", type=float, default=0.95)
    g.add_argument("--clip", type=float, default=0.1)
    g.add_argument("--value_coef", type=float, default=0.5)
    g.add_argument("--entropy_coef", type=float, default=1e-3)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--count_bonus_coef", type=float, default=1e-3)
    g.add_argument("--count_bonus_exponent", type=float, default=0.5)

    g = p.add_argument_group("augmentation (one random similarity per rollout)")
    g.add_argument("--no_aug", action="store_true")
    g.add_argument("--aug_scale", type=float, nargs=2, default=(0.9, 1.1), metavar=("MIN", "MAX"))
    g.add_argument("--aug_shift_std", type=float, default=0.05)
    g.add_argument("--aug_reflect_prob", type=float, default=0.1)

    g = p.add_argument_group("model")
    g.add_argument("--hidden_channels", type=int, default=64)
    g.add_argument("--out_channels", type=int, default=64)
    g.add_argument("--num_layers", type=int, default=3, help="EGNN layers.")
    g.add_argument("--snn_layers", type=int, default=2)

    g = p.add_argument_group("runtime / output")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--num_workers", type=int, default=0, help="Processes for flip transitions (0 = in-process).")
    g.add_argument("--flip_workers", type=int, default=0, help="Processes for building flip lookups (0 = all CPUs).")
    g.add_argument("--torch_threads", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--output_dir", default=None, help="Default: runs/<objective>_<d>d_seed<seed>.")
    g.add_argument("--save_interval", type=int, default=1000)
    g.add_argument("--eval_interval", type=int, default=500)
    g.add_argument("--wandb_project", default=None, help="Log to Weights & Biases when set.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Visit counts and exploration bonus
# ---------------------------------------------------------------------------

def sample_start_states(table: StateTable, num_states: int) -> List[TriangulationState]:
    """Sample start states with probability ∝ max(1, visits)^(-1/2)."""
    states = list(table.keys())
    weights = np.array([(1.0 / max(1, s.visitation)) ** 0.5 for s in states])
    chosen = np.random.choice(len(states), size=num_states, p=weights / weights.sum())
    return [states[i] for i in chosen]


def visit_count(state: TriangulationState, table: StateTable, runtime_counts: Dict[str, int]) -> int:
    tracked = table.get(state)
    return int(tracked.visitation) if tracked is not None else int(runtime_counts.get(state.key, 0))


def record_visits(states: Sequence[TriangulationState], table: StateTable, runtime_counts: Dict[str, int]) -> None:
    """Visits of seed states are stored on the state; newly reached states are counted by key."""
    for state in states:
        tracked = table.get(state)
        if tracked is not None:
            tracked.visitation += 1
        else:
            runtime_counts[state.key] = runtime_counts.get(state.key, 0) + 1


def count_bonus(
    states: Sequence[TriangulationState],
    table: StateTable,
    runtime_counts: Dict[str, int],
    coef: float,
    exponent: float,
) -> List[float]:
    if coef <= 0.0:
        return [0.0] * len(states)
    return [coef / (visit_count(s, table, runtime_counts) + 1.0) ** exponent for s in states]


# ---------------------------------------------------------------------------
# Rollouts and PPO
# ---------------------------------------------------------------------------

def rollout_step(
    states: List[TriangulationState],
    policy: TriSearchPolicy,
    table: StateTable,
    lookups,
    objective: Objective,
    *,
    device: torch.device,
    transforms: Optional[List[SimilarityTransform]] = None,
    pool: Optional[TransitionPool] = None,
    runtime_counts: Optional[Dict[str, int]] = None,
    bonus_coef: float = 0.0,
    bonus_exponent: float = 0.5,
    deterministic: bool = False,
):
    """Apply one policy flip to every state; return data, actions and rewards."""
    data = [featurize(s, None if transforms is None else transforms[i]) for i, s in enumerate(states)]
    with torch.no_grad():
        action_index, value, log_prob = policy(Batch.from_data_list(data).to(device), deterministic=deterministic)
    actions = [s.subcomplex_actions()[i] for s, i in zip(states, action_index.tolist())]

    next_states = []
    for state, (simplices, edges, key) in zip(states, batch_transitions(states, actions, pool)):
        next_state = table.get(key)
        if next_state is None:
            next_state = state.successor(simplices, edges)
            next_state.find_available_actions(lookups[state.point_config_index])
        next_states.append(next_state)

    rewards = [objective.reward(s, n) for s, n in zip(states, next_states)]
    bonus = count_bonus(next_states, table, runtime_counts or {}, bonus_coef, bonus_exponent)
    return {
        "data": data,
        "action_index": action_index.cpu(),
        "value": value.cpu(),
        "log_prob": log_prob.cpu(),
        "next_states": next_states,
        "reward": torch.tensor(rewards, dtype=torch.float),
        "bonus": torch.tensor(bonus, dtype=torch.float),
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, bootstrap: torch.Tensor, gamma: float, lam: float):
    """GAE(λ) over ``[T, B]`` rewards/values; returns ``(advantages, value_targets)``."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(rewards.size(0))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


def ppo_update(policy, optimizer, data, action_index, old_log_prob, value_targets, advantages, args, device):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    policy.train()
    for _ in range(args.num_epochs):
        order = torch.randperm(len(data)).tolist()
        for start in range(0, len(order), args.batch_size):
            idx = order[start: start + args.batch_size]
            batch = Batch.from_data_list([data[i] for i in idx]).to(device)
            value, logits = policy.get_value_and_logits(batch)
            log_prob, entropy = policy.log_prob_and_entropy(logits, action_index[idx].to(device))
            adv = advantages[idx].to(device)
            ratio = torch.exp(log_prob - old_log_prob[idx].to(device))
            policy_loss = torch.max(-ratio * adv, -torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
            value_loss = 0.5 * (value - value_targets[idx].to(device)).pow(2).mean()
            entropy = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            for name, value_ in (("loss", loss), ("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy)):
                stats[name] += float(value_.item())
    return stats


def save_checkpoint(policy: TriSearchPolicy, path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save({"config": policy.config, "state_dict": policy.state_dict()}, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(args.torch_threads)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = None if str(args.flip_cache_dir).lower() == "none" else args.flip_cache_dir
    rows, lookups = load_datasets(args.datasets, flip_cache_dir=cache_dir, num_workers=args.flip_workers)
    dim = coordinate_dim(rows)
    num_polytopes = len(set(rows["point_config_index"]))
    num_test = min(20, max(1, num_polytopes // 10))
    train_rows, test_rows = split_by_polytope(rows, num_test=num_test, seed=args.seed)
    print(f"{dim}D data: {num_polytopes} polytopes ({num_polytopes - num_test} train / {num_test} held out)")

    output_dir = args.output_dir or os.path.join("runs", f"{args.objective}_{dim}d_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump({"args": vars(args), "coordinate_dim": dim, "trisearch_version": __version__}, handle, indent=2)

    policy = TriSearchPolicy(
        in_channels=dim,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        snn_layers=args.snn_layers,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    objective = Objective(args.objective)

    print("Building triangulation states ...")
    table: StateTable = {s: s for s in actionable_states(tqdm(train_rows), lookups)}
    eval_states = actionable_states(tqdm(test_rows), lookups, one_per_polytope=True, max_polytopes=num_test)
    eval_table: StateTable = {s: s for s in eval_states}
    if not table or not eval_states:
        raise ValueError("No triangulations with available flips in the training or held-out data.")
    runtime_counts: Dict[str, int] = {}

    wandb = None
    if args.wandb_project:
        import wandb

        wandb.init(project=args.wandb_project, name=os.path.basename(output_dir), config=vars(args))
    pool = TransitionPool(args.num_workers) if args.num_workers > 0 else None

    try:
        for iteration in range(args.num_iterations):
            started = time.perf_counter()
            states = sample_start_states(table, args.num_states)
            transforms = None
            if not args.no_aug:
                transforms = [
                    sample_similarity_transform(
                        featurize(s).x,
                        scale_range=tuple(args.aug_scale),
                        shift_std=args.aug_shift_std,
                        reflect_prob=args.aug_reflect_prob,
                    )
                    for s in states
                ]

            buffers = {k: [] for k in ("data", "action_index", "value", "log_prob", "reward", "bonus")}
            for _ in range(args.rollout_length):
                record_visits(states, table, runtime_counts)
                out = rollout_step(
                    states, policy, table, lookups, objective,
                    device=device, transforms=transforms, pool=pool, runtime_counts=runtime_counts,
                    bonus_coef=args.count_bonus_coef, bonus_exponent=args.count_bonus_exponent,
                )
                for k in buffers:
                    buffers[k].append(out[k])
                states = out["next_states"]

            last = [featurize(s, None if transforms is None else transforms[i]) for i, s in enumerate(states)]
            with torch.no_grad():
                bootstrap = policy.get_value(Batch.from_data_list(last).to(device)).cpu()
            extrinsic = torch.stack(buffers["reward"])
            bonus = torch.stack(buffers["bonus"])
            advantages, targets = compute_gae(
                extrinsic + bonus, torch.stack(buffers["value"]), bootstrap, args.gamma, args.gae_lambda
            )
            # Flatten time-major: sample (t, i) -> t * num_states + i.
            data = [d for step in buffers["data"] for d in step]
            stats = ppo_update(
                policy, optimizer, data,
                torch.stack(buffers["action_index"]).reshape(-1),
                torch.stack(buffers["log_prob"]).reshape(-1),
                targets.reshape(-1), advantages.reshape(-1), args, device,
            )
            log = {
                "rollout/return": extrinsic.sum(0).mean().item(),
                "rollout/bonus_mean": bonus.mean().item(),
                "rollout/new_states_visited": len(runtime_counts),
                **{f"train/{k}": v for k, v in stats.items()},
                "time/iteration_s": time.perf_counter() - started,
            }

            if iteration % args.save_interval == 0 or iteration == args.num_iterations - 1:
                save_checkpoint(policy, os.path.join(output_dir, f"{iteration}.pt"))

            if iteration % args.eval_interval == 0:
                eval_return = torch.zeros(len(eval_states))
                current = list(eval_states)
                for _ in range(args.rollout_length):
                    out = rollout_step(current, policy, eval_table, lookups, objective, device=device, pool=pool)
                    eval_return += out["reward"]
                    current = out["next_states"]
                log["eval/return_mean"] = eval_return.mean().item()

            print(f"[{iteration}] " + " ".join(f"{k}={v:.4g}" for k, v in log.items()))
            if wandb is not None:
                wandb.log(log, step=iteration)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
