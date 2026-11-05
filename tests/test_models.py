import torch
from torch_geometric.data import Batch

from conftest import make_state
from trisearch.data import featurize, sample_similarity_transform
from trisearch.models import TriSearchPolicy, load_policy
from trisearch.topology import down_laplacian
from trisearch.train import save_checkpoint


def _states(rows, lookups_for):
    lookups = lookups_for(rows)
    return [make_state(r, lookups) for r in rows]


def _reference_snn(actor, node_embeddings, data_list):
    """Per-graph, loop-based SNN actor used as a reference for the batched one."""
    candidates, offset = [], 0
    for data in data_list:
        h = node_embeddings[offset: offset + data.num_nodes]
        offset += data.num_nodes
        z = h[data.simplex_vertices].amax(dim=1)
        laplacian = down_laplacian(data.simplex_vertices, dtype=h.dtype)
        for i, layer in enumerate(actor.layers):
            z = layer(laplacian, z)
            if i != len(actor.layers) - 1:
                z = actor.act(z)
        for row in data.subcomplex_vertices.tolist():
            members = [j for j, s in enumerate(data.simplex_vertices.tolist()) if set(s) <= set(row)]
            candidates.append(z[members].amax(dim=0))
    return torch.stack(candidates)


def test_featurize_layout(rows, lookups_for):
    for state in _states(rows, lookups_for):
        data = featurize(state)
        d = state.dim
        assert data.x.shape == (len(state.vertices), d)
        assert data.subcomplex_vertices.shape == (len(state.subcomplex_actions()), d + 2)
        for row, action in zip(data.subcomplex_vertices.tolist(), state.subcomplex_actions()):
            assert tuple(v for v in row if v >= 0) == action
        assert data.simplex_vertices.shape == (len(state.simplices), d + 1)
        assert featurize(state).x.data_ptr() == data.x.data_ptr()  # cached tensors are shared


def test_batched_snn_matches_reference(rows, lookups_for):
    torch.manual_seed(0)
    data_list = [featurize(s) for s in _states(rows, lookups_for)]
    batch = Batch.from_data_list(data_list)
    policy = TriSearchPolicy(in_channels=data_list[0].x.size(1))
    h, _ = policy.encoder(batch.x, batch.x, batch.edge_index)
    candidate, _, _ = policy.actor(h, batch)
    assert torch.allclose(candidate, _reference_snn(policy.actor, h, data_list), atol=1e-5)


def test_policy_outputs_and_batch_invariance(rows, lookups_for):
    torch.manual_seed(0)
    states = _states(rows, lookups_for)
    policy = TriSearchPolicy(in_channels=states[0].dim).eval()
    batch = Batch.from_data_list([featurize(s) for s in states])
    with torch.no_grad():
        value, logits = policy.get_value_and_logits(batch)
        assert value.shape == (len(states),)
        for i, state in enumerate(states):
            n = len(state.subcomplex_actions())
            assert torch.isfinite(logits[i, :n]).all() and torch.isinf(logits[i, n:]).all()
            v1, l1 = policy.get_value_and_logits(Batch.from_data_list([featurize(state)]))
            assert torch.allclose(l1[0, :n], logits[i, :n], atol=1e-5)
            assert torch.allclose(v1, value[i: i + 1], atol=1e-5)
        action, _, log_prob = policy(batch, deterministic=True)
        assert torch.equal(action, logits.argmax(dim=1))
        assert torch.all(log_prob <= 0)


def test_similarity_augmentation_preserves_shape(rows_3d, lookups_for):
    torch.manual_seed(0)
    state = _states(rows_3d, lookups_for)[0]
    x = featurize(state).x
    transform = sample_similarity_transform(x, scale_range=(1.0, 1.0), shift_std=0.0, reflect_prob=0.0)
    y = transform.apply(x)
    assert torch.allclose(torch.cdist(x, x), torch.cdist(y, y), atol=1e-4)  # pure rigid motion
    assert torch.allclose(y.mean(0), x.mean(0), atol=1e-5)
    aug = featurize(state, transform)
    assert torch.allclose(aug.x, y) and torch.equal(featurize(state).x, x)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    policy = TriSearchPolicy(in_channels=4, hidden_channels=16, out_channels=8, num_layers=2, snn_layers=1)
    path = str(tmp_path / "0.pt")
    save_checkpoint(policy, path)
    loaded = load_policy(path)
    assert loaded.config == policy.config
    for (k, a), (_, b) in zip(policy.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k
