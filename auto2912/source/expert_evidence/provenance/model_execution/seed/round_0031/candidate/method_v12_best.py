"""Bias calibration with conservative anchor blending + graph-smoothed node and node-horizon biases."""
import numpy as np

GRAPH_PATH = '/workspace/public_assets/graph.npz'


def _graph_smooth(values_n, adj, alpha=0.1, n_iter=2):
    result = values_n.copy()
    deg = np.maximum(adj.sum(axis=1), 1.0)
    for _ in range(n_iter):
        neighbor_sum = adj.T @ result
        neighbor_avg = neighbor_sum / deg
        result = (1 - alpha) * result + alpha * neighbor_avg
    return result


def fit(features, targets, seed):
    rng = np.random.RandomState(seed)
    base = features['base'].astype(np.float64)
    target = targets.astype(np.float64)
    anchor = features['anchor'].astype(np.float64)

    n_samples, n_horizons, n_nodes = base.shape
    mask = np.isfinite(target) & (target > 0)

    graph = np.load(GRAPH_PATH)
    adj = graph['adjacency'].astype(np.float64)

    # Step 0: Conservative blend of base and anchor forecasts
    anchor_weight = 0.02
    blended = (1.0 - anchor_weight) * base + anchor_weight * anchor

    # Step 1: Per-horizon mean bias
    h_bias = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        h_bias[h] = (blended[:, h, :][m] - target[:, h, :][m]).mean()

    corrected = blended - h_bias[np.newaxis, :, np.newaxis]

    # Step 2: Per-node residual bias (graph-smoothed, shrunken)
    residual = corrected - target
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 5:
            node_bias[n] = residual[:, :, n][m].mean()

    node_bias = 0.5 * node_bias
    node_bias = _graph_smooth(node_bias, adj, alpha=0.2, n_iter=3)
    corrected = corrected - node_bias[np.newaxis, np.newaxis, :]

    # Step 3: Per-node-horizon bias (shrunken, graph-smoothed)
    residual = corrected - target
    bias_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 0:
                raw = residual[:, h, n][m].mean()
                bias_nh[h, n] = 0.3 * raw

    for h in range(n_horizons):
        bias_nh[h] = _graph_smooth(bias_nh[h], adj, alpha=0.15, n_iter=2)

    state = {
        'method': 'anchor_blend_bias_cal_graph',
        'version': 12,
        'anchor_weight': np.float32(anchor_weight),
        'h_bias': h_bias.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float64)
    anchor = np.array(features['anchor'], dtype=np.float64)

    anchor_weight = float(state['anchor_weight'])
    h_bias = state['h_bias'].astype(np.float64)
    node_bias = state['node_bias'].astype(np.float64)
    bias_nh = state['bias_nh'].astype(np.float64)

    blended = (1.0 - anchor_weight) * base + anchor_weight * anchor
    result = blended - h_bias[np.newaxis, :, np.newaxis]
    result = result - node_bias[np.newaxis, np.newaxis, :]
    result = result - bias_nh[np.newaxis, :, :]
    result = np.maximum(result, 0.0)

    return result.astype(np.float32)
