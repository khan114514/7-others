"""Backup of v10: per-horizon ridge + graph-smoothed node bias + nh bias."""
import numpy as np
from sklearn.linear_model import Ridge

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

    ridge_alpha = 100.0
    coefs_h = np.zeros((n_horizons, 2), dtype=np.float64)
    intercept_h = np.zeros(n_horizons, dtype=np.float64)

    for h in range(n_horizons):
        b_flat = base[:, h, :][mask[:, h, :]]
        a_flat = anchor[:, h, :][mask[:, h, :]]
        t_flat = target[:, h, :][mask[:, h, :]]
        X = np.column_stack([b_flat, a_flat])
        model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X, t_flat)
        coefs_h[h] = model.coef_
        intercept_h[h] = model.intercept_

    pred = np.zeros_like(base)
    for h in range(n_horizons):
        pred[:, h, :] = coefs_h[h, 0] * base[:, h, :] + coefs_h[h, 1] * anchor[:, h, :] + intercept_h[h]

    residual = pred - target
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 10:
            node_bias[n] = residual[:, :, n][m].mean()

    node_bias = 0.5 * node_bias
    node_bias = _graph_smooth(node_bias, adj, alpha=0.2, n_iter=3)
    pred = pred - node_bias[np.newaxis, np.newaxis, :]

    residual = pred - target
    bias_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 3:
                bias_nh[h, n] = residual[:, h, n][m].mean()

    bias_nh = 0.3 * bias_nh
    for h in range(n_horizons):
        bias_nh[h] = _graph_smooth(bias_nh[h], adj, alpha=0.15, n_iter=2)

    state = {
        'method': 'per_horizon_ridge_node_nhbias_graph',
        'version': 10,
        'coefs_h': coefs_h.astype(np.float32),
        'intercept_h': intercept_h.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float64)
    anchor = np.array(features['anchor'], dtype=np.float64)

    coefs_h = state['coefs_h'].astype(np.float64)
    intercept_h = state['intercept_h'].astype(np.float64)
    node_bias = state['node_bias'].astype(np.float64)
    bias_nh = state['bias_nh'].astype(np.float64)

    n_samples, n_horizons, n_nodes = base.shape
    result = np.zeros_like(base)

    for h in range(n_horizons):
        result[:, h, :] = coefs_h[h, 0] * base[:, h, :] + coefs_h[h, 1] * anchor[:, h, :] + intercept_h[h]

    result = result - node_bias[np.newaxis, np.newaxis, :]
    result = result - bias_nh[np.newaxis, :, :]
    result = np.maximum(result, 0.0)

    return result.astype(np.float32)
