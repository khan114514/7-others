"""Per-horizon slope+intercept + anchor blend + history correction + per-node graph bias (no nh bias)."""
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
    history = features['history'].astype(np.float64)

    n_samples, n_horizons, n_nodes = base.shape
    mask = np.isfinite(target) & (target > 0)

    graph = np.load(GRAPH_PATH)
    adj = graph['adjacency'].astype(np.float64)

    # Step 0: Conservative anchor blend
    anchor_weight = 0.02
    blended = (1.0 - anchor_weight) * base + anchor_weight * anchor

    # Step 1: Per-horizon slope + intercept
    ridge_alpha = 1000.0
    slopes = np.ones(n_horizons)
    intercepts = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        b_flat = blended[:, h, :][m]
        t_flat = target[:, h, :][m]
        X = b_flat.reshape(-1, 1)
        model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X, t_flat)
        slopes[h] = model.coef_[0]
        intercepts[h] = model.intercept_

    corrected = np.zeros_like(blended)
    for h in range(n_horizons):
        corrected[:, h, :] = slopes[h] * blended[:, h, :] + intercepts[h]

    # Step 2: Per-node bias
    residual = corrected - target
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 5:
            node_bias[n] = residual[:, :, n][m].mean()
    node_bias = 0.5 * node_bias
    node_bias = _graph_smooth(node_bias, adj, alpha=0.2, n_iter=3)
    corrected = corrected - node_bias[np.newaxis, np.newaxis, :]

    # Step 3: History-based correction
    h_last = history[:, -1, :]
    base_h0 = base[:, 0, :]
    init_error = h_last - base_h0
    
    residual = corrected - target
    hist_coefs = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        ie = init_error[m]
        res = residual[:, h, :][m]
        if len(ie) > 10 and np.std(ie) > 0:
            X = ie.reshape(-1, 1)
            model = Ridge(alpha=1000.0, fit_intercept=False).fit(X, res)
            hist_coefs[h] = model.coef_[0]
    hist_coefs = 0.3 * hist_coefs

    state = {
        'method': 'per_h_slope_anchor_hist_nodebias_graph',
        'version': 20,
        'anchor_weight': np.float32(anchor_weight),
        'slopes': slopes.astype(np.float32),
        'intercepts': intercepts.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'hist_coefs': hist_coefs.astype(np.float32),
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float64)
    anchor = np.array(features['anchor'], dtype=np.float64)
    history = np.array(features['history'], dtype=np.float64)

    anchor_weight = float(state['anchor_weight'])
    slopes = state['slopes'].astype(np.float64)
    intercepts = state['intercepts'].astype(np.float64)
    node_bias = state['node_bias'].astype(np.float64)
    hist_coefs = state['hist_coefs'].astype(np.float64)

    blended = (1.0 - anchor_weight) * base + anchor_weight * anchor
    n_horizons = len(slopes)
    result = np.zeros_like(blended)
    for h in range(n_horizons):
        result[:, h, :] = slopes[h] * blended[:, h, :] + intercepts[h]

    result = result - node_bias[np.newaxis, np.newaxis, :]

    # History correction
    h_last = history[:, -1, :]
    base_h0 = base[:, 0, :]
    init_error = h_last - base_h0
    for h in range(n_horizons):
        result[:, h, :] = result[:, h, :] - hist_coefs[h] * init_error

    result = np.maximum(result, 0.0)

    return result.astype(np.float32)
