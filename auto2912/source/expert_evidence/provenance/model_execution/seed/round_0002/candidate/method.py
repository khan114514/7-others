"""Shared-slope calibration: global base+anchor weights + node/horizon intercepts + per-node-horizon bias."""
import numpy as np


GRAPH_PATH = '/workspace/public_assets/graph.npz'


def _graph_smooth(values_n, adj, alpha=0.1, n_iter=2):
    """Lightly smooth 1D array across nodes."""
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

    # --- Step 1: Global regression: target = w_base * base + w_anchor * anchor + intercept ---
    # Shared weights across all nodes and horizons
    b_flat = base[mask]
    a_flat = anchor[mask]
    t_flat = target[mask]

    X_global = np.column_stack([b_flat, a_flat])
    # Ridge with moderate regularization
    from sklearn.linear_model import Ridge
    global_model = Ridge(alpha=100.0, fit_intercept=True).fit(X_global, t_flat)
    w_base = global_model.coef_[0]
    w_anchor = global_model.coef_[1]
    global_intercept = global_model.intercept_

    # Compute global prediction
    global_pred = w_base * base + w_anchor * anchor + global_intercept

    # --- Step 2: Per-node intercept correction (shrunken, graph-smoothed) ---
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 5:
            node_bias[n] = (global_pred[:, :, n][m] - target[:, :, n][m]).mean()

    # Shrink toward 0
    shrink_node = 0.4
    node_bias = shrink_node * node_bias  # shrink toward 0 (global intercept already accounts for average)

    # Graph smooth node biases
    node_bias = _graph_smooth(node_bias, adj, alpha=0.15, n_iter=2)

    pred_after_node = global_pred - node_bias[np.newaxis, np.newaxis, :]

    # --- Step 3: Per-horizon intercept correction ---
    horizon_bias = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        if m.sum() > 0:
            horizon_bias[h] = (pred_after_node[:, h, :][m] - target[:, h, :][m]).mean()

    pred_after_h = pred_after_node - horizon_bias[np.newaxis, :, np.newaxis]

    # --- Step 4: Per-node-horizon bias correction (light, shrunken) ---
    bias_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 3:
                bias_nh[h, n] = (pred_after_h[:, h, n][m] - target[:, h, n][m]).mean()
            else:
                bias_nh[h, n] = 0.0

    shrink_nh = 0.2  # heavy shrinkage
    bias_nh = shrink_nh * bias_nh

    state = {
        'method': 'shared_slope_base_anchor_node_h_nhbias',
        'version': 9,
        'w_base': np.float32(w_base),
        'w_anchor': np.float32(w_anchor),
        'global_intercept': np.float32(global_intercept),
        'node_bias': node_bias.astype(np.float32),
        'horizon_bias': horizon_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float64)
    anchor = np.array(features['anchor'], dtype=np.float64)

    w_base = float(state['w_base'])
    w_anchor = float(state['w_anchor'])
    global_intercept = float(state['global_intercept'])
    node_bias = state['node_bias'].astype(np.float64)
    horizon_bias = state['horizon_bias'].astype(np.float64)
    bias_nh = state['bias_nh'].astype(np.float64)

    # Global prediction
    result = w_base * base + w_anchor * anchor + global_intercept

    # Per-node bias
    result = result - node_bias[np.newaxis, np.newaxis, :]

    # Per-horizon bias
    result = result - horizon_bias[np.newaxis, :, np.newaxis]

    # Per-node-horizon bias
    result = result - bias_nh[np.newaxis, :, :]

    # Ensure non-negative
    result = np.maximum(result, 0.0)

    return result.astype(np.float32)
