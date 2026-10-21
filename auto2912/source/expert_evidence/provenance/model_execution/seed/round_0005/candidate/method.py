"""Global per-horizon LAD with base+anchor+history + strong graph-smoothed biases."""
import numpy as np

GRAPH_PATH = '/workspace/public_assets/graph.npz'


def _lad_fit(X, y, max_iter=50, tol=1e-6, min_weight=1e-6):
    n, p = X.shape
    X_aug = np.column_stack([X, np.ones(n)])
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    for _ in range(max_iter):
        resid = y - X_aug @ beta
        w = 1.0 / np.maximum(np.abs(resid), min_weight)
        XtWX = (X_aug.T * w) @ X_aug
        XtWy = (X_aug.T * w) @ y
        try:
            beta_new = np.linalg.solve(XtWX + 1e-10 * np.eye(p + 1), XtWy)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    return beta[:-1], beta[-1]


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

    # History-based features (broadcast to match horizon dim)
    h_last = history[:, -1, :]  # (sample, node)
    h_avg = history.mean(axis=1)  # (sample, node)

    # Step 1: Global per-horizon LAD with base + anchor + last history
    n_features = 4  # base, anchor, h_last, h_avg
    coefs_h = np.zeros((n_horizons, n_features), dtype=np.float64)
    intercepts_h = np.zeros(n_horizons, dtype=np.float64)
    for h in range(n_horizons):
        m = mask[:, h, :]
        X = np.column_stack([
            base[:, h, :][m],
            anchor[:, h, :][m],
            h_last[m],
            h_avg[m],
        ])
        y = target[:, h, :][m]
        c, i = _lad_fit(X, y)
        coefs_h[h] = c
        intercepts_h[h] = i

    pred = np.zeros_like(base)
    for h in range(n_horizons):
        pred[:, h, :] = (
            coefs_h[h, 0] * base[:, h, :]
            + coefs_h[h, 1] * anchor[:, h, :]
            + coefs_h[h, 2] * h_last
            + coefs_h[h, 3] * h_avg
            + intercepts_h[h]
        )

    # Step 2: Per-node residual bias (graph-smoothed, shrunken)
    residual = pred - target
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 10:
            node_bias[n] = residual[:, :, n][m].mean()
    node_bias = 0.6 * node_bias
    node_bias = _graph_smooth(node_bias, adj, alpha=0.2, n_iter=3)
    pred = pred - node_bias[np.newaxis, np.newaxis, :]

    # Step 3: Per-node-horizon bias (conservative)
    residual = pred - target
    bias_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        global_h_res = residual[:, h, :][mask[:, h, :]].mean()
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 3:
                raw = residual[:, h, n][m].mean()
                bias_nh[h, n] = 0.25 * raw + 0.75 * global_h_res
            else:
                bias_nh[h, n] = global_h_res
    for h in range(n_horizons):
        bias_nh[h] = _graph_smooth(bias_nh[h], adj, alpha=0.15, n_iter=2)

    state = {
        'method': 'global_lad_ba_hist_nodebias_nhbias',
        'version': 4,
        'coefs_h': coefs_h.astype(np.float32),
        'intercepts_h': intercepts_h.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float64)
    anchor = np.array(features['anchor'], dtype=np.float64)
    history = np.array(features['history'], dtype=np.float64)

    coefs_h = state['coefs_h'].astype(np.float64)
    intercepts_h = state['intercepts_h'].astype(np.float64)
    node_bias = state['node_bias'].astype(np.float64)
    bias_nh = state['bias_nh'].astype(np.float64)

    h_last = history[:, -1, :]
    h_avg = history.mean(axis=1)

    n_horizons = len(intercepts_h)
    result = np.zeros_like(base)
    for h in range(n_horizons):
        result[:, h, :] = (
            coefs_h[h, 0] * base[:, h, :]
            + coefs_h[h, 1] * anchor[:, h, :]
            + coefs_h[h, 2] * h_last
            + coefs_h[h, 3] * h_avg
            + intercepts_h[h]
        )

    result -= node_bias[np.newaxis, np.newaxis, :]
    result -= bias_nh[np.newaxis, :, :]
    result = np.maximum(result, 0.0)
    return result.astype(np.float32)
