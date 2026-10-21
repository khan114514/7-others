"""Anchor blend + per-horizon Ridge + per-node-horizon shrunken bias + history correction."""
import numpy as np
from sklearn.linear_model import Ridge

GRAPH_PATH = '/workspace/public_assets/graph.npz'


def fit(features, targets, seed):
    rng = np.random.RandomState(seed)
    base = features['base'].astype(np.float64)
    target = targets.astype(np.float64)
    anchor = features['anchor'].astype(np.float64)
    history = features['history'].astype(np.float64)

    n_samples, n_horizons, n_nodes = base.shape
    mask = np.isfinite(target) & (target > 0)

    # Step 1: Anchor blend
    anchor_weight = 0.12
    blended = (1.0 - anchor_weight) * base + anchor_weight * anchor

    # Step 2: Per-horizon Ridge regression (slope + intercept)
    ridge_alpha = 1000.0
    slopes = np.zeros(n_horizons)
    intercepts = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        b_flat = blended[:, h, :][m].reshape(-1, 1)
        t_flat = target[:, h, :][m]
        model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(b_flat, t_flat)
        slopes[h] = model.coef_[0]
        intercepts[h] = model.intercept_

    # Apply regression to get initial prediction
    pred = np.zeros_like(blended)
    for h in range(n_horizons):
        pred[:, h, :] = slopes[h] * blended[:, h, :] + intercepts[h]

    # Step 3: Per-node-horizon shrunken bias correction
    shrink = 0.4
    node_horizon_bias = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        m = mask[:, h, :]
        global_h_res = (pred[:, h, :][m] - target[:, h, :][m]).mean()
        for n in range(n_nodes):
            m_n = mask[:, h, n]
            if m_n.sum() > 0:
                node_res = (pred[:, h, n][m_n] - target[:, h, n][m_n]).mean()
                node_horizon_bias[h, n] = shrink * node_res + (1 - shrink) * global_h_res
            else:
                node_horizon_bias[h, n] = global_h_res

    # Step 4: History-based correction
    h_last = history[:, -1, :]
    base_h0 = base[:, 0, :]
    init_error = h_last - base_h0

    # Compute residuals after bias correction and fit history coef
    corrected = pred.copy()
    for h in range(n_horizons):
        corrected[:, h, :] -= node_horizon_bias[h, :][np.newaxis, :]

    hist_coefs = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :].flatten()
        res_flat = (corrected[:, h, :] - target[:, h, :]).flatten()[m]
        ie_flat = init_error.flatten()[m]
        if len(ie_flat) > 10 and np.std(ie_flat) > 0:
            X = ie_flat.reshape(-1, 1)
            model = Ridge(alpha=1000.0, fit_intercept=False).fit(X, res_flat)
            hist_coefs[h] = 0.3 * model.coef_[0]

    state = {
        'method': 'anchor_blend_perh_ridge_nhbias_hist',
        'version': 21,
        'anchor_weight': np.float32(anchor_weight),
        'slopes': slopes.astype(np.float32),
        'intercepts': intercepts.astype(np.float32),
        'node_horizon_bias': node_horizon_bias.astype(np.float32),
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
    node_horizon_bias = state['node_horizon_bias'].astype(np.float64)
    hist_coefs = state['hist_coefs'].astype(np.float64)

    n_horizons = len(slopes)
    blended = (1.0 - anchor_weight) * base + anchor_weight * anchor

    result = np.zeros_like(blended)
    for h in range(n_horizons):
        result[:, h, :] = slopes[h] * blended[:, h, :] + intercepts[h]

    for h in range(n_horizons):
        result[:, h, :] -= node_horizon_bias[h, :][np.newaxis, :]

    # History correction
    h_last = history[:, -1, :]
    base_h0 = base[:, 0, :]
    init_error = h_last - base_h0
    for h in range(n_horizons):
        result[:, h, :] -= hist_coefs[h] * init_error

    result = np.maximum(result, 0.0)

    return result.astype(np.float32)
