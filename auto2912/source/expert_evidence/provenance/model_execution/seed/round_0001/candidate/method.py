"""Calibrated forecasting: per-node linear scaling + shrunken per-node-horizon bias + anchor blending."""
import numpy as np


def fit(features, targets, seed):
    rng = np.random.RandomState(seed)
    base = features['base'].astype(np.float64)
    target = targets.astype(np.float64)
    anchor = features['anchor'].astype(np.float64)
    
    n_samples, n_horizons, n_nodes = base.shape
    mask = np.isfinite(target) & (target > 0)
    
    # --- Step 1: Per-node linear scaling (target = slope * base + intercept) ---
    b_flat = base[mask]
    t_flat = target[mask]
    var_b = np.var(b_flat)
    global_slope = np.cov(b_flat, t_flat)[0, 1] / var_b if var_b > 1e-6 else 1.0
    global_intercept = np.mean(t_flat) - global_slope * np.mean(b_flat)
    
    node_slope = np.full(n_nodes, global_slope, dtype=np.float64)
    node_intercept = np.full(n_nodes, global_intercept, dtype=np.float64)
    
    for n in range(n_nodes):
        m = mask[:, :, n]
        b_n = base[:, :, n][m]
        t_n = target[:, :, n][m]
        if len(b_n) > 10:
            var_b_n = np.var(b_n)
            if var_b_n > 1e-6:
                s = np.cov(b_n, t_n)[0, 1] / var_b_n
                i = np.mean(t_n) - s * np.mean(b_n)
                node_slope[n] = s
                node_intercept[n] = i
    
    # Shrink toward global
    shrink_slope = 0.3
    shrink_intercept = 0.3
    node_slope = shrink_slope * node_slope + (1.0 - shrink_slope) * global_slope
    node_intercept = shrink_intercept * node_intercept + (1.0 - shrink_intercept) * global_intercept
    
    # Apply scaling to get scaled base
    scaled_base = base * node_slope[np.newaxis, np.newaxis, :] + node_intercept[np.newaxis, np.newaxis, :]
    
    # --- Step 2: Per-node-horizon residual bias correction ---
    bias_h = np.zeros(n_horizons, dtype=np.float64)
    for h in range(n_horizons):
        m = mask[:, h, :]
        if m.sum() > 0:
            bias_h[h] = (scaled_base[:, h, :][m] - target[:, h, :][m]).mean()
    
    bias_nh = np.zeros((n_horizons, n_nodes), dtype=np.float64)
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 0:
                bias_nh[h, n] = (scaled_base[:, h, n][m] - target[:, h, n][m]).mean()
            else:
                bias_nh[h, n] = bias_h[h]
    
    shrink_nh = 0.3
    shrunk_bias = shrink_nh * bias_nh + (1.0 - shrink_nh) * bias_h[:, np.newaxis]
    
    # --- Step 3: Anchor blending ---
    anchor_scale = 0.06
    anchor_coeffs = anchor_scale * np.linspace(0, 1, n_horizons)
    
    state = {
        'method': 'node_linear_nh_bias_anchor',
        'version': 4,
        'node_slope': node_slope.astype(np.float32),
        'node_intercept': node_intercept.astype(np.float32),
        'shrunk_bias': shrunk_bias.astype(np.float32),
        'anchor_coeffs': anchor_coeffs.astype(np.float32),
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float32, copy=True)
    anchor = np.array(features['anchor'], dtype=np.float32, copy=True)
    
    node_slope = state['node_slope']  # (170,)
    node_intercept = state['node_intercept']  # (170,)
    bias = state['shrunk_bias']  # (12, 170)
    anchor_coeffs = state['anchor_coeffs']  # (12,)
    
    # Step 1: linear scaling
    result = base * node_slope[np.newaxis, np.newaxis, :] + node_intercept[np.newaxis, np.newaxis, :]
    
    # Step 2: bias correction
    result = result - bias[np.newaxis, :, :]
    
    # Step 3: anchor blending
    anchor_diff = anchor - result
    result = result + anchor_coeffs[np.newaxis, :, np.newaxis] * anchor_diff
    
    # Ensure non-negative
    result = np.maximum(result, 0.0)
    
    return result.astype(np.float32)
