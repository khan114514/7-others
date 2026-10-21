"""v25: v23 base + conservative per-sample bias correction via base-anchor diff."""
import numpy as np
from sklearn.linear_model import Ridge

GRAPH_PATH = '/workspace/public_assets/graph.npz'


def fit(features, targets, seed):
    rng = np.random.RandomState(seed)
    base = features['base'].astype(np.float64)
    target = targets.astype(np.float64)
    anchor = features['anchor'].astype(np.float64)
    history = features['history'].astype(np.float64)
    clock = features['clock'].astype(np.float64)

    n_samples, n_horizons, n_nodes = base.shape
    mask = np.isfinite(target) & (target > 0)

    # ---- Step 1: Per-horizon optimal anchor blending ----
    anchor_h = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        best_mae = float('inf')
        for w in np.linspace(0, 0.30, 31):
            pred = (1 - w) * base[:, h, :][m] + w * anchor[:, h, :][m]
            mae = np.abs(pred - target[:, h, :][m]).mean()
            if mae < best_mae:
                best_mae = mae
                anchor_h[h] = w

    # ---- Step 2: Per-node-horizon anchor blending (strongly shrunken) ----
    pna_shrink = 0.2
    anchor_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 15:
                best_mae = float('inf')
                best_w = anchor_h[h]
                for w in np.linspace(0, 0.40, 21):
                    pred = (1 - w) * base[:, h, n][m] + w * anchor[:, h, n][m]
                    mae = np.abs(pred - target[:, h, n][m]).mean()
                    if mae < best_mae:
                        best_mae = mae
                        best_w = w
                anchor_nh[h, n] = best_w
            else:
                anchor_nh[h, n] = anchor_h[h]

    anchor_nh = pna_shrink * anchor_nh + (1 - pna_shrink) * anchor_h[:, np.newaxis]

    blended = (1 - anchor_nh[np.newaxis, :, :]) * base + anchor_nh[np.newaxis, :, :] * anchor

    # ---- Step 3: Per-horizon median bias ----
    h_bias = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        h_bias[h] = np.median(blended[:, h, :][m] - target[:, h, :][m])

    corrected = blended - h_bias[np.newaxis, :, np.newaxis]
    residual = corrected - target

    # ---- Step 4: Per-node median bias (shrunken) ----
    node_shrink = 0.4
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 5:
            node_bias[n] = np.median(residual[:, :, n][m])

    node_bias = node_shrink * node_bias
    corrected = corrected - node_bias[np.newaxis, np.newaxis, :]
    residual = corrected - target

    # ---- Step 5: Per-node-horizon median bias (strongly shrunken) ----
    pnh_bias_shrink = 0.15
    bias_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 0:
                raw = np.median(residual[:, h, n][m])
                bias_nh[h, n] = pnh_bias_shrink * raw

    corrected = corrected - bias_nh[np.newaxis, :, :]
    residual = corrected - target

    # ---- Step 6: Multiplicative per-node correction ----
    mult_shrink = 0.5
    mult_factor = np.ones(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        p = corrected[:, :, n][m]
        tgt = target[:, :, n][m]
        if m.sum() > 20 and np.min(p) > 1:
            ratios = tgt / p
            mult_factor[n] = np.median(ratios)

    mult_factor = 1.0 + mult_shrink * (mult_factor - 1.0)
    corrected = corrected * mult_factor[np.newaxis, np.newaxis, :]
    residual = residual * mult_factor[np.newaxis, np.newaxis, :]

    # ---- Step 6b: Per-sample bias correction via base-anchor diff ----
    # Simple per-horizon slope model with ridge regularization.
    # Captures sample-level forecast bias that base-anchor divergence signals.
    sample_shrink = 0.2
    sample_slope = np.zeros(n_horizons)
    sample_intercept = np.zeros(n_horizons)

    sample_ba_diff = np.zeros((n_samples, n_horizons))
    sample_resid = np.zeros((n_samples, n_horizons))
    for h in range(n_horizons):
        for i in range(n_samples):
            mi = mask[i, h, :]
            if mi.sum() > 0:
                sample_ba_diff[i, h] = (base[i, h, :][mi] - anchor[i, h, :][mi]).mean()
                sample_resid[i, h] = residual[i, h, :][mi].mean()

    sample_alpha = 10.0
    for h in range(n_horizons):
        valid = np.isfinite(sample_resid[:, h]) & np.isfinite(sample_ba_diff[:, h])
        if valid.sum() < 10:
            continue
        x = sample_ba_diff[valid, h]
        y = sample_resid[valid, h]
        # Ridge regression with intercept
        X = np.column_stack([np.ones(len(x)), x])
        coef = np.linalg.lstsq(X.T @ X + sample_alpha * np.eye(2), X.T @ y, rcond=None)[0]
        sample_intercept[h] = coef[0]
        sample_slope[h] = coef[1]

    # Apply correction in-place
    pred_sample_bias = sample_intercept[np.newaxis, :] + sample_ba_diff * sample_slope[np.newaxis, :]
    corrected = corrected - sample_shrink * pred_sample_bias[:, :, np.newaxis]
    residual = residual - sample_shrink * pred_sample_bias[:, :, np.newaxis]

    # ---- Step 7: Ridge residual correction (per horizon, strongly regularized) ----
    ridge_alpha = 1000.0
    ridge_shrink = 0.3
    ridge_models = []

    hist_last = history[:, -1, :]
    hist_avg = history.mean(axis=1)
    hist_trend = (history[:, -1, :] - history[:, 0, :]) / 12.0

    for h in range(n_horizons):
        m = mask[:, h, :]
        y = residual[:, h, :][m].flatten()
        if len(y) < 100:
            ridge_models.append(None)
            continue

        node_idx_flat = np.tile(np.arange(n_nodes), (n_samples, 1)).reshape(-1, 1)
        X = np.hstack([
            base[:, h, :].reshape(-1, 1),
            anchor[:, h, :].reshape(-1, 1),
            hist_last.reshape(-1, 1),
            hist_avg.reshape(-1, 1),
            hist_trend.reshape(-1, 1),
            np.sin(2 * np.pi * np.repeat(clock[:, h, 0], n_nodes)).reshape(-1, 1),
            np.cos(2 * np.pi * np.repeat(clock[:, h, 0], n_nodes)).reshape(-1, 1),
            np.sin(2 * np.pi * node_idx_flat / n_nodes),
            np.cos(2 * np.pi * node_idx_flat / n_nodes),
        ])
        X_valid = X[m.flatten()]

        mu = X_valid.mean(axis=0)
        sd = X_valid.std(axis=0)
        sd[sd < 1e-6] = 1.0
        Xs = (X_valid - mu) / sd

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(Xs, y)

        ridge_models.append({
            'coef': model.coef_.astype(np.float32),
            'intercept': np.float32(model.intercept_),
            'mu': mu.astype(np.float32),
            'sd': sd.astype(np.float32),
        })

    state = {
        'method': 'v25_sample_bias',
        'version': 25,
        'anchor_nh': anchor_nh.astype(np.float32),
        'h_bias': h_bias.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
        'mult_factor': mult_factor.astype(np.float32),
        'sample_slope': sample_slope.astype(np.float32),
        'sample_intercept': sample_intercept.astype(np.float32),
        'sample_shrink': np.float32(sample_shrink),
        'ridge_shrink': np.float32(ridge_shrink),
        'ridge_models': ridge_models,
        'n_nodes': n_nodes,
        'n_horizons': n_horizons,
    }
    return state


def predict(state, features):
    base = np.array(features['base'], dtype=np.float64)
    anchor = np.array(features['anchor'], dtype=np.float64)
    history = np.array(features['history'], dtype=np.float64)
    clock = np.array(features['clock'], dtype=np.float64)

    n_samples, n_horizons, n_nodes = base.shape

    anchor_nh = state['anchor_nh'].astype(np.float64)
    h_bias = state['h_bias'].astype(np.float64)
    node_bias = state['node_bias'].astype(np.float64)
    bias_nh = state['bias_nh'].astype(np.float64)
    mult_factor = state['mult_factor'].astype(np.float64)
    sample_slope = state['sample_slope'].astype(np.float64)
    sample_intercept = state['sample_intercept'].astype(np.float64)
    sample_shrink = float(state['sample_shrink'])
    ridge_shrink = float(state['ridge_shrink'])
    ridge_models = state['ridge_models']

    # Anchor blending
    result = (1 - anchor_nh[np.newaxis, :, :]) * base + anchor_nh[np.newaxis, :, :] * anchor

    # Additive bias corrections
    result = result - h_bias[np.newaxis, :, np.newaxis]
    result = result - node_bias[np.newaxis, np.newaxis, :]
    result = result - bias_nh[np.newaxis, :, :]

    # Multiplicative correction
    result = result * mult_factor[np.newaxis, np.newaxis, :]

    # Per-sample bias correction
    sample_ba_diff = np.zeros((n_samples, n_horizons))
    for h in range(n_horizons):
        sample_ba_diff[:, h] = (base[:, h, :] - anchor[:, h, :]).mean(axis=1)
    pred_sample_bias = sample_intercept[np.newaxis, :] + sample_ba_diff * sample_slope[np.newaxis, :]
    result = result - sample_shrink * pred_sample_bias[:, :, np.newaxis]

    # Ridge correction
    hist_last = history[:, -1, :]
    hist_avg = history.mean(axis=1)
    hist_trend = (history[:, -1, :] - history[:, 0, :]) / 12.0

    for h in range(n_horizons):
        if ridge_models[h] is not None:
            rm = ridge_models[h]
            coef = rm['coef'].astype(np.float64)
            intercept = float(rm['intercept'])
            mu = rm['mu'].astype(np.float64)
            sd = rm['sd'].astype(np.float64)

            node_idx_flat = np.tile(np.arange(n_nodes), (n_samples, 1)).reshape(-1, 1)
            X = np.hstack([
                base[:, h, :].reshape(-1, 1),
                anchor[:, h, :].reshape(-1, 1),
                hist_last.reshape(-1, 1),
                hist_avg.reshape(-1, 1),
                hist_trend.reshape(-1, 1),
                np.sin(2 * np.pi * np.repeat(clock[:, h, 0], n_nodes)).reshape(-1, 1),
                np.cos(2 * np.pi * np.repeat(clock[:, h, 0], n_nodes)).reshape(-1, 1),
                np.sin(2 * np.pi * node_idx_flat / n_nodes),
                np.cos(2 * np.pi * node_idx_flat / n_nodes),
            ])
            Xs = (X - mu) / sd
            correction = ridge_shrink * (Xs @ coef + intercept)
            result[:, h, :] = result[:, h, :] - correction.reshape(n_samples, n_nodes)

    result = np.maximum(result, 0.0)
    return result.astype(np.float32)
