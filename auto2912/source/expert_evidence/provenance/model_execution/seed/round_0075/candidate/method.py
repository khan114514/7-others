"""v26: v25 base + improved per-sample bias predictor + expanded ridge features + optimized pna_shrink."""
import numpy as np
from sklearn.linear_model import Ridge

GRAPH_PATH = '/workspace/public_assets/graph.npz'


def _build_sample_feats(base, anchor, history, clock, n_h, n_nodes, n_s):
    """Build per-sample features for sample-level bias prediction.
    Returns array of shape [n_s, n_h, n_feat].
    """
    feats_list = []
    feats_list.append(base.mean(axis=2)[:, :, np.newaxis])
    feats_list.append(anchor.mean(axis=2)[:, :, np.newaxis])
    feats_list.append((base - anchor).mean(axis=2)[:, :, np.newaxis])
    hist_last = history[:, -1, :]
    feats_list.append(np.tile(hist_last.mean(axis=1)[:, np.newaxis, np.newaxis], (1, n_h, 1)))
    feats_list.append(np.tile(history.mean(axis=(1, 2))[:, np.newaxis, np.newaxis], (1, n_h, 1)))
    hist_trend = (history[:, -1, :] - history[:, 0, :]) / 12.0
    feats_list.append(np.tile(hist_trend.mean(axis=1)[:, np.newaxis, np.newaxis], (1, n_h, 1)))
    feats_list.append(np.sin(2 * np.pi * clock[:, :, 0:1]))
    feats_list.append(np.cos(2 * np.pi * clock[:, :, 0:1]))
    feats_list.append(np.sin(4 * np.pi * clock[:, :, 0:1]))
    feats_list.append(np.cos(4 * np.pi * clock[:, :, 0:1]))
    return np.concatenate(feats_list, axis=2)


def _build_node_feats(base_h, anchor_h, hist_last, hist_avg, hist_trend,
                      clock_h, n_nodes, n_s):
    """Build per-node per-sample features for ridge correction at horizon h.
    Returns array of shape [n_s * n_nodes, n_feat].
    """
    node_idx = np.tile(np.arange(n_nodes), (n_s, 1)).reshape(-1, 1)
    X = np.hstack([
        base_h.reshape(-1, 1),
        anchor_h.reshape(-1, 1),
        (base_h - anchor_h).reshape(-1, 1),
        hist_last.reshape(-1, 1),
        hist_avg.reshape(-1, 1),
        hist_trend.reshape(-1, 1),
        (hist_last - base_h).reshape(-1, 1),
        np.sin(2 * np.pi * np.repeat(clock_h, n_nodes)).reshape(-1, 1),
        np.cos(2 * np.pi * np.repeat(clock_h, n_nodes)).reshape(-1, 1),
        np.sin(4 * np.pi * np.repeat(clock_h, n_nodes)).reshape(-1, 1),
        np.cos(4 * np.pi * np.repeat(clock_h, n_nodes)).reshape(-1, 1),
        np.sin(2 * np.pi * node_idx / n_nodes),
        np.cos(2 * np.pi * node_idx / n_nodes),
        np.sin(4 * np.pi * node_idx / n_nodes),
        np.cos(4 * np.pi * node_idx / n_nodes),
    ])
    return X


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

    # ---- Step 2: Per-node-horizon anchor blending ----
    pna_shrink = 0.3
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

    # ---- Step 4: Per-node median bias ----
    node_shrink = 0.4
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 5:
            node_bias[n] = np.median(residual[:, :, n][m])

    node_bias = node_shrink * node_bias
    corrected = corrected - node_bias[np.newaxis, np.newaxis, :]
    residual = corrected - target

    # ---- Step 5: Per-node-horizon median bias ----
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

    # ---- Step 7: Improved per-sample bias correction ----
    # Uses multiple features: base, anchor, history, clock (with harmonics)
    sample_shrink = 0.4
    sample_alpha = 100.0

    sample_resid = np.full((n_samples, n_horizons), np.nan)
    for i in range(n_samples):
        for h in range(n_horizons):
            mi = mask[i, h, :]
            if mi.sum() > 0:
                sample_resid[i, h] = residual[i, h, :][mi].mean()

    sample_feat = _build_sample_feats(base, anchor, history, clock, n_horizons, n_nodes, n_samples)

    sample_models = []
    for h in range(n_horizons):
        valid = np.isfinite(sample_resid[:, h])
        if valid.sum() < 10:
            sample_models.append(None)
            continue
        X = sample_feat[valid, h, :]
        y = sample_resid[valid, h]
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        Xs = (X - mu) / sd
        model = Ridge(alpha=sample_alpha, fit_intercept=True)
        model.fit(Xs, y)
        sample_models.append({
            'coef': model.coef_.astype(np.float32),
            'intercept': np.float32(model.intercept_),
            'mu': mu.astype(np.float32),
            'sd': sd.astype(np.float32),
        })

    # Apply sample bias correction (in-place on corrected/residual for downstream steps)
    sample_pred_bias = np.zeros((n_samples, n_horizons))
    for h in range(n_horizons):
        if sample_models[h] is not None:
            sm = sample_models[h]
            coef = sm['coef'].astype(np.float64)
            intercept = float(sm['intercept'])
            mu = sm['mu'].astype(np.float64)
            sd = sm['sd'].astype(np.float64)
            X = sample_feat[:, h, :]
            Xs = (X - mu) / sd
            sample_pred_bias[:, h] = Xs @ coef + intercept

    corrected = corrected - sample_shrink * sample_pred_bias[:, :, np.newaxis]
    residual = residual - sample_shrink * sample_pred_bias[:, :, np.newaxis]

    # ---- Step 8: Ridge correction with expanded features ----
    ridge_shrink = 0.3
    ridge_alpha = 100.0

    hist_last = history[:, -1, :]
    hist_avg = history.mean(axis=1)
    hist_trend = (history[:, -1, :] - history[:, 0, :]) / 12.0

    ridge_models = []
    for h in range(n_horizons):
        m = mask[:, h, :]
        y = residual[:, h, :][m].flatten()
        if len(y) < 100:
            ridge_models.append(None)
            continue

        X = _build_node_feats(
            base[:, h, :], anchor[:, h, :], hist_last, hist_avg, hist_trend,
            clock[:, h, 0], n_nodes, n_samples
        )
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
        'method': 'v26_improved_sample_bias',
        'version': 26,
        'anchor_nh': anchor_nh.astype(np.float32),
        'h_bias': h_bias.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
        'mult_factor': mult_factor.astype(np.float32),
        'sample_shrink': np.float32(sample_shrink),
        'sample_models': sample_models,
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
    sample_shrink = float(state['sample_shrink'])
    sample_models = state['sample_models']
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

    # Per-sample bias correction (improved multi-feature)
    sample_feat = _build_sample_feats(base, anchor, history, clock, n_horizons, n_nodes, n_samples)
    sample_pred_bias = np.zeros((n_samples, n_horizons))
    for h in range(n_horizons):
        if sample_models[h] is not None:
            sm = sample_models[h]
            coef = sm['coef'].astype(np.float64)
            intercept = float(sm['intercept'])
            mu = sm['mu'].astype(np.float64)
            sd = sm['sd'].astype(np.float64)
            X = sample_feat[:, h, :]
            Xs = (X - mu) / sd
            sample_pred_bias[:, h] = Xs @ coef + intercept

    result = result - sample_shrink * sample_pred_bias[:, :, np.newaxis]

    # Ridge correction with expanded features
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

            X = _build_node_feats(
                base[:, h, :], anchor[:, h, :], hist_last, hist_avg, hist_trend,
                clock[:, h, 0], n_nodes, n_samples
            )
            Xs = (X - mu) / sd
            correction = ridge_shrink * (Xs @ coef + intercept)
            result[:, h, :] = result[:, h, :] - correction.reshape(n_samples, n_nodes)

    result = np.maximum(result, 0.0)
    return result.astype(np.float32)
