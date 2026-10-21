"""Per-node-horizon optimal anchor blending + multi-stage bias correction + ridge residual model."""
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
    clock = features['clock'].astype(np.float64)

    n_samples, n_horizons, n_nodes = base.shape
    mask = np.isfinite(target) & (target > 0)

    graph = np.load(GRAPH_PATH)
    adj = graph['adjacency'].astype(np.float64)
    deg = np.maximum(adj.sum(axis=1), 1.0)

    # Step 1: Per-horizon optimal anchor blending (global)
    anchor_h = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        best_mae = float('inf')
        for w in np.linspace(0, 0.2, 41):
            pred = (1 - w) * base[:, h, :][m] + w * anchor[:, h, :][m]
            mae = np.abs(pred - target[:, h, :][m]).mean()
            if mae < best_mae:
                best_mae = mae
                anchor_h[h] = w

    # Step 2: Per-node-horizon optimal anchor blending (shrunken to global)
    pna_shrink = 0.4
    anchor_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 10:
                best_mae = float('inf')
                best_w = anchor_h[h]
                for w in np.linspace(0, 0.4, 41):
                    pred = (1 - w) * base[:, h, n][m] + w * anchor[:, h, n][m]
                    mae = np.abs(pred - target[:, h, n][m]).mean()
                    if mae < best_mae:
                        best_mae = mae
                        best_w = w
                anchor_nh[h, n] = best_w
            else:
                anchor_nh[h, n] = anchor_h[h]

    anchor_nh = pna_shrink * anchor_nh + (1 - pna_shrink) * anchor_h[:, np.newaxis]
    for h in range(n_horizons):
        anchor_nh[h] = _graph_smooth(anchor_nh[h], adj, alpha=0.1, n_iter=2)

    blended = (1 - anchor_nh[np.newaxis, :, :]) * base + anchor_nh[np.newaxis, :, :] * anchor

    # Step 3: Per-horizon median bias
    h_bias = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        h_bias[h] = np.median(blended[:, h, :][m] - target[:, h, :][m])

    corrected = blended - h_bias[np.newaxis, :, np.newaxis]
    residual = corrected - target

    # Step 4: Per-node median bias (graph-smoothed, shrunken)
    node_bias = np.zeros(n_nodes)
    for n in range(n_nodes):
        m = mask[:, :, n]
        if m.sum() > 5:
            node_bias[n] = np.median(residual[:, :, n][m])

    node_bias = 0.5 * node_bias
    node_bias = _graph_smooth(node_bias, adj, alpha=0.2, n_iter=3)
    corrected = corrected - node_bias[np.newaxis, np.newaxis, :]
    residual = corrected - target

    # Step 5: Per-node-horizon median bias (shrunken, graph-smoothed)
    bias_nh = np.zeros((n_horizons, n_nodes))
    for h in range(n_horizons):
        for n in range(n_nodes):
            m = mask[:, h, n]
            if m.sum() > 0:
                raw = np.median(residual[:, h, n][m])
                bias_nh[h, n] = 0.3 * raw

    for h in range(n_horizons):
        bias_nh[h] = _graph_smooth(bias_nh[h], adj, alpha=0.15, n_iter=2)

    corrected = corrected - bias_nh[np.newaxis, :, :]
    residual = corrected - target

    # Step 6: Time-of-day (clock) bias correction
    n_clock_bins = 24
    clock_bins = np.floor(clock * n_clock_bins).astype(int) % n_clock_bins
    clock_bias = np.zeros(n_clock_bins)
    for cb in range(n_clock_bins):
        bin_mask = (clock_bins == cb) & mask
        if bin_mask.sum() > 100:
            clock_bias[cb] = np.median(residual[bin_mask])

    for _ in range(2):
        clock_smooth = np.zeros(n_clock_bins)
        for i in range(n_clock_bins):
            clock_smooth[i] = np.mean(clock_bias[[(i - 1) % n_clock_bins, i, (i + 1) % n_clock_bins]])
        clock_bias = clock_smooth

    clock_bias = 0.25 * clock_bias
    corrected = corrected - clock_bias[clock_bins]
    residual = corrected - target

    # Step 7: History-based per-sample correction
    last_hist = history[:, -1, :]
    base_h0 = base[:, 0, :]
    hist_dev = last_hist - base_h0

    hist_w = np.zeros(n_horizons)
    for h in range(n_horizons):
        m = mask[:, h, :]
        x = hist_dev[m].reshape(-1)
        y = residual[:, h, :][m]
        if len(x) > 10 and np.abs(x).mean() > 0:
            med_x = np.median(x)
            med_y = np.median(y)
            valid = np.abs(x - med_x) > 1
            if valid.sum() > 20:
                hist_w[h] = np.median((y[valid] - med_y) / (x[valid] - med_x + 1e-8))

    hist_w = 0.15 * hist_w
    for h in range(n_horizons):
        corrected[:, h, :] = corrected[:, h, :] - hist_w[h] * hist_dev
    residual = corrected - target

    # Step 8: Pooled ridge regression for residual prediction
    hist_last = history[:, -1, :]
    hist_avg = history.mean(axis=1)
    hist_trend = (history[:, -1, :] - history[:, 0, :]) / 12.0
    hist_hm1 = history[:, -2, :]

    ridge_shrink = 0.25
    ridge_models = []
    alphas = [500.0, 200.0, 100.0, 50.0, 10.0]

    for h in range(n_horizons):
        neighbor_base_h = (base[:, h, :] @ adj) / deg
        m = mask[:, h, :]

        feats_base = base[:, h, :][m].reshape(-1, 1)
        feats_anchor = anchor[:, h, :][m].reshape(-1, 1)
        feats_hist_last = hist_last[m].reshape(-1, 1)
        feats_hist_avg = hist_avg[m].reshape(-1, 1)
        feats_clock = clock[:, h, 0][np.where(m)[0]].reshape(-1, 1)
        feats_clock_sin = np.sin(2 * np.pi * feats_clock)
        feats_clock_cos = np.cos(2 * np.pi * feats_clock)
        feats_clock_sin2 = np.sin(4 * np.pi * feats_clock)
        feats_clock_cos2 = np.cos(4 * np.pi * feats_clock)
        feats_nbase = neighbor_base_h[m].reshape(-1, 1)
        feats_hist_trend = hist_trend[m].reshape(-1, 1)
        feats_hm1 = hist_hm1[m].reshape(-1, 1)
        node_idx = np.where(m)[1].reshape(-1, 1)
        node_sin = np.sin(2 * np.pi * node_idx / n_nodes)
        node_cos = np.cos(2 * np.pi * node_idx / n_nodes)

        X = np.hstack([
            feats_base, feats_anchor, feats_hist_last, feats_hist_avg,
            feats_clock_sin, feats_clock_cos, feats_clock_sin2, feats_clock_cos2,
            feats_nbase, feats_hist_trend, feats_hm1, node_sin, node_cos,
        ])
        y = residual[:, h, :][m]

        if len(y) > 100:
            mu = X.mean(axis=0)
            sd = X.std(axis=0) + 1e-8
            Xs = (X - mu) / sd

            # Inner CV for alpha
            rng_h = np.random.RandomState(h * 100 + seed)
            perm = rng_h.permutation(len(y))
            n_val = min(int(0.2 * len(y)), 2000)
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]

            best_alpha = alphas[0]
            best_mae = float('inf')
            for alpha in alphas:
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(Xs[tr_idx], y[tr_idx])
                pred = model.predict(Xs[val_idx])
                mae = np.abs(pred - y[val_idx]).mean()
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha

            model = Ridge(alpha=best_alpha, fit_intercept=True)
            model.fit(Xs, y)
            ridge_models.append({
                'coef': model.coef_.astype(np.float32),
                'intercept': np.float32(model.intercept_),
                'mu': mu.astype(np.float32),
                'sd': sd.astype(np.float32),
                'alpha': np.float32(best_alpha),
            })
        else:
            ridge_models.append(None)

    state = {
        'method': 'pna_ridge_combined',
        'version': 20,
        'anchor_nh': anchor_nh.astype(np.float32),
        'h_bias': h_bias.astype(np.float32),
        'node_bias': node_bias.astype(np.float32),
        'bias_nh': bias_nh.astype(np.float32),
        'clock_bias': clock_bias.astype(np.float32),
        'n_clock_bins': n_clock_bins,
        'hist_w': hist_w.astype(np.float32),
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
    clock_bias = state['clock_bias'].astype(np.float64)
    n_clock_bins = state['n_clock_bins']
    hist_w = state['hist_w'].astype(np.float64)
    ridge_shrink = float(state['ridge_shrink'])
    ridge_models = state['ridge_models']

    # Anchor blending
    result = (1 - anchor_nh[np.newaxis, :, :]) * base + anchor_nh[np.newaxis, :, :] * anchor

    # Bias corrections
    result = result - h_bias[np.newaxis, :, np.newaxis]
    result = result - node_bias[np.newaxis, np.newaxis, :]
    result = result - bias_nh[np.newaxis, :, :]

    # Clock correction
    clock_bins = np.floor(clock * n_clock_bins).astype(int) % n_clock_bins
    result = result - clock_bias[clock_bins]

    # History correction
    last_hist = history[:, -1, :]
    base_h0 = base[:, 0, :]
    hist_dev = last_hist - base_h0
    for h in range(n_horizons):
        result[:, h, :] = result[:, h, :] - hist_w[h] * hist_dev

    # Ridge correction
    graph = np.load(GRAPH_PATH)
    adj = graph['adjacency'].astype(np.float64)
    deg = np.maximum(adj.sum(axis=1), 1.0)

    hist_last = history[:, -1, :]
    hist_avg = history.mean(axis=1)
    hist_trend = (history[:, -1, :] - history[:, 0, :]) / 12.0
    hist_hm1 = history[:, -2, :]

    for h in range(n_horizons):
        if ridge_models[h] is not None:
            rm = ridge_models[h]
            coef = rm['coef'].astype(np.float64)
            intercept = float(rm['intercept'])
            mu = rm['mu'].astype(np.float64)
            sd = rm['sd'].astype(np.float64)

            neighbor_base_h = (base[:, h, :] @ adj) / deg
            feats_base = base[:, h, :].reshape(-1, 1)
            feats_anchor = anchor[:, h, :].reshape(-1, 1)
            feats_hist_last = hist_last.reshape(-1, 1)
            feats_hist_avg = hist_avg.reshape(-1, 1)
            feats_clock = np.repeat(clock[:, h, 0], n_nodes).reshape(-1, 1)
            feats_clock_sin = np.sin(2 * np.pi * feats_clock)
            feats_clock_cos = np.cos(2 * np.pi * feats_clock)
            feats_clock_sin2 = np.sin(4 * np.pi * feats_clock)
            feats_clock_cos2 = np.cos(4 * np.pi * feats_clock)
            feats_nbase = neighbor_base_h.reshape(-1, 1)
            feats_hist_trend = hist_trend.reshape(-1, 1)
            feats_hm1 = hist_hm1.reshape(-1, 1)
            node_idx = np.tile(np.arange(n_nodes), (n_samples, 1)).reshape(-1, 1)
            node_sin = np.sin(2 * np.pi * node_idx / n_nodes)
            node_cos = np.cos(2 * np.pi * node_idx / n_nodes)

            X = np.hstack([
                feats_base, feats_anchor, feats_hist_last, feats_hist_avg,
                feats_clock_sin, feats_clock_cos, feats_clock_sin2, feats_clock_cos2,
                feats_nbase, feats_hist_trend, feats_hm1, node_sin, node_cos,
            ])
            Xs = (X - mu) / sd
            correction = ridge_shrink * (Xs @ coef + intercept)
            result[:, h, :] = result[:, h, :] - correction.reshape(n_samples, n_nodes)

    result = np.maximum(result, 0.0)
    return result.astype(np.float32)
