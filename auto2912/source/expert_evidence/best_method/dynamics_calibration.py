"""Residual trees with explicit history dynamics and sensor-level scaling."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import tree_calibration
import spatial_calibration


def _profile(features):
    history = np.asarray(features['history'], dtype=np.float32)
    flat = history.reshape(-1, history.shape[-1])
    return {'mean': np.maximum(np.mean(flat, axis=0), 10),
            'std': np.maximum(np.std(flat, axis=0), 10),
            'median': np.median(flat, axis=0),
            'quartiles': np.percentile(flat, [10, 90], axis=0).astype(np.float32)}


def _features(f, state):
    if state.get('neighbors'):
        x = spatial_calibration._features(f, state['graph'])
    else:
        x = tree_calibration._features(f)
    base = np.asarray(f['base'], dtype=np.float32)
    anchor = np.asarray(f['anchor'], dtype=np.float32)
    history = np.asarray(f['history'], dtype=np.float32)
    def broadcast(v):
        return np.broadcast_to(v, base.shape)
    recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
    extra = []
    if state['variant'] in ('lags', 'both', 'robust'):
        extra.extend([broadcast(history[:, j:j+1, :]) - base for j in (0, 3, 6, 8, 10)])
        extra.extend([broadcast(base[:, :1, :]) - base,
                      broadcast(base[:, -1:, :]) - base,
                      broadcast(np.mean(base, axis=1, keepdims=True)) - base,
                      broadcast(anchor[:, :1, :]) - anchor,
                      broadcast(anchor[:, -1:, :]) - anchor,
                      broadcast(np.median(history[:, -3:, :], axis=1)[:, None, :]) - base,
                      broadcast(np.median(history, axis=1)[:, None, :]) - base])
    if state['variant'] in ('profile', 'both', 'robust'):
        profile = state['profile']
        mean, std = profile['mean'], profile['std']
        extra.extend([broadcast(mean), broadcast(std),
                      base / mean, anchor / mean,
                      (base - profile['median']) / std,
                      (anchor - base) / std,
                      (recent - base) / std])
    if state['variant'] == 'robust':
        spread = np.percentile(history, 75, axis=1) - np.percentile(history, 25, axis=1)
        extra.extend([broadcast(spread[:, None, :]),
                      broadcast(np.mean(history == 0, axis=1)[:, None, :]),
                      (anchor - base) / np.maximum(anchor, 30),
                      np.sin(2 * np.pi * f['clock']), np.cos(2 * np.pi * f['clock'])])
    return np.column_stack([x] + [v.reshape(-1) for v in extra])


def fit(features, targets, seed, variant='both', neighbors=False, max_iter=90):
    with threadpool_limits(limits=2):
        state = {'variant': variant, 'profile': _profile(features), 'neighbors': neighbors}
        if neighbors:
            state['graph'] = spatial_calibration._neighborhood(features)
        x = _features(features, state)
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        valid = np.isfinite(y) & (y > 0)
        residual = y - np.asarray(features['base'])
        model = HistGradientBoostingRegressor(
            loss='absolute_error', max_iter=max_iter, learning_rate=0.06,
            max_leaf_nodes=15, min_samples_leaf=250, l2_regularization=20,
            categorical_features=[14], early_stopping=False, random_state=seed)
        model.fit(x[valid.ravel()], residual[valid])
        state['model'] = model
        return state


def predict(state, features):
    with threadpool_limits(limits=2):
        residual = state['model'].predict(_features(features, state)).reshape(features['base'].shape)
        return np.maximum(np.asarray(features['base']) + residual, 0).astype(np.float32)
