"""Median residual calibration using anchor-aligned historical innovations."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import spatial_calibration


def _basis(clock, harmonics=8):
    arr = [np.ones_like(clock)]
    for k in range(1, harmonics + 1):
        arr.extend([np.sin(2 * np.pi * k * clock), np.cos(2 * np.pi * k * clock)])
    return np.stack(arr, axis=-1)


def _template(features):
    clock = np.asarray(features['clock'], dtype=np.float64)[:, :, 0].reshape(-1)
    anchor = np.asarray(features['anchor'], dtype=np.float64).reshape(-1, features['base'].shape[-1])
    x = _basis(clock)
    penalty = np.diag(np.r_[0.1, np.repeat(np.arange(1, 9) ** 2, 2) * 2])
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ anchor)
    return coef.astype(np.float32)


def _features(features, state):
    x = spatial_calibration._features(features, state['graph'])
    base = np.asarray(features['base'], dtype=np.float32)
    anchor = np.asarray(features['anchor'], dtype=np.float32)
    history = np.asarray(features['history'], dtype=np.float32)
    future_clock = np.asarray(features['clock'])[:, :, 0]
    history_clock = (future_clock[:, :1] - np.arange(12, 0, -1)[None, :] / 288) % 1
    template_future = _basis(future_clock) @ state['template']
    template_history = _basis(history_clock) @ state['template']
    def broadcast(v):
        return np.broadcast_to(v[:, None, :], base.shape)
    aligned_history = template_history + np.mean(anchor - template_future, axis=1)[:, None, :]
    innovation = history - aligned_history
    last = innovation[:, -1, :]
    recent = np.mean(innovation[:, -3:, :], axis=1)
    older = np.mean(innovation[:, :6, :], axis=1)
    slope = np.mean(innovation[:, -3:, :], axis=1) - np.mean(innovation[:, :3, :], axis=1)
    extra = [broadcast(last), broadcast(recent), broadcast(older), broadcast(slope),
             broadcast(np.std(innovation, axis=1)), anchor - template_future,
             template_future - base]
    if state['variant'] == 'neighbors':
        graph = state['graph']
        def aggregate(v):
            normalized = v / graph['scale']
            return np.sum(normalized[..., graph['neighbors']] * graph['weights'], axis=-1) * graph['scale']
        extra += [broadcast(aggregate(last)), broadcast(aggregate(recent)), broadcast(aggregate(slope))]
    if state['variant'] == 'extrapolated':
        slope_anchor = np.mean(np.diff(anchor[:, :4, :], axis=1), axis=1)
        expected_last = anchor[:, 0, :] - slope_anchor
        extra += [broadcast(history[:, -1, :] - expected_last),
                  broadcast(np.mean(history[:, -3:, :], axis=1) - (expected_last - slope_anchor)),
                  broadcast(slope_anchor)]
    return np.column_stack([x] + [v.reshape(-1) for v in extra])


def fit(features, targets, seed, variant='local', max_iter=90):
    with threadpool_limits(limits=2):
        state = {'template': _template(features),
                 'graph': spatial_calibration._neighborhood(features), 'variant': variant}
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
