"""Heteroscedastic residual trees with weights preserving the raw MAE objective."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import spatial_calibration


def _scale(features, state):
    base = np.asarray(features['base'], dtype=np.float32)
    if state['scaling'] == 'poisson':
        return np.sqrt(np.maximum(base, 20))
    if state['scaling'] == 'flow':
        return np.maximum(base, 30)
    return np.broadcast_to(state['node_scale'], base.shape)


def fit(features, targets, seed, scaling='node', max_iter=90):
    with threadpool_limits(limits=2):
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        valid = np.isfinite(y) & (y > 0)
        residual = y - np.asarray(features['base'])
        masked = np.where(valid, np.abs(residual), np.nan)
        node_scale = np.maximum(np.nanmedian(masked, axis=(0, 1)), 2).astype(np.float32)
        node_scale = np.nan_to_num(node_scale, nan=10)
        state = {'scaling': scaling, 'node_scale': node_scale,
                 'graph': spatial_calibration._neighborhood(features)}
        scale = _scale(features, state)
        x = spatial_calibration._features(features, state['graph'])
        x = np.column_stack([x, scale.reshape(-1)])
        model = HistGradientBoostingRegressor(
            loss='absolute_error', max_iter=max_iter, learning_rate=0.06,
            max_leaf_nodes=15, min_samples_leaf=250, l2_regularization=20,
            categorical_features=[14], early_stopping=False, random_state=seed)
        weights = scale[valid]
        weights = weights / np.mean(weights)
        model.fit(x[valid.ravel()], (residual / scale)[valid], sample_weight=weights)
        state['model'] = model
        return state


def predict(state, features):
    with threadpool_limits(limits=2):
        scale = _scale(features, state)
        x = spatial_calibration._features(features, state['graph'])
        x = np.column_stack([x, scale.reshape(-1)])
        residual = state['model'].predict(x).reshape(features['base'].shape) * scale
        return np.maximum(np.asarray(features['base']) + residual, 0).astype(np.float32)
