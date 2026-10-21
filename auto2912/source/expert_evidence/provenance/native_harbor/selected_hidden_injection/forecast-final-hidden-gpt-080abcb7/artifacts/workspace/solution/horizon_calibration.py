"""Partial pooling of median residual trees specialized by forecast horizon."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import spatial_calibration


def _model(seed):
    return HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=90, learning_rate=0.06,
        max_leaf_nodes=15, min_samples_leaf=250, l2_regularization=20,
        categorical_features=[14], early_stopping=False, random_state=seed)


def fit(features, targets, seed, groups=3, graph=None):
    with threadpool_limits(limits=1):
        pooled = spatial_calibration.fit(features, targets, seed, graph=graph)
        x = spatial_calibration._features(features, pooled['graph'])
        shape = features['base'].shape
        x = x.reshape(*shape, -1)
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        valid = np.isfinite(y) & (y > 0)
        residual = y - np.asarray(features['base'])
        models = []
        for horizons in np.array_split(np.arange(shape[1]), groups):
            xx = x[:, horizons].reshape(-1, x.shape[-1])
            ok = valid[:, horizons].reshape(-1)
            yy = residual[:, horizons].reshape(-1)
            model = _model(seed)
            model.fit(xx[ok], yy[ok])
            models.append((horizons, model))
        return {'pooled': pooled, 'models': models, 'expert_weight': 0.5}


def predict(state, features):
    with threadpool_limits(limits=1):
        shape = features['base'].shape
        x = spatial_calibration._features(features, state['pooled']['graph']).reshape(*shape, -1)
        expert = np.zeros(shape, dtype=np.float32)
        for horizons, model in state['models']:
            expert[:, horizons] = model.predict(x[:, horizons].reshape(-1, x.shape[-1])).reshape(shape[0], len(horizons), shape[2])
        pooled = state['pooled']['model'].predict(x.reshape(-1, x.shape[-1])).reshape(shape)
        weight = state['expert_weight']
        correction = pooled * (1 - weight) + expert * weight
        return np.maximum(np.asarray(features['base']) + correction, 0).astype(np.float32)
