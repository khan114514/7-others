"""Origin bootstrap regularization of the pooled residual-tree component."""
import numpy as np
from threadpoolctl import threadpool_limits
import blended_calibration
import horizon_calibration
import spatial_calibration
import neural_calibration


def fit(features, targets, seed, network=True, copies=3):
    with threadpool_limits(limits=2):
        state = (blended_calibration.fit(features, targets, seed) if network else
                 {'tree': horizon_calibration.fit(features, targets, seed, groups=2)})
        tree = state['tree']
        x = spatial_calibration._features(features, tree['pooled']['graph'])
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        valid = np.isfinite(y) & (y > 0)
        residual = y - np.asarray(features['base'])
        rng = np.random.default_rng(seed)
        models = []
        for _ in range(copies):
            counts = rng.poisson(1, size=y.shape[0]).astype(np.float32)
            weights = np.broadcast_to(counts[:, None, None], y.shape)[valid]
            weights = weights / np.mean(weights)
            model = horizon_calibration._model(seed)
            model.fit(x[valid.ravel()], residual[valid], sample_weight=weights)
            models.append(model)
        state['bags'] = models
        state['bag_weight'] = 0.5
        return state


def predict(state, features):
    with threadpool_limits(limits=2):
        tree = state['tree']
        shape = features['base'].shape
        x = spatial_calibration._features(features, tree['pooled']['graph'])
        pooled = tree['pooled']['model'].predict(x).reshape(shape)
        bag = np.mean([model.predict(x).reshape(shape) for model in state['bags']], axis=0)
        pooled = pooled * (1 - state['bag_weight']) + bag * state['bag_weight']
        x = x.reshape(*shape, -1)
        expert = np.zeros(shape, dtype=np.float32)
        for horizons, model in tree['models']:
            xx = x[:, horizons].reshape(-1, x.shape[-1])
            expert[:, horizons] = model.predict(xx).reshape(shape[0], len(horizons), shape[2])
        correction = pooled * (1 - tree['expert_weight']) + expert * tree['expert_weight']
        forecast = np.maximum(np.asarray(features['base']) + correction, 0)
        if 'network' in state:
            forecast = 0.75 * forecast + 0.25 * neural_calibration.predict(state['network'], features)
        return forecast.astype(np.float32)
