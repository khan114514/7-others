"""Cross-fitted tree forecasts with a smooth residual correction network."""
import numpy as np
from threadpoolctl import threadpool_limits
import horizon_calibration
import neural_calibration


def _with_base(features, base):
    return {**features, 'base': base}


def fit(features, targets, seed):
    with threadpool_limits(limits=2):
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        out_of_fold = np.zeros_like(y)
        index = np.arange(len(y))
        for held in np.array_split(index, 3):
            train = np.setdiff1d(index, held)
            fold = horizon_calibration.fit({k: v[train] for k, v in features.items()}, y[train], seed, groups=2)
            out_of_fold[held] = horizon_calibration.predict(fold, {k: v[held] for k, v in features.items()})
        network = neural_calibration.fit(_with_base(features, out_of_fold), y, seed)
        tree = horizon_calibration.fit(features, y, seed, groups=2)
        return {'tree': tree, 'network': network, 'correction_weight': 0.25}


def predict(state, features):
    with threadpool_limits(limits=2):
        tree = horizon_calibration.predict(state['tree'], features)
        corrected = neural_calibration.predict(state['network'], _with_base(features, tree))
        weight = state['correction_weight']
        return (tree * (1 - weight) + corrected * weight).astype(np.float32)
