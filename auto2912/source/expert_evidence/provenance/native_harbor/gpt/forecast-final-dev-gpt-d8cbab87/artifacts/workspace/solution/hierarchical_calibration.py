"""Pooled median tree followed by regularized sensor-specific corrections."""
import numpy as np
from threadpoolctl import threadpool_limits
import spatial_calibration


def _design(features, mode, center):
    base = np.asarray(features['base'], dtype=np.float64)
    history = np.asarray(features['history'], dtype=np.float64)
    anchor = np.asarray(features['anchor'], dtype=np.float64)
    arr = [np.ones_like(base)]
    if mode == 'affine':
        horizon = np.broadcast_to(np.linspace(-1, 1, base.shape[1])[None, :, None], base.shape)
        recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
        arr += [horizon, np.clip((anchor - base) / 50, -5, 5),
                np.clip((recent - base) / 50, -5, 5),
                (base - center) / 200]
    return np.stack(arr, axis=-1)


def _sensor_fit(x, residual, valid):
    xx = x.transpose(2, 0, 1, 3).reshape(x.shape[2], -1, x.shape[-1])
    yy = residual.transpose(2, 0, 1).reshape(x.shape[2], -1)
    mask = valid.transpose(2, 0, 1).reshape(x.shape[2], -1)
    yy = np.where(mask, yy, 0)
    coef = np.zeros((x.shape[2], x.shape[-1]))
    strength = np.array([10, 50, 60, 60, 60], dtype=np.float64)[:x.shape[-1]]
    for _ in range(12):
        error = yy - np.einsum('nrp,np->nr', xx, coef)
        weights = mask / np.maximum(np.abs(error), 3)
        gram = np.einsum('nrp,nr,nrq->npq', xx, weights, xx, optimize=True)
        rhs = np.einsum('nrp,nr,nr->np', xx, weights, yy, optimize=True)
        coef = np.linalg.solve(gram + np.diag(strength), rhs[..., None])[..., 0]
    return coef.astype(np.float32)


def fit(features, targets, seed, mode='offset', crossfit=False):
    with threadpool_limits(limits=2):
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        tree = spatial_calibration.fit(features, y, seed)
        if crossfit:
            prediction = np.zeros_like(y)
            for held in np.array_split(np.arange(y.shape[0]), 3):
                train = np.setdiff1d(np.arange(y.shape[0]), held)
                fold = spatial_calibration.fit({k: v[train] for k, v in features.items()}, y[train], seed)
                prediction[held] = spatial_calibration.predict(fold, {k: v[held] for k, v in features.items()})
        else:
            prediction = spatial_calibration.predict(tree, features)
        valid = np.isfinite(y) & (y > 0)
        center = np.mean(features['base'], axis=(0, 1), keepdims=True)
        x = _design(features, mode, center)
        coef = _sensor_fit(x, y - prediction, valid)
        return {'tree': tree, 'coef': coef, 'mode': mode, 'center': center}


def predict(state, features):
    with threadpool_limits(limits=2):
        baseline = spatial_calibration.predict(state['tree'], features)
        correction = np.einsum('shnp,np->shn', _design(features, state['mode'], state['center']), state['coef'])
        return np.maximum(baseline + correction, 0).astype(np.float32)
