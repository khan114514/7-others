"""Robust hierarchical residual calibration of fixed traffic forecasts."""
import numpy as np
from threadpoolctl import threadpool_limits


def _design(features):
    b = np.asarray(features['base'], dtype=np.float64)
    a = np.asarray(features['anchor'], dtype=np.float64)
    hist = np.asarray(features['history'], dtype=np.float64)
    clock = np.asarray(features['clock'], dtype=np.float64)
    hh = np.broadcast_to(np.linspace(-1.0, 1.0, b.shape[1])[None, :, None], b.shape)
    recent = np.broadcast_to(np.median(hist[:, -3:, :], axis=1)[:, None, :], b.shape)
    older = np.broadcast_to(np.median(hist[:, :6, :], axis=1)[:, None, :], b.shape)
    allmean = np.broadcast_to(np.mean(hist, axis=1)[:, None, :], b.shape)
    sine = np.sin(2 * np.pi * clock)
    cosine = np.cos(2 * np.pi * clock)
    anchor = np.clip((a - b) / 50, -5, 5)
    level = b / 300
    recent_delta = np.clip((recent - b) / 50, -5, 5)
    trend = np.clip((recent - older) / 50, -5, 5)
    arr = [np.ones_like(b), hh, hh * hh, sine, cosine,
           np.sin(4 * np.pi * clock), np.cos(4 * np.pi * clock),
           level, anchor, anchor * hh, recent_delta, recent_delta * hh,
           np.clip((allmean - b) / 50, -5, 5), trend, trend * hh,
           sine * hh, cosine * hh, sine * level, cosine * level]
    return np.stack(arr, axis=-1)


def _global_fit(x, y, valid, iterations=8):
    xx = x.reshape(-1, x.shape[-1])
    yy = y.reshape(-1)
    ok = valid.reshape(-1)
    xx = xx[ok]
    yy = yy[ok]
    coef = np.zeros(xx.shape[1])
    penalty = np.eye(xx.shape[1]) * 2.0
    for _ in range(iterations):
        residual = yy - xx @ coef
        w = 1.0 / np.maximum(np.abs(residual), 4.0)
        coef = np.linalg.solve(xx.T @ (w[:, None] * xx) + penalty,
                               xx.T @ (w * yy))
    return coef


def fit(features, targets, seed):
    with threadpool_limits(limits=2):
        x = _design(features)
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets,
                       dtype=np.float64)
        base = np.asarray(features['base'], dtype=np.float64)
        valid = np.isfinite(y) & (y > 0)
        y = np.where(valid, y - base, 0)
        global_coef = _global_fit(x, y, valid)
        # Every sensor gets a regularized deviation from the pooled model.
        xx = x.transpose(2, 0, 1, 3).reshape(x.shape[2], -1, x.shape[-1])
        yy = y.transpose(2, 0, 1).reshape(x.shape[2], -1)
        mask = valid.transpose(2, 0, 1).reshape(x.shape[2], -1)
        prior = np.broadcast_to(global_coef, (x.shape[2], x.shape[-1])).copy()
        coef = prior.copy()
        strength = np.array([3, 20, 40, 15, 15, 30, 30,
                             60, 40, 80, 60, 100, 60, 80, 100,
                             50, 50, 60, 60], dtype=np.float64)
        reg = np.diag(strength)
        for _ in range(8):
            residual = yy - np.einsum('nrp,np->nr', xx, coef)
            w = mask / np.maximum(np.abs(residual), 4.0)
            gram = np.einsum('nrp,nr,nrq->npq', xx, w, xx, optimize=True)
            rhs = np.einsum('nrp,nr,nr->np', xx, w, yy, optimize=True)
            coef = np.linalg.solve(gram + reg, (rhs + prior * strength)[..., None])[..., 0]
        return {'version': 2, 'coef': coef.astype(np.float32)}


def predict(state, features):
    with threadpool_limits(limits=2):
        x = _design(features)
        correction = np.einsum('shnp,np->shn', x, np.asarray(state['coef']))
        return np.maximum(np.asarray(features['base']) + correction, 0).astype(np.float32)
