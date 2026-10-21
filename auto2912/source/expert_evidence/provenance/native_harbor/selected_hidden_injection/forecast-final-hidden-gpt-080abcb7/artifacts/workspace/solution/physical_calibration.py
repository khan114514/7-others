"""Learned and public physical-neighbor context for residual calibration."""
import numpy as np
from threadpoolctl import threadpool_limits
import spatial_calibration
import horizon_calibration
import blended_calibration


def _graph(features, mode='augment'):
    graph = spatial_calibration._neighborhood(features)
    adjacency = np.asarray(np.load('/workspace/public_assets/graph.npz')['adjacency'], dtype=np.float32)
    weights = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-8)
    if mode == 'augment':
        graph['physical'] = weights
    else:
        neighbors = np.argsort(weights, axis=1)[:, -9:]
        graph['neighbors'] = neighbors
        graph['weights'] = np.take_along_axis(weights, neighbors, axis=1)
    return graph


def fit(features, targets, seed, mode='augment', network=True):
    with threadpool_limits(limits=2):
        graph = _graph(features, mode)
        if network:
            return {'blend': blended_calibration.fit(features, targets, seed, graph=graph)}
        return {'tree': horizon_calibration.fit(features, targets, seed, groups=2, graph=graph)}


def predict(state, features):
    if 'blend' in state:
        return blended_calibration.predict(state['blend'], features)
    return horizon_calibration.predict(state['tree'], features)
