"""Median residual trees with neighborhoods learned from calibration history."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import tree_calibration


def _neighborhood(features, count=8):
    history = np.asarray(features['history'], dtype=np.float64)
    changes = np.diff(history, axis=1).reshape(-1, history.shape[2])
    changes -= np.mean(changes, axis=0)
    scale = np.maximum(np.std(history.reshape(-1, history.shape[2]), axis=0), 10)
    norm = np.sqrt(np.sum(changes * changes, axis=0))
    correlation = (changes.T @ changes) / np.maximum(norm[:, None] * norm[None, :], 1)
    np.fill_diagonal(correlation, -1)
    neighbors = np.argsort(correlation, axis=1)[:, -count:]
    weights = np.maximum(np.take_along_axis(correlation, neighbors, axis=1), 0) ** 2
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return {'neighbors': neighbors, 'weights': weights.astype(np.float32),
            'scale': scale.astype(np.float32)}


def _prediction_neighborhood(features, prediction, count=8, mode='pooled'):
    """Learn edges from forecast innovations, with clock/horizon effects removed."""
    shape = features['base'].shape
    clock = np.asarray(features['clock'], dtype=np.float64)
    horizon = np.broadcast_to(np.linspace(-1, 1, shape[1])[None, :, None], shape)
    design = np.stack([np.ones(shape), horizon, np.sin(2 * np.pi * clock),
                       np.cos(2 * np.pi * clock), np.sin(4 * np.pi * clock),
                       np.cos(4 * np.pi * clock)], axis=-1)
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(features['base'])
    gram = np.einsum('shni,shnj->nij', design, design, optimize=True)
    rhs = np.einsum('shni,shn->ni', design, residual, optimize=True)
    ridge = np.diag([0, 1, 1, 1, 1, 1]) * shape[0] * shape[1] * 0.05
    coef = np.linalg.solve(gram + ridge, rhs[..., None])[..., 0]
    seasonal = np.einsum('shni,ni->shn', design, coef)
    innovation = residual - seasonal
    flat = innovation.reshape(-1, shape[2])
    flat -= np.mean(flat, axis=0)
    scale = np.maximum(np.std(flat, axis=0), 10.0)
    normalized = flat / scale
    norm = np.sqrt(np.sum(normalized * normalized, axis=0))
    correlation = (normalized.T @ normalized) / np.maximum(norm[:, None] * norm[None, :], 1)
    confidence = None
    if mode in ('median', 'median_confidence', 'confidence_fallback'):
        # Each OOF block uses a different fitted expert. Require edge support
        # in a typical block instead of pooling those experts' offset shifts.
        block_correlations = []
        for block in np.array_split(np.arange(shape[0]), 3):
            values = innovation[block].reshape(-1, shape[2]).copy()
            values -= values.mean(axis=0)
            block_norm = np.sqrt(np.sum(values * values, axis=0))
            block_correlations.append(
                (values.T @ values) /
                np.maximum(block_norm[:, None] * block_norm[None, :], 1))
        block_correlations = np.asarray(block_correlations)
        median_correlation = np.median(block_correlations, axis=0)
        mean_correlation = np.mean(block_correlations, axis=0)
        if mode == 'confidence_fallback':
            stable_correlation = mean_correlation - np.std(block_correlations, axis=0)
        else:
            stable_correlation = median_correlation
        correlation = stable_correlation
    np.fill_diagonal(correlation, -1)
    neighbors = np.argsort(correlation, axis=1)[:, -count:]
    weights = np.maximum(np.take_along_axis(correlation, neighbors, axis=1), 0) ** 2
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    result = {'neighbors': neighbors, 'weights': weights.astype(np.float32),
              'scale': scale.astype(np.float32),
              'source': ('prediction_confidence_fallback' if mode == 'confidence_fallback'
                         else 'prediction_median_confidence' if mode == 'median_confidence'
                         else 'prediction_median' if mode == 'median' else 'prediction')}
    if mode in ('median_confidence', 'confidence_fallback'):
        evidence = np.maximum(np.take_along_axis(stable_correlation, neighbors, axis=1), 0)
        support = np.maximum(np.take_along_axis(mean_correlation, neighbors, axis=1), 0)
        confidence = np.clip((evidence ** 2).sum(axis=1) /
                             np.maximum((support ** 2).sum(axis=1), 1e-8),
                             0, 1).astype(np.float32)
    if confidence is not None:
        result['confidence'] = confidence
    return result


def _features(features, graph, variant='neighbors'):
    x = tree_calibration._features(features)
    base = np.asarray(features['base'], dtype=np.float32)
    anchor = np.asarray(features['anchor'], dtype=np.float32)
    history = np.asarray(features['history'], dtype=np.float32)
    scale = graph['scale']
    def aggregate(value):
        normalized = value / scale
        mean = np.sum(normalized[..., graph['neighbors']] * graph['weights'], axis=-1)
        return mean * scale
    recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
    older = np.mean(history[:, :3, :], axis=1)[:, None, :]
    last = history[:, -1:, :]
    spread = np.std(history, axis=1)[:, None, :]
    extra = [aggregate(anchor - base), aggregate(recent - base),
             aggregate(last - base), np.broadcast_to(aggregate(recent - older), base.shape),
             np.broadcast_to(aggregate(spread), base.shape)]
    if variant == 'relative':
        extra.extend([(anchor - base) / scale, (recent - base) / scale,
                      np.broadcast_to(spread / scale, base.shape),
                      base / np.maximum(anchor, 30),
                      aggregate(anchor - base) - (anchor - base)])
    if 'physical' in graph:
        def physical(value):
            return ((value / scale) @ graph['physical'].T) * scale
        extra.extend([physical(anchor - base), physical(recent - base),
                      physical(last - base),
                      np.broadcast_to(physical(recent - older), base.shape),
                      np.broadcast_to(physical(spread), base.shape)])
    return np.column_stack([x] + [v.reshape(-1) for v in extra])


def fit(features, targets, seed, variant='neighbors', max_iter=90, graph=None):
    with threadpool_limits(limits=1):
        if graph is None:
            graph = _neighborhood(features)
        x = _features(features, graph, variant)
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        valid = np.isfinite(y) & (y > 0)
        residual = y - np.asarray(features['base'])
        model = HistGradientBoostingRegressor(
            loss='absolute_error', max_iter=max_iter, learning_rate=0.06,
            max_leaf_nodes=15, min_samples_leaf=250, l2_regularization=20,
            categorical_features=[14], early_stopping=False, random_state=seed)
        model.fit(x[valid.ravel()], residual[valid])
        return {'graph': graph, 'variant': variant, 'model': model}


def predict(state, features):
    with threadpool_limits(limits=1):
        x = _features(features, state['graph'], state['variant'])
        residual = state['model'].predict(x).reshape(features['base'].shape)
        return np.maximum(np.asarray(features['base']) + residual, 0).astype(np.float32)


def diffusion_fit(features, prediction, robust=False, graph_mode='history'):
    shape = features['base'].shape
    clock = np.asarray(features['clock'], dtype=np.float64)
    horizon = np.broadcast_to(np.linspace(-1, 1, shape[1])[None, :, None], shape)
    x = np.stack([np.ones(shape), horizon, np.sin(2 * np.pi * clock),
                  np.cos(2 * np.pi * clock), np.sin(4 * np.pi * clock),
                  np.cos(4 * np.pi * clock)], axis=-1)
    residual = prediction - features['base']
    gram = np.einsum('shni,shnj->nij', x, x, optimize=True)
    rhs = np.einsum('shni,shn->ni', x, residual, optimize=True)
    ridge = np.diag([0, 1, 1, 1, 1, 1]) * shape[0] * shape[1] * 0.05
    coef = np.linalg.solve(gram + ridge, rhs[..., None])[..., 0]
    prediction_mode = {'prediction': 'pooled',
                       'prediction_median': 'median',
                       'prediction_median_confidence': 'median_confidence',
                       'prediction_confidence_fallback': 'confidence_fallback'}.get(graph_mode)
    graph = (_prediction_neighborhood(features, prediction, mode=prediction_mode)
             if prediction_mode is not None else _neighborhood(features))
    state = {'graph': graph, 'coef': coef.astype(np.float32), 'graph_mode': graph_mode}
    if graph_mode in ('prediction_median_confidence', 'prediction_confidence_fallback'):
        state['confidence'] = graph.pop('confidence')
        state['fallback_graph'] = _neighborhood(features)
    if robust:
        seasonal = np.einsum('shni,ni->shn', x, state['coef'])
        normalized = (prediction - features['base'] - seasonal) / state['graph']['scale']
        difference = (normalized[..., state['graph']['neighbors']]
                      - normalized[..., None])
        # A forecast-only scale for each sensor bounds the influence of large
        # disagreements without fitting any additional target-dependent rule.
        state['threshold'] = np.maximum(
            3 * np.median(np.abs(difference), axis=(0, 1, 3)), 0.05)
    return state


def diffusion_predict(state, features, prediction):
    if 'fallback_graph' in state:
        primary = dict(state)
        primary.pop('fallback_graph')
        primary.pop('confidence')
        fallback = dict(primary)
        fallback['graph'] = state['fallback_graph']
        primary_value = diffusion_predict(primary, features, prediction)
        fallback_value = diffusion_predict(fallback, features, prediction)
        confidence = state['confidence'][None, None, :]
        return np.maximum(confidence * primary_value +
                          (1 - confidence) * fallback_value, 0)
    shape = features['base'].shape
    clock = np.asarray(features['clock'], dtype=np.float64)
    horizon = np.broadcast_to(np.linspace(-1, 1, shape[1])[None, :, None], shape)
    x = np.stack([np.ones(shape), horizon, np.sin(2 * np.pi * clock),
                  np.cos(2 * np.pi * clock), np.sin(4 * np.pi * clock),
                  np.cos(4 * np.pi * clock)], axis=-1)
    seasonal = np.einsum('shni,ni->shn', x, state['coef'])
    innovation = prediction - features['base'] - seasonal
    graph = state['graph']
    normalized = innovation / graph['scale']
    pooling = state.get('pooling', 'static')
    if pooling == 'static':
        neighbor = np.sum(normalized[..., graph['neighbors']] * graph['weights'], axis=-1)
        neighbor *= graph['scale']
        return np.maximum(features['base'] + seasonal + 0.75 * innovation + 0.25 * neighbor, 0)
    # A fixed history graph can connect sensors experiencing different current
    # deviations. Share changing corrections preferentially along edges whose
    # known anchor/base and recent-history/base disagreements currently match.
    scale = graph['scale']
    neighbors = graph['neighbors']
    anchor = (np.asarray(features['anchor']) - features['base']) / scale
    recent = (np.mean(features['history'][:, -3:, :], axis=1)[:, None, :]
              - features['base']) / scale
    distance = ((anchor[..., neighbors] - anchor[..., None]) ** 2
                + (recent[..., neighbors] - recent[..., None]) ** 2) / 2
    affinity = 1 / (1 + distance)
    weights = graph['weights'] * affinity
    if pooling == 'context_normalize':
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
    difference = normalized[..., neighbors] - normalized[..., None]
    if 'threshold' in state:
        threshold = state['threshold'][:, None]
        difference = threshold * np.tanh(difference / threshold)
    shared = normalized + 0.25 * np.sum(weights * difference, axis=-1)
    return np.maximum(features['base'] + seasonal + shared * scale, 0)
