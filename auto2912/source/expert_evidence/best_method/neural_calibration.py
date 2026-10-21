"""Small NumPy residual network trained with a smooth absolute-error objective."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
import spatial_calibration


def _features(features, graph, mode='summary'):
    raw = spatial_calibration._features(features, graph)
    node = raw[:, 14].astype(np.int64)
    clock = np.asarray(features['clock']).reshape(-1)
    periodic = [np.sin(2 * np.pi * clock), np.cos(2 * np.pi * clock),
                np.sin(4 * np.pi * clock), np.cos(4 * np.pi * clock)]
    columns = [np.delete(raw, 14, axis=1)] + periodic
    if mode == 'network_context':
        base = np.asarray(features['base'], dtype=np.float32)
        anchor = np.asarray(features['anchor'], dtype=np.float32)
        history = np.asarray(features['history'], dtype=np.float32)
        recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
        scale = np.maximum(np.mean(history, axis=1)[:, None, :], 30)
        anchor_difference = np.clip((anchor - base) / scale, -2, 2)
        recent_difference = np.clip((recent - base) / scale, -2, 2)
        summaries = [np.mean(np.clip(base / scale, 0, 3), axis=-1, keepdims=True),
                     np.mean(anchor_difference, axis=-1, keepdims=True),
                     np.mean(recent_difference, axis=-1, keepdims=True),
                     np.std(anchor_difference, axis=-1, keepdims=True)]
        columns += [np.broadcast_to(value, base.shape).reshape(-1) for value in summaries]
    if mode in ('history', 'relative'):
        base = np.asarray(features['base'], dtype=np.float32)
        history = np.asarray(features['history'], dtype=np.float32)
        recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
        # The shared network can learn a temporal filter rather than relying
        # only on fixed three/six-step averages and one trend statistic.
        columns += [np.broadcast_to(history[:, j:j+1, :] - recent, base.shape).reshape(-1)
                    for j in range(history.shape[1])]
        columns += [(base[:, :1, :] - base).reshape(-1),
                    (base[:, -1:, :] - base).reshape(-1)]
        if mode == 'relative':
            anchor = np.asarray(features['anchor'], dtype=np.float32)
            scale = graph['scale']
            columns += [np.broadcast_to(scale, base.shape).reshape(-1),
                        (base / scale).reshape(-1),
                        ((anchor - base) / scale).reshape(-1),
                        ((recent - base) / scale).reshape(-1)]
    return np.column_stack(columns), node


def _head_features(features, mode=True):
    shape = features['base'].shape
    clock = np.asarray(features['clock'], dtype=np.float32).reshape(-1)
    horizon = np.broadcast_to(np.linspace(-1, 1, shape[1])[None, :, None], shape).reshape(-1)
    if mode == 'trend':
        return np.column_stack([np.ones_like(clock), horizon]).astype(np.float32)
    columns = [np.ones_like(clock), horizon,
               np.sin(2 * np.pi * clock), np.cos(2 * np.pi * clock),
               np.sin(4 * np.pi * clock), np.cos(4 * np.pi * clock)]
    if mode == 'innovation':
        base = np.asarray(features['base'], dtype=np.float32)
        anchor = np.asarray(features['anchor'], dtype=np.float32)
        history = np.asarray(features['history'], dtype=np.float32)
        scale = np.maximum(np.mean(history, axis=1)[:, None, :], 30)
        recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
        last = history[:, -1:, :]
        columns.extend([np.clip((anchor - base) / scale, -1.5, 1.5).reshape(-1),
                        np.clip((recent - base) / scale, -1.5, 1.5).reshape(-1),
                        np.broadcast_to(np.clip((last - recent) / scale, -1, 1), shape).reshape(-1)])
    return np.column_stack(columns).astype(np.float32)


def _forward(x, node, params, head_x=None):
    xx = np.column_stack([x, params['embedding'][node]])
    h1 = np.tanh(xx @ params['w1'] + params['b1'])
    h2 = np.tanh(h1 @ params['w2'] + params['b2'])
    output = (h2 @ params['w3'] + x @ params['w0'] + params['b3']).reshape(-1)
    if 'site_head' in params:
        output += np.sum(head_x * params['site_head'][node], axis=1)
    return output, (xx, h1, h2)


def _node_gradient(node, values, nodes):
    return np.column_stack([
        np.bincount(node, weights=values[:, j], minlength=nodes)
        for j in range(values.shape[1])]).astype(np.float32)


def fit(features, targets, seed, epochs=100, graph=None, mode='summary', site_head=False, average=False,
        graph_strength=0.005, embedding_strength=0, replicas=1):
    if replicas > 1:
        members = [fit(features, targets, seed + 1009 * j,
                       epochs[j] if isinstance(epochs, (tuple, list)) else epochs, graph,
                       mode[j] if isinstance(mode, (tuple, list)) else mode,
                       site_head, average, graph_strength, embedding_strength)
                   for j in range(replicas)]
        return {'members': members}
    with threadpool_limits(limits=1):
        start = time.monotonic()
        rng = np.random.default_rng(seed)
        if graph is None:
            graph = spatial_calibration._neighborhood(features)
        x, node = _features(features, graph, mode)
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        valid = (np.isfinite(y) & (y > 0)).reshape(-1)
        residual = (y - np.asarray(features['base'])).reshape(-1)[valid]
        if mode == 'scaled_summary':
            normalizer = graph['scale'][node[valid]]
            residual = residual / normalizer
        else:
            residual = residual / 20
        x, node = x[valid], node[valid]
        head_x = _head_features(features, site_head)[valid] if site_head else None
        laplacian = None
        if site_head in ('graph', 'graph_horizon', 'innovation') or embedding_strength:
            adjacency = np.zeros((features['base'].shape[-1], features['base'].shape[-1]), dtype=np.float32)
            np.put_along_axis(adjacency, graph['neighbors'], graph['weights'], axis=1)
            adjacency = (adjacency + adjacency.T) / 2
            laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        center = np.mean(x, axis=0)
        scale = np.maximum(np.std(x, axis=0), 0.01)
        x = np.clip((x - center) / scale, -8, 8).astype(np.float32)
        dim = x.shape[1] + 6
        params = {'embedding': (rng.normal(size=(features['base'].shape[-1], 6)) * 0.05).astype(np.float32),
                  'w1': (rng.normal(size=(dim, 48)) / np.sqrt(dim)).astype(np.float32),
                  'b1': np.zeros(48, dtype=np.float32),
                  'w2': (rng.normal(size=(48, 24)) / np.sqrt(48)).astype(np.float32),
                  'b2': np.zeros(24, dtype=np.float32),
                  'w0': np.zeros((x.shape[1], 1), dtype=np.float32),
                  'w3': np.zeros((24, 1), dtype=np.float32),
                  'b3': np.array([np.median(residual)], dtype=np.float32)}
        if site_head:
            params['site_head'] = np.zeros((features['base'].shape[-1], head_x.shape[1]), dtype=np.float32)
        first = {k: np.zeros_like(v) for k, v in params.items()}
        second = {k: np.zeros_like(v) for k, v in params.items()}
        average_params = {k: np.zeros_like(v) for k, v in params.items()} if average else None
        average_count = 0
        step = 0
        for epoch in range(epochs):
            order = rng.permutation(x.shape[0])
            rate = 0.003 * (0.35 + 0.65 * (1 - epoch / epochs))
            for offset in range(0, len(order), 4096):
                index = order[offset:offset + 4096]
                batch_x, batch_node = x[index], node[index]
                batch_head = head_x[index] if site_head else None
                output, (xx, h1, h2) = _forward(batch_x, batch_node, params, batch_head)
                error = output - residual[index]
                if mode == 'scaled_summary':
                    local_scale = normalizer[index]
                    dout = (error / np.sqrt(error * error + (2 / local_scale) ** 2)
                            * local_scale / 20 / len(index))[:, None]
                else:
                    dout = (error / np.sqrt(error * error + 0.01) / len(index))[:, None]
                gradients = {'w0': batch_x.T @ dout, 'w3': h2.T @ dout, 'b3': dout.sum(axis=0)}
                dh2 = (dout @ params['w3'].T) * (1 - h2 * h2)
                gradients.update(w2=h1.T @ dh2, b2=dh2.sum(axis=0))
                dh1 = (dh2 @ params['w2'].T) * (1 - h1 * h1)
                gradients.update(w1=xx.T @ dh1, b1=dh1.sum(axis=0))
                dembed = dh1 @ params['w1'][-6:, :].T
                gradients['embedding'] = _node_gradient(batch_node, dembed, len(params['embedding']))
                if site_head:
                    gradients['site_head'] = _node_gradient(
                        batch_node, batch_head * dout, len(params['site_head']))
                step += 1
                for key, gradient in gradients.items():
                    if key == 'site_head':
                        gradient = gradient + params[key] * 0.0005
                        if laplacian is not None:
                            first_shared = 1 if site_head == 'graph_horizon' else 2
                            gradient[:, first_shared:] += graph_strength * (laplacian @ params[key][:, first_shared:])
                    elif key.startswith('w') or key == 'embedding':
                        gradient = gradient + params[key] * 0.0002
                        if key == 'embedding' and embedding_strength:
                            gradient += embedding_strength * (laplacian @ params[key])
                    first[key] = 0.9 * first[key] + 0.1 * gradient
                    second[key] = 0.999 * second[key] + 0.001 * gradient * gradient
                    corrected = first[key] / (1 - 0.9 ** step)
                    variance = second[key] / (1 - 0.999 ** step)
                    params[key] -= rate * corrected / (np.sqrt(variance) + 1e-7)
            if average and epoch >= 3 * epochs // 4:
                for key in params:
                    average_params[key] += params[key]
                average_count += 1
            if time.monotonic() - start > 150:
                break
        state = {'graph': graph, 'center': center, 'scale': scale, 'params': params, 'mode': mode,
                 'epochs': epoch + 1, 'fit_seconds': time.monotonic() - start, 'head_mode': site_head}
        if mode == 'scaled_summary':
            state['target_scale'] = graph['scale']
        if average_count:
            state['last_params'] = params
            state['params'] = {key: value / average_count for key, value in average_params.items()}
        return state


def replica_disagreement(state, features):
    prediction = np.stack([predict(member, features) for member in state['members']])
    return 2 * np.std(prediction, axis=0)


def predict(state, features):
    if 'members' in state:
        return np.mean([predict(member, features) for member in state['members']], axis=0)
    with threadpool_limits(limits=1):
        x, node = _features(features, state['graph'], state.get('mode', 'summary'))
        x = np.clip((x - state['center']) / state['scale'], -8, 8).astype(np.float32)
        head_x = _head_features(features, state.get('head_mode', True)) if 'site_head' in state['params'] else None
        residual = np.empty(x.shape[0], dtype=np.float32)
        for offset in range(0, len(x), 8192):
            end = offset + 8192
            multiplier = state['target_scale'][node[offset:end]] if 'target_scale' in state else 20
            residual[offset:end] = _forward(x[offset:end], node[offset:end], state['params'],
                                          head_x[offset:end] if head_x is not None else None)[0] * multiplier
        return np.maximum(np.asarray(features['base']) + residual.reshape(features['base'].shape), 0).astype(np.float32)
