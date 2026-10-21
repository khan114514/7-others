"""Cross-fitted, regularized conditional blending of residual experts."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor
import blended_calibration
import horizon_calibration
import neural_calibration
import spatial_calibration


def _design(features, tree, network, mode='standard', context_graph=None, replica_gap=None):
    use_history_trend = mode == 'site_history_trend'
    if use_history_trend:
        mode = 'site_neighbor_horizon'
    use_replica_gap = mode in ('site_replica_gap', 'site_replica_shared')
    if use_replica_gap:
        mode = 'site_neighbor_horizon'
    base = np.asarray(features['base'], dtype=np.float64)
    anchor = np.asarray(features['anchor'], dtype=np.float64)
    history = np.asarray(features['history'], dtype=np.float64)
    scale = np.maximum(np.mean(history, axis=1)[:, None, :], 30)
    recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
    std = np.std(history, axis=1)[:, None, :]
    horizon = np.broadcast_to(np.arange(base.shape[1])[None, :, None] / 11, base.shape)
    def broadcast(v):
        return np.broadcast_to(v, base.shape)
    variables = [np.ones_like(base), horizon,
                 np.clip(np.abs(anchor - base) / scale, 0, 2),
                 np.clip(np.abs(recent - base) / scale, 0, 2),
                 broadcast(np.clip(std / scale, 0, 2)),
                 np.clip(np.abs(network - tree) / 20, 0, 3),
                 np.clip(base / 200, 0, 3)]
    if mode == 'agreement':
        last = history[:, -1:, :]
        for reference in (anchor, recent, last):
            variables.append(np.clip((np.abs(tree - reference) -
                                      np.abs(network - reference)) / 20, -3, 3))
    if mode in ('interactions', 'site', 'site_horizon', 'site_context', 'site_direction',
                'site_direction_horizon', 'site_reliability', 'site_expert_direction', 'site_expert_shared',
                'site_expert_horizon', 'site_expert_horizon_shared',
                'site_neighbor_expert', 'site_neighbor_horizon',
                'site_context_signed', 'periodic'):
        variables.extend([(horizon - 0.5) * variables[j] for j in (2, 3, 4, 5)])
    if mode in ('site_direction', 'site_direction_horizon', 'site_reliability',
                'site_expert_direction', 'site_expert_shared', 'site_expert_horizon',
                'site_expert_horizon_shared', 'site_neighbor_expert', 'site_neighbor_horizon',
                'site_context_signed'):
        variables.extend([np.clip((anchor - base) / scale, -2, 2),
                          np.clip((recent - base) / scale, -2, 2)])
    if mode == 'site_direction_horizon':
        variables.extend([(horizon - 0.5) * variables[j] for j in (11, 12)])
    if mode in ('site_expert_direction', 'site_expert_shared', 'site_expert_horizon',
                'site_expert_horizon_shared', 'site_neighbor_expert', 'site_neighbor_horizon'):
        variables.append(np.clip((network - tree) / 20, -3, 3))
    if mode in ('site_expert_horizon', 'site_expert_horizon_shared', 'site_neighbor_horizon'):
        variables.append((horizon - 0.5) * variables[13])
    if mode in ('site_neighbor_expert', 'site_neighbor_horizon'):
        normalized = (network - tree) / context_graph['scale']
        neighbor_gap = np.sum(normalized[..., context_graph['neighbors']] *
                              context_graph['weights'], axis=-1) * context_graph['scale']
        variables.append(np.clip(neighbor_gap / 20, -3, 3))
    if mode == 'periodic':
        clock = np.asarray(features['clock'], dtype=np.float64)
        variables.extend([np.sin(2 * np.pi * clock), np.cos(2 * np.pi * clock),
                          np.sin(4 * np.pi * clock), np.cos(4 * np.pi * clock)])
    if use_replica_gap:
        uncertainty = np.clip(replica_gap / 20, 0, 3)
        variables.extend([uncertainty, (horizon - 0.5) * uncertainty])
    if use_history_trend:
        coordinate = np.arange(history.shape[1]) - (history.shape[1] - 1) / 2
        slope = (np.sum(history * coordinate[None, :, None], axis=1)[:, None, :]
                 / np.sum(coordinate ** 2))
        steps = np.arange(1, base.shape[1] + 1)[None, :, None]
        persistence = np.broadcast_to(history[:, -1:, :], base.shape)
        trend = persistence + steps / (1 + steps / 6) * slope
        variables.extend([np.clip((np.abs(tree - reference)
                                   - np.abs(network - reference)) / 20, -3, 3)
                          for reference in (persistence, trend)])
    return np.stack(variables, axis=-1).reshape(-1, len(variables))


def _gate_fit(features, target, tree, network, penalty=0.01, mode='standard', replica_gap=None):
    design_mode = mode
    if mode == 'site_history_trend':
        mode = 'site_neighbor_horizon'
    context_graph = (spatial_calibration._neighborhood(features)
                     if mode in ('site_neighbor_expert', 'site_neighbor_horizon',
                                 'site_replica_gap', 'site_replica_shared') else None)
    x = _design(features, tree, network, design_mode, context_graph, replica_gap)
    valid = (np.isfinite(target) & (target > 0)).reshape(-1)
    x = x[valid]
    center = x.mean(axis=0)
    center[0] = 0
    scale = np.maximum(x.std(axis=0), 0.1)
    scale[0] = 1
    x = np.clip((x - center) / scale, -4, 4)
    residual = (tree - target).reshape(-1)[valid]
    difference = (network - tree).reshape(-1)[valid]
    weight0 = np.broadcast_to(np.where(np.arange(tree.shape[1]) < 6, 0.18, 0.32)[None, :, None], tree.shape).reshape(-1)[valid]
    offset = np.log(weight0 / (1 - weight0))
    use_site = mode in ('site', 'site_horizon', 'site_context', 'site_direction',
                       'site_direction_horizon', 'site_reliability',
                       'site_expert_direction', 'site_expert_shared', 'site_expert_horizon',
                       'site_expert_horizon_shared', 'site_neighbor_expert', 'site_neighbor_horizon',
                       'site_replica_gap', 'site_replica_shared',
                       'site_context_signed')
    if use_site:
        nodes = tree.shape[2]
        node = np.broadcast_to(np.arange(nodes)[None, None, :], tree.shape).reshape(-1)[valid]
        width = {'site': 1, 'site_horizon': 2, 'site_context': 4, 'site_direction': 6,
                 'site_direction_horizon': 10,
                 'site_reliability': 8, 'site_expert_direction': 7,
                 'site_expert_shared': 6,
                 'site_expert_horizon': 8, 'site_expert_horizon_shared': 7,
                 'site_neighbor_expert': 8,
                 'site_neighbor_horizon': 9,
                 'site_replica_shared': 9, 'site_replica_gap': 11,
                 'site_context_signed': 4}[mode]
        basis = np.ones((len(node), width))
        if width >= 2:
            basis[:, 1] = np.broadcast_to(np.linspace(-1, 1, tree.shape[1])[None, :, None], tree.shape).reshape(-1)[valid]
        site_columns = [2, 3] if width >= 4 else []
        if width >= 6:
            site_columns.extend([11, 12])
        if width == 10:
            site_columns.extend([7, 8, 13, 14])
        if mode == 'site_reliability':
            site_columns.extend([4, 5])
        if mode in ('site_expert_direction', 'site_expert_horizon', 'site_expert_horizon_shared',
                    'site_neighbor_expert', 'site_neighbor_horizon',
                    'site_replica_gap', 'site_replica_shared'):
            site_columns.append(13)
        if mode in ('site_expert_horizon', 'site_neighbor_expert', 'site_neighbor_horizon',
                    'site_replica_gap', 'site_replica_shared'):
            site_columns.append(14)
        if mode in ('site_neighbor_horizon', 'site_replica_gap', 'site_replica_shared'):
            site_columns.append(15)
        if mode == 'site_replica_gap':
            site_columns.extend([16, 17])
        if site_columns:
            basis[:, 2:] = x[:, site_columns]
        graph = context_graph if context_graph is not None else spatial_calibration._neighborhood(features)
        adjacency = np.zeros((nodes, nodes))
        np.put_along_axis(adjacency, graph['neighbors'], graph['weights'], axis=1)
        adjacency = (adjacency + adjacency.T) / 2
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        site_shape = (nodes, basis.shape[1])
    def objective(coef):
        global_coef = coef[:x.shape[1]]
        logit = offset + x @ global_coef
        if use_site:
            site = coef[x.shape[1]:].reshape(site_shape)
            centered_site = site - site.mean(axis=0)
            logit += np.sum(centered_site[node] * basis, axis=1)
        weight = expit(logit)
        error = residual + weight * difference
        smooth = np.sqrt(error * error + 4)
        derivative = error / smooth * difference * weight * (1 - weight)
        value = np.mean(smooth) / 20 + penalty * np.dot(global_coef, global_coef) / 2
        gradient = x.T @ derivative / (len(x) * 20) + penalty * global_coef
        if use_site:
            site_gradient = np.column_stack([
                np.bincount(node, weights=derivative * basis[:, j], minlength=nodes)
                for j in range(basis.shape[1])]) / (len(x) * 20)
            site_gradient -= site_gradient.mean(axis=0)
            regularizer = 0.02 * site + 0.05 * (laplacian @ site)
            site_gradient += regularizer / nodes
            value += np.sum(site * regularizer) / (2 * nodes)
            gradient = np.r_[gradient, site_gradient.reshape(-1)]
        return value, gradient
    count = x.shape[1] + nodes * basis.shape[1] if use_site else x.shape[1]
    result = minimize(objective, np.zeros(count), jac=True, method='L-BFGS-B',
                      options={'maxiter': 100 if use_site else 60, 'ftol': 1e-10})
    state = {'coef': result.x[:x.shape[1]].astype(np.float32), 'center': center.astype(np.float32),
             'scale': scale.astype(np.float32), 'penalty': penalty, 'mode': design_mode}
    if use_site:
        site = result.x[x.shape[1]:].reshape(site_shape)
        state['site'] = (site - site.mean(axis=0)).astype(np.float32)
        state['site_columns'] = site_columns
    if context_graph is not None:
        state['context_graph'] = context_graph
    return state


def _weight(state, features, tree, network, replica_gap=None):
    x = _design(features, tree, network, state.get('mode', 'standard'),
                state.get('context_graph'), replica_gap)
    x = np.clip((x - state['center']) / state['scale'], -4, 4)
    initial = np.broadcast_to(np.where(np.arange(tree.shape[1]) < 6, 0.18, 0.32)[None, :, None], tree.shape)
    offset = np.log(initial / (1 - initial))
    logit = offset + (x @ state['coef']).reshape(tree.shape)
    if 'site' in state:
        site = state['site']
        logit += site[None, None, :, 0]
        if site.shape[1] >= 2:
            logit += np.linspace(-1, 1, tree.shape[1])[None, :, None] * site[None, None, :, 1]
        for j, column in enumerate(state.get('site_columns', range(2, site.shape[1])), 2):
            logit += x[:, column].reshape(tree.shape) * site[None, None, :, j]
    return np.clip(expit(logit), 0.05, 0.65)


def _amplitude_design(features, correction):
    base = np.asarray(features['base'], dtype=np.float64)
    history = np.asarray(features['history'], dtype=np.float64)
    scale = np.maximum(np.mean(history, axis=1)[:, None, :], 30)
    anchor = np.asarray(features['anchor'])
    recent = np.mean(history[:, -3:, :], axis=1)[:, None, :]
    horizon = np.broadcast_to(np.linspace(0, 1, base.shape[1])[None, :, None], base.shape)
    variables = [np.ones_like(base), horizon,
                 np.clip(np.abs(anchor - base) / scale, 0, 2),
                 np.clip(np.abs(recent - base) / scale, 0, 2),
                 np.broadcast_to(np.clip(np.std(history, axis=1)[:, None, :] / scale, 0, 2), base.shape),
                 np.clip(np.abs(correction) / 20, 0, 3),
                 np.clip(np.sign(correction) * (anchor - base) / scale, -2, 2)]
    return np.stack(variables, axis=-1).reshape(-1, len(variables))


def _amplitude_fit(features, target, tree, network, gate):
    base = np.asarray(features['base'], dtype=np.float64)
    correction = tree + _weight(gate, features, tree, network) * (network - tree) - base
    valid = (np.isfinite(target) & (target > 0)).reshape(-1)
    x = _amplitude_design(features, correction)[valid]
    center = x.mean(axis=0)
    center[0] = 0
    scale = np.maximum(x.std(axis=0), 0.1)
    scale[0] = 1
    x = np.clip((x - center) / scale, -4, 4)
    residual = (base - target).reshape(-1)[valid]
    correction = correction.reshape(-1)[valid]
    def objective(coef):
        activation = np.tanh(x @ coef)
        error = residual + (1 + 0.5 * activation) * correction
        smooth = np.sqrt(error * error + 4)
        derivative = error / smooth * correction * 0.5 * (1 - activation * activation)
        return (np.mean(smooth) / 20 + 0.02 * np.dot(coef, coef) / 2,
                x.T @ derivative / (len(x) * 20) + 0.02 * coef)
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 60, 'ftol': 1e-10})
    return {'coef': result.x.astype(np.float32), 'center': center.astype(np.float32),
            'scale': scale.astype(np.float32)}


def _blend(gate, features, tree, network, amplitude=None, replica_gap=None):
    blend = tree + _weight(gate, features, tree, network, replica_gap) * (network - tree)
    if amplitude is not None:
        base = np.asarray(features['base'], dtype=np.float64)
        correction = blend - base
        x = _amplitude_design(features, correction)
        x = np.clip((x - amplitude['center']) / amplitude['scale'], -4, 4)
        factor = (1 + 0.5 * np.tanh(x @ amplitude['coef'])).reshape(base.shape)
        blend = np.maximum(base + factor * correction, 0)
    return blend


def _residual_tree_design(features, tree, network, blend):
    base = np.asarray(features['base'], dtype=np.float32)
    horizon = np.broadcast_to(np.arange(base.shape[1])[None, :, None], base.shape)
    node = np.broadcast_to(np.arange(base.shape[2])[None, None, :], base.shape)
    return np.column_stack([value.reshape(-1) for value in
                            (horizon, node, tree - base, network - base,
                             network - tree, blend - base)]).astype(np.float32)


def _residual_tree_fit(features, y, tree, network, seed, gate_mode, diffuse=False):
    # Hold out origins from the gate as well as the two primary experts when
    # producing the errors used to train the final, deliberately shallow stage.
    index = np.arange(len(y))
    blend_oof = np.empty_like(y)
    for held in np.array_split(index, 3):
        keep = np.setdiff1d(index, held)
        train_features = {k: v[keep] for k, v in features.items()}
        held_features = {k: v[held] for k, v in features.items()}
        gate = _gate_fit(train_features, y[keep], tree[keep], network[keep], mode=gate_mode)
        blend_oof[held] = _blend(gate, held_features, tree[held], network[held])
    x = _residual_tree_design(features, tree, network, blend_oof)
    valid = np.isfinite(y) & (y > 0)
    model = HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=40, learning_rate=0.05,
        max_leaf_nodes=7, min_samples_leaf=2000, l2_regularization=50,
        categorical_features=[1], early_stopping=False, random_state=seed)
    model.fit(x[valid.reshape(-1)], (y - blend_oof)[valid])
    if diffuse:
        # Preserve the sensor's seasonal correction while sharing only its
        # changing component with neighbors learned from calibration history.
        correction = np.clip(model.predict(x).reshape(y.shape), -10, 10)
        diffusion = spatial_calibration.diffusion_fit(
            features, np.asarray(features['base']) + correction,
            robust=diffuse == 'context_smooth')
        if diffuse in ('context', 'context_smooth'):
            diffusion['pooling'] = 'context_attenuate'
        return {'model': model, 'diffusion': diffusion}
    return model


def _residual_tree_predict(model, features, tree, network, blend):
    x = _residual_tree_design(features, tree, network, blend)
    if isinstance(model, dict):
        base = np.asarray(features['base'])
        correction = np.clip(model['model'].predict(x).reshape(blend.shape), -10, 10)
        correction = spatial_calibration.diffusion_predict(
            model['diffusion'], features, base + correction) - base
        return np.maximum(blend + correction, 0)
    correction = np.clip(model.predict(x).reshape(blend.shape), -10, 10)
    return np.maximum(blend + 0.5 * correction, 0)


def fit(features, targets, seed, auxiliary_mode=None, site_head=False, average=False,
        gate_mode='standard', amplitude=False, network_mode='summary', network_replicas=1,
        network_diffusion=False, network_epochs=100, residual_tree=False):
    with threadpool_limits(limits=1):
        y = np.asarray(targets['target'] if isinstance(targets, dict) else targets)
        index = np.arange(len(y))
        tree_oof, network_oof = np.empty_like(y), np.empty_like(y)
        use_replica_gap = gate_mode in ('site_replica_gap', 'site_replica_shared')
        replica_gap_oof = np.empty_like(y) if use_replica_gap else None
        for held in np.array_split(index, 3):
            train = np.setdiff1d(index, held)
            train_features = {k: v[train] for k, v in features.items()}
            test_features = {k: v[held] for k, v in features.items()}
            fold = blended_calibration.fit(train_features, y[train], seed,
                                           site_head=site_head, average=average,
                                           network_mode=network_mode, network_replicas=network_replicas,
                                           network_epochs=network_epochs)
            tree_oof[held] = horizon_calibration.predict(fold['tree'], test_features)
            network_oof[held] = neural_calibration.predict(fold['network'], test_features)
            if use_replica_gap:
                replica_gap_oof[held] = neural_calibration.replica_disagreement(
                    fold['network'], test_features)
        diffusion = None
        if network_diffusion:
            diffusion_graph = ('prediction_confidence_fallback'
                               if network_diffusion == 'prediction_confidence_fallback'
                               else 'prediction_median_confidence'
                               if network_diffusion == 'prediction_median_confidence'
                               else 'prediction_median' if network_diffusion == 'prediction_median'
                               else 'prediction' if network_diffusion in
                               ('prediction', 'prediction_context',
                                'prediction_context_smooth') else 'history')
            diffusion = spatial_calibration.diffusion_fit(
                features, network_oof,
                robust=network_diffusion in ('context_smooth', 'prediction_context_smooth'),
                graph_mode=diffusion_graph)
            if network_diffusion in ('context', 'context_smooth',
                                     'prediction_context', 'prediction_context_smooth'):
                diffusion['pooling'] = 'context_normalize'
            network_oof = spatial_calibration.diffusion_predict(diffusion, features, network_oof)
        gate = _gate_fit(features, y, tree_oof, network_oof, mode=gate_mode,
                         replica_gap=replica_gap_oof)
        state = blended_calibration.fit(features, y, seed, site_head=site_head, average=average,
                                        network_mode=network_mode, network_replicas=network_replicas,
                                        network_epochs=network_epochs)
        state['gate'] = gate
        if diffusion is not None:
            state['diffusion'] = diffusion
        if residual_tree:
            state['residual_tree'] = _residual_tree_fit(
                features, y, tree_oof, network_oof, seed, gate_mode,
                diffuse=(residual_tree if residual_tree in ('context', 'context_smooth')
                         else residual_tree == 'diffused'))
        if amplitude:
            state['amplitude'] = _amplitude_fit(features, y, tree_oof, network_oof, gate)
        if auxiliary_mode is not None:
            state['aux_network'] = neural_calibration.fit(
                features, y, seed, graph=state['tree']['pooled']['graph'], mode=auxiliary_mode)
        return state


def predict(state, features):
    with threadpool_limits(limits=1):
        tree = horizon_calibration.predict(state['tree'], features)
        network = neural_calibration.predict(state['network'], features)
        if 'aux_network' in state:
            network = (network + neural_calibration.predict(state['aux_network'], features)) / 2
        if 'diffusion' in state:
            network = spatial_calibration.diffusion_predict(state['diffusion'], features, network)
        replica_gap = (neural_calibration.replica_disagreement(state['network'], features)
                       if state['gate'].get('mode') in ('site_replica_gap', 'site_replica_shared')
                       else None)
        blend = _blend(state['gate'], features, tree, network,
                       state.get('amplitude'), replica_gap)
        if 'residual_tree' in state:
            blend = _residual_tree_predict(state['residual_tree'], features, tree, network, blend)
        return blend.astype(np.float32)
