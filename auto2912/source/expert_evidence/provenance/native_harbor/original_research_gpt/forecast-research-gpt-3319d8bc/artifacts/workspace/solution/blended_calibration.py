"""Horizon-specific median trees blended with a smooth residual network."""
import numpy as np
from threadpoolctl import threadpool_limits
import horizon_calibration
import neural_calibration


def fit(features, targets, seed, graph=None, network_mode='summary', auxiliary_mode=None,
        site_head=False, average=False, network_replicas=1, network_epochs=100):
    with threadpool_limits(limits=1):
        state = {'tree': horizon_calibration.fit(features, targets, seed, groups=2, graph=graph),
                'network': neural_calibration.fit(features, targets, seed, graph=graph, mode=network_mode,
                                                  site_head=site_head, average=average,
                                                  replicas=network_replicas, epochs=network_epochs),
                # The smooth model is more useful at longer horizons on both
                # public origin blocks; keep a conservative horizon schedule.
                'network_weight': np.where(np.arange(features['base'].shape[1]) < 6,
                                           0.18, 0.32).astype(np.float32)}
        if auxiliary_mode is not None:
            state['aux_network'] = neural_calibration.fit(
                features, targets, seed, graph=state['tree']['pooled']['graph'], mode=auxiliary_mode)
        return state


def predict(state, features):
    with threadpool_limits(limits=1):
        tree = horizon_calibration.predict(state['tree'], features)
        network = neural_calibration.predict(state['network'], features)
        if 'aux_network' in state:
            network = (network + neural_calibration.predict(state['aux_network'], features)) / 2
        weight = np.asarray(state.get('network_weight', 0.25), dtype=np.float32)
        return (tree * (1.0 - weight[None, :, None]) +
                network * weight[None, :, None]).astype(np.float32)
