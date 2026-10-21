"""Seasonal experts with confidence-gated innovation diffusion and trend gating."""
import conditional_calibration


def fit(features, targets, seed):
    return conditional_calibration.fit(features, targets, seed, site_head='graph',
                                       gate_mode='site_history_trend', network_replicas=2,
                                       network_epochs=(100, 80), network_diffusion='prediction_confidence_fallback',
                                       residual_tree='context_smooth')


def predict(state, features):
    return conditional_calibration.predict(state, features)
