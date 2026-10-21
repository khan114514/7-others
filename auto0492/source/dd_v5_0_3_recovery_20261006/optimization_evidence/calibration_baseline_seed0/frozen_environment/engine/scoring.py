"""Frozen scalar reward; there is deliberately no clipping or Reference cap."""
import math


def normalize(metric, anchors):
    if anchors.get('calibration_status') != 'FROZEN':
        raise ValueError('Formal scoring requires frozen calibration evidence')
    baseline = anchors.get('baseline')
    upper = anchors.get('upper_bound')
    for value in (metric, baseline, upper):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Metric and anchors must be finite scalars')
    if anchors.get('direction') != 'minimize' or metric < 0 or not 0 <= upper < baseline:
        raise ValueError('Invalid FID metric or minimization anchors')
    return (baseline - metric) / (baseline - upper)
