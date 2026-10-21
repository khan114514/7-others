"""Input-detection-gated intensity shrinkage and detection uncertainty.

Fits only training pseudobulk statistics. Input control cells select the
nearest training context and provide unsupervised gene/donor adaptation.
No target data, filesystem access, external models, or stochastic inference.
"""

from types import SimpleNamespace

import numpy as np
import torch


DONOR_BLEND = 0.5
EXPRESSION_FLOOR = 0.001
RATIO_POWER = 0.25
HURDLE_PRIOR = 0.005
HURDLE_BLEND = 0.75
DETECTION_SHRINKAGE = 10.0
RULE_RIDGE = 0.0001
RULE_STEPS = 400
TRAIN_EXPRESSION_SCALE = 5.0
CONTEXT_TEMPERATURE = 0.1
COMPATIBILITY_STRENGTH = 0.5
CONTEXT_BLEND = 0.25
COMPATIBILITY_SUPPORT = 0.1
SUPPORT_POWER = 2.0
QUERY_DONOR_PRIOR = 32.0
DONOR_TRANSFER_SUPPORT = 0.1
DISTANCE_DONOR_BLEND = 1.0
LEVEL_RIDGE_MULTIPLIER = 4.0
LEVEL_INFERENCE_BLEND = 1.0
LEVEL_CENTERING_STRENGTH = 1.0
LOCAL_RULE_RIDGE = 0.001
LOCAL_RULE_RAW_BOUND = 0.25
LOCAL_RULE_STEPS = 400
LOCAL_RULE_TEMPERATURE = 0.1
LOCAL_TARGET_STANDARD_ERROR = 0.1
LOCAL_TARGET_MIN_CONFIDENCE = 0.1
INTENSITY_SHRINKAGE_STRENGTH = 0.5
INTENSITY_INPUT_SUPPORT = 0.05
INTENSITY_GATE_POWER = 4.0
DETECTION_BETA_SUPPORT = 0.05
DETECTION_BETA_GATE_POWER = 8.0


class ResponseState(torch.nn.Module):
    """All fitted expression statistics live in the saved state dictionary."""

    def __init__(self, n_contexts, n_donors, n_genes):
        super().__init__()
        self.register_buffer(
            "control_mean", torch.zeros(n_contexts, n_genes, dtype=torch.float64)
        )
        self.register_buffer(
            "response", torch.zeros(n_contexts, n_genes, dtype=torch.float64)
        )
        self.register_buffer(
            "donor_response",
            torch.zeros(n_contexts, n_donors, n_genes, dtype=torch.float64),
        )
        self.register_buffer(
            "control_rate", torch.zeros(n_contexts, n_genes, dtype=torch.float64)
        )
        self.register_buffer(
            "stimulated_rate", torch.zeros(n_contexts, n_genes, dtype=torch.float64)
        )
        for name in ["donor_control_mean", "donor_control_rate", "donor_stimulated_rate"]:
            self.register_buffer(
                name, torch.zeros(n_contexts, n_donors, n_genes, dtype=torch.float64)
            )
        self.register_buffer(
            "detection_uncertainty",
            torch.zeros(n_contexts, n_genes, dtype=torch.float64),
        )
        self.register_buffer(
            "beta_detection_uncertainty",
            torch.zeros(n_contexts, n_genes, dtype=torch.float64),
        )
        self.register_buffer(
            "positive_intensity_uncertainty",
            torch.zeros(n_contexts, n_genes, dtype=torch.float64),
        )
        for name in ["control_variance", "stimulated_variance"]:
            self.register_buffer(name, torch.zeros(n_contexts, n_genes, dtype=torch.float64))
        self.register_buffer("transfer_coefficients", torch.zeros(5, dtype=torch.float64))
        self.register_buffer(
            "local_transfer_adjustments", torch.zeros(n_contexts, 5, dtype=torch.float64)
        )


def build_model(train_scaled, device, seed):
    torch.manual_seed(seed)
    contexts = sorted(set(train_scaled.obs["cell_type"].astype(str)))
    donors = sorted(set(train_scaled.obs["replicate"].astype(str)))
    state = ResponseState(len(contexts), len(donors), train_scaled.shape[1])
    return SimpleNamespace(
        model=state,
        train_data=train_scaled,
        contexts=contexts,
        donors=donors,
        device=torch.device("cpu"),
        is_trained=False,
    )


def _summaries(matrix, mask):
    index = torch.as_tensor(np.flatnonzero(mask), device=matrix.device)
    selected = matrix.index_select(0, index)
    return selected.mean(dim=0), (selected > 0).to(torch.float64).mean(dim=0)


def _sampling_variance(matrix, mask):
    index = torch.as_tensor(np.flatnonzero(mask), device=matrix.device)
    return matrix.index_select(0, index).var(dim=0, unbiased=False)


def _positive_cv2(mean, rate, variance):
    positive_mean = mean / rate.clamp_min(1e-12)
    positive_second = (variance + mean.square()) / rate.clamp_min(1e-12)
    positive_variance = (positive_second - positive_mean.square()).clamp_min(0.0)
    return torch.where(
        mean > 0,
        positive_variance / positive_mean.square().clamp_min(1e-12),
        torch.ones_like(mean),
    )


def train_model(model, seed):
    """Fit training statistics and a reliability rule from fresh state."""
    torch.manual_seed(seed)
    train = model.train_data
    x = train.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = torch.as_tensor(x, dtype=torch.float64, device=model.device)
    cell_type = np.asarray(train.obs["cell_type"].astype(str))
    condition = np.asarray(train.obs["condition"].astype(str))
    donor_id = np.asarray(train.obs["replicate"].astype(str))
    pooled_control = model.model.control_mean
    pooled_response = model.model.response
    donor_response = model.model.donor_response

    for i, context in enumerate(model.contexts):
        control = (cell_type == context) & (condition == "control")
        stimulated = (cell_type == context) & (condition == "stimulated")
        if not control.any() or not stimulated.any():
            raise ValueError("Training contexts must contain both conditions")
        c, cf = _summaries(x, control)
        s, sf = _summaries(x, stimulated)
        pooled_control[i] = c
        pooled_response[i] = s - c
        model.model.control_rate[i] = cf
        model.model.stimulated_rate[i] = sf
        model.model.control_variance[i] = _sampling_variance(x, control)
        model.model.stimulated_variance[i] = _sampling_variance(x, stimulated)
        donor_counts = []
        for j, donor in enumerate(model.donors):
            dc = control & (donor_id == donor)
            ds = stimulated & (donor_id == donor)
            donor_counts.append([int(dc.sum()), int(ds.sum())])
            if dc.any() and ds.any():
                donor_c, donor_cf = _summaries(x, dc)
                donor_s, donor_sf = _summaries(x, ds)
            else:
                donor_c, donor_cf = c, cf
                donor_s, donor_sf = s, sf
            donor_response[i, j] = donor_s - donor_c
            model.model.donor_control_mean[i, j] = donor_c
            model.model.donor_control_rate[i, j] = donor_cf
            model.model.donor_stimulated_rate[i, j] = donor_sf

        # Estimate disagreement beyond finite-count detection noise. Pair
        # weights reflect the effective sample size of each donor response.
        counts = torch.as_tensor(donor_counts, dtype=torch.float64)
        valid = (counts[:, 0] > 0) & (counts[:, 1] > 0)
        counts = counts.clamp_min(1.0)
        weights = valid.to(torch.float64) / (1.0 / counts[:, 0] + 1.0 / counts[:, 1])
        if weights.sum() == 0:
            model.model.detection_uncertainty[i].fill_(1e6)
            model.model.beta_detection_uncertainty[i].fill_(1e6)
            model.model.positive_intensity_uncertainty[i].fill_(1e6)
            continue
        weights /= weights.sum()
        pc = model.model.donor_control_rate[i].clamp(HURDLE_PRIOR, 1.0 - HURDLE_PRIOR)
        ps = model.model.donor_stimulated_rate[i].clamp(HURDLE_PRIOR, 1.0 - HURDLE_PRIOR)
        effects = torch.log(ps) - torch.log1p(-ps) - torch.log(pc) + torch.log1p(-pc)
        average_effect = (weights[:, None] * effects).sum(dim=0)
        denominator = (1.0 - weights.square().sum()).clamp_min(1e-12)
        between_variance = (weights[:, None] * (effects - average_effect).square()).sum(dim=0) / denominator
        sampling_noise = 1.0 / (counts[:, 0, None] * pc * (1.0 - pc))
        sampling_noise += 1.0 / (counts[:, 1, None] * ps * (1.0 - ps))
        average_noise = (weights[:, None] * sampling_noise).sum(dim=0)
        biological_variance = (between_variance - average_noise).clamp_min(0.0)
        pooled_sampling_variance = (weights[:, None].square() * sampling_noise).sum(dim=0)
        model.model.detection_uncertainty[i] = biological_variance + pooled_sampling_variance

        # Jeffreys posterior logit variance handles sparse detection counts.
        nc = counts[:, 0, None]
        ns = counts[:, 1, None]
        kc = torch.round(nc * model.model.donor_control_rate[i]).clamp_min(0.0)
        ks = torch.round(ns * model.model.donor_stimulated_rate[i]).clamp_min(0.0)
        beta_noise = torch.polygamma(1, kc + 0.5)
        beta_noise += torch.polygamma(1, nc - kc + 0.5)
        beta_noise += torch.polygamma(1, ks + 0.5)
        beta_noise += torch.polygamma(1, ns - ks + 0.5)
        beta_average_noise = (weights[:, None] * beta_noise).sum(dim=0)
        # Match Beta disagreement effects to posterior expected log odds.
        beta_effects = torch.digamma(ks + 0.5) - torch.digamma(ns - ks + 0.5)
        beta_effects -= torch.digamma(kc + 0.5) - torch.digamma(nc - kc + 0.5)
        beta_average_effect = (weights[:, None] * beta_effects).sum(dim=0)
        beta_between = (
            weights[:, None] * (beta_effects - beta_average_effect).square()
        ).sum(dim=0) / denominator
        beta_biological = (beta_between - beta_average_noise).clamp_min(0.0)
        beta_pooled_noise = (weights[:, None].square() * beta_noise).sum(dim=0)
        model.model.beta_detection_uncertainty[i] = beta_biological + beta_pooled_noise

        # Estimate uncertainty in positive-cell intensity responses separately
        # from changes in detection probability, using pooled conditional CV.
        cv_control = _positive_cv2(c, cf, model.model.control_variance[i])
        cv_stimulated = _positive_cv2(s, sf, model.model.stimulated_variance[i])
        donor_cf = model.model.donor_control_rate[i]
        donor_sf = model.model.donor_stimulated_rate[i]
        donor_c = model.model.donor_control_mean[i]
        donor_s = donor_c + model.model.donor_response[i]
        intensity_prior = HURDLE_PRIOR / TRAIN_EXPRESSION_SCALE
        intensity_effects = torch.log((donor_s + intensity_prior) / (donor_sf + HURDLE_PRIOR))
        intensity_effects -= torch.log((donor_c + intensity_prior) / (donor_cf + HURDLE_PRIOR))
        average_intensity_effect = (weights[:, None] * intensity_effects).sum(dim=0)
        intensity_between = (
            weights[:, None] * (intensity_effects - average_intensity_effect).square()
        ).sum(dim=0) / denominator
        intensity_noise = cv_control / (counts[:, 0, None] * donor_cf).clamp_min(1.0)
        intensity_noise += cv_stimulated / (counts[:, 1, None] * donor_sf).clamp_min(1.0)
        average_intensity_noise = (weights[:, None] * intensity_noise).sum(dim=0)
        intensity_biological = (intensity_between - average_intensity_noise).clamp_min(0.0)
        intensity_pooled_noise = (weights[:, None].square() * intensity_noise).sum(dim=0)
        model.model.positive_intensity_uncertainty[i] = intensity_biological + intensity_pooled_noise

    _fit_transfer_rule(model, cell_type, condition, donor_id)
    model.is_trained = True


def _log_odds(probability):
    probability = np.clip(probability, HURDLE_PRIOR, 1.0 - HURDLE_PRIOR)
    return np.log(probability) - np.log1p(-probability)


def _control_distances(pooled_controls, matched_controls, query):
    geometry = pooled_controls + DISTANCE_DONOR_BLEND * (matched_controls - pooled_controls)
    return np.mean((geometry - query) ** 2, axis=1)


def _rule_features(effect, uncertainty, input_rate, control_rate,
                   input_intensity, control_intensity):
    log_ratio = np.log((uncertainty + 1e-12) / (effect ** 2 + 1e-12))
    features = np.stack([
        np.ones_like(effect),
        np.clip(log_ratio, -8.0, 8.0) / 4.0,
        np.abs(_log_odds(input_rate) - _log_odds(control_rate)) / 3.0,
        np.clip(np.log(input_intensity / control_intensity), -3.0, 3.0) / 3.0,
        np.broadcast_to(np.clip(_log_odds(input_rate), -6.0, 6.0) / 3.0, effect.shape),
    ], axis=-1)
    return features, -np.log(DETECTION_SHRINKAGE) - log_ratio


def _query_detection_uncertainty(delta_uncertainty, beta_uncertainty, input_rate):
    gate = DETECTION_BETA_SUPPORT ** DETECTION_BETA_GATE_POWER / (
        input_rate ** DETECTION_BETA_GATE_POWER
        + DETECTION_BETA_SUPPORT ** DETECTION_BETA_GATE_POWER
    )
    return (1.0 - gate) * delta_uncertainty + gate * beta_uncertainty


def _transfer_intensity(input_intensity, control_intensity, stimulated_intensity,
                        uncertainty, input_rate):
    effect = np.log(stimulated_intensity / (control_intensity + 1e-12))
    reliability = effect ** 2 / (effect ** 2 + uncertainty + 1e-12)
    gate = INTENSITY_INPUT_SUPPORT ** INTENSITY_GATE_POWER / (
        input_rate ** INTENSITY_GATE_POWER + INTENSITY_INPUT_SUPPORT ** INTENSITY_GATE_POWER
    )
    multiplier = 1.0 - INTENSITY_SHRINKAGE_STRENGTH * gate * (1.0 - reliability)
    return input_intensity * np.exp(multiplier * effect)


def _fit_transfer_rule(model, cell_type, condition, donor_id):
    """Fit five bounded coefficients on donor-level training pseudo-transfers."""
    state = model.model
    c = state.control_mean.detach().cpu().numpy()
    d = state.response.detach().cpu().numpy()
    donor_d = state.donor_response.detach().cpu().numpy()
    donor_c = state.donor_control_mean.detach().cpu().numpy()
    cf = state.control_rate.detach().cpu().numpy()
    sf = state.stimulated_rate.detach().cpu().numpy()
    donor_cf = state.donor_control_rate.detach().cpu().numpy()
    donor_sf = state.donor_stimulated_rate.detach().cpu().numpy()
    uncertainty = state.detection_uncertainty.detach().cpu().numpy()
    beta_uncertainty = state.beta_detection_uncertainty.detach().cpu().numpy()
    intensity_uncertainty = state.positive_intensity_uncertainty.detach().cpu().numpy()
    examples = []
    example_weights = []
    target_confidences = []
    control_variance = state.control_variance.detach().cpu().numpy()
    stimulated_variance = state.stimulated_variance.detach().cpu().numpy()
    profile_distances = []
    prior = HURDLE_PRIOR / TRAIN_EXPRESSION_SCALE
    floor = EXPRESSION_FLOOR / TRAIN_EXPRESSION_SCALE
    for i, context in enumerate(model.contexts):
        pair_weights = np.zeros(len(model.donors), dtype=np.float64)
        pair_counts = np.zeros((len(model.donors), 2), dtype=np.float64)
        for k, donor in enumerate(model.donors):
            group = (cell_type == context) & (donor_id == donor)
            nc = int(np.sum(group & (condition == "control")))
            ns = int(np.sum(group & (condition == "stimulated")))
            pair_counts[k] = [nc, ns]
            if nc > 0 and ns > 0:
                pair_weights[k] = 1.0 / (1.0 / nc + 1.0 / ns)
        if pair_weights.sum() == 0:
            continue
        pair_weights /= pair_weights.sum()

        for k, weight in enumerate(pair_weights):
            if weight == 0:
                continue
            q, f = donor_c[i, k], donor_cf[i, k]
            matched_controls = (1.0 - DONOR_BLEND) * c + DONOR_BLEND * donor_c[:, k]
            distances = _control_distances(c, matched_controls, q)
            profile_distances.append(distances.copy() * TRAIN_EXPRESSION_SCALE ** 2)
            distances[i] = np.inf
            nearest = int(np.argmin(distances))

            def pool(pooled, donor_values):
                return (1.0 - DONOR_BLEND) * pooled + DONOR_BLEND * donor_values[k]

            cc = pool(c[nearest], donor_c[nearest])
            ss = pool(c[nearest] + d[nearest], donor_c[nearest] + donor_d[nearest])
            fc = pool(cf[nearest], donor_cf[nearest])
            fs = pool(sf[nearest], donor_sf[nearest])
            effect = _log_odds(fs) - _log_odds(fc)
            input_intensity = (q + prior) / (f + HURDLE_PRIOR)
            control_intensity = (cc + prior) / (fc + HURDLE_PRIOR)
            stimulated_intensity = (ss + prior) / (fs + HURDLE_PRIOR)
            features, base_logits = _rule_features(
                effect, _query_detection_uncertainty(
                    uncertainty[nearest], beta_uncertainty[nearest], f
                ), f, fc, input_intensity, control_intensity
            )
            level_mean = features[..., 4].mean(axis=-1, keepdims=True)
            features[..., 4] -= LEVEL_CENTERING_STRENGTH * level_mean
            mean_transfer = np.maximum(q + pool(d[nearest], donor_d[nearest]) * (
                (q + floor) / (c[nearest] + floor)
            ) ** RATIO_POWER, 0.0)
            examples.append((
                features, base_logits, _log_odds(f), effect,
                _transfer_intensity(
                    input_intensity, control_intensity, stimulated_intensity,
                    intensity_uncertainty[nearest], f,
                ),
                mean_transfer, donor_c[i, k] + donor_d[i, k],
            ))
            example_weights.append(weight)
            # Pooled cell variances stabilize finite-count pseudo-target weights.
            nc, ns = pair_counts[k]
            sampling_variance = control_variance[i] / nc + stimulated_variance[i] / ns
            prior_variance = (LOCAL_TARGET_STANDARD_ERROR / TRAIN_EXPRESSION_SCALE) ** 2
            confidence = np.maximum(
                LOCAL_TARGET_MIN_CONFIDENCE, 1.0 / (1.0 + sampling_variance / prior_variance)
            )
            target_confidences.append(confidence / confidence.mean())

    features, base_logits, input_logits, effects, intensities, means, targets = [
        torch.as_tensor(np.stack([example[k] for example in examples]), dtype=torch.float64)
        for k in range(7)
    ]
    weights = torch.as_tensor(example_weights, dtype=torch.float64)
    weights /= weights.sum()
    theta = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.03)
    for _ in range(RULE_STEPS):
        coefficients = 2.0 * torch.tanh(theta)
        reliability = torch.sigmoid(base_logits + (features * coefficients).sum(dim=-1))
        predicted = (1.0 - HURDLE_BLEND) * means + HURDLE_BLEND * (
            torch.sigmoid(input_logits + reliability * effects) * intensities
        )
        loss = ((predicted - targets).square().mean(dim=-1) * weights).sum()
        penalty = coefficients[:4].square().sum()
        penalty += LEVEL_RIDGE_MULTIPLIER * coefficients[4].square()
        loss += RULE_RIDGE * penalty / 4.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        state.transfer_coefficients.copy_(2.0 * torch.tanh(theta))

    # Joint local fitting transfers only differences between control profiles.
    # Removing the common component leaves the frozen global rule unchanged.
    distances = np.stack(profile_distances, axis=1)
    distances -= distances.min(axis=1, keepdims=True)
    local_weights = np.exp(-distances / LOCAL_RULE_TEMPERATURE)
    local_weights *= np.asarray(example_weights)[None, :]
    local_weights /= local_weights.sum(axis=1, keepdims=True)
    local_weights = torch.as_tensor(local_weights, dtype=torch.float64)
    frozen_logits = base_logits + (
        features * state.transfer_coefficients.detach()
    ).sum(dim=-1)
    gene_confidence = torch.as_tensor(np.stack(target_confidences), dtype=torch.float64)
    flattened_features = features.reshape(-1, 5)
    local_theta = torch.zeros(len(model.contexts), 5, dtype=torch.float64, requires_grad=True)
    local_optimizer = torch.optim.Adam([local_theta], lr=0.03)
    for _ in range(LOCAL_RULE_STEPS):
        raw_adjustments = LOCAL_RULE_RAW_BOUND * torch.tanh(local_theta)
        adjustments = raw_adjustments - raw_adjustments.mean(dim=0, keepdim=True)
        local_logits = (flattened_features @ adjustments.T).T.reshape(
            len(model.contexts), *base_logits.shape
        )
        reliability = torch.sigmoid(frozen_logits[None] + local_logits)
        predicted = (1.0 - HURDLE_BLEND) * means[None] + HURDLE_BLEND * (
            torch.sigmoid(input_logits[None] + reliability * effects[None]) * intensities[None]
        )
        per_profile_loss = (
            ((predicted - targets[None]).square() * gene_confidence[None]).mean(dim=-1)
            * local_weights
        ).sum(dim=-1)
        penalty = adjustments[:, :4].square().sum(dim=-1)
        penalty += LEVEL_RIDGE_MULTIPLIER * adjustments[:, 4].square()
        loss = (per_profile_loss + LOCAL_RULE_RIDGE * penalty / 4.0).mean()
        local_optimizer.zero_grad()
        loss.backward()
        local_optimizer.step()
    with torch.no_grad():
        raw_adjustments = LOCAL_RULE_RAW_BOUND * torch.tanh(local_theta)
        state.local_transfer_adjustments.copy_(
            raw_adjustments - raw_adjustments.mean(dim=0, keepdim=True)
        )


def _context_predictions(model, q, f, fractions, scale):
    """Predict means from each context and measure control compatibility."""
    state = model.model
    c = state.control_mean.detach().cpu().numpy() * scale
    d = state.response.detach().cpu().numpy() * scale
    donor_d = state.donor_response.detach().cpu().numpy() * scale
    donor_c = state.donor_control_mean.detach().cpu().numpy() * scale
    cf = state.control_rate.detach().cpu().numpy()
    sf = state.stimulated_rate.detach().cpu().numpy()
    donor_cf = state.donor_control_rate.detach().cpu().numpy()
    donor_sf = state.donor_stimulated_rate.detach().cpu().numpy()
    missing_fraction = 1.0 - fractions.sum()

    def pool(pooled, donor_values):
        matched = (fractions[None, :, None] * donor_values).sum(axis=1)
        matched += missing_fraction * pooled
        return (1.0 - DONOR_BLEND) * pooled + DONOR_BLEND * matched

    response = pool(d, donor_d)
    ratio = ((q + EXPRESSION_FLOOR) / (c + EXPRESSION_FLOOR)) ** RATIO_POWER
    mean_transfer = np.maximum(q + response * ratio, 0.0)
    cc = pool(c, donor_c)
    ss = pool(c + d, donor_c + donor_d)
    fc = pool(cf, donor_cf)
    fs = pool(sf, donor_sf)
    detection_effect = _log_odds(fs) - _log_odds(fc)
    uncertainty = state.detection_uncertainty.detach().cpu().numpy()
    beta_uncertainty = state.beta_detection_uncertainty.detach().cpu().numpy()
    uncertainty = _query_detection_uncertainty(uncertainty, beta_uncertainty, f)
    input_intensity = (q + HURDLE_PRIOR) / (f + HURDLE_PRIOR)
    control_intensity = (cc + HURDLE_PRIOR) / (fc + HURDLE_PRIOR)
    stimulated_intensity = (ss + HURDLE_PRIOR) / (fs + HURDLE_PRIOR)
    features, base_logits = _rule_features(
        detection_effect, uncertainty, f, fc, input_intensity, control_intensity
    )
    # Match feature centering in training pseudo-queries and input controls.
    level_mean = features[..., 4].mean(axis=-1, keepdims=True)
    features[..., 4] -= LEVEL_CENTERING_STRENGTH * level_mean
    features[..., 4] *= LEVEL_INFERENCE_BLEND
    coefficients = state.transfer_coefficients.detach().cpu().numpy()
    distances = _control_distances(c, cc, q)
    profile_weights = np.exp(-(distances - distances.min()) / LOCAL_RULE_TEMPERATURE)
    profile_weights /= profile_weights.sum()
    adjustments = state.local_transfer_adjustments.detach().cpu().numpy()
    coefficients = coefficients + (profile_weights[:, None] * adjustments).sum(axis=0)
    logits = base_logits + (features * coefficients).sum(axis=-1)
    reliability = 1.0 / (1.0 + np.exp(-logits))
    odds = _log_odds(f) + reliability * detection_effect
    target_rate = 1.0 / (1.0 + np.exp(-odds))
    intensity_uncertainty = state.positive_intensity_uncertainty.detach().cpu().numpy()
    hurdle_mean = target_rate * _transfer_intensity(
        input_intensity, control_intensity, stimulated_intensity, intensity_uncertainty, f
    )
    context_means = (1.0 - HURDLE_BLEND) * mean_transfer + HURDLE_BLEND * hurdle_mean

    distances = _control_distances(c, cc, q)
    rate_mismatch = np.clip(_log_odds(f) - _log_odds(fc), -6.0, 6.0) / 3.0
    intensity_mismatch = np.clip(np.log(input_intensity / control_intensity), -3.0, 3.0)
    compatibility_cost = rate_mismatch ** 2 + intensity_mismatch ** 2
    return context_means, distances, compatibility_cost, fc


def _pooled_prediction_mean(context_means, distances, compatibility_cost, fc, f):
    nearest = int(np.argmin(distances))
    cost = (distances - distances[nearest])[:, None] / CONTEXT_TEMPERATURE
    cost = cost + COMPATIBILITY_STRENGTH * compatibility_cost
    cost -= cost.min(axis=0)
    weights = np.exp(-cost)
    weights /= weights.sum(axis=0)
    pooled_mean = (weights * context_means).sum(axis=0)
    support = np.minimum(f, fc[nearest])
    strength = support ** SUPPORT_POWER
    blend = CONTEXT_BLEND * strength / (strength + COMPATIBILITY_SUPPORT ** SUPPORT_POWER)
    return context_means[nearest] + blend * (pooled_mean - context_means[nearest])


def _query_prediction_mean(model, q, f, fractions, scale):
    context_means, distances, compatibility_cost, fc = _context_predictions(
        model, q, f, fractions, scale
    )
    return _pooled_prediction_mean(context_means, distances, compatibility_cost, fc, f)


def _calibrate_cells(x, q, target_mean):
    """Preserve cell ranks while matching nonnegative target column means."""
    prediction = np.maximum(x + (target_mean - q), 0.0)
    column_mean = prediction.mean(axis=0)
    factor = np.divide(
        target_mean, column_mean, out=np.zeros_like(target_mean), where=column_mean > 0
    )
    prediction *= factor
    return prediction


def predict(model, controls_scaled, scaler, inference_seed):
    """Transfer donor responses and return cells in their supplied order."""
    if not model.is_trained:
        raise RuntimeError("The response model has not been fitted")
    x = controls_scaled.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64) * scale
    q = x.mean(axis=0)
    f = (x > 0).mean(axis=0)
    if "replicate" not in controls_scaled.obs:
        fractions = np.zeros(len(model.donors), dtype=np.float64)
        target_mean = _query_prediction_mean(model, q, f, fractions, scale)
        prediction = _calibrate_cells(x, q, target_mean)
    else:
        input_donors = np.asarray(controls_scaled.obs["replicate"].astype(str))
        aggregate_fractions = np.asarray(
            [(input_donors == donor).mean() for donor in model.donors], dtype=np.float64
        )
        aggregate_mean = _query_prediction_mean(model, q, f, aggregate_fractions, scale)
        aggregate_prediction = _calibrate_cells(x, q, aggregate_mean)
        prediction = np.empty_like(x)
        for donor in sorted(set(input_donors)):
            mask = input_donors == donor
            donor_x = x[mask]
            donor_q = donor_x.mean(axis=0)
            donor_f = (donor_x > 0).mean(axis=0)
            weight = len(donor_x) / (len(donor_x) + QUERY_DONOR_PRIOR)
            smooth_q = weight * donor_q + (1.0 - weight) * q
            smooth_f = weight * donor_f + (1.0 - weight) * f
            fractions = np.asarray(
                [float(donor == source) for source in model.donors], dtype=np.float64
            )
            smooth_target = _query_prediction_mean(model, smooth_q, smooth_f, fractions, scale)
            target_mean = np.maximum(donor_q + smooth_target - smooth_q, 0.0)
            prediction[mask] = _calibrate_cells(donor_x, donor_q, target_mean)
        support = f ** 2 / (f ** 2 + DONOR_TRANSFER_SUPPORT ** 2)
        prediction = (1.0 - support) * aggregate_prediction + support * prediction
    return np.ascontiguousarray(prediction, dtype=np.float32)
