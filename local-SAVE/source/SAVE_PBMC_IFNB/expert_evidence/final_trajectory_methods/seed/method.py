"""Method v59: Signed log-ratio power modulation for BOTH up and down genes.

v39 baseline: Dev MSE = 0.001871, Score = 0.4882
v56 baseline: Dev MSE = 0.001607, Score = 0.5604
v59: Add RATIO_MOD_DOWN = 0.01 on top of v56 best (RATIO_MOD_UP=0.03)
  - Isolates effect of down-ratio modulation from up-ratio reduction
"""
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ENSEMBLE_TEMPS = [0.05, 0.07, 0.09]
TEMP_WEIGHTS = [0.3, 0.4, 0.3]

UP_POWER_CONFIGS = [
    (0.80, -0.55),
    (1.00, -0.65),
    (1.20, -0.85),
]
UP_POWER_WEIGHTS = [1.0/3.0, 1.0/3.0, 1.0/3.0]

BASE_DOWN_POW = 1.40
SLOPE_DOWN_POW = -0.75
RATIO_CLIP_MAX = 60.0
UP_SHRINK_STRENGTH = 0.5
DOWN_SHRINK_STRENGTH = 0.0
UP_POW_CLIP_MAX = 2.0
RATIO_MOD_UP = 0.035
RATIO_MOD_DOWN = 0.01
DOWN_POW_CLIP_MAX = 3.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rankdata(x):
    """Fast rank data (average ranks for ties)."""
    sorter = np.argsort(x, kind='mergesort')
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    sorted_x = x[sorter]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[i:j] = avg_rank
        i = j
    out = ranks[inv]
    return out


def _spearman_corr(x, y):
    """Spearman rank correlation between two 1D arrays."""
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = rx.mean()
    my = ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx)**2) * np.sum((ry - my)**2))
    if den < 1e-12:
        return -1.0
    return num / den


class ShiftModel(nn.Module):
    def __init__(self, n_genes, n_cell_types):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.register_buffer('ctrl_means', torch.zeros(n_cell_types, n_genes))
        self.register_buffer('stim_diffs', torch.zeros(n_cell_types, n_genes))

    def forward(self, x):
        return x


class ShiftWrapper:
    def __init__(self, adata, device, seed):
        self.adata = adata
        self.device = device
        self.seed = seed
        self.is_trained = False
        self.cell_types = sorted(adata.obs['cell_type'].unique())
        self.n_genes = adata.shape[1]
        self.model = ShiftModel(self.n_genes, len(self.cell_types))

    def train(self, **kwargs):
        adata = self.adata
        n_ct = len(self.cell_types)
        ctrl_means = np.zeros((n_ct, self.n_genes), dtype=np.float32)
        stim_diffs = np.zeros((n_ct, self.n_genes), dtype=np.float32)

        for i, ct in enumerate(self.cell_types):
            mask_ct = adata.obs['cell_type'] == ct
            ctrl_mask = mask_ct & (adata.obs['condition'] == 'control')
            stim_mask = mask_ct & (adata.obs['condition'] == 'stimulated')
            if hasattr(adata.X, 'toarray'):
                ctrl_mean = np.asarray(adata[ctrl_mask].X.mean(axis=0)).flatten()
                stim_mean = np.asarray(adata[stim_mask].X.mean(axis=0)).flatten()
            else:
                ctrl_mean = adata[ctrl_mask].X.mean(axis=0)
                stim_mean = adata[stim_mask].X.mean(axis=0)
            ctrl_means[i] = ctrl_mean.astype(np.float32)
            stim_diffs[i] = (stim_mean - ctrl_mean).astype(np.float32)

        self.model.ctrl_means = torch.from_numpy(ctrl_means).to(self.device)
        self.model.stim_diffs = torch.from_numpy(stim_diffs).to(self.device)
        self.is_trained = True


def build_model(train, device, seed):
    return ShiftWrapper(adata=train, device=device, seed=seed)


def train_model(model, seed):
    model.train()


def _predict_single_config(target_ctrl_mean, ctrl_means_np, stim_diffs_np,
                           temperature, base_up_pow, slope_up_pow):
    """Compute predicted mean expression in scaled space for a single config."""
    n_ct = ctrl_means_np.shape[0]

    corrs = np.zeros(n_ct, dtype=np.float64)
    for i in range(n_ct):
        c = _spearman_corr(target_ctrl_mean, ctrl_means_np[i])
        corrs[i] = c if np.isfinite(c) else -1.0

    logits = corrs / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    ref_ctrl = np.zeros_like(target_ctrl_mean)
    ref_diff = np.zeros_like(target_ctrl_mean)
    for i in range(n_ct):
        ref_ctrl += weights[i] * ctrl_means_np[i]
        ref_diff += weights[i] * stim_diffs_np[i]

    w = weights.reshape(-1, 1)
    wmean = (w * stim_diffs_np).sum(axis=0)
    wvar = (w * (stim_diffs_np - wmean.reshape(1, -1)) ** 2).sum(axis=0)
    wstd = np.sqrt(np.maximum(wvar, 0))
    w_mean_abs = (w * np.abs(stim_diffs_np)).sum(axis=0)
    w_ratio = np.where(w_mean_abs > 1e-10, wstd / np.maximum(w_mean_abs, 1e-10), 0)

    shrink = np.ones_like(w_ratio)
    up_mask_diff = ref_diff > 0
    down_mask_diff = ref_diff < 0
    shrink[up_mask_diff] = 1.0 / (1.0 + UP_SHRINK_STRENGTH * w_ratio[up_mask_diff])
    shrink[down_mask_diff] = 1.0 / (1.0 + DOWN_SHRINK_STRENGTH * w_ratio[down_mask_diff])
    ref_diff = ref_diff * shrink

    ratio = np.where(
        ref_ctrl > 1e-8,
        target_ctrl_mean / np.maximum(ref_ctrl, 1e-8),
        1.0
    )
    ratio = np.clip(ratio, 1.0 / RATIO_CLIP_MAX, RATIO_CLIP_MAX)

    sorted_vals = np.sort(ref_ctrl)
    ranks = np.searchsorted(sorted_vals, ref_ctrl)
    n_genes = len(ref_ctrl)
    quantile = ranks / (n_genes - 1)

    up_pow_gene = base_up_pow + slope_up_pow * quantile

    # v53: signed log-ratio modulation for up-genes
    # ratio > 1 → log2 positive → increases power
    # ratio < 1 → log2 negative → decreases power
    log2_ratio = np.log2(np.maximum(ratio, 1e-12))
    up_pow_gene = up_pow_gene + RATIO_MOD_UP * log2_ratio

    up_pow_gene = np.clip(up_pow_gene, 0.0, UP_POW_CLIP_MAX)

    down_pow_gene = BASE_DOWN_POW + SLOPE_DOWN_POW * quantile
    down_pow_gene = down_pow_gene + RATIO_MOD_DOWN * log2_ratio
    down_pow_gene = np.clip(down_pow_gene, 0.0, DOWN_POW_CLIP_MAX)

    up_mask = ref_diff > 0
    down_mask = ref_diff < 0

    scaled_diff = np.zeros_like(ref_diff)
    scaled_diff[up_mask] = ref_diff[up_mask] * (ratio[up_mask] ** up_pow_gene[up_mask])
    scaled_diff[down_mask] = ref_diff[down_mask] * (ratio[down_mask] ** down_pow_gene[down_mask])

    pred_mean_scaled = target_ctrl_mean + scaled_diff
    pred_mean_scaled = np.clip(pred_mean_scaled, 0, 1)
    return pred_mean_scaled


def predict(model, controls, scaler, inference_seed):
    set_seed(inference_seed)

    ctrl_means_np = model.model.ctrl_means.cpu().numpy()
    stim_diffs_np = model.model.stim_diffs.cpu().numpy()

    if hasattr(controls.X, 'toarray'):
        target_ctrl_mean = np.asarray(controls.X.mean(axis=0)).flatten()
    else:
        target_ctrl_mean = controls.X.mean(axis=0)

    ensemble_pred = np.zeros_like(target_ctrl_mean)
    total_weight = 0.0

    for temp, temp_w in zip(ENSEMBLE_TEMPS, TEMP_WEIGHTS):
        for (base_up, slope_up), up_w in zip(UP_POWER_CONFIGS, UP_POWER_WEIGHTS):
            pred = _predict_single_config(
                target_ctrl_mean, ctrl_means_np, stim_diffs_np,
                temp, base_up, slope_up
            )
            w = temp_w * up_w
            ensemble_pred += w * pred
            total_weight += w

    ensemble_pred /= total_weight

    n_cells = controls.shape[0]
    pred_scaled = np.tile(ensemble_pred.reshape(1, -1), (n_cells, 1))
    pred_unscaled = scaler.inverse_transform(pred_scaled)
    pred_unscaled = np.clip(pred_unscaled, 0, None)

    return pred_unscaled.astype(np.float64)
