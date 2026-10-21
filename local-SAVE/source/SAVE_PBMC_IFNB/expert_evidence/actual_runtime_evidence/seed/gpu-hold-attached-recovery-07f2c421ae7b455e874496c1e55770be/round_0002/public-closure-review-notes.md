# Closure Review Notes

## Summary
This is a bounded own-closure review of the retained best method (v67) against the public API, saved-state reload contract, and measured public Dev outcomes. No new experiments were run; the evaluator is disabled for this review phase. All conclusions are based on own measured public results only.

## Selected Best Method

**File**: `/workspace/solution/method.py` (byte-for-byte identical to `method_v67_best.py`)
**SHA256**: `2dea1252c57e8c2c6d3ff4382c8074509b6e0c3d76d326be452efc7b37ef8042`
**Dev MSE**: 0.0015913
**Dev Score**: 0.56485
**Evaluation run ID**: `d0daf2ea99aa4084ba94fe0e615f3123`
**Quality gates**: passed
**Reload verification**: confirmed (fresh + reload max abs difference = 0.0)

### Key Algorithm
A statistical cell-type reference ensemble with power-law modulation:

1. **Reference construction**: For each cell type in training data, compute per-gene control means and stimulated-expression diffs (stim_mean - ctrl_mean). Stored as two buffers in `ShiftModel` (`ctrl_means`, `stim_diffs`).

2. **Cell-type weighting**: Compute Spearman correlation between target-context control mean and each cell-type reference control profile. Convert to weights via softmax at multiple temperatures (0.05, 0.07, 0.09), with weights [0.3, 0.4, 0.3] favoring the mid-temperature.

3. **Consensus shrinkage**: Weighted cross-cell-type coefficient of variation of stim diffs (wstd / w_mean_abs) shrinks stim diffs more when cell types disagree. Strength: UP=0.5, DOWN=0.0.

4. **Power-law scaling**: Reference stim diff is scaled by `(target_ctrl / ref_ctrl) ** pow_gene`, where per-gene power depends on expression quantile of the **reference** control profile.

5. **Signed log-ratio modulation**: Per-gene power is additionally modulated by `RATIO_MOD_UP * log2(ratio)` for up-genes and `RATIO_MOD_DOWN * log2(ratio)` for down-genes. UP=0.035, DOWN=0.01.

6. **Ensemble**: 3 temperatures × 3 up-power configs (base,slope) = 9 member ensemble with uniform weights across power configs and weighted temps.

### State Dictionary / Reload Contract
- `ShiftModel` extends `nn.Module` with two registered buffers: `ctrl_means` (n_cell_types × n_genes) and `stim_diffs` (n_cell_types × n_genes).
- Both buffers are populated during `train_model()` from the training data computed in `ShiftWrapper.train()`.
- `predict()` reads only `model.model.ctrl_means` and `model.model.stim_diffs` from the model state — all learned state is in the state dict.
- Constants (temperatures, power configs, ratio mod strengths, etc.) are module-level Python variables — they are part of the source, not the state dict. This is correct because the same source file is used for both fresh training and reload.
- The `set_seed(inference_seed)` call in `predict()` ensures deterministic output across reloads.
- The reconciled results confirm `reload_max_abs_difference = 0.0` for all seeds, so the reload contract is fully satisfied.

### API Compliance
- `build_model(train_scaled, device, seed)` → returns `ShiftWrapper` with `.model` attribute (a `nn.Module`) ✓
- `train_model(model, seed)` → calls `model.train()` which populates buffers and sets `.is_trained = True` ✓
- `predict(model, controls_scaled, scaler, inference_seed)` → returns numpy float64 matrix of shape (n_cells, n_genes) ✓
- Prediction is nonnegative (clipped with `np.clip(pred_unscaled, 0, None)`) ✓
- Prediction is finite (all operations are clipped and bounded) ✓
- Checkpoint size: two float32 matrices of ~27 cell types × 2000 genes ≈ 432 KiB — well under 256 MiB ✓
- File size: 7,732 bytes — well under 1 MiB ✓

## Own Alternatives That Were Worse

All alternatives below were tested with actual public Dev evaluation. The method file for each is preserved in `/workspace/solution/`.

| Version | SHA256 (prefix) | Change | Dev MSE | Score | Δ MSE vs v67 | Verdict |
|---------|-----------------|--------|---------|-------|--------------|---------|
| v67 (best) | 2dea1252 | RATIO_MOD_UP=0.035, DOWN=0.01 | 0.001591 | 0.5648 | 0 | **RETAINED** |
| v69 | 42eb80ee | Asymmetric ratio mod: POS=0.050, NEG=0.020 (5 runs, all identical) | 0.001612 | 0.5593 | +0.000021 | WORSE |
| v68 | f8da2cc1 | RATIO_MOD_UP=0.040 | 0.001601 | 0.5623 | +0.000010 | WORSE |
| v64 | — (file: method_v64_worse.py) | RATIO_MOD_UP=0.06 | 0.001953 | 0.4659 | +0.000362 | WORSE |
| v63 | — (file: method_v63_test.py) | Per-gene regression-based power (log-log across cell types) | 0.008641 | -1.363 | +0.007050 | CATASTROPHIC |
| v62 | — (file: method_v62_worse.py) | DOWN_SHRINK_STRENGTH=0.3 | 0.001606 | 0.5609 | +0.000015 | WORSE |
| v61 | — (file: method_v61_test.py) | RATIO_MOD_DOWN=0.02 | 0.001601 | 0.5621 | +0.000010 | WORSE (no gain) |
| v60 | — (file: method_v60_test.py) | Target-ctrl quantile instead of ref-ctrl quantile for power | 0.002361 | 0.3543 | +0.000770 | CATASTROPHIC |
| v59 | 41e35a3a | RATIO_MOD_UP=0.03, DOWN=0.01 | 0.001601 | 0.5621 | +0.000010 | WORSE |
| v56 | f66fa308 | RATIO_MOD_UP=0.03, no down mod | 0.001607 | 0.5604 | +0.000016 | WORSE |
| baseline (VAE) | — anchor | Initial method (anchor B) | 0.003657 | 0.0000 | +0.002066 | WORSE |

### Why v67 Was Selected
1. **Lowest measured Dev MSE** (0.001591) among all tested variants, confirmed by run `d0daf2ea`.
2. **Clean quality gates and reload verification**: both fresh and reload produce identical output (max abs diff = 0.0).
3. **Robust peak in the ratio-modulation landscape**: RATIO_MOD_UP of 0.035 sits at the optimum between 0.03 (v59, worse), 0.04 (v68, worse), and 0.06 (v64, much worse). The peak is sharp — a ±0.005 change away from 0.035 in either direction increases MSE by ~1e-5.
4. **Down-ratio modulation at 0.01 adds small but consistent improvement** (v56→v59: 0.001607→0.001601), while 0.02 gives no further gain (v61 = v59).
5. **Ref-ctrl quantile for power calculation is correct** (v60 shows target-ctrl quantile is catastrophically wrong).
6. **Down-gene shrinkage is harmful** (v62 regression confirms DOWN_SHRINK_STRENGTH=0.0 is optimal).
7. **Per-gene regression power is catastrophically noisy** (v63), confirming the quantile-based approach is well-regularized.

## Limitations of the Measured Advantage

### 1. Single-Run Best (No Replication of the Peak)
- v67 was measured only **once** (run `d0daf2ea`). While v69 was replicated 5 times (all identical, as expected for a deterministic method) and v68 was measured once, the v67 peak itself has only one public measurement.
- However, the method is fully deterministic given training data and seed (all seeds produce identical MSE — confirmed by per-seed rows showing exactly the same value for v68/v69 runs). So seed-to-seed variability is zero for this method; the 5 seeds don't provide independent samples of method performance.
- The improvement from v59 (0.001601) to v67 (0.001591) is ~0.000010 (1e-5) in MSE. The improvement from v67 to v68 is +0.000010. This forms a clean peak with v67 at the center.

### 2. Improvement vs. Baseline Standard Deviation
- Baseline anchor B = 0.003657. The v67 MSE = 0.001591 → reduction of 0.002066 (56.5% reduction).
- However, the task specification requires that a "reference improvement must exceed three times the Baseline sample standard deviation." The baseline sample SD across 5 seeds is not directly measured by my experiments (since my method shows zero seed-to-seed variance — all seeds get identical MSE because the method computes a single mean prediction tiled across cells).
- The baseline VAE method likely has nonzero seed variance. Without measuring the baseline's per-seed variance directly, I cannot confirm whether the v67 advantage over baseline exceeds the 3σ threshold.
- The improvement from my own v56 (RATIO_MOD_UP=0.03) to v67 is only ~1e-5 in MSE, which is very small in absolute terms.

### 3. Deterministic Mean Prediction = Zero Cell-Level Variance
- The method predicts a **single mean expression vector** tiled across all cells (`np.tile(ensemble_pred, (n_cells, 1))`). This means:
  - Per-cell predictions are identical — no cell-level heterogeneity is captured.
  - The metric is **mean-expression MSE**, which only depends on the per-gene mean, so this is correct and optimal for the metric.
  - But if the evaluation metric ever changed to include cell-level structure, this method would be poorly calibrated.
  - All 5 seeds produce the same MSE because the method's output doesn't depend on training seed (the computation is deterministic given the training data, and training data is identical across seeds — only weight init changes, but this is a statistical method with no learned parameters beyond reference means).

### 4. Overfitting to Dev Context via Hyperparameter Tuning
- I performed a line search over RATIO_MOD_UP (tried 0.01, 0.03, 0.035, 0.04, 0.06) with the Dev context score as feedback.
- The peak at 0.035 may be partially overfit to the Dev context. The true optimum on the withheld evaluation context could be different.
- The landscape is relatively flat between 0.03 and 0.04 (MSE changes by only ~1e-5), so even if the true optimum differs, the loss should be small.
- However, v64 (RATIO_MOD_UP=0.06) shows a large regression (MSE=0.001953), confirming that going too far is damaging. The risk of overshooting is asymmetric.

### 5. Cell Type Number and Composition Sensitivity
- The method relies on cell type labels in `adata.obs['cell_type']`. If the withheld evaluation context has cell types not present in training, the Spearman-correlation-based weighting will still work (it's a soft matching), but accuracy may degrade.
- The number of cell types affects the reference diversity — more cell types = richer reference but potentially noisier.
- The power law parameters (base, slope) were tuned on the Dev context and may not transfer perfectly.

### 6. Up-Power Config Grid Is Coarse
- Three up-power configs are used: (0.80,-0.55), (1.00,-0.65), (1.20,-0.85).
- These were not independently fine-tuned; the grid spacing of 0.20 in base power is coarse. Finer grid search might yield small additional gains, but the improvement would likely be within the same order as the ratio-mod tuning (~1e-5 MSE).

### 7. No Cell-Type-Specific or Gene-Cluster-Specific Modulation
- RATIO_MOD_UP and RATIO_MOD_DOWN are global scalar parameters.
- Different gene modules or cell types might benefit from different modulation strengths.
- However, v63 showed that per-gene regression-based power is catastrophically noisy, suggesting that gene-level specialization requires strong regularization.
- A middle ground (e.g., gene-cluster-level or cell-type-level parameters) might work but wasn't tested.

### 8. Memory and Timing Are Comfortable
- Per-run peak RSS: ~1.15 GiB — well within 32 GiB limit.
- Training is near-instant (just computing mean vectors per cell type).
- Checkpoint: ~432 KiB — well within 256 MiB.
- This leaves substantial headroom for more complex methods, but my attempts at more complexity (v63 per-gene regression, v60 target-ctrl quantile) all failed.

## Reproducibility and Saved State
- **Determinism**: The method is fully deterministic given the training data. All 5 seeds produce identical predictions and therefore identical MSE.
- **Reload accuracy**: `reload_max_abs_difference = 0.0` for all seeds — exact bitwise agreement, well within the 1e-7 tolerance.
- **State dict keys**: Only `ctrl_means` and `stim_diffs` buffers — no parameters that depend on initialization seed.
- **Reproduction steps**: Running `bash /workspace/solution/solve.sh` with the v67 source should produce MSE=0.001591 on Dev.

## What Future Evaluators Should Know
1. The method's predictions are **cell-invariant** — every cell gets the same predicted expression vector. This is optimal for the mean-expression MSE metric but means the method provides no cell-level heterogeneity.
2. All seeds produce identical results. The formal seed set of 5 provides no additional statistical power for this method; one seed would be sufficient.
3. The measured improvement over the baseline anchor is large (56.5% MSE reduction), but the incremental improvements from my hyperparameter tuning are very small (~1e-5 MSE steps).
4. The method is simple, fast, and robust — unlikely to fail on the hidden evaluation context due to runtime issues.
5. The main risk is **Dev overfitting** from hyperparameter tuning (ratio mod strength, power configs, shrinkage strengths). The withheld context may have a slightly different optimal ratio mod strength.
6. V69 (asymmetric ratio modulation) being worse than symmetric v67 suggests that the log-ratio modulation's linear approximation is already adequate — splitting it into pos/neg branches adds noise without benefit.

## File Inventory
- `method.py` — retained best (v67), SHA256: `2dea1252...`
- `method_v67_best.py` — backup of best
- `method_v69_test.py` — worse asymmetric variant (5 runs confirmed)
- `method_v68_test.py` — worse (RATIO_MOD_UP=0.04)
- `method_v64_worse.py` — much worse (RATIO_MOD_UP=0.06)
- `method_v63_test.py` — catastrophic (per-gene regression power)
- `method_v60_test.py` — catastrophic (target-ctrl quantile)
- `reconciled-own-dev-results.json` — actual public measurements
- `research-notes.md` — detailed experiment history
