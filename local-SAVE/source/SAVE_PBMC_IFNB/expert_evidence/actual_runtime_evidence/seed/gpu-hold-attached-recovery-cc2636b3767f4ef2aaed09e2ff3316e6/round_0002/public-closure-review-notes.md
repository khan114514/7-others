# Closure Review Notes

## Summary
This is a bounded own-closure review of the retained best method (v67) against the public API, saved-state reload contract, and measured public Dev outcomes. No new experiments were run; the evaluator is disabled for this review phase. All conclusions are based on own measured public results only.

**Evidence tier principle**: Claims about method performance are classified by how directly they are supported by reconciled public measurements. See "Evidence Tiers" below.

## Evidence Tiers

| Tier | Source | Meaning |
|------|--------|---------|
| **Tier 1 — Direct reconciled public** | `reconciled-own-dev-results.json` → `actual_public_results_only[]` | Directly measured by public Dev in this session, with run IDs and quality gate status. Highest confidence. |
| **Tier 2 — Prior-session best** | `reconciled-own-dev-results.json` → `previously_selected_own_best` | Recorded as prior session's best, with an evaluation run ID, but not replicated in `actual_public_results_only[]` of this reconciliation. Used to select the retained method. |
| **Tier 3 — Local dev_result.json** | `/workspace/solution/dev_result.json` | Locally stored full result from a Dev run. Includes per-seed details, RSS, checkpoint hash, etc. |
| **Tier 4 — Research notes only** | `research-notes.md` + file hash comparisons | Results documented in research notes and verified by file identity (e.g., identical hashes imply identical results). No independent JSON record of the run. |
| **Tier 5 — Recovered earliest** | `recovered-own-dev.json` | Earliest recovered best from session restoration. Older record, lower provenance. |

## Selected Best Method

**File**: `/workspace/solution/method.py` (byte-for-byte identical to `method_v67_best.py`)
**SHA256**: `2dea1252c57e8c2c6d3ff4382c8074509b6e0c3d76d326be452efc7b37ef8042`
**Dev MSE**: 0.0015913
**Dev Score**: 0.56485
**Evaluation run ID**: `d0daf2ea99aa4084ba94fe0e615f3123`
**Evidence tier**: **Tier 2** (in `previously_selected_own_best`; NOT in `actual_public_results_only[]`)
**Quality gates**: passed (per prior record)
**Reload verification**: confirmed per prior record (reload max abs difference = 0.0)

**IMPORTANT CAVEAT**: v67 is the retained best based on the `previously_selected_own_best` record, which comes from a prior session. It was NOT re-measured in the reconciled `actual_public_results_only` array. The only method with Tier-1 direct reconciled measurement in this session is v69 (asymmetric ratio mod, MSE=0.001612, WORSE than v67). The selection of v67 as best relies on the prior-session record being accurate and the v69 result confirming that moving away from v67's ratio-mod parameters is detrimental.

### Key Algorithm
A statistical cell-type reference ensemble with power-law modulation:

1. **Reference construction**: For each cell type in training data, compute per-gene control means and stimulated-expression diffs (stim_mean - ctrl_mean). Stored as two buffers in `ShiftModel` (`ctrl_means`, `stim_diffs`).

2. **Cell-type weighting**: Compute Spearman correlation between target-context control mean and each cell-type reference control profile. Convert to weights via softmax at multiple temperatures (0.05, 0.07, 0.09), with weights [0.3, 0.4, 0.3] favoring the mid-temperature.

3. **Consensus shrinkage**: Weighted cross-cell-type coefficient of variation of stim diffs (wstd / w_mean_abs) shrinks stim diffs more when cell types disagree. Strength: UP=0.5, DOWN=0.0.

4. **Power-law scaling**: Reference stim diff is scaled by `(target_ctrl / ref_ctrl) ** pow_gene`, where per-gene power depends on expression quantile of the **reference** control profile.

5. **Signed log-ratio modulation**: Per-gene power is additionally modulated by `RATIO_MOD_UP * log2(ratio)` for up-genes and `RATIO_MOD_DOWN * log2(ratio)` for down-genes. UP=0.035, DOWN=0.01.

6. **Ensemble**: 3 temperatures × 3 up-power configs (base,slope) = 9 member ensemble with uniform weights across power configs and weighted temps.

7. **Output**: Single mean vector tiled across all cells — optimal for the mean-MSE metric since only the per-gene mean is scored.

### Docstring Note (Cosmetic)
The module docstring says "Method v59" but the constant `RATIO_MOD_UP = 0.035` is the v67 value (v59 had RATIO_MOD_UP=0.03). This is a cosmetic label mismatch only; the actual code constants define the method behavior.

### State Dictionary / Reload Contract
- `ShiftModel` extends `nn.Module` with two registered buffers: `ctrl_means` (n_cell_types × n_genes) and `stim_diffs` (n_cell_types × n_genes). Both are `torch.float32`.
- Both buffers are populated during `train_model()` from training data computed in `ShiftWrapper.train()`.
- `predict()` reads only `model.model.ctrl_means` and `model.model.stim_diffs` from the model state — all learned state is in the state dict.
- Constants (temperatures, power configs, ratio mod strengths, etc.) are module-level Python variables — they are part of the source, not the state dict. This is correct because the same source file is used for both fresh training and reload.
- The `set_seed(inference_seed)` call in `predict()` ensures deterministic output across reloads.
- **Tier-1 evidence** (v69, 5 runs): `independent_fresh_reload_verified = true` for all 5 runs, confirming the reload pattern works for this architecture.
- **Tier-3 evidence** (v68, dev_result.json): `reload_max_abs_difference = 0.0` for all 5 seeds — exact bitwise agreement.
- Since v67 differs from v68 by only the `RATIO_MOD_UP` constant (0.035 vs 0.040), and the reload contract is source-code invariant (constants are in source, state is in buffers), v67's reload contract is satisfied with high confidence.

### API Compliance (verified against `/tests/train_eval.py`)
- `build_model(train, device, seed)` → returns `ShiftWrapper` with `.model` attribute (a `nn.Module`) ✓
- `train_model(model, seed)` → calls `model.train()` which populates buffers and sets `.is_trained = True` ✓
- `predict(model, controls, scaler, inference_seed)` → returns numpy float64 matrix of shape (n_cells, n_genes) ✓
- Prediction is nonnegative: `np.clip(pred_unscaled, 0, None)` at end of predict ✓
- Prediction is finite: all operations are on mean vectors with clipping; `RATIO_CLIP_MAX = 60.0` bounds ratios ✓
- Checkpoint size: two float32 matrices of ~27 cell types × 2000 genes ≈ 432 KiB — well under 256 MiB ✓
- File size: 7,732 bytes — well under 1 MiB ✓
- No subprocess calls, no network, no file writes during candidate execution ✓
- Seed handling: `torch.manual_seed(seed)` and `np.random.seed(seed)` are set by the executor before `build_model`; `set_seed(inference_seed)` is called in `predict` ✓

## Own Alternatives and Their Evidence

### Directly Reconciled Public Results (Tier 1)

| Method SHA256 (prefix) | Description | Dev MSE | Score | Runs | Quality Gates | Reload | Verdict vs v67 |
|------------------------|-------------|---------|-------|------|---------------|--------|----------------|
| `42eb80ee` (v69) | Asymmetric ratio mod: POS=0.050, NEG=0.020 | 0.001612 | 0.5593 | 5 (all identical) | PASSED | VERIFIED | WORSE (+0.000021 MSE) |

**Note**: v69 is the ONLY method with Tier-1 reconciled evidence in this session. All 5 runs are byte-identical (same MSE, same checkpoint hash), confirming full determinism.

### Local dev_result.json (Tier 3)

| Method SHA256 (prefix) | Description | Dev MSE | Score | Seeds | Quality Gates | Reload | Verdict vs v67 |
|------------------------|-------------|---------|-------|-------|---------------|--------|----------------|
| `f8da2cc1` (v68) | RATIO_MOD_UP=0.040 (vs v67's 0.035) | 0.001601 | 0.5623 | 5 (all identical) | PASSED | VERIFIED (0.0) | WORSE (+0.000009 MSE) |

### Recovered Earliest (Tier 5)

| Method SHA256 (prefix) | Description | Dev MSE | Score | Verdict vs v67 |
|------------------------|-------------|---------|-------|----------------|
| `f66fa308` (v56) | RATIO_MOD_UP=0.03, no down-ratio mod | 0.001607 | 0.5604 | WORSE (+0.000016 MSE) |

### Research-Notes + File-Hash Evidence (Tier 4)

| Version | SHA256 (prefix) | Change | Dev MSE (from notes) | Verdict vs v67 | Notes |
|---------|-----------------|--------|---------------------|----------------|-------|
| v64 | `44836c60` | RATIO_MOD_UP=0.06 | 0.001953 | WORSE (+0.000362) | Large regression |
| v63 | `dce83827` | Per-gene regression-based power | 0.008641 | CATASTROPHIC | Noisy estimates |
| v62 | `41e35a3a` | DOWN_SHRINK_STRENGTH=0.3 | 0.001606 | WORSE (+0.000015) | v62 file is byte-identical to v59/v61/v65/v66 — research-notes label may have hash confusion |
| v61 | `41e35a3a` | RATIO_MOD_DOWN=0.02 | 0.001601 | WORSE (same as v59) | File is byte-identical to v59 — no-change rerun |
| v60 | `03de6eb5` | Target-ctrl quantile for power | 0.002361 | CATASTROPHIC | Wrong reference |
| v59 | `41e35a3a` | RATIO_MOD_UP=0.03, DOWN=0.01 | 0.001601 | WORSE (+0.000010) | Baseline for v67 improvement |

### Duplicate / No-Change Reruns (Important Finding)
The following files are **byte-identical** (SHA256 `41e35a3a`):
- `method_v59_confirmed_best.py`
- `method_v61_test.py`
- `method_v65_test.py`
- `method_v66_test.py`
- `method_v62_worse.py`

This means v61, v65, v66, and v62 are all the same method as v59. Several experiment iterations were no-change reruns or had labeling mismatches between file name and content. The actual distinct methods tested are fewer than the version numbering suggests.

### Version Progression (Distinct Methods Only)
1. **v56** — RATIO_MOD_UP=0.03, no down modulation (tier 5, recovered)
2. **v59** — Add RATIO_MOD_DOWN=0.01 (tier 4, research notes)
3. **v60** — Target-ctrl quantile (catastrophic, tier 4)
4. **v63** — Per-gene regression power (catastrophic, tier 4)
5. **v64** — RATIO_MOD_UP=0.06 (worse, tier 4)
6. **v67** — RATIO_MOD_UP=0.035 (best, tier 2)
7. **v68** — RATIO_MOD_UP=0.040 (worse, tier 3)
8. **v69** — Asymmetric POS/NEG ratio mod (worse, tier 1)

## Limitations of the Measured Advantage

### 1. Evidence Tier Gap for v67
The selected best method (v67) is only in `previously_selected_own_best`, not in the directly reconciled `actual_public_results_only[]`. While v69 (worse, tier 1) and v68 (worse, tier 3) bracket v67's RATIO_MOD_UP parameter, and the ordering is consistent (0.035 is best, 0.040 worse, 0.050/0.020 split worse), v67 itself has not been independently re-verified in the current reconciled session. The 0.001591 MSE value should be treated with appropriate caution pending re-verification.

### 2. Single-Context Optimization
All hyperparameter tuning (ratio mod strength, power configs, shrinkage strengths) was done against the single public Dev context. With only one Dev context, there is a real risk of overfitting to Dev-specific properties. The withheld evaluation context may have different cell-type composition, stimulation strength, or expression distribution that shifts the optimal hyperparameters.

### 3. Small Incremental Improvements
The improvement from v59 (MSE=0.001601) to v67 (MSE=0.001591) is only ~1e-5 MSE, roughly 0.6% relative improvement. This is very small compared to the ~56% MSE reduction from the baseline anchor (0.003657). The ratio-modulation tuning is a minor optimization on top of a strong core method.

### 4. Deterministic Method, Zero Seed Variance
All 5 seeds produce identical MSE because the method is fully deterministic given the training data (it computes reference mean vectors — no learned parameters with random initialization). This means the formal seed set provides no statistical power for this method; the sample standard deviation across seeds is exactly 0. The task's "3× baseline SD" significance criterion is undefined for a method with zero variance.

### 5. Cell-Invariant Predictions
The method outputs the same predicted expression vector for every cell (tiled mean). This is mathematically optimal for the mean-expression MSE metric, but it means the method provides no cell-level heterogeneity. If the evaluation metric or task structure changes to reward per-cell accuracy, this method would be poorly calibrated.

### 6. Cell Type Dependency
The method relies on `adata.obs['cell_type']` labels in the training data. If the withheld evaluation context has cell types not present in training, the Spearman-correlation-based soft matching still works but accuracy may degrade. The number and diversity of training cell types directly affects reference quality.

### 7. Coarse Power Config Grid
Three up-power configs are used: (0.80, -0.55), (1.00, -0.65), (1.20, -0.85). These were not independently fine-tuned; the grid spacing is coarse. Finer grid search might yield additional gains, but likely of similar magnitude to the ratio-mod tuning (~1e-5 MSE).

### 8. No Cell-Type-Specific or Gene-Cluster Modulation
RATIO_MOD_UP and RATIO_MOD_DOWN are global scalar parameters. Different gene modules or cell types might benefit from different modulation strengths. However, v63 (per-gene regression-based power) was catastrophically noisy, suggesting that gene-level specialization requires strong regularization. A middle ground (gene-cluster-level or cell-type-level parameters) might work but wasn't tested.

## Reproducibility and Saved State
- **Determinism**: The method is fully deterministic given training data. All 5 seeds produce identical predictions and therefore identical MSE. Confirmed by Tier-1 (v69, 5 runs identical), Tier-3 (v68, 5 seeds identical), and Tier-4 (v59 and variants) evidence.
- **Reload accuracy**: Exact bitwise agreement (0.0 max abs difference). Confirmed for v69 (tier 1, 5 runs) and v68 (tier 3, 5 seeds). Architecture-wise, v67 reload is guaranteed correct (same pattern — only constants differ).
- **State dict keys**: Only `ctrl_means` and `stim_diffs` buffers — no parameters that depend on initialization seed.
- **Checkpoint**: ~432 KiB — well within 256 MiB.
- **Memory**: Peak RSS ~1.15 GiB — well within 32 GiB.
- **Runtime**: ~115 seconds total for 5 seeds — well within 3,600 seconds.
- **Reproduction**: Running `bash /workspace/solution/solve.sh` with the v67 source should produce MSE≈0.001591 on Dev (per tier-2 record).

## API Contract Verification Details
Verified against `/tests/train_eval.py` (the executor):

| Contract Point | Code Location | Status |
|----------------|---------------|--------|
| `build_model(train_scaled, device, seed)` callable | `method.py:121` | ✓ |
| Returned object has `.model` (nn.Module) | `ShiftWrapper.__init__`, line 96 | ✓ |
| `train_model(model, seed)` callable | `method.py:125` | ✓ |
| `predict(model, controls_scaled, scaler, inference_seed)` callable | `method.py:202` | ✓ |
| Returns numpy-compatible matrix | `.astype(np.float64)` at end | ✓ |
| Nonnegative predictions | `np.clip(pred_unscaled, 0, None)` | ✓ |
| Finite predictions | All ops bounded; ratio clipped at 60.0 | ✓ |
| Shape matches controls | `np.tile(..., (n_cells, 1))` | ✓ |
| `.model.state_dict()` contains all learned state | Only `ctrl_means` and `stim_diffs` buffers | ✓ |
| `load_state_dict(..., weights_only=True, strict=True)` works | Both are registered buffers | ✓ |
| `.is_trained = True` set after reload | Executor sets it; method checks nothing | ✓ |
| `inference_seed = seed + 10000` | Executor passes `a.seed+10000` | ✓ |
| MaxAbsScaler inverse transform | `scaler.inverse_transform(pred_scaled)` | ✓ |

## What Future Evaluators Should Know
1. The method's predictions are **cell-invariant** — every cell gets the same predicted expression vector. This is optimal for the mean-expression MSE metric but means the method provides no cell-level heterogeneity.
2. All seeds produce identical results. The formal seed set of 5 provides no additional statistical power for this method; one seed is sufficient.
3. The measured improvement over the baseline anchor is large (56.5% MSE reduction), but the incremental improvements from hyperparameter tuning are very small (~1e-5 MSE steps).
4. The method is simple, fast, and robust — unlikely to fail on the hidden evaluation context due to runtime issues.
5. The main risk is **Dev overfitting** from hyperparameter tuning (ratio mod strength, power configs, shrinkage strengths). The withheld context may have a slightly different optimal ratio mod strength.
6. v69 (asymmetric ratio modulation) being worse than symmetric v67 suggests that the log-ratio modulation's linear approximation is already adequate — splitting it into pos/neg branches adds noise without benefit.
7. The v67 result comes from a prior session's best record (tier 2 evidence). It is bracketed by v68 (0.040, worse) and v59 (0.03, worse), consistent with 0.035 being the optimum, but direct re-verification would strengthen confidence.

## File Inventory
- `method.py` — retained best (v67), SHA256: `2dea1252...`
- `method_v67_best.py` — backup of best (byte-identical to method.py)
- `method_v69_test.py` — worse asymmetric variant (5 tier-1 runs confirmed)
- `method_v68_test.py` — worse (RATIO_MOD_UP=0.04, tier-3 via dev_result.json)
- `method_v64_worse.py` — much worse (RATIO_MOD_UP=0.06, tier 4)
- `method_v63_test.py` — catastrophic (per-gene regression power, tier 4)
- `method_v62_worse.py` — labeled worse, but hash `41e35a3a` = v59 (duplicate)
- `method_v61_test.py` — hash `41e35a3a` = v59 (no-change rerun)
- `method_v60_test.py` — catastrophic (target-ctrl quantile, tier 4)
- `method_v59_confirmed_best.py` — v59 baseline for v67 (tier 4)
- `method_v65_test.py` — hash `41e35a3a` = v59 (no-change rerun)
- `method_v66_test.py` — hash `41e35a3a` = v59 (no-change rerun)
- `reconciled-own-dev-results.json` — actual public measurements (source of truth)
- `dev_result.json` — v68 detailed per-seed results (tier 3)
- `recovered-own-dev.json` — v56 earliest recovered (tier 5)
- `research-notes.md` — experiment history notes
- `closure-review-notes.md` — this document
