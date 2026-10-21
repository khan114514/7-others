# Research Journal: Conditional Traffic Forecasting

## Baseline (identity method)
- **Method**: Returns base forecasts unchanged
- **Dev MAE**: 15.4219
- **Score**: 0.000
- **Date**: 2026-10-06

## Experiment 1: Shrunken per-node-horizon bias + anchor correction
- **Method**: 
  - Computes per-node-horizon additive bias from calibration data
  - Shrinks each bias toward the global per-horizon mean (alpha=0.3)
  - Adds small anchor-based correction growing linearly with horizon (scale=0.03)
  - Clips to non-negative
- **Rationale**: Base forecasts have ~2 unit negative bias overall, with per-node and per-horizon structure. Shrinkage prevents overfitting given only 74 calibration samples. Anchor provides complementary information especially for later horizons.
- **Cross-validation (5-fold)**: ~14.80 MAE (baseline ~14.98)
- **Dev MAE**: 15.2123
- **Score**: 0.4428
- **Result**: SUCCESS - improvement of ~0.21 MAE over baseline
- **Next**: Tune shrinkage alpha and anchor scale; try graph smoothing; try per-node bias with better regularization; try regression-based approaches.
