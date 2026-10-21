# Research Notes: Hierarchical Transport Valuation Speedup

## Goal
Reduce per-case median method time while maintaining NRMSE ≤ 0.01, Spearman ≥ 0.999, top-quarter overlap ≥ 0.98.

## Baseline (canonical)
- B = 0.5215s aggregate (from protocol.json) — formal baseline
- Local per-case (measured): ~0.7-2.1s across 30 dev cases
- Raw metric aggregate (from evaluator): ~1.04s (canonical method.py)

## Architecture
- 3 candidate batches × 2 reference batches = 6 batch pairs (for N=3000, M=2000, batch_size=1024)
- Per pair: POT sinkhorn for batchcost (~1s slow) + GeomLoss potentials for F (~50ms fast)
- Label cost: 190 GeomLoss calls (~0.7s)
- Q: 3×2 OT Sinkhorn (negligible)

## Key Findings

### H1: POT sinkhorn is the bottleneck
- POT sinkhorn per pair: ~0.2-1.0s (6 pairs = 1-6s total)
- GeomLoss potentials per pair: ~50-100ms
- Label cost: ~0.7s
- Total canonical: ~4-6s first run, ~3s steady state

### H2: GeomLoss at matched reg for batchcost
- POT effective reg = 0.1 * C_max (because C is normalized)
- GeomLoss default reg = 0.1 (no normalization) ≠ POT effective reg
- Use GeomLoss with blur = sqrt(0.1 * C_max) to match effective reg
- GeomLoss matched reg: ~23ms per pair (vs POT ~1000ms per pair) → ~40x faster

### H3: v10 (wrong: mismatched reg) failed quality gates
- v10 used default GeomLoss reg for batchcost → wrong Q → NRMSE ~0.9
- Root cause: GeomLoss reg (0.1) ≠ POT effective reg (0.1*C_max ≈ 2000)

### H4: v12 (matched reg without refinement) partially works
- Rank correlation of batchcost vs canonical: 0.71-1.0 across cases
- Q diff: up to 0.08 → NRMSE up to 9% on some cases → FAILS quality gate
- Issue: GeomLoss epsilon-scaling gives approximate potentials

### H5: v13 (matched reg + log-domain refinement) — under evaluation
- GeomLoss matched reg + 5 log-domain Sinkhorn iters
- Expected: batchcost accurate to ~1e-9 → Q matches → NRMSE < 1%
- Speed: ~2x faster than canonical (eliminates POT, adds GeomLoss + 5 log iters)

### H6: Q-level Sinkhorn reg sensitivity
- Q (3×2) with reg=0.01 is very sensitive to batchcost errors
- 1% batchcost error → 10%+ Q error at reg=0.01
- Higher reg (0.1-0.2) → less sensitive but further from canonical Q

## Version History
- v0: canonical (POT + GeomLoss, ~3s per case) — PASSES quality
- v10: single GeomLoss solve per pair, mismatched reg — FAILS quality (NRMSE ~0.9)
- v12: two GeomLoss solves (F + matched-reg batchcost) — FAILS quality (NRMSE up to 9%)
- v13: two GeomLoss solves + log-domain refinement for batchcost — EVALUATING

## Speed Estimates
- Canonical: ~3-4s per case (6 pairs × ~0.5s POT + ~0.05s GeomLoss each + 0.7s label)
- v13: ~1.5-2s per case (6 pairs × ~0.1s GeomLoss + ~0.4s log-refine + 0.7s label)
- Expected speedup: ~2x
- Label cost (~0.7s) is the next biggest target

## Next Steps
1. Evaluate v13 with official evaluator
2. If v13 passes: optimize further (accelerate label cost, reduce refinement iters)
3. If v13 fails: increase refinement iters or increase Q reg
4. Target: label cost acceleration (POT for small pairs, or simplified distance)
