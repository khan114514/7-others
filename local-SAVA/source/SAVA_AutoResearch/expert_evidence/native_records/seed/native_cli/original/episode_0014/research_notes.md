# Research Notes: Hierarchical Transport Valuation Speedup

## Goal
Reduce per-case median method time while maintaining NRMSE ≤ 0.01, Spearman ≥ 0.999, top-quarter overlap ≥ 0.98.

## Baseline (canonical)
- B = 0.5215s aggregate (from protocol.json)
- My local single-case measurement: ~5.7s for N=3000, M=2000
  - 3 candidate batches (1024, 1024, 952) × 2 reference batches (1024, 976) = 6 batch pairs
  - Label cost: ~0.8-1.0s (190 pairs of ~100 samples each)
  - Per pair: POT sinkhorn ~0.5-0.7s + GeomLoss potentials ~0.13s
  - 6 × (0.6 + 0.13) ≈ 4.4s + label cost ≈ 5.4s total ≈ matches measurement

## Key Optimization Hypotheses

### H1: Unify dual Sinkhorn solves
Canonical solves Sinkhorn twice per batch pair:
- POT `sinkhorn()` for batch-level cost (for inter-batch transport Q)
- GeomLoss `SamplesLoss(potentials=True)` for point-level potentials
Both solve the same Sinkhorn problem. Compute once, extract both coupling and potentials.
Expected speedup: ~40-50% per batch pair.

### H2: Accelerate label cost computation
190 GeomLoss calls at ~5ms each = ~1s. Can we:
- Use POT or simpler distance metric?
- Use smaller subsets per class?
- Use a proxy like average class centroid distance?

### H3: Larger batches / fewer pairs
Currently 1024 batch size → 6 pairs. 
Larger batches → fewer pairs but each pair more expensive.
Net effect depends on scalability.

### H4: Log-domain Sinkhorn
Already confirmed slightly faster. Use as baseline improvement.

### H5: Approximate the second-level transport
Inter-batch transport Q is small (3x2 matrix). Could we avoid OT here entirely?
Just use uniform weighting?

## Quality Gates
- NRMSE ≤ 0.01 (1% error)
- Spearman ≥ 0.999
- Top-quarter overlap ≥ 0.98
These are fairly loose, so significant approximation should be possible.
