# Hierarchical Transport Valuation - Research Notes

## Task Overview
- 30 cases (5 seeds x 6 corruption types), N=3000 candidate, M=2000 reference, 10-dim embeddings
- Labels 0-9, reference labels offset by +10
- Target: minimize time while maintaining NRMSE <= 0.01, Spearman >= 0.999, top-quartile overlap >= 0.98

## Baseline Method Analysis

The canonical method does:
1. **_label_cost**: Compute W (20x20) pairwise Sinkhorn distances between all label classes using GeomLoss SamplesLoss on first batch only
2. **Per-batch OT**: For each candidate batch x reference batch pair (up to 1024 each):
   - cdist + W label costs → cost matrix C
   - POT sinkhorn for coupling P, compute batchcost
   - GeomLoss SamplesLoss with custom cost to get potentials f
   - Store normalized F
3. **Batch-level OT**: Sinkhorn on batchcost matrix (numItermax=10000, stopThr=1e-10, reg=0.01)
4. **Aggregation**: Weighted sum of potentials using Q as weights

Key optimization opportunities:
- N=3000, batch_size=1024 → 3 candidate batches, 2 reference batches = 6 batch pairs
- Each pair does full Sinkhorn (POT) + GeomLoss potentials
- W is computed only from first batch - maybe can be simplified
- The double Sinkhorn (POT + GeomLoss) per pair is expensive
- Batch-level OT with very tight convergence (stopThr=1e-10)

## Experiments

### v0 - Baseline
- Date: 2026-10-10
- Status: Running initial baseline measurement
- Expected: ~seconds per case
