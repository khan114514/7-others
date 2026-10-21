# Additional Research Notes — CPU Sandbox Investigation (Turn 45)

## Summary
CPU-only deep dive into unresolved Inductor compilation questions.
No GPU evaluation possible — all analysis is structural/symbolic.

## Open Questions Investigated

### 1. Is `rotary` safe to compile?
**Builder created**: exp120_program.json (single rotary op in compiled region)
**Analysis**: 
- `rotate()` uses: torch.cat, torch.split, unsqueeze, negation, elementwise mul+add
- All operations are either view-like or bf16 elementwise
- No accumulation, no dtype changes
- Should be bit-exact with eager mode
**Conclusion**: Very likely safe, but needs GPU verification

### 2. Why does softmax compilation fail (exp112)?
**Evidence**: 1-6% per-stage RMSE from just scale→add→softmax→cast region
**Hypotheses**:
- Hypothesis A: F.softmax(dtype=fp32) has implicit cast that Inductor optimizes
  - `emulate_precision_casts=True` only protects EXPLICIT casts
  - The implicit bf16→fp32 cast inside softmax might not be preserved
  - Inductor might do softmax in bf16 or with different accumulation
- Hypothesis B: Triton softmax kernel has different numerical properties
  - Inductor generates Triton kernels for softmax
  - These may differ from cuDNN/PyTorch eager softmax
- Hypothesis C: Broadcasting of img_mask_rows differs
  - Unlikely: broadcasting is well-defined and shouldn't change precision
**Conclusion**: Root cause unclear without GPU debugging.
Implication: softmax cannot be safely compiled.

### 3. Is rms_norm safe to compile?
**Builder created**: Not yet — would need dedicated experiment
**Analysis**:
- rms_norm does: bf16→fp32 cast → pow(2) → mean → rsqrt → mul → fp32→bf16 cast
- All elementwise, with explicit dtype transitions
- With emulate_precision_casts=True, the explicit .to() calls should be preserved
- The mean operation is a reduction (like sum/N) — might have precision differences
- Reductions in bf16 are the known issue; fp32 reductions should be fine
**Conclusion**: Likely safe (all fp32 reductions in rms_norm), but unproven.
Worth testing if more turns are available.

### 4. Can fp32 casts force fp32 precision in compiled regions?
**Idea**: Insert explicit cast ops to force fp32 for matmuls/linears
**Analysis**:
- F.linear with fp32 input does fp32 matmul with fp32 accumulation
- This matches eager's allow_bf16_reduced_precision_reduction=False behavior
- But fp32 matmuls are ~2x slower than bf16 matmuls
- Since matmuls are 90%+ of compute, fusion benefits can't offset 2x slowdown
- For small ops (softmax, rms_norm), fp32 is fine
**Conclusion**: Not viable for compute-heavy ops. Might work for small elementwise chains.

### 5. Maximum Safe Region Analysis
Even if rotary + rms_norm are both safe, the graph structure limits fusion:

```
rms_norm (unsafe boundary)
   ↓ [1-2 safe ops: reshape]
linear QKV (unsafe boundary)
   ↓ [~13 safe ops: slices, reshapes, concat, transpose, rotary, repeat_heads, views]
matmul scores (unsafe boundary)
   ↓ [3 safe ops: reshape, scale, add]
softmax + cast (unsafe boundary)
   ↓ [0 safe ops]
matmul V (unsafe boundary)
   ↓ [2 safe ops: transpose, reshape]
linear O (unsafe boundary)
   ↓ [2 safe ops: reshape, add(residual)]
rms_norm (unsafe boundary — possibly safe)
   ↓ [1 safe op: reshape]
mlp (unsafe boundary — contains linears)
   ↓ [3 safe ops: reshape, add, concat]
output
```

**Key observation**: Each region has only 2-13 safe ops, mostly view ops.
View ops are already free in eager mode. Elementwise fusion of 2-3 ops
doesn't provide enough benefit to offset region boundary overhead.

## Compilation Safe Ops Summary
**Confirmed safe (GPU-tested, 0 RMSE)**:
- reshape, transpose, slice (views)
- add, mul, silu, scale (bf16 elementwise)
- concat, repeat_heads (memory/broadcast)

**Likely safe (analysis, untested)**:
- rotary (bf16 elementwise + view ops, no accumulation)
- rms_norm (fp32 elementwise + reductions, emulate_precision_casts)
- silu (confirmed — part of mlp_silu_fusion test)

**Confirmed unsafe (GPU-tested, quality failure)**:
- linear, matmul (bf16 reduction in Inductor vs fp32 in eager)
- softmax (different fp32 path — Triton kernel or implicit cast issue)
- cast (part of softmax failure — uncertain if cast alone is safe)
- mlp op (contains linears — fails for same reason as linear)

**Untested**:
- mix (3-input elementwise — likely safe)
- attention_eager / attention_sdpa (very likely unsafe — contains matmuls)

## Final Assessment
Inductor compilation cannot meaningfully improve on our ~71.9% speedup because:
1. The heavy compute ops (linear, matmul) can't be compiled with matching precision
2. The remaining safe ops are mostly views + small elementwise chains
3. View ops are already optimal in eager
4. Small elementwise fusions don't offset region boundary overhead
5. The graph's compute-heavy structure naturally limits fusion opportunities

The theoretical speedup from full compilation (exp110 raw: ~77%) shows there's
significant headroom IF precision could be maintained. But with the fixed
compiler options, this is not achievable.

## Artifacts Created
- exp120_program.json — rotary compilation test (builder: exp120_rotary_test.py)
- additional-research-notes.md — this document
- prior-research-exhausted.json — preserved from previous turn
- research-exhausted.json — final comprehensive assessment


## Additional Investigation — Session 2 (exp121, exp122)

### exp121: rms_norm compilation test
**Builder**: `exp121_rms_norm_test.py` / `exp121_program.json`
**Status**: Structurally valid, needs GPU verification

**Analysis of rms_norm safety**:
```python
def rms_norm(value,weight):
    dtype=value.dtype
    full=value.to(torch.float32)       # explicit cast bf16→fp32
    variance=full.pow(2).mean(-1,keepdim=True)  # fp32 reduction
    full=full*torch.rsqrt(variance+1e-6)  # fp32 elementwise
    return weight*full.to(dtype)       # explicit cast fp32→bf16
```
- All dtype transitions use explicit `.to()` calls
- `emulate_precision_casts=True` should preserve these exact casts
- `mean()` is a reduction in fp32 — should match eager (fp32 reduction)
- `pow(2)`, `rsqrt`, `*` are all elementwise in fp32 — should match exactly

**Verdict**: Likely safe. The explicit casts are protected. The fp32 mean
reduction should be bit-exact or very close (well within 0.001 RMSE).
Worth testing on GPU if more turns are available.

**Impact if safe**: Minimal. rms_norm is followed by reshape then linear (unsafe).
The region would be rms_norm + reshape (2 ops) — too small to be useful,
and adds a region boundary.

### exp122: Explicit fp32 softmax compilation test
**Builder**: `exp122_fp32_softmax.py` / `exp122_program.json`
**Status**: Structurally valid, needs GPU verification

**Design**:
```
scores (bf16) → cast fp32 → scale → add(mask_fp32) → softmax(dim=-1, dtype=fp32) → cast bf16
```
- Explicit casts both before and after softmax
- Mask is also explicitly cast to fp32 before addition
- Scale and add happen entirely in fp32
- Softmax input is already fp32, so no implicit cast needed

**Hypothesis**: Exp112 failed because softmax's internal implicit bf16→fp32 cast
was optimized away by Inductor. With EXPLICIT casts, `emulate_precision_casts=True`
should protect the precision boundaries.

**Alternative hypothesis**: Even with explicit casts, Inductor's Triton softmax
kernel has different numerical properties than eager's cuDNN/host softmax.
If so, fp32 softmax differences should be tiny (1e-6 to 1e-5 range),
well within the 0.001 RMSE threshold.

**Verdict**: Good candidate for GPU testing. If this passes, we can compile
the entire pre/post-softmax elementwise chain as one region.

### Compilation Safe Ops — Updated Classification

**Confirmed safe (GPU-tested, 0 RMSE)**:
- reshape, transpose, slice (views) — exp113, exp118
- add, mul, silu, scale (bf16 elementwise) — exp115, exp116
- concat, repeat_heads (memory/broadcast) — exp116, exp117

**Likely safe (analysis, untested)**:
- rotary (bf16 elementwise + view ops, no accumulation) — exp120 built
- rms_norm (explicit fp32/bf16 casts + fp32 elementwise + fp32 reduction) — exp121 built
- mix (3-input bf16 elementwise)

**Uncertain / needs testing**:
- softmax with explicit fp32 casts — exp122 built
- cast (depends on direction; bf16↔fp32 should be exact with emulation)

**Confirmed unsafe (GPU-tested, quality failure)**:
- linear, matmul (bf16 reduction in Inductor vs fp32 in eager) — exp110, exp111
- softmax with implicit fp32 cast — exp112 (1-6% RMSE)
- mlp op (contains linears) — exp111

### Structural Analysis: Region Boundary Counting

If we assume {rotary, rms_norm, softmax(fp32)} are ALL safe:

```
input → slice → rms_norm → reshape → linear → [SAFE REGION 1: ~15 ops] → matmul →
  [SAFE REGION 2: ~6 ops (reshape+cast+scale+add+softmax+cast)] → matmul →
  [SAFE REGION 3: ~6 ops (transpose+reshape)] → linear →
  [SAFE REGION 4: ~4 ops (reshape+add+rms_norm+reshape)] → mlp →
  [SAFE REGION 5: ~3 ops (reshape+add+concat)] → output
```

5 region boundaries for 33 safe ops = 6.6 safe ops per region on average.
Compare to exp119 (6 regions, 26 safe ops) which was 2.2% SLOWER.

Even with optimistic assumptions about rotary/rms_norm/softmax safety,
we still have ~5 region boundaries and relatively few ops per region.
The net effect is likely still slower than pure eager.

### Why Compilation Can't Help — Summary

1. **Compute-heavy ops can't be compiled** (precision mismatch)
2. **Safe ops are mostly memory-bound views** (already optimal in eager)
3. **Elementwise fusion opportunities are small** (2-3 ops between compute ops)
4. **Region boundary overhead exceeds fusion benefit** for small regions
5. **Graph structure naturally limits fusion** (compute ops separate safe ops)

The only way compilation would help is if we could compile large regions
including the compute-heavy ops, which requires matching precision —
not possible with fixed compiler options and `allow_bf16_reduced_precision_reduction=False`.

### All Draft Candidates (CPU-verified structure, GPU-unverified quality)

| Exp | Description | Nodes | Regions | Status |
|-----|-------------|-------|---------|--------|
| exp120 | Rotary compilation test | 65 | 1 | Structurally valid |
| exp121 | RMS norm compilation test | 65 | 1 | Structurally valid |
| exp122 | Explicit fp32 softmax chain | 67 | 1 | Structurally valid |

All three are single-region tests to isolate precision of individual op types.
If they pass, they can be combined into larger multi-region graphs.
But based on structural analysis, even if all pass, the net speed impact
is likely neutral to slightly negative due to region boundary overhead.
