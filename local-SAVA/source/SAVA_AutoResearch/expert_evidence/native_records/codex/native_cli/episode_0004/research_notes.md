# Continuing research notes: episode 4 complete

## Best fully evaluated state
- Best is **162_precomputed_norm_geometry**, raw **0.06487423613192125 seconds**, score **0.8755994146299784**, all 30 quality-valid.
- Exact evaluator source hash: **f57def419ac069782eeb981cefb9157e8d2a80d927c6794c0cd116203a4b37a6**. Worst nRMSE 0.0039617933199789625, minimum Spearman 0.9999831555536839, minimum top-quarter overlap 0.996.
- **Current method.py is restored to the exact best source. No evaluation pending.** Only method.py and frozen solve.sh are root submission files. Verify profile/cleanup completion in episode_4_working_notes.md.
- Starting best123 raw .07845975276820051. Best measured raw improved 17.32% in episode 4.
- Frozen launcher SHA256 bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1. Use `sh ./solve.sh`.

## Official results
| Variant | Raw seconds | Score | All cases valid | Exact evaluator source hash |
|---|---:|---:|---|---|
| 136_all_torch_reductions | 0.0735952235995986 | 0.8588763515055424 | True | f6fb552bc2afd0b84bb2daa3a8d912d159992a8d9604b98d8f396fa402bfe60c |
| 145_inplace_matrix_scaling | 0.06843345973272032 | 0.8687743708868532 | True | 1da84d688b4a1dde5586eb23b05e48769639b4f0b56931b8bbb9d6a92b1a6492 |
| 162_precomputed_norm_geometry | 0.06487423613192125 | 0.8755994146299784 | True | f57def419ac069782eeb981cefb9157e8d2a80d927c6794c0cd116203a4b37a6 |
| 165_addmm_baddbmm_geometry | 0.07197932919904512 | 0.8619749345688746 | True | 45380e25239e9c12ec4828fcf80f9314d981ca792c4d485523e4e7849f6bcae7 |

All original result reports and evaluator logs are preserved unchanged in the corresponding history directories. Retention uses actual complete official raw results, not local screens or self-reported scores. Timing noise remains substantial. Formal baseline B=.5214946210980997 was inferred from the evaluator's score formula in episode 3; do not infer its sample SD from earlier local baseline runs.

## Current numerical method
- Same contiguous 1024 batches and 100 empirical cross-class transport costs from the first candidate/reference batches. Float32 geometry is shared by point potentials and coupling. No output lookup, learned constants, persistent input cache, or solver I/O.
- Retains best123's shared residual exponential kernel recurrence: reciprocal row/column factors, two squares when temperature quarters, scalar shift bookkeeping. Kernel maximum normalization is omitted; relative positive-mass checks trigger stable direct fallback. Mask class padding to zero.
- Initial full soft transform, two duplicate coarse symmetric updates skipped; three late point updates removed. Class hard minima only when all eps<=8; point hard minima at eps<=2; final candidate transform uses hard min at .1. This finite annealed recipe approximates the canonical finite schedule, not a fully converged alternative objective.
- Coupling now normalizes the dense kernel's columns to 1/m and rows to 1/n, twice, then sums K*C. Algebraically equivalent to two alternating Sinkhorn vector scaling steps up to float32 rounding; removes transpose copies and matrix-vector overhead. Outer entropy regularization remains .01 after batch-cost max normalization. For two reference batches, safeguarded scalar Newton solves the balancing condition; other counts use canonical POT.
- Vectorize all 100 class bounding boxes and repeated padded feature assignments. Compute feature squared norms once and use direct squared-distance matrix products. Add W/2 to point costs.
- Reuse final cost buffers by subtracting point g or class f,g only after their last other use. Supplied feature arrays are never modified.
- Direct dense exponentials use exp2(z*log2(e)). Two Torch threads, flush-denormal flag set before thread initialization.
-162 uses ordinary matrix products plus in-place feature norm additions;165 uses addmm feature and baddbmm class products. See the best source identified above for which implementation was retained.

## Experiments and decisions
-126-131: vectorized class setup, first-distance sharing, global distance sharing, packed class geometry, in-place coupling energy/mass. All 30 local-valid, worst nRMSE about .0039615. Sharing geometry is accurate; gather/copy costs offset some savings.
-132: one coupling update failed six cases (worst .0117195863); reject.133-136: initial column sum, strided transpose, weighted column reduction, all Torch reductions. All 30 valid.136 improved the full official result.
-137-139: NumPy point/class/both recurrences passed all 30 but were slower.140: unpadded class blocks via sorting and NumPy reduceat passed all 30, but fresh median .16799 versus136 .07630; reject for speed. All supports retained.
-141-146: global geometry, vectorized setup, pow4, exp2, in-place coupling balance, streamed hard reductions. All valid. Single pow4 much slower than two squares.145 became the next measured best. Streamed hard transforms had little effect because coarse point schedules rarely enter that branch.
-147-156: separate/both final-buffer mutation, streamed final point transform, combinations of geometry sharing, matrix scaling, exp2. All valid; local speed changes were noisy.
-157-160: NumPy einsum fused kernel broadcast scaling and fused first square. All valid, but slower, especially square fusion (.20-.24 seconds versus dense .07); reject.
-161-165: combined vectorized setup, matrix scaling, final buffer reuse and exp2; precomputed feature norms; addmm features; baddbmm classes; both fused geometries. All local 30-valid. Fresh round4 medians:145 .07841,151 .06501,153 .07443,155 .06752,161 .06829,162 .06322,163 .06543,165 .06457.162 and165 received complete official evaluations; choose by reported raw above.
-166-168: structured Taylor expansion of exp(centered feature dot/beta) to degree1/2/3 with exact radial and empirical label factors. Grouped polynomial moments replace dense kernel applications and energy contraction; all supports retained. Fail30/27/19 cases, worst nRMSE .04324/.03411/.01691.
-169-170: class-wise matrix energy contractions accelerate degree2/3 Taylor variants but preserve the same failures.171 degree4 passes27/30, worst .0110019202, local mean .28121. Reject structured polynomial compression: insufficient accuracy at cheap degrees, slower than dense coupling at higher degrees.

## Research artifacts and continuation
-46 new source attempts126-171,39 local all-30 valid,7 with quality failures,4 complete official evaluations. Sources, builders, local logs, fresh-process timing logs, exact reports and hashes preserved under /tmp/research_history. `episode_4_index.json` is the machine-readable summary; `episode_4_working_notes.md` records chronology. Prior episode notes are preserved.
-`screen_all.py` compares public arrays against outputs computed by the saved canonical initial source, not protected targets. These screens are not scores. `fresh_bench.py` runs eight fresh processes per variant in alternating order. First paired round1 overlapped a local screen and is marked diagnostic; subsequent fresh rounds had no concurrent numerical work. Official evaluations had no concurrent numerical workloads.
-Only public inputs and local canonical calculations used. No protected targets, evaluator source, private inputs, other trajectories, credentials, environment variables, network or subagents accessed. No protected assets or frozen launcher changed.
-For a fresh episode, start from the exact measured best. Profile it before choosing the next bottleneck. Remaining point-dual kernel recurrence and class symmetric transforms likely dominate. Possible next hypothesis: group class pairs by support size while retaining Torch contiguous reductions, reducing padding and dense working sets. The full NumPy segmented solver and Taylor compression were slower; repeating them is unlikely useful. Public nRMSE margin is about .006.

Final checks completed: exact best162 source and frozen launcher hashes verified; all270 official quality records valid; root submission files exactly method.py and solve.sh. Own generated method pycache removed. Final local profile303_gaussian016 .06931s: point dual .022s, class .021s, coupling .007s. Next focus is class/point reductions and broadcast kernel arithmetic. No evaluation pending.
