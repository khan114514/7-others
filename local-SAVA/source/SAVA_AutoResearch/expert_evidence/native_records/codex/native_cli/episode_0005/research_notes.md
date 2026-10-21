# Continuing research notes: episode 5 complete

## Best fully evaluated state
- Best: **265_mean_skip3_linear_column**, raw **0.04625840150013876 seconds**, score **0.9112964935233014**, all 30 quality-valid.
- Exact evaluator source hash: **cbb503193aabdeefdcb35fa0d64660e7ed23e9237f69481998f6c2c3ca1dbece**.
- Worst nRMSE **0.00788923531667515**, minimum Spearman **0.9999780382197817**, minimum top-quarter overlap **0.996**. All 270 quality records valid.
- Current method.py is restored to the exact measured best. No evaluation pending. Only method.py and frozen solve.sh are root submission files; generated method cache removed.
- Episode start source162 official repeat172: raw0.05629953476842881, score0.892041964594227, same source hashf57def419ac069782eeb981cefb9157e8d2a80d927c6794c0cd116203a4b37a6. Best raw is **17.84%** lower than that repeat. Prior episode162 measured0.0648742; timing drift is substantial.
- Frozen solve.sh SHA256 **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Run `sh ./solve.sh`.

## Official results
| Variant | Raw seconds | Score | Quality valid | Exact evaluator source hash |
|---|---:|---:|---|---|
| 172_start162_repeat | 0.05629953476842881 | 0.892041964594227 | True | f57def419ac069782eeb981cefb9157e8d2a80d927c6794c0cd116203a4b37a6 |
| 190_fuse_direct_both | 0.062160325032891706 | 0.8808035164351224 | True | 60489b287ca6e1ef56ce61b31413985781b0b1ec5db15861090abae9d62f92c4 |
| 224_mean_initial | 0.06020816953338605 | 0.8845469021202808 | True | 173297cfd168e6e336844f253f9baadbd31c77a8ecd2f627a584de962ef32b44 |
| 241_numpy_label_take | 0.049951216733218944 | 0.9042152790990676 | True | d26af51d56550306674075a9be4114694acf97d6f835602aebeaa9139587b25b |
| 264_mean_hard16_linear_column | 0.0488753250654554 | 0.9062783716492805 | True | 530ce257edf08b325935c8d67349714134490d5e92d12f438f63a6528f41a725 |
| 265_mean_skip3_linear_column | 0.04625840150013876 | 0.9112964935233014 | True | cbb503193aabdeefdcb35fa0d64660e7ed23e9237f69481998f6c2c3ca1dbece |

All original reports and evaluator logs are preserved unchanged in the respective history directories. Retention uses complete official quality and raw results. Local screens compare against values computed from the saved canonical initial implementation using public arrays; they are not evaluator scores.

## Current method
The retained source keeps all empirical class supports and contiguous batches of at most 1024. Class geometry uses 100 padded cross-class transport problems from the first candidate/reference batches. Feature costs use precomputed squared norms and matrix products, then add empirical label cost divided by 2. Point potentials are calibrated by subtracting their batch mean and multiplying by n/(n-1). Outer coupling uses the safeguarded scalar Newton solve for two reference batches, with canonical POT fallback otherwise.

The first point soft transform is approximated by mean costs. The residual exponential kernels use reciprocal row/column factors and two squares when temperature quarters, with stable direct fallback. Direct exponent scaling constants are fused. Class potentials initialize analytically from feature means and norms; later class transforms still use all empirical supports. Label costs are gathered with NumPy np.take from contiguous class rows, avoiding two-dimensional Torch advanced indexing. The second coupling column balancing factor uses the first-order reciprocal approximation 2-t at t=m*column_mass; the first balancing update and the final row balance are exact. No input caches, learned parameters, lookup outputs, solver I/O, private material or corruption truth are used.

This retained265 variant uses point schedule[3:-3], omitting one more coarse update than241, and class hard transforms at eps<=8. It retains both point dual transforms in each update.

## Hypotheses, ablations and decisions
- 173-175 group class pairs by support size into2/4/10 working sets, preserving original schedules and simultaneous hard-transform switch. All 30 valid, but dispatch/copy overhead offsets smaller padding; not retained.
- 176-179 store point/class kernels in float16 or bfloat16. Broad point half fails10 cases; broad bfloat fails all 30 with large errors. Half class passes but is slower.255/256 defer point reduced precision until eps<=128: both pass, but still slower, especially half due fallback.257-259 balance coupling around unit element means to avoid half underflow; float32/half/bfloat all pass, reduced precision still slower. These failures and slow successes are preserved.
- 180-183 use structured degree1/2 feature moments for early point transforms, retaining exact radial/label factors. First two transformations pass; extending degree2 to three or four fails.198 cubic first three passes but costs more.199/200 initial-only moments pass.224 mean initialization and225 mean-minus-variance initialization pass; mean is cheapest.230 analytical point means avoid dense reduction but do not improve fresh timings.231/232 remove one further coarse point update, both pass.236/237 analytical class mean initialization passes with low error.238/239 earlier class hard threshold16/32 passes; the two cutoffs give identical public outputs.
- 184-187 remove1/2/3/4 late class updates; first three pass, four fails10.233/234 combine this with point means.188-190 fuse exponent constants and remove negligible tiny-tail clamping/redundant padding masks; all pass, but official190 is slower than the starting-source repeat.201/235 eliminate the unused last candidate dual, with235 preserving the kernel recurrence and column-only stable fallback.
- 202-205 subsample class support by strides2/3, optionally preserving means; all 30 fail even with mean correction.206-210 deterministic weighted barycenters compress classes to64/32/16 centers.64 without variance correction passes; variance corrections and smaller supports generally fail.211 full-support weighted control matches the original.217-223 weighted reduction and moment combinations pass at64/80;48 fails9.226-229 use NumPy or batched Torch clustering to reduce setup overhead, but fresh medians still give no speed benefit.246/247 with label take also pass but are slower than simpler methods. Reject barycenter compression for retention.
- 212-216 replace Torch reductions with matrix-vector/batched matrix-vector products and hybrid coupling contractions. All pass but matrix-vector versions are slower.250-252 one-thread controls on the new workloads pass but are slower.
- 240-243 test label gathering with NumPy advanced indices, contiguous np.take, cached Torch indices and SciPy feature geometry. All pass.241 np.take is the largest measured gain; SciPy geometry is slower.260 two Torch index_select operations and261 a global label matrix match outputs but show no fresh speed gain over np.take.
- 244-249 combine class initialization, last-column updates, earlier hard transforms and label take. All pass.253/254 linearize the second coupling column or both column/row factors; both pass.253 reduces worst local error to0.0035153.262-265 combine this with the class/point improvements; all pass, and264/265 received complete official evaluation.

## Fresh measurements and artifacts
Five isolated fresh-process benchmark rounds used8 rounds per variant in alternating order, on public303_gaussian016. Final round medians:241 .0441859,249 .0435620,254 .0444583,262 .0424352,263 .0413659,264 .0413984,265 .0400741. Official results determine retention; these diagnostics selected promising candidates.

Episode 5 preserved **94 source snapshots172-265**, **93 complete local screens**, **78 all 30 local-valid attempts**, **15 attempts with numerical failures**, and **6 complete official evaluations**. All sources, builders, screen logs, fresh timing logs, exact reports and source hashes remain under `/tmp/research_history`. `episode_5_index.json` and `episode_5_final_selection.json` summarize them. `episode_5_working_notes.md` records chronology. Earlier episode notes remain preserved.

## Continuation
Start from the exact retained source and profile it before changing the next bottleneck. Matrix construction, point kernel scaling/reductions and the remaining padded class operations are likely the next limits. General grouping, lower precision, class subsampling, weighted barycenters and structured polynomial coupling have been explored and did not improve retained official performance. A cheaper fused reduction implementation or geometry reuse may help, but preserve the canonical finite annealing recipe and measure the full protocol. Prior episode4 Taylor coupling degrees1-4 were too inaccurate or expensive; do not repeat without a materially different approximation.

No protected assets or frozen launcher changed. No protected targets, private cases, evaluator source, other trajectories, credentials, environment variables, network or subagents accessed. Only public arrays, local canonical calculations, and official report outputs were used.

Final diagnostic profile (public303_gaussian016, all inputs loaded before timing):0.061117 seconds including profiling overhead; class solves0.027, point duals0.018, coupling0.007, NumPy label take0.002. This is not an evaluator score. Remaining class direct transforms and kernel broadcasts/reductions dominate. No evaluation pending.
