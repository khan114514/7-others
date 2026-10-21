# Continuing research notes: episode 12 complete

Selected exact source: **736_episode12_full_cost_point_terminal_hard_inplace**, restored byte-for-byte to `/workspace/solution/method.py`. No evaluation is pending.

The selected source uses full augmented cost construction and disposable terminal point costs. Adding the 25-pair class cache blocks in source 743 passed quality but lost the official aggregate. Source 736 is 4.693% faster than the episode's initial repeat and faster in 25 of 30 cases, with lower means for all five seeds. This is a descriptive comparison of measured reports, not an additional statistical-significance claim.

- Best complete official raw: **0.033623949997612118 seconds**; score: **0.93552387956214988**.
- Exact evaluator source hash: **af839345946785e31a26e6b03d25b2203d1d1ece87402356c942afb7d752c3ba**.
- All 30 cases and 270 quality records pass. Worst nRMSE **0.0070255392887100332**; minimum Spearman **0.99995660399517805**; minimum top-quarter overlap **0.99199999999999999**.
- Frozen launcher SHA256 **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Use `sh ./solve.sh`.
- Ordinary submission files are exactly method.py and solve.sh; method size 12711 bytes.

Retention uses the fastest complete official quality-valid aggregate, including the historical best. Initial 689 repeats previous selected 671 bytes; its improved raw is timing variation, not an algorithmic gain. No own numerical workload overlapped any official evaluation.

## Complete official measurements

| Source | Raw seconds | Score | Exact evaluator source hash |
|---|---:|---:|---|
| 689_episode12_start671_repeat | 0.035279755699351274 | 0.93234876397178656 | c84af4a358d47b35c429d0e0a5a14f2f3efc0ea14a2ae1c341aea6f4fa25e404 |
| 693_episode12_class_implicit_row_factor | 0.042163712534238583 | 0.91914832708062189 | 1d400d78c955a300cb0328effa8290c2249ba66fe59e666c39ce72abd14a0ab2 |
| 705_episode12_full_cost_control_current_recipe | 0.036573858499953836 | 0.92986723732079724 | 5176aaffddcdf5001bf38bb42bec7c16f0e3368b464e2690ec374db009a1f16d |
| 722_episode12_class_hard_all_shifts_tile25 | 0.036106729400732243 | 0.93076298788144141 | a7ea6f26fbd572130bdc9830b09c1144d08a7215a23c7ece313d22de8a53eb5c |
| 736_episode12_full_cost_point_terminal_hard_inplace | 0.033623949997612118 | 0.93552387956214988 | af839345946785e31a26e6b03d25b2203d1d1ece87402356c942afb7d752c3ba |
| 743_episode12_full_cost_cache_terminal_inplace | 0.03610079859936377 | 0.93077436058046559 | d287a270e10cac9927c763338d1ac9fbd699584ddb93ebb08d16de240a8c1950 |

Original evaluator stdout and reports are archived unchanged in each source directory; source bytes match every reported hash. No source with only local diagnostics is retained.

## New mechanisms and decisions

- 690-694/699-704: implicit separable kernel factors with weighted GEMM/BMM marginals. Point variants are slower. Class row 693 looked promising locally but lost in the full evaluator. Both logarithmic and algebraic normalization controls are preserved.
- 695-698: shared first class/point feature geometry, padded gathers and reconstruction from class BMM. Gather overhead offsets the saved product; no established improvement.
- 705-708: full augmented point costs and grouped coupling response GEMMs. Grouping is slower. Independent full-cost 705 lost its official comparison.
- 709-714/721-725: complete hard class suffix in 10/20/25/50/100-pair cache blocks, including localized entry shifts. This changes loop order across independent pairs. Row-major copies are slower. Standalone 722 passed the evaluator but lost at 0.03610672940073224 seconds. Combination 725 remains a useful controlled comparison.
- 715-720/726: native NumPy fused final fourth powers and coupling energy, with operand-order and float64 controls. No robust gain. Traced public 303 Gaussian016 uses no point fallback and has a hard final stage in all six pairs, limiting final power fusion's activity.
- 727-734: native einsum fuses actual intermediate balancing and powers. Point variants average about 0.159 seconds and combined 734 about 0.181 seconds. Separate output allocation does not help.
- 735-737: disposable terminal point costs use in-place subtraction and restore the candidate unary vector after the final minimum. Clone-first initialization is a control. Source 736 received a complete official evaluation; see its exact result above.
- 738-742: NumPy complete hard cache blocks and terminal minima. Completing all hard stages inside each resident block differs from earlier entire-matrix NumPy sweeps. Fresh timings provide no established gain.
- 743-745: combine class cache blocks and terminal in-place transforms, controlling full-cost versus per-pair geometry and NumPy versus Torch terminal reductions. All local and official evidence is preserved in the index.

All methods preserve empirical first-batch supports, contiguous 1024-point batches and the selected finite annealing. There are no learned weights, output tables, input caches or solver I/O. Equations are in episode12_derivations.md.

The selected full-cost candidates construct C in one exact 22-dimensional homogeneous GEMM, copy each original point-pair slice, and calculate coupling before disposable in-place point transforms. Terminal hard subtraction restores the candidate unary contribution before existing calibration. Source 743 additionally completes each 25-pair class hard block before advancing to the next block. Every retained change has a complete official report.

## Validation and continuation

- 57 source snapshots; 56 complete local screens; 56 locally valid; 1680 local records; 6 complete official evaluations.
- 3 fresh-process rounds contain 144 independent calls over three public cases. Large timing variation is preserved; local timing is never an official score.
- Nine public-array resampling/size diagnostics use fresh canonical 000 calculations. Seven defined targets: all four candidates pass, worst nRMSE about 0.0067112, minimum rho 0.99995669, minimum top overlap 0.99. Two canonical outer couplings did not converge, so their targets are explicitly undefined. Original failures and corrected complete attempt records are retained. This is separate from the fixed 30-case official protocol.
- The index draft used wrong report field names; corrected before execution, original draft preserved. No evaluator report was altered.

The index, official summary, final selection and audit JSON preserve exact evidence. `python /tmp/research_history/episode12_index.py --restore` reproduces restoration. Builders 1-11, manifests, screens, fresh timings, trace, stress, slower sources and working notes are under `/tmp/research_history`. Prior notes are preserved as episode12_prior_notes.md and episode_11_continuing_notes.md.

Next episode should start from selected exact bytes. Complete hard-suffix cache blocks and disposable terminal cost arithmetic are useful building blocks, but confirm combinations through full official evaluation. Native multioperand einsum recurrence fusion is slow. Grouped coupling and implicit point factors do not win. Read earlier summaries before repeating generic groups, sparse graphs, coresets, reduced precision, polynomial coupling or aggressive truncation, which already have extensive failures/slower controls.

No protected assets, evaluator code/targets, private inputs, credentials, environment variables, hidden tests, other trajectories, network, installations or subagents were accessed. Only method.py changed as a submission asset; solve.sh remains frozen. Research artifacts are outside the submission directory.
