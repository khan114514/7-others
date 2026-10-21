# Continuing research notes: episode 9 complete

## Best fully evaluated state

- Best: **554_episode9_shared_coupling_cancel_gemm_columns**, raw **0.037601041732220133 seconds**, score **0.92789754637728683**.
- Exact evaluator source hash: **a73d8c270d13c6defb541b70facabbcf7f909dd3911b6a774e200d531691823d**.
- All 30 cases and all 270 quality records pass. Worst normalized RMSE **0.007025775756825209**, minimum Spearman **0.99995659866184416**, minimum top-quarter overlap **0.99199999999999999**.
- Current `/workspace/solution/method.py` is restored byte-for-byte to this measured source. No evaluation is pending.
- Frozen solve.sh SHA256: **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Invoke `sh ./solve.sh`; the frozen launcher has mode0444.

Previous historical best432 was raw0.03942903473265081, score0.924392250394403, hash7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354. The new selected method reduces that measured aggregate by 4.6362%. All selection decisions use complete official valid aggregates, including historical best; diagnostic timing is never treated as a score. Timing noise remains substantial.

## Complete official episode9 measurements

| Source | Raw seconds | Score | Exact evaluator source hash |
|---|---:|---:|---|
| 497_episode9_start432_repeat | 0.049396719331950105 | 0.90527856408578766 | 7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354 |
| 515_episode9_class_column_major | 0.040048826466954776 | 0.92320375925906029 | d55f04d527fb23000cd61107bdbd3facd172e52bb43a5969e195a87f3ae463d3 |
| 547_episode9_class_column_shared_vector_zero | 0.038594802833298067 | 0.92599194455346467 | 592df6dad60e09f62656181a75ed1687b9e58c4dfa0e8ad799e2a3f4217a33b4 |
| 554_episode9_shared_coupling_cancel_gemm_columns | 0.037601041732220133 | 0.92789754637728683 | a73d8c270d13c6defb541b70facabbcf7f909dd3911b6a774e200d531691823d |
| 564_episode9_coupling_lowrank_cost_energy_label_fields | 0.061441357034103326 | 0.88218218453581054 | cec185be3b526a4ac4beb2d90522f8f89d08d13022ae21f1ccdac22f06d9ee73 |

All reports passed all 30 cases. Original evaluator stdout and reports are archived unchanged inside each source directory. No own numerical work ran concurrently with official evaluations. The baseline repeat497 used exactly the saved432 source.

## Retained mechanisms and numerical recipe

All empirical supports, contiguous1024 point batches, class/point finite annealing, late class extrapolation, and the linearized second coupling column refinement remain derived from supplied arrays. No learned parameters, output tables or input caches are used.

547 combines column-major class geometry (batched matrix product with reversed physical axes), in-place distance assembly, vectorized annealing schedules/active flags, guarded initial soft-kernel shift removal, shared analytic point initialization, and exact final point column-factor cancellation in vector logarithms. Schedule construction uses the effective np.arange step `(start+delta)-start` and float32 temperatures. The initial unshifted exponential is allowed only when `(max(f)+max(g))/epsilon<60`; positive-support checks retain stable dense fallbacks.

554 adds exact first coupling row-scale cancellation. Let K=exp(-10*C/max(C)), v=1/(m*column_sum(K)), u=1/(n*(K @ v)), and w=v*clamp(2-m*v*(K.T @ u),.01). The exact current expectation equals mean(((K*C) @ w)/(K @ w)), because the first row factor cancels in final row normalization. Two-dimensional right operands invoke faster GEMM kernels; bare GEMV and single-batch BMM were much slower. Full derivation is in episode9_coupling_cancellation_derivation.md.

564 computes the same numerator using the exact low-rank cost factors. One 22-column response holds weighted mass, reference squared norm, ten feature coordinates and ten label-cost fields. Energy is candidate_norm*mass + reference_norm_response - candidate dot feature_response + observed_label_response. The kernel and supports stay exact; only float32 reduction order changes. Its final complete official result was raw 0.061441357034103326, score 0.8821821845358105. It passes quality but is slower, so it is rejected; source 554 remains selected.

## Hypotheses and controlled ablations

- **498/499, unpadded Torch segmented/scatter class reductions:** all 30 pass with base accuracy. Means0.0608/0.0719 seconds are slower; reject for speed. Prior segmented NumPy failures were already covered in episodes4/6.
- **500–502, final retained class stage:** g-only fails one case (worst0.010111); f-only fails five (0.012131). Separate exact transforms on both sides pass but have no established speed gain. Initial pre-screen padding-mask mistakes are preserved as method_initial.py and initial builder; corrected measured sources retain valid masks.
- **503–507/512, initial scalar shift removal and gauge controls:** all 30 pass. Recentered point costs506 and virtual cached class507 also pass, with507 worst0.007261. Guard512 preserves fallback outside the high-temperature bound. Freshround1/2/3 favored512 at times, but it was not retained without complete official measurement.
- **508–511/520–524, twenty unique supports and setup combinations:** packed global geometry, unique means/norms, repeated bmm geometry and in-place class assembly all pass. One class first-cumulant stage passes with worst0.007173; two first-cumulant stages525 fail11 cases (0.013614). Unique setup is not consistently faster than original means.
- **513–519, physical layout and row energy controls:** point column-major variants pass but are not compelling. Class column-major515 passed official raw 0.0400488, better than today’s baseline repeat but slower than historical best 432. Row energy einsum/bmm/vecdot are accurate but slower.
- **526–529, direct support-size buckets:** all 30 pass while preserving global class hard-switch and ending stage. Building geometry directly per bucket avoids the previous full-pad-then-crop setup, but repeated dispatch loses the hoped-for gain.
- **530–536/543–547, exact layout/schedule/initialization combinations:** all 30 pass. Vectorized schedule534 matches515 quality at printed precision. Freshround2 selected547; its official raw 0.0385948 became the first new best.
- **537–542, cached empirical point variance and budget reallocation:** restoring an earlier coarse point update with exact variance statistics improves worst nRMSE  to0.006117. Hard8/16 variants pass0.008646. Pointlate4 fails22, hard8late4 fails13. Classlate4 variant542 passes0.008906 but statistics and extra work remain slower; reject for speed.
- **548–558, exact coupling cancellation and kernel dispatch:** all 30 pass. Bare GEMV548–552 was very slow and variable. Two-dimensional GEMM 553–555 restores speed; shared554 passes official raw 0.0376010 and becomes new best. Single-batch BMM556/557 and hybrid558 are slower.
- **559–561, actual column-major coupling storage:** all 30 pass, but fresh medians0.0496/0.0531 lose to5540.0426. A stride diagnostic showed transpose-only555 keeps original row-major strides; preserve it as a unary-expression control, not a physical layout change. Explicit `empty((m,n)).T` plus out= creates the real column layout.
- **562–564, exact low-rank cost energy:** all 30 pass with worst nRMSE 0.00702565–0.00702570. One 22-column GEMM is faster than splitting 12+10. Freshround5 medians5540.042632,5620.042365,5640.041586 justified one final complete official564 evaluation.

## Validation, artifacts and continuation

Synthetic checks use18 permutations of public arrays (two public seeds, three settings, three independent permutations), with canonical targets freshly computed by saved initial source 000. Sources432/547/512/554/555 all pass18 with worst nRMSE  about0.00851313, minimumrho0.99995404 and top overlap 0.994667. 564 also passes18, with its original records in episode9_energy_shuffle_stress.json. These are diagnostics, not protected targets or evaluator scores; no tuning was done against them.

- **68 source snapshots**, **67 complete all 30 local screens**, **62 valid sources**, **5 failed sources**, and **5 complete official evaluations**.
- Builders1–13, manifests, per-case screenJSON/logs, five alternating fresh-process benchmark rounds, stress scripts/results, starting/final profiles, failed sources and original reports remain in `/tmp/research_history`.
- episode_9_index.json contains every source/hash and local/official evidence; episode9_official_summary.json and episode_9_final_selection.json preserve exact source selection. Run `python /tmp/research_history/episode9_index.py --restore` to reproduce best-source restoration.
- A final profile excludes NPZ reads by loading all arguments before timing. It measured 0.128227 seconds (point dual 0.058, class 0.021, coupling 0.020), much slower than the formal aggregate. This is diagnostic only; no formal score should be inferred from it. Point kernel operations remain a major cost.
- Next research should begin from the selected exact source and profile it. Kernel dispatch, matrix layout and cost-energy factorization are productive exact mechanisms. More aggressive finite-schedule truncation, frozen class graphs, Gaussian class geometry, generic grouping and polynomial coupling already have extensive failures or slow controls in prior notes. Read those summaries before revisiting them.

No protected assets, evaluator code, protected targets, private cases, credentials, environment variables, other trajectories, network, external packages or subagents were accessed. Only method.py changed in the submission. Frozen solve.sh remains byte-identical, and the submission directory contains exactly these two ordinary files; pre-existing protected infrastructure directories are untouched.
