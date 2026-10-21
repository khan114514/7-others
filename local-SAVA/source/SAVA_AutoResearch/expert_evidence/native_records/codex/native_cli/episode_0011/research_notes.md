# Continuing research notes: episode 11 complete

The selected source is **671_episode11_point_gemm_columns_point_recurrence_residual_increments**, restored byte-for-byte to `/workspace/solution/method.py`. No evaluation is pending.

- Best complete official raw: **0.036028623731787474 seconds**; score: **0.93091276060350758**.
- Exact evaluator source hash: **c84af4a358d47b35c429d0e0a5a14f2f3efc0ea14a2ae1c341aea6f4fa25e404**.
- All 30 cases and 270 quality records pass. Worst normalized RMSE 0.0070255387888521932; minimum Spearman 0.99995660399517805; minimum top-quarter overlap 0.99199999999999999.
- Frozen launcher SHA256: **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Invoke `sh ./solve.sh`; launcher mode is frozen.
- Ordinary submission files are exactly method.py and solve.sh; method size 12421 bytes.

Retention uses the fastest complete official quality-valid aggregate, including the historical best. Local timings are diagnostic only. The starting source644 is identical to the prior episode winner; a faster repeat of those bytes is timing variation, not an algorithmic gain.

## Complete official measurements

| Source | Raw seconds | Score | Exact evaluator source hash |
|---|---:|---:|---|
| 644_episode11_start554_repeat | 0.03667739993397845 | 0.92966868985772533 | a73d8c270d13c6defb541b70facabbcf7f909dd3911b6a774e200d531691823d |
| 651_episode11_class_persistent_fold_final_delta | 0.059034108334647804 | 0.8867982411585742 | ce0bae80f347f153ec795682e50c3580e959af7cfd70cdea5be4d43fe279ffb0 |
| 656_episode11_point_recurrence_residual_increments | 0.036817420866282188 | 0.92940019057386136 | 42450446e384142251d975fffa7fec856cdde17780fd71b1e6d309ef3e30dd3d |
| 671_episode11_point_gemm_columns_point_recurrence_residual_increments | 0.036028623731787474 | 0.93091276060350758 | c84af4a358d47b35c429d0e0a5a14f2f3efc0ea14a2ae1c341aea6f4fa25e404 |
| 686_episode11_class_unique_centered_point_increments_gemm | 0.044770907000929581 | 0.91414886138872065 | ea02e4397ee9d379a0011887f5117ee8ae57606d9b61fa30d067c94569c50313 |

All original evaluator stdout and reports are archived unchanged in each source directory. No own numerical workloads ran concurrently with official evaluations. The first persistent-residual source651 had temporary slow case measurements concentrated in seed303, but it loses under the required aggregate and was reverted.

## Retained recipe

The selected source uses persistent hard class residuals, direct point log-marginal increments and point column GEMM reductions. It retains the class vectorized schedule, the final reference-only class extrapolation0.25, the point schedule[3:-3], analytic batch initialization and exact row-balanced coupling expectation.

## New mechanisms and decisions

- **645–651: rank-two residuals and persistent hard class geometry.** Rank-two addmm/baddbmm forms C-f-g in one native call; aliasing and separate-output controls all pass but are slower. Keeping R=C-f-g in-place over the hard suffix gives df=rowmin(R)-eps*lb, dg=colmin(R)-eps*la, then R-=.5*df+.5*dg. Final g extrapolation folds into a .75 reference shift. All30 pass. Fresh diagnostics favored651, but its full official result lost. This is a new dataflow mechanism, distinct from merely reusing an allocated direct-transform buffer.
- **652–653: one-sided class padding.** Pad candidate supports only; sort actual reference supports and use segmented row reductions with native column reductions. Both physical layouts pass all30 but are slow (~.092–.100 local means), rejecting the hoped-for benefit from reduced padding.
- **654–658: mean-dual state and direct increments.** After hard residual activation, scalar mean-dual sums suffice for the class objective; scalar controls pass but reductions add overhead. Direct class and point log-marginal increments avoid readding then cancelling current potentials; all pass.656 full official loses narrowly.
- **659–664: native power/reduction fusion.** The last point stage can use sum(K^4)=norm4(K)^4=norm2(K^2)^2, cancelling roots directly in logs if desired. All pass, but fresh timings reject the native norm paths. Square-root kernel storage with norm2 marginals also passes and is slower (~.0725).
- **665–668: reuse first point kernel for coupling.** Exact temperature conversion and unary/shift cancellation pass but are slower (~.0516). Same-temperature, dyadic and quarter-step temperature approximations fail all30, worst nRMSE .2275/.2591/.1503. Correct coupling regularization is necessary. Initial unscreened builder omitted the scalar shift factor; original bytes are saved as method_initial.py and the corrected builder is explicit.
- **669–674: point dispatch combinations and isolated controls.** Column GEMM reductions combine with persistent class residuals and direct increments; all pass.671/672 were close to baseline in fresh diagnostics;671 received the complete official evaluation. Pure norm/increment point controls with original class geometry also pass.
- **675–677: certified cached minima on resident residuals.** Track winner indices and outside-neighbor lower bounds through additive shifts; refresh only uncertified rows/columns with full top-two minima. All pass with baseline accuracy but are slower (.0558–.0699). This confirms that certificate/gather overhead remains expensive even without repeated residual construction.
- **678–680: NumPy only for the hard class suffix.** Torch handles coarse soft transforms; NumPy views handle in-place hard residuals, hard reductions and optionally final transforms. All pass. Fresh timings provide no speed win. Keeping just NumPy minima is cheaper than the full NumPy suffix.
- **681–688: centered initial class geometry.** Exact gauge control forms original C-f0-g0, then direct centered products replace that construction. R0=-(a-Ea)·(b-Eb)+Ea·Eb-E||a||²/2-E||b||²/2. Solve from zero duals and restore2mean(C) to the objective.684 obtains this scalar from the offset and guards the actual exponent range;685 centers twenty unique supports before packing and shares unique means/norms.686 combines this with point increments and column GEMM.687 vectorizes the twenty unique statistics in two padded slices and688 combines that with retained point dispatch. All pass. See complete local screens and official reports for final decisions.

Full equations and tradeoffs are in `episode11_derivations.md`. These mechanisms retain all empirical supports, original contiguous1024 point batches and the selected finite annealing recipe. There are no learned parameters, output tables, input caches, solver I/O or private material.

## Evidence and continuation

- 45 numbered source snapshots; 44 complete all30 local screens; 41 locally valid sources; 3 numerical failures; 5 complete official evaluations.
- Builders round1–11, manifests, per-case quality screens, five fresh-process timing rounds (180 independent calls), failures, actual reports and working notes are in `/tmp/research_history`.
- `episode_11_index.json` records source hashes and evidence. `episode11_official_summary.json`, `episode_11_final_selection.json` and `episode11_final_audit.json` preserve selection and exact bytes. `python /tmp/research_history/episode11_index.py --restore` reproduces restoration.
- New synthetic stress uses seeds202/505, uniform labels/Gaussian016/mixed, three independent public-array permutations each. Canonical000 freshly recomputes targets from those permitted arrays. Original records and reproducible RNG seeds are in episode11_shuffle_stress.json/.log; these are diagnostics, not protected targets.

Stress summaries: [{"source": "644_episode11_start554_repeat", "cases": 18, "valid": true, "worst_nrmse": 0.0075730066088854076, "min_rho": 0.999971762663529, "min_top": 0.9946666666666667}, {"source": "671_episode11_point_gemm_columns_point_recurrence_residual_increments", "cases": 18, "valid": true, "worst_nrmse": 0.007572877982644854, "min_rho": 0.9999717635524179, "min_top": 0.9946666666666667}, {"source": "686_episode11_class_unique_centered_point_increments_gemm", "cases": 18, "valid": true, "worst_nrmse": 0.00757263611914155, "min_rho": 0.999971762663529, "min_top": 0.9946666666666667}, {"source": "688_episode11_class_vector_centered_point_increments_gemm", "cases": 18, "valid": true, "worst_nrmse": 0.007572653232128127, "min_rho": 0.999971762663529, "min_top": 0.9946666666666667}]

Next episode should start from the selected exact bytes. Persistent class residuals and centered initial geometry are useful new building blocks; confirm gains in full official measurements before retaining combinations. Native vector norms, certified minima and one-sided segmentation were slower. Read prior episode summaries before revisiting sparse graphs, coresets, lower precision, polynomial coupling, generic grouping or aggressive schedule truncation: these families have extensive failure/slow controls already.

No protected assets, evaluator code/targets, private inputs, credentials, environment variables, hidden tests, other trajectories, network, external installs or subagents were accessed. Only method.py changed as a submission asset; solve.sh remains frozen. Research artifacts remain outside the submission directory.

Final decision: centered candidate686 passed all30/all270 but measured0.04477090700092958 seconds, score0.9141488613887206. It loses to671;671 is restored byte-for-byte. Final diagnostic profile ran after all evaluations with arguments/imports loaded before timing:0.06785737500467803 seconds, class0.026, point0.020, coupling0.008. Profile timing includes instrumentation and is not an evaluator score. Dense class geometry and point kernels remain the main costs. Source and launcher hashes were verified again after profiling. No evaluation is pending.
