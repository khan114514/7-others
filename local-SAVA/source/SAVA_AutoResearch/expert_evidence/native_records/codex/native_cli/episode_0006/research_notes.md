# Continuing research notes: episode 6 complete

## Best fully evaluated state
- Best: **367_simplified_359_row_energy_353_absolute_mass_344_global_coupling_308_combo_hard256_late2_skip2_last1_linear0**, raw **0.04491611540094406 seconds**, score **0.9138704147966757**, all 30 quality-valid.
- Exact evaluator source hash: **ca9ba4b195760dc34509b9f79d287bb79d1b161478e76c8d53f19b0ef25c2804**.
- Worst nRMSE **0.007052958964993517**, minimum Spearman **0.9999570711063411**, minimum top-quarter overlap **0.992**. All 270 quality records valid.
- Current method.py restored to the exact measured best. No evaluation pending.
- Frozen solve.sh SHA256 **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Run `sh ./solve.sh`.

## Official results in episode 6
| Variant | Raw seconds | Score | Quality valid | Exact evaluator source hash |
|---|---:|---:|---|---|
| 266_start265_repeat | 0.05008928056631703 | 0.9039505326807684 | True | cbb503193aabdeefdcb35fa0d64660e7ed23e9237f69481998f6c2c3ca1dbece |
| 303_combo_hard64_late2_skip2_last0 | 0.05072115703320985 | 0.9027388682813114 | True | 3b49deca1907c9ce9ffafc3990eee737f244b09afb283e1bc4ffb929e4ff17a5 |
| 344_global_coupling_308_combo_hard256_late2_skip2_last1 | 0.04829915573354811 | 0.9073832139786109 | True | 9dffe0a408fc8d8922f8cc936a706c17b159a12f5741a23309ae22fbf1db3db8 |
| 359_row_energy_353_absolute_mass_344_global_coupling_308_combo_hard256_late2_skip2_last1_linear0 | 0.049546916665470536 | 0.904990550887868 | True | a3db49d67e5ee1817e37a3b14771f73c520f121160e08e2f9c6df656b8313b8d |
| 367_simplified_359_row_energy_353_absolute_mass_344_global_coupling_308_combo_hard256_late2_skip2_last1_linear0 | 0.04491611540094406 | 0.9138704147966757 | True | ca9ba4b195760dc34509b9f79d287bb79d1b161478e76c8d53f19b0ef25c2804 |

All original official reports and evaluator logs are preserved unchanged in their source directories. Historical265 (previous episode best) raw0.04625840150013876 was included in final selection. Episode6 start-source repeat266 raw0.05008928056631703 demonstrates substantial timing variation. All retention decisions use complete official quality and raw results, never diagnostic local timings.

## Method and mathematical findings
The retained method derives empirical class geometry from all first-batch supports and retains contiguous1024 point batches. The selected source uses class hard transforms at eps<=256, omits two final class refinements, uses in-place class dual updates, and computes the final point iteration only for the needed reference dual. Point annealing remains schedule[3:-3]. Coupling expectation is computed exactly as mean(row_energy/row_mass) after the linearized second column correction. Positive-mass safeguards use1e-34 with stable direct fallback; unreachable full-kernel initialization branches and redundant setup are removed. Read its source for the final combination; do not assume every experiment below is retained. It contains no input caches, learned constants, I/O, lookup outputs, private material, or corruption truth.

Class transport:
-267/268/269 test NumPy hard refinement, contiguous separate hard transforms, and small in-place dual updates; all pass, fresh speed gains inconsistent.
-270 degree2 feature moments for one early class update passes. Extending to2/3 updates271/272 fails21/30 cases.295/296 compute the first class residual covariance analytically (agree with dense cumulant286), but covariance setup offsets savings.297 first-order feature moments pass but accuracy margin shrinks.
-277 solves all100 class pairs on an unpadded sorted global matrix with NumPy reduceat and exact independent annealing. It passes but averages~0.276s, much slower.278 uses cropped NumPy blocks only for hard refinements and also fails to improve speed.
-273-276/289-291 use fixed reduced-cost class neighborhoods during late hard refinement, retain final dense transform. Top1 at eps<=8 passes; earlier top1 at32 fails3. Union of row and column choices298/299 improves accuracy (union32 passes), but union64 fails25 unless refreshed every2updates301/302. Refresh and scatter overhead erase speed savings.
-279-282/321 switch to hard class transforms at32/64/128/256/512, all30 pass with current point recipe.1024 fails10. All-hard323-326/328 fails27. Mean initialization cannot replace all high-temperature empirical transport.
-283 skips one more coarse class update, passes.284 skips two, fails1.303-308 combine earlier hard transforms, truncated late class updates, and point last-column simplification; all pass, official303 slower than baseline repeat.
-364/365 switch classhard by schedule step rather than temperature; fail26/13, so retain input-scale-sensitive thresholds.

Point transport:
-309/310 compute global residual variance densely/analytically and approximate first retained point c-transform with second cumulant. They agree numerically and fail20 cases, maxnRMSE0.02035.311 applies that approximation at one earlier coarser temperature, then resumes retained exact updates, passes with max0.0036294 but costs more.
-329-333 approximate point soft transforms using exact10-label mixture and conditional feature means/covariances. Pooled within-class covariance329 fails1; full class-specific covariance330 passes all30 max0.00740816, but~0.077smean. Means-only331 fails24. Earlier coarse temperature332/333 passes with~0.00359worsterror but slower. Adaptive dense fallback334-336 all pass and cost more.
-340-342/347 share batch statistics betweenpairs; accuracy identical and no speed gain.337-339 solve2/3/6 pointpairs together with padded dense kernels; accuracy matches baseline, larger working sets increase time. Reject grouped point solves.
-292/293 share first feature geometry using np.take/index_select; all valid, no robust fresh speed gain.294 packed class geometry allvalid but slower.
-349 exact last point column-only update matches retained accuracy.350-355 use absolute positive-mass checks1e-34 in place of relative min/max checks, retain stable direct fallback; allvalid.361-363 skip checks at high temperatures and also preserve reported outputs, no demonstrated retained speed improvement.

Transport weights:
-313 replaces the final row normalization after the second column correction with a scalar total-mass normalization. All30 pass max0.00972006.314 omits any normalization, fails5.315 linearizes last row factor (passes).316 computes first-order energy correction (passes, slower).317 one Sinkhorn step fails9. Earlier point cumulants318-320 do not make one step universally valid.
-343-346 combine global normalization with class simplifications. All pass locally. Official344 passes with raw0.04829915573354811 but loses to historical2650.0462584.
-356 computes exact row-balanced expectation as mean(row_energy/row_mass), avoiding final dense row scaling. Matches existing exact row balancing within rounding.357 linearizes reciprocal only on reduced rowenergy.358-360 combine this with class simplifications/mass checks. All30 pass.359 worstnRMSE0.00705296 versus3440.00912993 and received full official evaluation.
-366/367 remove unreachable initial-full-kernel branches after mean initialization and redundant setup.368/369 replace second dense column/row factors with exact rowenergy/mass matrix-vector contractions (Torch/NumPy). See index for final local/official evidence; these are controlled implementation ablations, not automatically retained.

## Artifacts and reproducibility
Episode6 preserved **104 source snapshots**, **103 complete local screens**, **79 all30 local-valid variants**, **24 numerical-failure variants**, and **5 complete official evaluations**. `episode_6_index.json`, `episode6_official_summary.json`, and `episode_6_final_selection.json` summarize them. Builders, fresh-process timing logs, source files and unchanged official reports remain under `/tmp/research_history`. `episode_6_working_notes.md` records chronology. One hard-only builder syntax error was caught before executing anycase; original invalid builder and initial sources are preserved before the correction.

`screen_all.py` compares against outputs previously computed from saved canonical initial implementation using only publicarrays; those caches are local research artifacts and never used inside submitted method. Local screens are not evaluator scores. Fresh benchmarkround5 had unusually large externalhost timingnoise, preserved in its rawtimelists. Official evaluations had no concurrent own numerical workloads.

## Continuation
Start from exact selected source and profile before extending. General sparseclass graphs, full segmented NumPy, Gaussian conditional point approximations, grouped point solves, and shared geometry were extensively explored this episode; repeating them without a materially new mechanism is unlikely useful. Exact rowenergy reduction is mathematically sound; whether it wins depends on kernel layout and reductioncost. Keep finite canonical annealing conventions: solving to convergence or removing all coarse class soft transforms changes the operator. Improvements must pass all30 gates and complete real `solve.sh` protocol before retention.

No protected assets or frozen launcher changed. No protected targets, private inputs, evaluator source, other trajectories, credentials, environment variables, network, external package installation or subagents accessed. Submission has exactly method.py and frozen solve.sh as root files; generated method cache removed. No evaluation pending.

Final diagnostic profile on public303_gaussian016:0.0754478seconds including profiling overhead, class transport0.033, point duals0.025, coupling0.007. This is not an evaluator score. Final source hash and frozen launcher hash verified, root submission files exactly method.py and solve.sh, and no evaluation pending. The selected source improves historical best time by2.9017percent and the episode6 start-source repeat by10.3279percent; compare full protocol results because timing noise is substantial.
