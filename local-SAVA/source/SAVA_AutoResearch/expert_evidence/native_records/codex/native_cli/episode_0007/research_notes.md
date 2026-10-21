# Continuing research notes: episode 7 complete

## Best fully evaluated state
- Best: **432_episode7_schedule_bounds**, raw **0.03942903473265081 seconds**, score **0.924392250394403**, all 30 cases and all 270 quality records valid.
- Exact evaluator source hash: **7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354**.
- Worst nRMSE **0.007025948263800726**, minimum Spearman **0.9999565959951772**, minimum top-quarter overlap **0.992**.
- Current `/workspace/solution/method.py` restored byte-for-byte to this measured source. No evaluation pending.
- Frozen solve.sh SHA256 **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**; run `sh ./solve.sh` (launcher has no executable permission and cannot be changed).
- Improvement is 11.7505% versus episode7 start-source repeat and 12.2163% versus historical episode6 winner. Timing noise is substantial; use complete official reports for decisions.

## Retained change this episode
Class transport still uses all empirical first-batch supports and the coarse class soft transforms. Its loop now ends at `max(map(len,schedules))-3` rather than -2. The final class update residual `dg = gt-g` is used to extrapolate only the reference class potential by `0.25*dg`, replacing one additional dense late refinement with a vector update. Both-side extrapolation 391 also passed, but g-only423 had better nRMSE. Point transport remains `schedule[3:-3]`, hard cutoff2, same dual recurrence and final reference-only point update. Coupling is unchanged exact mean(rowenergy/rowmass) after the linearized second column factor. Direct NumPy feature and label bounds construct each point schedule without concatenating labels into temporary tensors. Source contains no learned parameters, cached inputs, saved outputs, I/O or private material.

## Official episode7 results
| Variant | Raw seconds | Score | Quality valid | Exact evaluator source hash |
|---|---:|---:|---|---|
| 370_episode7_start367_repeat | 0.044679051434407785 | 0.9143250004375346 | True | ca9ba4b195760dc34509b9f79d287bb79d1b161478e76c8d53f19b0ef25c2804 |
| 391_episode7_class_late3_extrap0.25 | 0.03996076669718605 | 0.9233726196196587 | True | ebb3fa14a0fb032373582110ebb707ce6d38a6a8fb2d41eede149444fd464edb |
| 403_episode7_extrap0.15 | 0.04618020283184402 | 0.911446444577696 | True | 77425d2aeef83490a7604e88eba97b198fbc7a485f9123f0848e4b0f2e2752d4 |
| 426_episode7_gonly_schedule_numpy | 0.039484752933882795 | 0.9242854071040261 | True | 735c397eb50027f284c285eb09947f26b06afa92c9758be3052915f3a11e492b |
| 432_episode7_schedule_bounds | 0.03942903473265081 | 0.924392250394403 | True | 7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354 |
| 434_episode7_bounds_activebranch | 0.053173962633688156 | 0.8980354533250584 | True | 10a10446ca8de7200a14e2ee48dfbb27d74484cabfb4ea758e8c986d49dace98 |
| 440_episode7_bounds_gonly0.2 | 0.08985660549903211 | 0.827694089519422 | True | 4aa4bc29f2bd4768611f72351fdf29923ddd3229f80a892dcd67a31d2b949d4c |
| 449_episode7_bounds_hard512 | 0.09249506799921316 | 0.8226346653308747 | True | 4fa6eb602d0b2b5523f7b17bd8740588e337737f455264c4c634ea33fd6183c6 |

Original evaluator JSON reports are copied unchanged to their source directories alongside evaluator.log. Best432 is selected by complete actual official aggregate; no local timing is an evaluator score. Later official timing increased dramatically, including449 with no own numerical work concurrent. One diagnostic profile ran briefly during432; round10 local screens ran during440. Those overlaps are recorded honestly and are not grounds for claiming speed improvements. All official quality records are available in the archives.

## Substantial hypotheses and outcomes
- 371-375 test earlier point hard transforms (eps4/8/16/32/64).4 matches old public behavior;8 passes all30 but worstnRMSE0.009837 is too close to the gate,16 fails1,32 fails29,64 fails30.376 starts one more point stage later and fails all30 (worst0.063464). Initial diagnostic starts5/6 also fail all30.
- 377-379 change classhard128/512/768 on current base;128/512 pass,768 fails7. Classhard512 combined with final g-only recipe449 passes all30 but officialraw0.092495 loses. Retain input-scale-sensitive hard cutoff256. No claim512 intrinsically slower under timing noise.
- 380-383 omit3/4/5/6 late class stages.3 passes,4 fails17,5/6 failall.391-396 extrapolate the final class update to compensate truncation: late3coeff0.25 passes;0.5 fails3;1.0 fails25. Late4coeff0.5/1/2 failsall. Late4coeff0.25 in410 fails1. Extrapolation cannot replace multiple dense refinements safely.
- 384-389 replace linearized second coupling column factor by exact reciprocal and combine class truncation/point hard8/16. Exact coupling does not necessarily reduce final valuation error because finite-schedule approximations interact:384hard8 fails1;385hard16 fails3;386hard8late3 fails2.388/389point2 pass but no retained speed result.
- 397/398 use separate hard row/column transforms to eliminate same-side subtraction/readdition. Same accuracy, no demonstrated speed win.399/400 update class columns sequentially using the just-updated row potential; fail28/all30. Finite parallel annealing is part of target behavior.
- 401-414 controlled extrapolation/truncation/pointcutoff ablations. Both-side late3coeff0.1-0.35 pass,0.4 fails1.0.15 official403 passes but raw0.046180 loses. Late2coeff0.25 passes but uses more dense work. Pointcutoff8 with coeff0.25 passes at0.00996 worstmargin,16 fails2. Reject lowmargin truncations for currentbest.
- 415-417 move q=4 kernel row/column factors after fourth powers: class-only416 passes, point415 andboth417 returnNaN in all30. Tiny point masses cause overflow in squared reciprocals and underflow in the kernel; algebraic equivalence is not numerical stability.419-421 instead fuse row/column factors into a dense scaling matrix before powers; allpass but allocation and arithmetic cost more.
- 422/423 apply only f/g extrapolation. Both passall30 withworst0.0070346/0.0070260. Reference-only423 is sufficient and forms retained432.424 computes schedule from feature bounds withTorch; passes but kept unnecessary old tensors in that controlledsource.425/426 use directNumPy bounds;allpass.432 stores label bound scalars once perpair, allpassandfullbest.
- 427/434 avoid masked class update construction when all active, retaining active path otherwise. All30pass, localfreshmedians suggested improvement, official434raw0.053174 loses.418 was a redundant branch negativecontrol; sameoutput.
- 433 precomputes all six label term matrices before solving pairs;allpassbut slowerlocal.435/436 compute coupling rowenergy via matrix-vector contraction or explicit weighted sum;allpassbut slowerlocal. Prior episodes had similar contraction failures; no retention.
- 437-446 tune reference-only andboth extrapolation on directbounds. Reference-only0.1/0.15/0.2/0.3/0.35/0.4 allpass;0.2 has smallest observedmaxnRMSE0.006980 but official440raw0.089857 loses. Bothcoeff0.3/0.35 pass withshrinkingmargin.447-450 hardcutoff controls reproduce64/128/512pass and768fails7.

## Artifacts and research reproducibility
- Episode7 index: `episode_7_index.json`; official summary: `episode7_official_summary.json`; final selection: `episode_7_final_selection.json`.
- 87 preserved source snapshots, 84 complete direct local screens (56all30valid,28numericalfailures), 8 complete official evaluations including starting-source repeat. Two additional initial source aliases have byte-identical numbered snapshots already screened; starting source has official validation.
- `episode7_round1.log` through `episode7_round10.log`, `episode7_point_start.log`, `episode7_class_hard.log`, and `episode7_missing_screens.log` preserve actual results including failures. `episode7_build_round1.py`, `episode7_build_round2.py`, `episode7_official.py`, `episode7_screen_missing.py` and `finalize_episode7.py` preserve core drivers.
- Earlier `screen_all.py` treats its first argument as baseline and skips reporting it. Rounds3-10 initially omitted their first candidate from logs. Missing direct screens were completed by `episode7_screen_missing.py`, which compares every listed source to prior locally computed canonical public-array outputs and never changes those caches. Preserve original incomplete logs; index combines them with follow-up results.415NaN failure is now explicitly recorded for all30. No missing official cases.
- A round4 builder assertion happened after415/416/417 source creation and before418 creation; no officialsource was affected.418 was constructed afterward and rerun. Repeated diagnostic calls with just one argument to oldscreen_all did no screening; missingdriver resolves this.
- Fresh-process comparison432median0.0523977 versus4340.0510058 was diagnostic only; fullprotocol rejected434. Original observed timing lists are saved in `bench_episode7_round1_fresh.log`.

## Continuation
Start from exact432. Major speedgain is lateclass truncation plus vector extrapolation; point coarse transforms and empirical class geometry remain necessary. Moreaggressive classlate4 has only one failedcase but error risk acrossprivatecases; tuning constants to fixpublicfailure is less promising than a new mathematicalapproximation. Next work should study cheap estimates of the omitted class refinement based on final reducedcost geometry or an adaptive error indicator, preserving finiteannealing. Dense point kernels remain another largecost after classoptimization; kernel rearrangements need stability checks due tiny masses. Reuse and compressed geometry families were extensively explored in episodes1-6; read prior notes before repeating them.

No protected assets, frozen launcher, private material, evaluator source, credentials, environment variables, other trajectories, network, external installs or subagents accessed. Submission directory contains exactly method.py and frozen solve.sh as ordinary files; generated method cache removed. Noevaluationpending.
