# Continuing research notes: episode 8 complete

## Best fully evaluated state

- Best: **432_episode7_schedule_bounds**, raw **0.03942903473265081 seconds**, score **0.924392250394403**.
- Exact evaluator source hash: **7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354**.
- All 30 cases and all 270 quality records pass. Worst nRMSE **0.007025948263800726**, minimum Spearman **0.9999565959951772**, minimum top-quarter overlap **0.992**.
- Current `/workspace/solution/method.py` is restored byte-for-byte to that measured source. No evaluation is pending.
- Frozen solve.sh SHA256: **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Run `sh ./solve.sh`; the unchanged launcher is not directly executable.

No episode 8 candidate beat the historical best aggregate. The exact episode 7 source remains selected; none of the new screened approximations is included in the submission.

## Complete official measurements

| Source | Raw seconds | Score | Exact evaluator source hash |
|---|---:|---:|---|
| 432_episode7_schedule_bounds | 0.03942903473265081 | 0.92439225039440298 | 7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354 |
| 451_episode8_start432_repeat | 0.063189309765099705 | 0.8788303709977997 | 7d3f1dd18cf04a39aba10066b90def10384a1f4fd0f9b5562d00d89e1639a354 |
| 460_episode8_shared_initial_gauge | 0.041269054201742011 | 0.92086389287229342 | fab4ae464687ac8e3c80604ccd4be38b66476d47e031b62425cc7300a9f253a5 |
| 473_episode8_virtual_combine_point_last_logcolumn | 0.045238594365946481 | 0.91325204031694796 | fc9b3e4401eea47ac5d9df307a79d703e8a1bf9e0ea5f759c19ff992e6475cfb |
| 490_episode8_mean_first_combine_shared_point_logcolumn | 0.040081313966463007 | 0.92314146235666883 | 1ff73d90ffb05ae31910ecc236b7481fdc268b37573cd916036dd2058015cb90 |
| 491_episode8_mean_first_combine_point_foldcol | 0.04727764859999297 | 0.9093420206320797 | 0b6d2d08768addc7cc6342bb22a1c0fcf89ef73f797b287444b367dd3d87667f |

The first row is the historical episode 7 best. All episode 8 official evaluations passed quality; original evaluator reports and stdout are copied unchanged to each source directory. The starting-source repeat was much slower than its earlier measurement. Retention uses the fastest complete official valid aggregate, including historical best, rather than local diagnostic timings. No own numerical workloads ran concurrently with an official evaluation.

## Hypotheses and controlled ablations

- **452–455, late point extrapolation:** omit one additional late point stage and extrapolate its latest reference residual. Fail 24/30/30/30 public cases respectively, with worst nRMSE 0.01442/0.02691/0.06034/0.11430. Reject; extrapolation amplifies the missing point update error.
- **456/469, cached final class minimizers:** compute final class transforms only at the preceding stage's minimizers. All 30 pass, worst nRMSE 0.007117. Computing indices only once removes redundant work, but a robust full speed win was not established. Exact separate final transforms 457 and separate hard transforms 461 also pass.
- **458–460/483, shared initialization and cost gauge:** derive all six initial point potential pairs from shared batch means, label frequencies and norms, avoiding dense cost means. Equivalent unary cost recentering and skipping zero-dual subtraction also pass all 30. Full 460 raw 0.041269 loses to historical best.
- **462/463/465/466/472, half-power kernel rearrangement:** move one row or column factor between the two kernel squares, eliminating one square root. All 30 pass. This differs from previous full fourth-power factor folding, which overflowed. Original combined source 464 has a builder slicing SyntaxError and remains preserved; corrected source is 472.
- **467/468/484/485, final point column cancellation:** keep final row scaling and powers, then move the final column factor into vector logarithms before updating g. Zero sums use the stable dense column logsumexp fallback. All 30 pass with essentially base accuracy; 484 adds shared analytic means, and 485 adds half-folding. These point changes are algebraic reorganizations.
- **470/471/473–479, virtual class steps:** cached minimizers replace one dense late class annealing update; a dense final transform or cached final transform both pass (worst 0.007023/0.007261). Point combinations 473–475 pass. Full 473 raw 0.045239 loses. Two or three frozen single-minimizer updates 476–479 fail 13/5/30/30 cases; reject multi-stage fixed graphs.
- **480/481/482/489, certified adaptive class steps:** top-two reduced-cost neighbors and an outside-neighbor lower bound certify each virtual minimum; uncertain rows and columns use exact dense fallback. One and two virtual stages pass with base accuracy but local means about 0.047/0.050 seconds are slower. Three-stage 482 raises an uninitialized cached-cost error in five uniform-label cases; corrected 489 initializes graph costs when the first hard fallback becomes available, passes all 30, and averages about 0.052 seconds. Preserve the original implementation failure.
- **486–488/490–492, first class cumulants:** approximate the first retained class soft transform by second cumulants using covariance computed once for 20 unique supports. Analytic 486 and dense-variance control 488 agree within rounding and pass all 30, worst 0.007044. Mean-only control 487 passes, worst 0.007174, and is cheaper locally. Combinations 490/491/492 also pass. Covariance setup still offsets its saved dense work, consistent with earlier episodes.
- **493–496, wider fixed class graphs:** top-two neighbors for two virtual stages still fail five cases (worst 0.05978/0.01573); top-two/four for three stages fail all 30 (worst 0.15248/0.08827). Reject. Certificates recover accuracy, but their overhead eliminates the hoped-for speed gain.

## Additional diagnostics

Two fresh-process benchmark rounds used eight independent processes per source in alternating order. Round 1 favored shared initialization/gauge and exact point rearrangements; round 2 favored cached-class/log-column 473 and mean-first/shared-point 490. Raw timing lists are preserved in `episode8_fresh_round1.log` and `episode8_fresh_round2.log`. They are not evaluator scores.

Eighteen synthetic cases reorder only public input arrays: two public seeds, three settings, three independent permutations. Canonical outputs are computed from saved initial source 000, never protected targets. Sources 432, 473, 490 and 491 pass all 18, with worst nRMSE 0.0077797/0.0078114/0.00787285/0.00787290. All 72 candidate quality records and permutation seeds are in `episode8_shuffle_stress.json`; the script reproduces them. These diagnostics support the approximations beyond the fixed public batch layouts. No tuning was done against them.

The starting cProfile overlapped one local screen and includes NPZ argument reads in its total; it is explicitly diagnostic. It identified class transforms and point kernel operations as the remaining costs. The class-stage instrumented source/log shows no soft fallback on public 303 Gaussian016: temperatures about 3024–9829, 756–2457, 189–614, then hard stages down to 0.738–2.400.

## Artifacts and continuation

- **46 source snapshots**, **45 complete all-30 local screens**, **31 valid sources**, **14 failed sources**, and **5 complete episode 8 official evaluations** including the starting-source repeat.
- `episode_8_index.json`, `episode8_official_summary.json`, and `episode_8_final_selection.json` preserve source hashes, quality results, measured times and selection.
- Builders round 1–10, manifests, full per-case screen logs/JSON, failed sources, original driver before the import-error fix, actual 464 compile stderr, evaluator reports/stdout, profiles and fresh timing lists remain under `/tmp/research_history`.
- Read prior episode 7/6/5/4 summaries before revisiting segmented NumPy class solves, generic grouping, barycenters, lower precision, polynomial coupling or Gaussian point approximations; they generally failed or were slower. This episode confirms that extending fixed class minimizer graphs over several finite annealing steps is inaccurate.
- Next work should profile the selected source first. Exact point log-column cancellation is stable and available as a building block. Class covariance approximation is accurate but expensive even with unique supports. A materially different fused class reduction or compressed geometry mechanism would be more promising than further extrapolation constants. Global Torch segment reductions remain a possible new mechanism; previous global segmented **NumPy** versions were slow, so validate the kernel cost before implementing a complete variant.

No protected assets, evaluator code, private cases, protected target files, credentials, environment variables, other trajectories, network, external packages or subagents were accessed. Only the method submission file changed; frozen solve.sh is unchanged. Research scripts and diagnostic records remain outside the submission directory.
