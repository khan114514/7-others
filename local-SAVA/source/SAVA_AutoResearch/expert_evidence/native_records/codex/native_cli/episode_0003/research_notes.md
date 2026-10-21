# Continuing research notes: episode 3 complete

## Best fully evaluated state
- Best is123_flush_reciprocal_relative_check, raw **.07845975276820051**, score **.8495482990735561**, all30 quality-valid. Exact evaluator source hash **011e216f7713a4e107d99beedc0a24c769c46e7b53a7e657463191a50d6ac4cc**. Worst nRMSE .003961464539171208, minrho .9999831568870172, mintop .996.
- **Current method.py restored to exact best123. No evaluation pending.**125 fourth official run passed all30, raw .08285442496902154, score .841121228068361, slower than123; rejected. Exact reported source hash9a2123b1f10334586e1bbf44d6ef8138df7d753cd13de6f46c7276fce277cb63. Report copied unchanged. Final hashes and exactly two root submission files verified; protected infrastructure folders untouched.
- Frozen solve.sh unchanged SHA256 bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1, mode0444; use `sh ./solve.sh`.
- Only public inputs/local canonical computations used. No protected targets/evaluator source/private inputs, credentials, environment variables, network or agents accessed. No protected assets edited. Submission contains exactly method.py and frozen solve.sh; our generated workspace __pycache__ removed.

## Formal results in this episode
| Variant | Raw seconds | Score | Quality | Exact evaluator hash |
|---|---:|---:|---|---|
|086_recurrent_torch_kernel|.10838529869943159|.7921641100128568|all30 valid|6a1459c74bce4a5c80d0972b2627033fe8c8a6edced4c0c36c6c30f4f0c363b4|
|113_remove_redundant_reductions|.15102290423431744|.7104037163100325|all30 valid, slower|99714c8a2394f719327815a13c559ffcead26c03c584a6c7d605efa2a93285e1|
|123_flush_reciprocal_relative_check|.07845975276820051|.8495482990735561|all30 valid, best|011e216f7713a4e107d99beedc0a24c769c46e7b53a7e657463191a50d6ac4cc|
|125_numpy_coupling_reductions|.08285442496902154|.841121228068361|all30 valid, slower|9a2123b1f10334586e1bbf44d6ef8138df7d753cd13de6f46c7276fce277cb63|

Prior best075 raw .197422715400171 score .6214290475624421. All original reports copied unchanged to corresponding history directories. B from evaluator formula=.5214946210980997. Initial000 measured raw .7168641031 is an earlier noisy local/formal run, not the fixed baseline B; don't infer formal baseline SD from it. Timing noise substantial:113 strongly faster in local paired tests but slower formal. Retain by actual complete official raw.

## Current recipe and mathematical optimization
- Same contiguous1024 batches and empirical cross100 class transports. Torch float32 geometry reused for point potentials/coupling. No learned weights, lookup tables, input caches or files inside method.
- Preserve initial soft c-transform from zero, skip two duplicate coarse symmetric updates in both class and point solves, truncate three late point symmetric updates, final candidate transform remains hard-min at epsilon.1. Class hard-min only when all eps<=8; point hard-min when eps<=2. Finite annealed schedule is target; full convergence generally differs.
- Shared residual exponential kernel gives both row/column transforms. Reuse it across annealing instead of recomputing dense exp: divide by sqrt(row mean * column mean), square twice when temperature quarters, and update scalar dual shift. At initial full transform divide by actual row/column means and track negative old shift. Candidate/reference potentials still explicitly updated at each retained stage.
- Kernel normalization after each update omitted: balanced kernel entries<=sqrt(n*m), their fourth power<=(n*m)^2, safe for batch<=1024. Use reciprocal vector multiplication instead of dense division. Relative positive-mass checks trigger stable direct dense fallback. Mask padded label entries to zero. Only7 denseexp_ in profile, vs dozens before.
-123 enables Torch flush-denormal numeric mode before creating2-thread pool; outputs unchanged at reported precision in local all30 comparisons. Two dense alternating coupling updates at normalized epsilon.1. Scalar Newton outer solve for2 reference batches at reg.01; arbitrary reference batch counts use canonical POT.
-125 rejected: replaced only dense coupling with NumPy in-place exp, two row/column reduction scaling updates and row-calibrated expectation. Did not mutate C; no BLAS/process/env configuration operations.123 remains Torch coupling.

## Ablations and decisions
-085 NumPy point recurrence all30 valid but slower than086 Torch.086 full improvement.090 reuse point initialization kernel,091 class recurrence,092 both all30 valid. Paired round1 medians086 .22540,090 .21117,091 .19213,092 .18643.
-087/088/089 stride2/4/8 coupling support sampling fail all30, max nRMSE .1591/.3483/.5239. Outer batch weights amplify small expectation biases; reject.
-093/094/095 binomial coupling exp approximation k32/128/512 all30 valid, max .006998/.003299/.003177; no useful speed gain, reject.
-096/097/098 CUR ranks32/64/96 coupling pass26/5/1 out30, max .1491/.2448/.3606.100/101/102 recompute exact analytic cost/kernel factors in double pass26/10/6, max .1491/.2317/.2038. Compression not accurate; reject. Some rank32 passes likely dense fallback.
-104 dropping tiny kernel edges all30 valid but slower.105 another coarse skip all30 valid max .006735, no useful gain.106 soft threshold4 same reported accuracy, no gain.107 coupling2 all30 valid max .003963, incorporated.108 fused full outer-product scaling allocation slower; use broadcast reciprocals instead.
-109 both recurrence+coupling2;110 class quarter powers as two squares;111 both;112 reuse all initial kernels;113 remove duplicate point reductions. All30 valid. Paired round3 medians086 .242475,092 .196528,109 .187735,110 .186565,111 .166117,112 .162292,113 .150424.113 full valid but worse formal, rejected.
-114 omit both kernel normalizers,115point only,116label only,117both relative check all30 valid .003962max.118 reciprocal normalized,119 reciprocal unnormalized,120119 one thread all30 valid; one thread didn't beat2 in clean fresh comparison.
-103 framework denormal flag timed only, no all30 screen, not retained.121 flag before pool on113,122 on119,123122+relative mass check all30 valid,123/122 identical reported metrics. Fresh round4 medians086 .279516,113 .166680,119 .160292,120 .201003,121 .193634,122 .143205,123 .141675.123 full selected new best.
-124 strided Torch coupling transpose (avoid copy) all30 valid but slower: reject.125 NumPy coupling reductions all30 valid max .0039617694. Fresh round5 medians123 .091884,124 .148888,125 .082248.125 full evaluator valid but raw .082854425>123 .078459753; rejected and123 restored.

## History, tools and continuation
- All40 new source snapshots085-125 (099 unused), failures and successes retained in /tmp/research_history. Detailed chronological notes episode_3_working_notes.md. Index episode_3_index.json generated by make_episode3_index.py. Prior episode summaries episode_2_continuing_notes.md and episode_1_continuing_notes.md.
- Local screen_all.py uses all30 public NPZ arrays against outputs computed by canonical saved source, not protected targets. paired_bench.py alternating8 rounds; fresh_bench.py8 rounds with fresh processes. These are screening evidence, not scores.
-113 initial construction indentation bug caught before screen; initial source/builder preserved. Builder fixed, full-evaluated source unaffected.
-123 final profile303_gaussian016 .07968s: point dual .022s, coupling .022s, class .017s, geometry .006s; coupling transpose copy .006s drove124/125. No CPU work concurrent with formal timing.
- Next reasonable hypothesis: reuse first-batch feature squared-distance matrix for label geometry and first point pair, avoid class bmm and duplicate geometry; not tried. Prior episode Gaussian/anchor empirical class approximations failed; active sparse graphs had no consistent gain. Start from measured best, not an unvalidated snapshot.

Episode3 completed40 new source attempts,30 local all30-valid sources,9 numerical failures,1 flag variant benchmark-only,5 controlled timing rounds and4 complete official evaluations. Best measured raw improved60.26% from prior075. All reports/hashes preserved; no pending work.
