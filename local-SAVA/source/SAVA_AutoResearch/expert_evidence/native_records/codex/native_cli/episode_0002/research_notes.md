# Continuing research notes: episode 2 complete

## State and rules
- Only `/workspace/solution/method.py` is editable submission. Frozen launcher unchanged SHA256 `bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1`; mode0444, invoke `sh ./solve.sh`.
- No protected targets, evaluator source, private inputs, credentials, environment variables, other trajectories, network or agents accessed. Public inputs plus locally computed canonical outputs only. Official reports are copied unchanged from paths printed by our own evaluator.
- **Current best fully evaluated/restored:075_scalar_outer**, raw .197422715400171, score .6214290475624421, all30 quality-valid. Exact evaluator hash `1c8792d45354197af5834a1813b6f7fe554f418b811e07620270a678bc6b54fb`. Worst nRMSE .0031568536690847943, min Spearman .9999905964433994, min overlap .996.
- No evaluation pending. Full079_numpy_coupling passed all30 but raw .2315128375320152, score .5560590116068239 was worse than075. Exact reported hash `f406f42ecbad9d58e480f5b6b9452f00ccbe4a46148cd024c24fea4488d0c377`. Report copied unchanged to079/result.json;075 remains restored.
- History: snapshots, full logs and official result.json under `/tmp/research_history`. Full chronological notes `episode_2_working_notes.md`; summary `episode_2_index.json` can be regenerated with `make_episode2_index.py`. Prior concise notes preserved in `episode_1_continuing_notes.md`. No reports altered.

## Formal results
| Variant | Raw seconds | Score | Quality | Exact evaluator hash |
|---|---:|---:|---|---|
|030_torch_geometry (episode1 best)|.20901791953365317|.599194486237406|all30 valid|b92818cc1bb2e79042a1d59c7bf57eba0f5bc18597b8d94c8ed3e1fe82230e73|
|043_shared_exponential_all|.5545745612675091|-.06343294605753307|all30 valid|d5216ab4213ef2eed92d173dab42d30d33e2e56c045b23dc1adac67b484a47e6|
|075_scalar_outer|.197422715400171|.6214290475624421|all30 valid|1c8792d45354197af5834a1813b6f7fe554f418b811e07620270a678bc6b54fb|
|079_numpy_coupling|.2315128375320152|.5560590116068239|all30 valid|f406f42ecbad9d58e480f5b6b9452f00ccbe4a46148cd024c24fea4488d0c377|

043 ran with local screens concurrently, and host timing regime was much slower than episode1.075 ran alone. Keep official numerical outcomes/time distinct from local comparisons and avoid concurrent numerical work during formal timing. Earlier6 full reports listed in episode1 notes; none are faster than075.

## Current numerical recipe
- Canonical contiguous1024 batches, cross100 empirical class costs from first candidate/reference batches. Cost C=||x-r||²/2+W[y,ry]/2. Torch cdist float32 geometry reused for point potentials and normalized coupling.
- Shared residual R=C-f-g yields both soft c-transforms from one exp and row/column sums. Shift by global minimum, clamp exponent below-80 to avoid denormals, fallback stable logsumexp if a support sum <=2e-22. Missing tail then below1e-9 relative.075 uses Torch in-place;079 NumPy in-place (same computation).
- Skip the first2 duplicate coarse symmetric updates for both class/point solves, retain initial c-transform from zero. Skip final3 symmetric point updates; final candidate c-transform still at epsilon.1. Point hard-min only <=2, class hard-min when all eps<=8. Finite canonical schedule is important; full convergence is not target.
- Normalized coupling epsilon=.1 uses3 alternating scaling updates then row-corrected expectation. Outer entropy reg=.01 after max-normalization. For2 reference batches, scalar safeguarded Newton finds logistic probabilities satisfying column mass.5; arbitrary reference batch counts use canonical POT.
-075 Torch coupling;079 pure NumPy dense elementwise scaling reductions, no BLAS configuration/process operations. All values derive from supplied arrays, no lookup/training/saved coefficients. No per-input caches across invocations.

## Episode2 ablations and decisions
-035/036/037 truncate1/2/3 late point updates: all30 local valid, worst nRMSE .005784/.005836/.006206.044 truncate4 fails5,045 truncate5 fails30. Keep3;070 even with soft threshold2 truncate4 fails16, max .01321.
-038/039 union of row/column argmin active edges at eps8 fails29/30.040 union2 neighbors fails4.051/052 enter graph later at eps2: all30 valid, max .004872/.004260, but no clear speed win. Do not confuse fixed sparse support with exact c-transform.
-041 shared exponent only middle stages,042 also labels,043 almost all temperatures; all30 local valid ~.005777.046 direct squared matmul geometry and047 candidate-only final update valid.048/049/050 combine shared exponent/direct geometry/truncation/no transpose; all valid, geometry branch sometimes unexpectedly slower.
-053/054 batch point solves in groups2/6 with masks: all30 numerically same but no clear speed gain.055/056 single thread did not clearly beat2; prior episode2 threads were faster in clean comparisons.
-057/058 skip1/2 duplicate coarse point updates valid;059 also class duplicate2 valid.060 combines coarse2/late3 valid max .006306.061 in-place shared exp/strided cost transpose;062 adds short schedule; valid.063 removes coupling transpose copy;064 also2 coupling updates;065 short schedule+2-step coupling; valid but not faster consistently.
-066/067 skip3 coarse point updates with0/3 late removals valid max .008918/.009181, little speed win and less margin.068 skip4 fails all30 max .06248.069 uses soft minima down to2 with coarse2/late3, valid max .00315685 and substantially better accuracy.071 skips coarse3 with soft2 valid max .006737.
-072 only2 inner coupling updates: valid max .003963.073 coupling1 fails6 max .011721. No useful further gain identified for1 step.
-075 scalar outer equivalent to069 on all30. Random stress tests showed scalar column residual <6.4e-14; occasional generic POT unconverged stress examples differ, but public outputs match to rounding. Full075 valid and new best.
-076 pure NumPy shared point transforms;077 also class transforms;078 NumPy point+coupling;079 all3;080/081 scratch buffers. All30 local valid max .00315698. Clean alternating round4 medians075 .34325,076 .23944,077 .21913,078 .24411,079 .19722,080 .20458,081 .24826. NumPy shared approach is faster than prior episode NumPy double-logsumexp approach.079 selected for full report but formal timing was worse than075; reject for retention. Scratch buffer reuse did not improve time.
-082 Gaussian class W2 from empirical means/covariances fails all30 max .04067.083 Gaussian + empirical separable correction from19 anchor pairs fails21 max .03268.084 Gaussian+CUR residual from36 anchor pairs fails27 max .41626. These compressions do not preserve required empirical geometry; reject.

## Local tools/evidence
- `screen_all.py` uses public NPZ arrays and cached outputs computed by our saved canonical source (not protected targets). Reports all30 gates. `paired_bench.py` alternates source order8 rounds on seed303_gaussian016; not formal score.
- Clean round3 medians030 .63708,062 .30763,069 .33459,072 .31470,075 .33108. Runtime host noise substantial; full official075 .19742. Never infer official score from local speed.
- Suggested continuation if needed: start from formal winner075; optimize cost/memory passes with paired tests, consider input-derived entropy tail truncation without losing finite-schedule behavior. Gaussian geometry and sparse active graphs are rejected. Preserve failed attempts and evaluate every candidate to retain.

Episode2 complete after49 archived new sources, numerous all30 local ablations,4 clean/controlled timing rounds and3 complete official evaluations. New best075 improves measured raw over030 by .0115952041 seconds (5.55%) and improves accuracy margin. Exact best source and frozen launcher hashes verified, submission has only method.py and solve.sh. No protected assets changed.
