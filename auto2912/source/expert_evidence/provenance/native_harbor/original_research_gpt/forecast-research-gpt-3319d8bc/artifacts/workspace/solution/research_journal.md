# Forecasting Research Journal

## Confidence Fallback Turn, 2026-10-06
The unchanged retained median-innovation method reproduced protected masked MAE **13.902411027727524**, score **3.2100677547299794**, seeds 17/29/43, all gates, evaluation `6327888c7658482ba80a043512889425`. I tested a confidence-gated graph that blends innovation diffusion toward the history graph when selected-edge support is unstable. The first implementation mistakenly computed support over every possible edge and scored **13.903352477641612** (`3bff527e4ad644efad62cb05dfa629c4`), so it was rejected; after correcting support to the selected eight edges and matching the public calibration harness, the complete protected run scored **13.902094246542314**, score **3.210737004562953**, evaluation `efd5e3cb1fe04f2f8f62b414e142784d`, method SHA `655bf29ab73514e4e3a7d951460a005e4296d0ff58a478a81b182ec2d59f1ab5`, all gates passed. The corrected confidence-fallback graph is retained in `method.py`; next investigate whether the stability fraction should vary by short versus long horizon while requiring improvement on all three honest calibration blocks and a complete protected gain.

## Active Research, 2026-10-06 20:19 UTC
Read the task instructions and own journal, then ran the unchanged entry point first. It exactly reproduced **13.903431941524218**, score **3.2079109145519116**, all seeds 17/29/43 and quality gates passed, evaluation `e880a77aa3d148ce9ffa98bdebfdf30b`, unchanged method SHA `ba5b1497bc6fa73f58274e4c9e47e902c7d6765d5662c7d20b5c3a793c7235df`. Initial best sources saved as `/tmp/{method,spatial,conditional}.edge_turn_initial.py`; old journal `/tmp/journal.edge_turn_initial.md`.

Matched calibration caches reveal a prior diagnostic issue: some scripts paired raw OOF predictions from the budgeted100/80 ensemble with a different full model's validation predictions. This turn uses `temporal_validation.load` to match both experts,100/80epochs, all3honest outer origin blocks. Corrected prediction-graph control last/middle/first **13.3468183475 /13.0742951214 /13.3844295053**. The first attempted script stopped on a diagnostic-only missing `q['no']` key; repaired and reran the first block, without a submission failure.

Structural graph comparisons, always refitting the history-trend gate and honest shallow residual stage:
- Block-centered pooled correlation **13.3442144209 /13.0693675978 /13.3678178879**: all improve.
- Minimum block correlation **13.3522154530 /13.0629738916 /13.3827860818**: mixed, reject.
- Mean-minus-block-standard-deviation confidence **13.3476882075 /13.0648117542 /13.3692184528**: mixed, reject.
- Confidence graph blended with history using squared stable/mean support **13.3438690523 /13.0632179384 /13.3728483300**: all improve, reserve.
- Within-horizon innovation-change correlation **13.3398908846 /13.0701011667 /13.3648759898**: all improve, reserve.
- **Median across3origin-block innovation correlations 13.3444434054 /13.0664776525 /13.3601791698**: best block-average13.2570334092, candidate.

Implemented `network_diffusion='prediction_median'`: preserve seasonal forecast projection and pooled sensor scaling, but estimate correlations independently within each OOF origin block and select eight neighbors by their median correlation. Full implementation calibration **13.3444438985 /13.0664776292 /13.3601041553**; neighbors identical to prototype, network maxdiff below5e-7. Small differences alter a few residual-tree splits (first maximum final difference .926) but leave all-block gains intact; do not claim exact prediction equivalence. Public candidate evaluation and local seed43runtime/reload started20:19UTC. Retain only if complete protected MAE beats the unchanged baseline. Scripts/caches `/tmp/edge_turn_{validation,implementation_check,runtime}.py`, `/tmp/{last,middle,first}.edge_turn_motion`.

Local median-graph seed43fit passed: **121.9996wall /119.2050CPU seconds**, state946799bytes; independently reloaded in7.1390seconds with **maxdiff0**, finite `(148,12,170)`. Protected evaluation remains pending. The freed single-thread local worker now tests within-block innovation scales, edge stability reweighting, stability-weighted history fallback and reciprocal edge support against the implemented median graph. Never more than2CPU workers.

Follow-up calibration, same matched caches and gate/residual refits: implemented median-graph control **13.3444438985 /13.0664776292 /13.3601041553**; centered mean-block scale **13.3426380212 /13.0672179551 /13.3712638749** (reject,mixed); **median-block scale 13.3408242605 /13.0661036010 /13.3524377186** (all improve, next candidate); support reweighting **13.3433317954 /13.0669311547 /13.3572218887** (reject,mixed); stable history fallback **13.3394995397 /13.0683673268 /13.3749904433** (reject,mixed); reciprocal edges **13.3506031687 /13.0793013425 /13.3581456476** (reject,mixed). Scripts/caches `/tmp/edge_scale_turn_validation.py`, `/tmp/{last,middle,first}.edge_scale_turn`. Next residual-denoiser graph reuse diagnostic is running; no submitted source changed during this check.

Residual-denoiser graph tests are all mixed/rejected. History control **13.3444438985 /13.0664776292 /13.3601041553**; network edges/history scale **13.3470774257 /13.0671963589 /13.3583116063**; network graph+scale **13.3461454149 /13.0690384872 /13.3628013959**; own median residual edges/history scale **13.3467293622 /13.0664440555 /13.3622586854**; own residual graph+scale **13.3466468660 /13.0695183610 /13.3645678705**. Keep residual diffusion on history edges. Script/cache `/tmp/residual_edge_turn_validation.py`, `/tmp/{last,middle,first}.residual_edge_turn`.

Added executable option `prediction_median_scale`; neighbors match median graph while sensor scale becomes `max(sqrt(median(block_variances)),10)` using detrended OOF innovations. Implementation calibration **13.3408239203 /13.0661036149 /13.3539239528**, all improve against implemented median graph. Precision differences below2e-6 in scale/network can alter final tree splits; first final maxdiff .520, so use these implementation metrics. Main wrapper still evaluates median edges with pooled scale; its protected result is pending. Fresh matched seed29 outer-origin fits now check graph/scale gains across a second initialization; no more than2CPU workers.

**Median-graph protected result retained:** **13.902411027727524**, score **3.2100677547299794**, complete seeds17/29/43 and all quality gates pass, evaluation `b14e0fb54b4840dea8f5a5c4e5250e31`, SHA `16866114a257ff067608007d30b239dbd8d92a15d3893cdec83e461b19166627`. Gain **0.001020913796694** over unchanged initial. Best behavior backed up in `/tmp/{method,spatial,conditional}.edge_turn_median_best.py` (helpers now also contain inactive scale option). Main wrapper now activates `prediction_median_scale` for the next complete protected comparison; restore the median wrapper if it does not beat13.902411027727524.

Fresh seed29matched outer-origin experts, complete100/80epochs: last pooled/median/median-scale **13.4156199357 /13.4064722074 /13.4057495665**; middle **13.0868043221 /13.0827411754 /13.0867055207**. Median edges improve both; scale gains are mixed across seed/block. First block remains running. Script/caches `/tmp/edge_turn_seed_validation.py`, `/tmp/{last,middle,first}.edge_turn_seed_29{,.results}`. Prepared `/tmp/horizon_edge_turn_validation.py` to distinguish shared versus short/long-horizon innovation neighborhoods after this CPU worker finishes; it changes no submission source.

Seed29 first block completed: pooled/median/median-scale **13.4629372648 /13.4721450741 /13.4629870034**; median edges lose there, confirming block-dependent calibration uncertainty. Short/long horizon graph calibration worsened all blocks (**13.3442922663 /13.0679202886 /13.3929448405** split edges; **13.3424467315 /13.0671856472 /13.3765591606** split scales), reject. Replica-supported edge selection also worsened both checked blocks (13.3454488776 /13.0717826096 six-fit median;13.3425480566 /13.0702080000 replica consensus), reject. Diagnostics `/tmp/{edge_turn_diagnostics,replica_edge_turn_validation}.py`.

Median-scale protected evaluation **13.905305723647922** (`fff1549280e94bb6925d73519d3f58dd`) was worse than retained median edges, so scale changes were reverted. Final retained median-edge rerun exactly reproduced **13.902411027727524**, score **3.2100677547299794**, seeds17/29/43, all quality gates, evaluation `3108f07d8c464b2bb39cf6bd4e920656`, method SHA `16866114a257ff067608007d30b239dbd8d92a15d3893cdec83e461b19166627`. Next discriminating experiment for a future turn: test a confidence-gated blend between median innovation edges and the history graph, but require all three honest origin blocks and a complete protected improvement before retaining it.

## Prior Turn Status, 2026-10-06
The prior prediction-innovation graph result was masked MAE **13.903431941524218**, score **3.2079109145519116**. It has been superseded by the confidence-fallback graph recorded above; its backups remain under `/tmp/` for historical comparison.
Final rerun of the retained best produced the identical MAE and SHA under the complete protocol, evaluation `79090dbc52874d8495a47f2e72af7513`; all quality gates passed again.
The required unchanged invocation at the start of this turn returned scorer HTTP 409 before fitting; the prior exact unchanged result 13.911417634726083 remains the baseline comparison.

Read `/workspace/instruction.md`, `/workspace/task.toml`, own source and journal through actual tools before work. Only public calibration targets/public features used. No development labels, held-out/reference materials, another track, downloads, scorer changes or frozen edits. Sixteen Python source files, 71061 bytes. Each fit uses one BLAS/OpenMP thread; at most two CPU workers concurrently. Frozen solve.sh SHA256 is `d6e0c77a5c20b1afee86cdc5576402b45fea48f5623a0729b3c748d2ee5b4a4a` and remains unchanged.

## Retained Architecture
Median residual trees: pooled and two horizon specialists equally averaged, 90 stages/15 leaves/minimum leaf250/L2=20/LR=.06. Base, anchor, history summaries, forecast slopes, clock/horizon/categorical sensor and normalized neighbor summaries. Tree features use eight neighbors learned exclusively from within-history change correlations.

Average two independently initialized NumPy networks (seed s and s+1009), 100/80 epochs, 6D sensor embeddings, 48/24 tanh layers, linear skip, smooth absolute loss (2 raw units), Adam batches4096. Sensor intercept/horizon/two daily harmonics have ridge .0005; periodic coefficients have history-graph Laplacian .005. One thread and bincount node gradients keep fits under budget.

Preserve sensor intercept/horizon/daily-harmonic projection of OOF network corrections; learn eight prediction-innovation neighbors after removing clock/horizon harmonics and scale-normalizing, then diffuse 25% of the remaining innovations across those edges. Three origin cross-fits train a sigmoid expert gate with signed/absolute anchor/recent disagreements, variability, flow, expert gap, horizon interactions and neighbor gap; centered sensor coefficients ridge .02 and graph Laplacian .05. Network priors .18 short/.32 long, final weights clipped [.05,.65].

Final shallow median residual tree:40 stages/7 leaves/minimum leaf2000/L2=50/LR=.05, six expert/horizon/sensor inputs, trained on additional three-fold gate errors. Cap its correction to +/-10 and apply full strength after seasonal-preserving diffusion with context attenuation `1/(1+d)` for normalized anchor/recent differences. Tanh-bound edge differences at forecast-only per-sensor scale `max(3*median(abs(neighbor innovation disagreement)),.05)`.

## Protected Result History
All successful rows use the unchanged complete seed protocol and passed quality gates. Lower masked MAE wins. Detailed historical journal archived at `/tmp/research_journal.turn_baseline.md` and earlier `/tmp/forecast_turn_adaptive/journal.initial`.

| Method | Complete Masked MAE | Decision |
| --- | ---: | --- |
| Identity upstream forecasts |15.4218571566|Initial baseline|
| Robust pooled/sensor linear |15.0648913575|Superseded|
| Median residual trees |14.1452638897|Superseded|
| Learned-neighbor trees |14.1214289791|Superseded|
| Pooled/horizon trees |14.1101708074|Superseded|
| Trees+25% network |14.0354537227|Superseded|
| Horizon-scheduled network |14.0302026105|Superseded|
| Standard conditional gate |14.0180263118|Superseded|
| Graph-seasonal network head |14.0076506609|Superseded|
| Sensor/horizon gate |13.9928520282|Superseded|
| Sensor absolute/signed-context gate |13.9782379197 /13.9770593182|Superseded|
| Signed expert-gap gate |13.9763124847|Superseded|
| Horizon/neighbor gate |13.972877199845888|Superseded|
| Two networks100/80 |13.966280749980996|Superseded|
| Seasonal network diffusion |13.9474322555024|Superseded|
| Half-strength residual tree |13.924304561281838|Superseded|
| Signed-context residual tree |13.925918911711555|Rejected|
| Full seasonal residual diffusion |13.912218191255334|Superseded|
| Smooth context network pooling |13.919461925615863|Rejected|
| Bounded context residual diffusion |**13.911417634726083**|Best,27b1a2405f744eebb90bbff36f91a695|
| Forecast-only Mahalanobis support guard |13.911567688858696|Rejected,e8314d49ed9e4c1c9401b373db0134dc|
| Restored bounded best |13.911417634726083|Exact repeat,a46759b76aeb4efebcf17497370e7b7a|
| Unchanged initial this turn |13.911417634726083|Exact repeat,1462167db1a3401bb23bada8c73ffff1|
| **History-trend-aware global gate** |**13.906717605638235**|**Retain**,3b2a73c7d94b43aeab8c4704c0d0a809|
| **Prediction-innovation diffusion graph** |**13.903431941524218**|**Retain**,0a60532051fa4ce7b2fba3c85155304b|

Failures without a valid metric: infrastructure HTTP409 conflicts on earlier turns; two-thread fit exceeded180CPU s;100/100 network replica fit execution_error; one smooth-pooling evaluation execution_error. Runtime repairs retained one thread, faster node gradients,100/80 epochs and cached minibatch indexing (predictions exactly unchanged). Last retained local seed43 fit149.611wall/146.344CPU s, state946332bytes; independent reload maxdiff0, finite required `(148,12,170)`.

## Calibration Research
Diagnostic MAEs below use only public calibration labels, seed17, outer origin blocks: last train0:50/test50:74; middle train0:25+49:74/test25:49; first train25:74/test0:25. Primary experts' inner origin folds exclude each outer test. These are never protected metrics. Caches under `/tmp/forecast_turn_{adaptive,additive,graph,residual,resume,periodic}/`.

| Structural Comparison | Last | Middle | First | Decision |
| --- | ---: | ---: | ---: | --- |
| Static full residual diffusion |13.3225855637|13.0773360887|13.3473281546|Previous public winner|
| Bounded context residual (control) |13.3223871746|13.0769918483|13.3466131057|Public winner|
| Mahalanobis support guard |13.3194940511|13.0764871829|13.3477108663|Public rejection|
| Honest smooth monotone reliability |13.3224706922|13.0779113947|13.3416120006|Reject: mixed|
| Gate raw history-motion summaries |13.3244060512|13.0746445991|13.3547888726|Reject: mixed|
| Gate persistence+short linear trend |13.3252777653|13.0698316941|13.3336428354|Mixed|
| Gate relative trajectory agreement |13.3174330219|13.0740277299|13.3556891217|Reject: mixed|
| Gate local expert agreement |13.3273109008|13.0933651668|13.3524420166|Reject|
| Gate damped short trend |13.3287174070|13.0721750646|13.3414948955|Mixed|
| Gate damped anchor-aligned trend |13.3229319428|13.0648458845|13.3414852127|Mixed|
| **Gate damped all-history OLS trend** |**13.3219343092**|**13.0738409876**|**13.3462438952**|Current public candidate|
| Gate damped Theil-Sen trend |13.3212087360|13.0736405048|13.3476333691|Reject: mixed against OLS|
| Gate history-linearity interactions |13.3219500056|13.0794425409|13.3496508489|Reject against OLS|

This turn's reliability model added an honest three-fold residual-stage cross-fit, then fit sigmoid factors in[0,1] with negative monotone coefficients on log disagreement/correction/context distance and regularization toward .95 support. Global and gap-only versions are also mixed; reject under all-block criterion. Initial scratch prototype failed on flattened origin indexing before a valid result; corrected implementation `/tmp/reliability_turn_validation.py`.

Retained substantive implementation `gate_mode='site_history_trend'`: add two global gate signals, clipped relative absolute tree/network distance to the last observation and to `last+(h/(1+h/6))*OLS_history_slope`. OLS uses all12 observed steps, damping half-life6steps. Existing site graph regularization, experts and residual stage retained. Exact implemented coefficients/predictions match the prototype with maxdiff0 on all3blocks. Scripts/caches `/tmp/{momentum_turn_validation.py,*momentum_turn,*damped_turn}`. Public MAE13.906717605638235 confirms retention. Local one-thread seed43 fit passes:121.434wall/118.758CPU seconds,100/80epochs,946353-byte state; separate reload7.491seconds,maxdiff0,finite `(148,12,170)`, script `/tmp/trend_turn_runtime.py`.

Additional structural calibration tests while public scoring runs: adding graph-regularized sensor coefficients to the two agreement signals gives last/middle/first **13.3185225660 /13.0738163939 /13.3750496091**, severe first-block overfit; reject. Adding normalized neighbor averages of these agreement signals gives **13.3287474971 /13.0763155124 /13.3495460687**, reject. Own scripts/caches `/tmp/{site_trend_turn_validation.py,*site_trend_turn,neighbor_trend_turn_validation.py,*neighbor_trend_turn}`; neither changed the submitted source.

Theil-Sen median of66 pairwise slopes also implemented as a separate calibration-only method, same two gate signals/damping and honest residual refit. Last/middle/first **13.3212087360 /13.0736405048 /13.3476333691**, improving last/middle slightly but losing first versus retained OLS; reject under all-block rule. Own script/caches `/tmp/{robust_trend_turn_validation.py,*robust_trend_turn}`. Retained OLS source is unchanged; final protected rerun is starting.

Calibration-only diagnostic: OLS gate minus old bounded control has benefit in the highest history-linearity R2 quartile **-.01065/-.01206/-.00988** on last/middle/first, while lower quartiles generally benefit. A shared gate extension using R2 and its interactions with both agreement signals was tested honestly; **13.3219500056 /13.0794425409 /13.3496508489**, all worse than OLS. Reject. Thus the quartile association alone does not establish a useful fitted reliability rule. Own script/caches `/tmp/{reliability_interaction_turn_validation.py,*reliability_interaction_turn}`; no submitted source change. Final OLS verification is session3444.

Earlier rejected directions: rich chronological/relative histories, robust summaries, target scaling, physical graph, more horizon groups, bootstrap trees, short networks, epoch averaging, independent seasonal/spline heads, convex output refits, innovation heads, embedding regularization, network context, graph residual heads, nonlinear/replica gates, learned final correction strength, joint horizon networks, three narrow networks, tree/both-expert diffusion, extra residual trajectory/flow/clock features, median-leaf randomized forest, factor preservation, horizon-chain smoothing, retaining fold networks, additive residual splines, graph-resolvent network smoothing, error-derived graph, horizon-clock projection and tree-aligned projection. Final-stage origin bootstrap is mixed (last13.33901/middle13.07526/first13.34985). Do not repeat without a distinct hypothesis.

## Next Decision
Confidence-fallback innovation diffusion is retained by the complete protected comparison: mean-minus-block-standard-deviation edges are blended toward history edges using selected-edge support. Next discriminating experiment: test whether this support fraction should be horizon-specific for short versus long forecasts, requiring improvement on all three honest origin blocks and a complete protected gain.

## Support Transform Turn, 2026-10-06
The unchanged confidence-fallback method was run first under the complete protected protocol and reproduced masked MAE **13.902094246542314**, score **3.210737004562953**, evaluation `68b6f3f37023424bad009c820913bc69`, all quality gates passed. A calibration-only horizon-specific confidence rule was mixed across honest origin blocks and rejected. A substantive nonlinear support transform, raising each selected-edge stability ratio to power 2.5 before blending with the history graph, improved the three public blocks (**13.3356443524 / 13.0635245312 / 13.3578440594** versus **13.3431658688 / 13.0640154477 / 13.3728483495** for the matched control), but its protected complete result was **13.902499149584505**, score **3.2098815835152124**, evaluation `f37c318fb62b42b89d4db082ba0bc353`; it is rejected and the source was restored exactly. Next investigate a conservative support transform around the successful public range, then require a fresh complete protected comparison.
