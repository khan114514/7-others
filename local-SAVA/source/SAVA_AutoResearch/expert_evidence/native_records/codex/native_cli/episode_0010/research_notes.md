# Continuing research notes: episode 10 complete

Current method is the unchanged episode 9 winner, source554, repeated as565 this episode. No new candidate beat its complete official measurement. Current `/workspace/solution/method.py` matches the preserved source and evaluator hash byte-for-byte. No evaluation is pending.

- Best official raw: **0.03685865550020632 seconds**; score: **0.9293211204698644**.
- Exact evaluator source hash: **a73d8c270d13c6defb541b70facabbcf7f909dd3911b6a774e200d531691823d**.
- All30cases and all270qualityrecords pass. Worst nRMSE0.007025775756825209, minimum Spearman0.9999565986618442, minimum top-quarter overlap0.992.
- Frozen solve.sh SHA256: **bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1**. Invoke `sh ./solve.sh`; mode0444 is frozen.
- Submission ordinary files are exactly method.py and solve.sh; method size11541bytes. Generated workspace method cache removed after final profiling.

The same source's previous historical report554 measured0.037601041732220133 and score0.92789754637728683. Its faster repeat565 is timing variation, not a new algorithmic improvement. All selections use complete actual official quality-valid aggregates, including the historical best; local timing is never treated as a score.

## Complete official episode10 results

| Source | Raw seconds | Score | Exact evaluator source hash |
|---|---:|---:|---|
| 565_episode10_start554_repeat | 0.036858655500206318 | 0.92932112046986437 | a73d8c270d13c6defb541b70facabbcf7f909dd3911b6a774e200d531691823d |
| 571_episode10_point_gemm_columns | 0.037166569266507091 | 0.92873067571004619 | 11ab757b0756b88114ae8a0842f61511bb2edea6af6a9970fcf3cb01e904deff |
| 591_episode10_point_gemm_columns_class_tile20 | 0.038183655467582865 | 0.92678034648338203 | a81f881c4909fe7a4c7746449c8aa8f2b08ef266c8a5a5cf63c4259bd964d654 |
| 620_episode10_class_and_point_augmented_geometry | 0.048834316631352216 | 0.90635700800034547 | 36f6be7911e35c677ae394915c4ccfdc84eafbf796dba9fde9cfc194641d605b |
| 627_episode10_augmented_full_cost_contiguous_pairs | 0.037873630533188894 | 0.92737483954592048 | e82bb0d4d623f828bebcd1673a651e4acb54a0edf07f99dcfc761a1d0b911b00 |
| 642_episode10_full_augmented_class_factor_initialization | 0.03851975833482963 | 0.92613584728118692 | a40a6a5de217330ae595311ed7e92968bcf86f213346651d8c314223a84d0193 |

All six reports pass all30cases/all270records. Original stdout and JSON reports are archived unchanged inside the source directories. Exact reported source hashes match archived bytes. No own solver numerical work ran concurrently with official evaluations.

## Retained recipe

The selected source preserves all empirical class supports and contiguous1024point batches. It uses class column-major geometry, vectorized float32 annealing and guarded initial unshifted exponentials. Class loop ends three stages early with reference-only residual extrapolation0.25. Point schedule remains[3:-3], with a reference-only final retained update and complete final candidate c-transform. Shared batch statistics initialize all point potentials analytically. Coupling cancels the exact first row factor and computes mean(rowenergy/rowmass) after the linearized second column factor; its right operands are2D columns to dispatch GEMM. Final point column factors cancel into vector logarithms. No learned parameters, output tables, input caches, solver I/O or private material.

## New hypotheses and outcomes

- **566–584, cache blocks and marginal dispatch:** point recurrence tiles64/128/256/512rows, tiled initial residual exponentials, class tiles10/20/50pairs, and point/class sums replaced by GEMM/BMM against ones. All19sources pass all30. Small tiles and class BMM are slow. Point-column GEMM571 and its class20pair combination591 looked faster in fresh diagnostics, but official0.03716657/0.03818366 lost to565.
- **585–590, cost allocation and coupling controls:** label matrix as addmm bias/output passes but is slower in fresh processes. Explicit temporary lifetime control also passes. Half/full quadratic column reciprocal correction and exact second reciprocal pass but worsen worst nRMSE to0.0071915/0.0074416/0.0073406. They establish no additional accuracy budget.
- **591–613, layout, exponential basis, fusion and buffers:** exp/exp2, actual column-major point kernels, Torch einsum/outer scaling, final-point tiles and reusable direct residual buffers. Five generated sources595–597/604/606 have fallback indentation errors; preserved unchanged with all30import-failure records. Corrected609–613 compile and pass. Column-major point kernels are slower, means0.046–0.052. Both buffers613 are sometimes faster locally but have no retained full official report.
- **614–621, exact cost factors:** packed22dimensional point factors create complete feature/norm/label costs in one GEMM. Controls change accumulation order, retain only12dimensional feature factors, pad to32, or combine point GEMM columns. Class homogeneous12dimensional BMM and combinations also pass. Combination620 was consistent in fresh benchmarks but official0.04883432 lost.
- **622–624, exact nearest-neighbor trees:** augment reference features with sqrt(2*(max_g-g)), query candidates with zero last coordinate separately per reference label, then add empirical label offsets. This computes the full final c-transform with zero query approximation and no support deletion. All30pass, but means0.0559–0.0586 are slower. Reject for speed.
- **625–639, shared cost construction and combinations:** one augmented GEMM per reference batch, one full cost GEMM with contiguous original-pair copies, strided views,2048row construction blocks, and combinations with class geometry/tiles or residual buffers. All15pass. Full-cost627 had fresh median0.0341846 but official0.03787363 lost. Gains do not reliably add.
- **640–643, factor-derived initialization:** batch means of22dimensional point factors compute exact initial potentials; masked12dimensional class means/BMM compute class initial potentials. All30pass with only float32 reduction-order changes.642 was strongest in final fresh median0.0355864, but official0.03851976 lost.

Exact factorization: A=[-X,||X||²/2,1,L[y,:]], B=[R,1,||R||²/2,onehot(ry)], so C=A @ B.T. L is computed from current empirical class transport. This changes cost construction, not the exponential kernel or transport supports. Full derivations, including blocking and trees, are in `episode10_derivations.md`.

## Validation and artifacts

Four isolated fresh-process benchmark rounds contain192calls: six alternating rounds over three public cases, eight sources per round. All timings and output std are archived in episode10_freshround1–4.json/.log. Timing noise is substantial. These comparisons selected candidates for full evaluation and are not formal scores.

Sources565/613/620/642 pass18independently permuted public-array diagnostics each. Saved canonical000 recomputes every synthetic target from the permuted permitted arrays. Worst nRMSE about0.0085131, minimum Spearman0.99995404 and top overlap0.994667. No tuning against these permutations. Script, seed formula,72records and summaries are in episode10_stress.py and episode10_shuffle_stress.json/.log; these are own public-input calculations, not protected targets.

-79source snapshots;78complete all30local screens;73valid local sources and five preserved import failures; six complete official evaluations.
- Builders1–13, manifests, screens, benchmarks, failed attempts, reports, derivations and working notes are in `/tmp/research_history`.
- `episode_10_index.json` records every source/hash and evidence. `episode10_official_summary.json` and `episode_10_final_selection.json` preserve selection. `episode10_final_audit.json` confirms exact source bytes, frozen launcher and all270valid records.
- Run `python /tmp/research_history/episode10_index.py --restore` to reproduce best restoration. The detailed notes are in `episode_10_detailed_notes.md`; the prior episode notes are preserved as `episode_9_continuing_notes.md`.

Final selected-source profile loaded arguments before timing and ran after all evaluations:0.04342821seconds total, class0.016, point0.013, coupling0.006. Original profile is in episode10_final_profile.log. This is diagnostic only. Class geometry and dense point kernels remain the main costs.

Next episode should start from exact selected bytes. New augmented cost/dataflow variants have plausible local gains but no established official improvement. Revisit only with a concrete new mechanism or justified paired control. Exact trees are slower. Earlier episodes extensively cover failed or slower sparse graphs, coresets, reduced precision, cumulants, polynomial coupling and aggressive schedule truncation; read prior summaries before repeating them.

No protected assets, evaluator code/targets, private cases, credentials, environment variables, hidden tests, other trajectories, network, external installs or subagents were accessed. Only method.py changed during submission experiments; frozen launcher and pre-existing protected directories remain untouched.
