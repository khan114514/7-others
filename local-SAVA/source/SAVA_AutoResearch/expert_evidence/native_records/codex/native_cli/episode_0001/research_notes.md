# Continuing research notes: episode 1

## Rules and state
- Task: numerical approximation of canonical hierarchical transport. Only `/workspace/solution/method.py` is editable submission; frozen `solve.sh` unchanged.
- Read instruction, canonical source, public manifest and all public array schemas. No private/hidden targets, evaluator source, credentials, environment variables, network, or other trajectories accessed.
- Launcher mode0444 prevents direct execution; use `sh ./solve.sh` (unchanged bytes). Launcher SHA256 `bca6ef61c3a1bc0ded2aa6b2da4734b62072846def679d12a2d7ae669fc88bd1`.
- Full logs, exact reported hashes, method snapshots, successful/worse/failed variants are under `/tmp/research_history`. `result.json` files are untouched copies of reports from evaluator-supplied result paths. Expanded chronological notes saved as `episode_1_working_notes.md`.
- **Best fully measured and restored: 030_torch_geometry**, raw .20901791953365317, score .599194486237406, all30 valid. Source hash `b92818cc1bb2e79042a1d59c7bf57eba0f5bc18597b8d94c8ed3e1fe82230e73`.
- Current method.py exactly matches fully evaluated030. Source and launcher hashes verified; only method.py and solve.sh are present in submission directory. Episode1 complete after35 archived sources, numerous controlled local measurements, and6 complete official evaluations. No evaluation is pending.

## Full evaluator results (all quality valid so far)
| Variant | Raw seconds | Score | Worst nRMSE | Exact evaluator-reported SHA256 |
|---|---:|---:|---:|---|
| 000_baseline | 0.716864103102 | -0.374633743283 | 0.000000599 | 92b12caf57128aac8eb7ca32c0e9e3d6f0fb45ec479b78a8243c093922bc5993 |
| 001_cross_only | 0.524686492000 | -0.006120620947 | 0.000000599 | f855f22591fb86743419256f78afae10b9b36baa4f4f3de50d4df270a38fe504 |
| 009_gibbs_early | 0.271582475132 | 0.479222864158 | 0.005723606 | ea63f829bbdba097fc488d3a5115108125e09949410e77f09cc2151420a43ba5 |
| 015_sparse_8_later | 0.304177083401 | 0.416720573721 | 0.002259987 | 494783a939f65bbd7cd6654e3cfb44ae0d7c26028aa05dca54fb3179b543389e |
| 023_coupling_3 | 0.259020773768 | 0.503310747055 | 0.005776985 | fb46e5d3b5dc019ba7d3858876ad64af9a951d943f9e5d0229b19707e0e99ad0 |
| 030_torch_geometry | 0.209017919534 | 0.599194486237 | 0.005777037 | b92818cc1bb2e79042a1d59c7bf57eba0f5bc18597b8d94c8ed3e1fe82230e73 |

## Canonical recipe and main findings
- Inputs x(N,10), r(M,10), labels0..9; public N3000,M2000 across5 seeds x6 settings. Contiguous1024 batches gives3 candidate x2 reference batches. Seed unused by canonical; retain deterministic calculations from arrays.
- Canonical W20x20 computes190 class-pair OT values, but only100 cross candidate/reference costs used. Removing90 within-side solves is exact (001).
- GeomLoss label OT and point potentials are **finite symmetric epsilon-scaling**, not fully converged OT: initial softmin, repeated coarse temperatures, scaling.5, target epsilon.1, final full softmin extrapolation. Need preserve schedule per pair (diameter includes appended labels for point potentials).
- Batch C = squared Euclidean/2 + W/2. Inner coupling uses epsilon=.1 after C/max(C); expectation/(n+m) forms outer costs. Outer3x2 POT uses regularization.01, max normalization,10000 iterations,tol1e-10. Center point potentials and multiply n/(n-1); aggregate outer-weighted per batch.
- Hybrid approximation: high temperatures use explicit Gibbs kernel + matvec whenever cost span/eps<60, otherwise log-domain; eps<=8 uses hard minima. At these units this passes all30: nRMSE~.00572, rank>=.999978, overlap>=.993333. Class costs batched100 padded with masks; each own schedule. Reuse kernel at repeated coarse temperatures; share C-f-g for row/column minima. All kernels per call from actual arrays.
- Normalized inner coupling sufficiently regularized:3 scaling iterations adds little error. 023 has max nRMSE .0057769853, min rank .9999779964, overlap .9933333.2 iterations locally pass but less margin;1 iteration fails8 public cases.
- 030 additionally uses same Torch cdist float32 C for coupling and potentials, avoiding double SciPy geometry. Locally all30 pass with same error as023. Full result best among6 official evaluations. Max nRMSE .0057770371, min Spearman .9999779964, min overlap .9933333.

## Controlled ablations and rejected branches
- 002 explicit dual solver/cost reuse: local six cases nRMSE<5e-7.003 hard min eps<=2,004 batched labels,005 float32 inner coupling: local errors<.000751; slower than min8 branch.
- 006 min8: all30 local max.005723;007 min32 and008 min128 fail RMSE (max.0312/.1326); do not submit.009 early Gibbs matvec full valid and faster; archived predecessor best.
- 010 graph32 at eps<=32,011 also sparse labels: six-case error<=.0014 but graph selection slower.012 graph8/entry32 and013 graph4/entry32 fail RMSE.014 graph16/entry32 all30 local valid max.006543.015 graph8/entry8 improves accuracy max.002260, but full aggregate worse (.304177 vs009 .271582); reverted.016 graph4/entry2 locally valid but not faster.
- 017 fused dense reductions did not speed up locally.018 extends early Gibbs to batched labels;019 graph8 hard-min low stages locally valid but no useful speedup.020 shared reduced-cost row/column minima;021 cached initial kernels; both locally valid, incorporated023.
- 022 moving thread setup to source initialization: no meaningful local gain, never retained/evaluated.
- 023/024/025/026 fix inner iterations at3/1/2/5. All30 local pass except024 (max.01236).027/028 adaptive every-step marginal checks with reused transpose product, tol1e-6/1e-5, all30 local pass but not faster than fixed3.
- 029 Torch1 thread initially looked faster while contended; clean alternating benchmark overturned it.030 Torch2/shared geometry faster than031 Torch1/shared geometry.
- 032 NumPy point duals + SciPy logsumexp and033 also NumPy class duals: all30 local pass but much slower. Keep Torch2.
- 034 NumPy point duals with hand-written in-place stable logsumexp (clip shifted logits below-80, negligible <1e-31 total contribution) + Torch2: all30 local pass max.0057772007. Clean alternating runtime comparison completed:030 median .163145s,034 .221808s (bench_round_5.log); reject034 as slower without retaining it. No full034 evaluation needed.

## Evidence and continuation
- `screen.py` six-case, `screen_all.py` all30, `bench.py` grouped timings, `paired_bench.py` alternates order; local targets are freshly computed from saved canonical on allowed arrays, stored as `*_canonical.npy`. These are not protected targets. Official results define validity/time.
- Runtime host noise is substantial (.17 to .4 for same local methods). Use full evaluator to retain candidates; never infer official score from local timing. Clean `bench_round_4.log` medians:023 .21498,029 .25002,030 .18192,031 .25223,032 .39500,033 .38525. Earlier contended benchmarks are not directly comparable.
- Next episode: start from030, already fully validated. Do not repeat finished benchmarks unless a new hypothesis warrants it. Formal result is saved030/result.json; exact reported hash matches current source. Local bench_round_5 supports rejecting remaining NumPy branch.
- Further research if another episode: improve low-temperature dual kernels without graph selection overhead; accuracy margin currently ~.0042 to gate. Batch geometry and potentials must remain array-derived. No trained/saved weights or output lookup.
