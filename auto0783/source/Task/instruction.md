## Goal

Optimize the execution algorithm of the pretrained Bifrost-1 patch-latent decoder while preserving its numerical predictions. The model bridges multimodal language representations to image patch latents. This task covers all 28 decoder layers, final image RMS normalization and projection head; the FLUX image renderer and retraining are outside this benchmark.

## Task Setting

The canonical sampler has batch size 2 (unconditional and conditional), 300 padded text positions, 256 image positions, seeded reveal ordering, a cosine reveal schedule and per-patch classifier-free guidance weight 4. This benchmark explicitly uses canonical per-patch CFG; it does not claim to reproduce the published script's multi-patch indexing or rendered image quality unchanged. Public Dev has 3 disclosed prompts, sampling seeds 101/202/303 and 32 stages. Final evaluation uses 6 withheld prompts, 2 for each of those sampling seeds, 256 stages and 3 paired repetitions per prompt. The scoring code and rules are public; final prompt embeddings remain private.

Your local Docker environment has no direct network or GPU access.

## Objective and Metrics

The score measures the fractional reduction in paired decoder latency: `(B - T) / (B - U)`, with the metric lower bound U = 0 ms; equivalently `(B - T) / B` without clipping. Here B and T are the Baseline and candidate means measured during the same evaluation. First take the median of 3 repetitions per prompt, then average the 2 prompts within each seed, then average the 3 seed means. Latency sums CUDA-event times of all 256 decoder forwards, including first-stage intermediate construction, all layers, final normalization and projection. CFG/reveal, input preparation, quality diagnostics, checkpoint loading, transport and GPU queue waits are excluded equally and recorded separately. The unchanged Starter has the Baseline semantics; paired timing noise is retained in its measured score rather than forced to zero by identity. A measured candidate slower than B receives a negative score. An effective improvement requires `B - T >= 3 * sigma_B`, where sigma_B is the sample standard deviation (n-1) of the three Baseline seed means; a smaller positive change is reported as insufficient evidence, not relabeled as a verified improvement.

## Allowed Scope

Use graph factoring, partitioning and recombination, alternative attention computation, intermediate reuse, or equivalent operator compositions. The graph is an open sequence of up to 256 operator nodes with a single layer output, applied to each pretrained layer. Its complete numerical interpreter, operator catalog, published decoder source, configuration, frozen public inputs and public checkpoint tensor manifest are in `/opt/bifrost/public_eval/`. You may inspect all of them. The trusted A100 worker validates allocation-free tensor shapes before executing a graph. It accepts JSON operations, never arbitrary Python, shell, modules, filesystem paths or network targets. Checkpoint weights and evaluator rules are fixed.

Each node names its operation and earlier inputs; weights use `w:<layer-weight-name>`. Node `lifetime` can be `stage` or `prefix`. The latter retains a node's first value within one prompt rollout and is reset between prompts and repetitions. You are responsible for whether that reuse preserves predictions as image tokens change. The interpreter enforces a 7 GiB retained-tensor budget and bounded tensor dimensions. Exact graph syntax, allowed attributes, tensor shapes and weight names are documented in the public catalog and source.

The optional `bifrost-tensor-dag/2` format adds `compile_regions` to the three graph fields. Declare 1–16 nonoverlapping regions by region `id` and an ordered, contiguous list of existing node IDs. A region must contain only `stage` nodes or only `prefix` nodes. Its input/output boundary is derived from the graph. All operators, graph/tensor/cache limits and numerical checks remain the same. The trusted worker compiles these regions with a fixed full-graph PyTorch Inductor backend; custom code, backends, flags and fallback are unavailable. The version1 eager graph remains supported. See the public operator catalog for the exact contract.

## Hard Boundaries

The Agent uses the task environment's shell, filesystem and local Python tools. Read the public files in `/opt/bifrost/public_eval/` and the task-local Starter; write your builder and graph artifact in `/workspace/solution/`. `/tmp/` and the sandbox user's home directory (`$HOME`) are disposable writable scratch directories. The evaluator, checkpoint assets, Baseline and trusted worker implementation are read-only and are outside your editable submission.

WebSearch, WebFetch, external downloads, SSH and direct GPU or trusted-worker control are unavailable and are not permitted. Use only `python /opt/bifrost/public_eval/evaluate.py` for Dev evaluation through the provided bounded queue. Local shell and Python processes share the sandbox CPU, memory and filesystem restrictions. The provided evaluator submits a candidate through the bounded queue and waits for its recorded result; the trusted worker serializes GPU evaluations under its resource locks. Do not modify trusted services or bypass the queue, resource locks or evaluator. The worker executes only the submitted data-only graph under the fixed numerical and memory gates.


Every stage must produce finite predictions with relative RMSE and scaled maximum absolute error at most 0.001 against the published-shape eager Baseline. An additional autonomous candidate rollout checks final latents with both limits. Relative RMSE is `rms(candidate-baseline)/(rms(baseline)+1e-12)`; scaled maximum error is `max(abs(candidate-baseline))/(1+max(abs(baseline)))`. All cases must pass, and peak CUDA allocation must not exceed 34 GiB. Any quality or memory failure is a hard-gate invalid result with score -1, excluded from valid ranking; it is recorded separately from valid negative improvement and from infrastructure failure. The Baseline uses the same pretrained BF16 weights, hardware, prefix lengths, sampler, seeds, warmup and repetition budget. All prefix caches are discarded before timing; their first construction is included.

## Submission Instructions

The runnable Starter in `/workspace/solution/method.py` builds the published eager dual-branch computation as a data-only tensor DAG. Edit it or construct your own `program.json`. Run `python method.py` to emit the artifact, then run `python /opt/bifrost/public_eval/evaluate.py` for public Dev feedback. The Verifier reads only your submitted `/workspace/solution/program.json`; it never imports your Python builder. Keep the builder as an explanation and reproducibility artifact. Changes to files outside the submission have no effect on final scoring.

## Workflow & Iteration

Compilation and first execution during the existing disposable warmup are recorded separately. All numerical prefix values are cleared after warmup and between repetitions/prompts, so their reconstruction remains inside the first timed forward. A reusable compiled executable never substitutes cached numerical results. A compiler or resource failure invalidates the run and is preserved; it does not silently use an eager candidate.

Public Dev runs through its isolated bounded file queue to the trusted remote A100 executor. Model transport uses a separate trusted relay with the fixed lane model and no credentials in your tools. Final scoring uses a separate Docker Verifier image and queue.

## Completion Criteria

Save the best public Dev candidate and regenerate `program.json` before ending.
