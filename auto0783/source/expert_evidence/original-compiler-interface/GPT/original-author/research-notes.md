Recovery operation notice: this fresh workspace restores your own original native CLI UUID/history and the last captured method.py/program.json from round11. Your own captured artifact builders are under recovered-own-artifacts; your own historical Dev feedback is in own-previous-Dev-evidence.json. Loose files that were not in the private backup are not claimed restored. Continue independent method research using your own history, tools and new public Dev experiments. Keep evaluation commands in the foreground and inspect actual exit status/result files; do not background them or poll results that were never submitted. Historical scores and preparation are not new experiments.

## Recovered starting point and operational handling

Read the recovery notice and current AGENTS.md before new work. The original
tool session IDs from before recovery were unavailable, and no evaluator was
running in the recovered workspace. Do not infer their outcome from missing
loose files. Recovery evidence supplies a completed, passing historical Dev
result for experiment 30 (text-only input RMS with a stage-lifetime input view):
run f137e3fc0c5c4a1ebecbc98516bda7c7, reward 0.7140471529484217, artifact
589d8aee0438612fc3248e38226f63b037fa2d504cbe34a34e456f66caeedc9f.
Its builder/program are recovered under recovered-own-artifacts/dev-f137e3fc0c5c4a1ebecbc98516bda7c7/.
This exceeds historical experiment 23's 0.7132044104257124 and is the starting
best; these historical results are not counted as this turn's new evaluations.
Earlier unresolved directions and failed numerical attempts described in the
native history remain historical, without claiming their loose files restored.

## 31: Text-only cached MLP after text-only input RMS

Start from recovered valid 30. Feed the cached contiguous [2,300,3584]
text_post_norm directly into the cached bias-free text MLP. Remove the prefix
full text/image normalized_post concatenation and the redundant text_mlp_rows
slice. Gate/up/down GEMMs now consume 600 rather than 1112 flattened rows;
previous image outputs were discarded. Keep text Q/K/V projection dimensions,
attention, image computations and cache semantics otherwise unchanged. This
combines the previously unresolved text-only MLP direction with validated 30.
Row-local algebra supports the change, but different GEMM dimensions still
require measured numerical validation.

Saved builder/program: experiments/31_text_only_mlp_after_input_rms/.
Public validator syntax/topology checks passed locally for 95 nodes; GPU meta
shape and actual numerical validation were performed by the trusted worker.
Artifact SHA256: 9bff4f03ff87966524d92d7e8a790b432befb945ff266419bb384392c445fee7.

An initial local subprocess logging wrapper failed before launching Dev because
its Python environment could not resolve the python executable. Preserve this
as local-wrapper-error.json; it is neither a worker submission nor a numerical
failure. Run the documented evaluator directly in the foreground afterward:
python /opt/bifrost/public_eval/evaluate.py --program program.json --output experiments/31_text_only_mlp_after_input_rms/dev-result.json
Actual CLI exit code 0, execution_status COMPLETED, quality_pass true.
Run ID: 453a46373819498b81ad683487872e94. No stderr was emitted.

Baseline mean 7077.473602294922 ms; candidate mean 1987.7133687337239 ms
across the three public 32-stage cases. Reward 0.7191493065987284, versus
historical best 30's 0.7140471529484217 (+0.0051021536503067 absolute reward).
All per-stage, autonomous final-latent and source-alignment errors are exactly
zero for seeds 101/202/303. Peak CUDA allocation 29480879616 bytes (27.46 GiB),
below 34 GiB. noise_3sigma_pass true. Actual worker feedback and local CLI
exit receipt are retained; no numerical failure was observed for this candidate.
Historical reward comparison is across separate runs, not a new paired 30-vs-31
timing measurement. The real paired baseline is the baseline in this Dev result.

## Best restoration and next decision after 31

Adopt 31 as the highest completed valid public Dev method. Restore its saved
builder to method.py, run python method.py, and verify regenerated program.json
matches both the saved program and feedback digest above. No evaluator rules,
controller code or checkpoint weights were changed. Stop this turn after one
concrete new method evaluation, with its real completed feedback.

Next decision: partition the cached bias-free text attention output projection.
Currently the cached full [2,556,3584] attended tensor is projected and only its
300 text rows survive. Slice these rows before the output linear and test actual
Dev fidelity/latency. This is independent of prior failed biased value projection
row splits; bias-free image output projection partitioning passed historically,
but text GEMM shape changes still need actual feedback. Retain 31 unless a
completed candidate passes quality/allocation and improves its valid reward.
Hidden prompts and the 256-stage formal trajectory remain untested.

## 32: Cached text output projection on text rows

Read AGENTS.md and the recovery notice before continuing. Saved experiment 32
had its builder/program restored, but no completed Dev result was available;
the original pre-recovery tool cell could not establish numerical outcome here.
This turn's foreground command resolves that missing feedback, rather than
repeating an unchanged completed graph to occupy time. No nonexistent process
or result file was polled as a queue-wait strategy.

Start from valid 31. Slice cached full attention output to [2,300,3584] before
the bias-free text output linear, instead of projecting all 556 rows and then
slicing. Keep the input view stage-lifetime and the output projection cached.
Saved builder/program: experiments/32_text_output_rows/. Graph has 95 nodes.
SHA256: 5e219e9cfd7db1e066239f50a3e68e44beae166e6fb1e62abcaf9a4fa12740d6.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true.
Run c26ed9e1397e4d1f83c8a3eae67378f5. All per-stage, final-latent and source
alignment errors zero. Baseline 7125.314554850261 ms; candidate 2014.99169921875
ms; reward 0.7172066322535714. Peak allocation 29376869888 bytes (27.36 GiB).
noise_3sigma_pass true. Actual worker result and CLI receipt retained.

Decision: this is numerically valid but fails to improve the best measured
reward, 31's 0.7191493065987284. Preserve it as a valid alternative and use its
validated narrower output projection to test text query attention separately.
This is a failed optimization decision, not a numerical hard-gate failure.

## 33: Shared key/value construction and cached text-query attention

Start from 32. Remove prefix full-Q concatenation and full eager attention.
Compute text Q rotary for only 300 queries; reuse stage full rotary/repeated
keys and repeated values from the image attention branch. Explicit BF16
scores, BF16 scale and mask addition, FP32 softmax and BF16 probabilities
preserve the published eager composition. Keep all 556 key/value positions
and the original text-query mask rows, including padded query behavior.
Prefix-cache the text attention nodes and text output projection. Image
attention, both MLPs and biased text Q/K/V projection shapes remain unchanged.
The interpreter executes every node even if consumers are cached, so prefix
intermediates must also be retained to avoid repeated computation.

Saved builder/program: experiments/33_text_query_attention/. Public graph
syntax/topology validation passed locally for 112 nodes; trusted worker also
validated shapes and allocations. SHA256:
36f9688eff8cf0243f158e8c62a55e5eb07a30113092561640566f0524f56d22.
Foreground exit 0; execution_status COMPLETED, quality_pass true, no stderr.
Run d91901df120c4ca1a9a12e72455add2a. All stage, autonomous final and source
alignment errors exactly zero for seeds 101/202/303. Baseline mean
7134.268391927083 ms; candidate 1997.3652534484863 ms; reward
0.7200322242279751. Peak 32908100608 bytes (30.65 GiB), below 34 GiB.
noise_3sigma_pass true. Worker feedback and true CLI exit receipt retained.

Adopt 33 as highest completed valid own Dev reward: +0.0008829176292467
absolute over 31, and +0.0028255919744037 over 32. These candidate comparisons
are across separate runs; their small differences do not establish a paired
33-vs-31 statistical claim. Each reported reward uses its own actual paired
baseline. Current worker/source binding differs from the recovered historical
31 run; no evaluator or controller files were changed by this research.
No numerical hard-gate or infrastructure failures occurred in this turn's
two foreground evaluations. The non-improving 32 is retained unchanged.

## Restoration and next decision after 32/33

Restore saved 33 builder into method.py, regenerate program.json, and verify
its canonical SHA256 against saved graph and completed Dev feedback. Stop
after these two distinct experiments. No frozen weights or evaluator rules
were modified. Hidden prompts/256-stage formal trajectories remain untested.

Next decision: restrict cached text attention to 300 text keys as well as
300 queries and use attention_eager on matching shorter Q/K/V. Valid causal
text queries cannot attend image columns. Padded text queries may change
when image keys are removed, but padding columns are masked from valid text
and image queries at every layer; safety must be confirmed by actual Dev.
This could eliminate the large prefix score/probability caches introduced by
33 while reducing first-stage arithmetic. Slice attention mask in both query
and key dimensions and slice rotary cos/sin positions consistently. Keep full
stage image keys/values unchanged. GEMM and softmax dimension changes require
real numerical feedback; retain 33 unless a tested candidate passes gates and
improves measured reward. Useful directions remain, so no exhaustion claim.

## 34: Cached eager attention on text queries and text keys

Read current AGENTS.md and latest research/controller notes first. Start from
33, replacing its explicit cached 300-query/556-key attention with a single
prefix attention_eager node on 300 text queries, keys and values. Slice the
original mask in both query/key dimensions and original cos/sin positions to
300. Keep stage image attention's full 556 keys/values and both branches'
projection/residual/MLP calculations unchanged. Valid causal text queries
cannot attend image columns; padded text query values may differ, but padded
key columns are masked from all valid queries. This argument depends on masks
and selections rather than asserting every padded intermediate is invariant.

Saved builder/program: experiments/34_text_only_eager_attention/. Local public
syntax/topology validation passed for 97 nodes; actual worker validates shapes.
SHA256: 47f604c6e3ceb57f4bf43f2f6c6183d63b980249b8d91f8586a940baf3377932.
Actual foreground Dev exit 0, execution_status COMPLETED, quality_pass true,
no stderr. Run 6bb64651743c431ab84218e03bc6e6f1. All stage, autonomous final
and source alignment errors are zero for seeds 101/202/303. Baseline mean
7144.8637288411455 ms; candidate 2001.7510414123535 ms; reward
0.7198335591297519. Peak CUDA allocation 29073389056 bytes (27.08 GiB),
down from 33's 32908100608 bytes. noise_3sigma_pass true.

Decision: preserve as valid memory-saving alternative, but reward falls short
of 33's 0.7200322242279751 by 0.0001986650982232. This failed to improve the
best measured speed score; it is not a numerical failure. The eager node
internally frees its scores/probabilities instead of retaining each explicit
graph node in the prefix cache, accounting for a substantial allocation drop.
Use the smaller cache as a base to test the unresolved text projection direction.

## 35: Fuse cached text Q/K/V on contiguous text rows

Start from 34. Remove normalized full-sequence concat and the three separate
cached biased text projections over 1112 flattened rows. Concatenate text
Q/K/V weights and biases once per layer/prompt. Reshape contiguous text RMS
output to [600,3584], project once to [600,4608], reshape to [2,300,4608],
and slice Q/K/V channels. Keep stage vision Q/K/V projection, image attention,
cached text-only eager attention and both MLPs unchanged. Different GEMM row
and output dimensions require actual numerical verification despite algebraic
equivalence. This is a new combined graph, not a repeat of prior incomplete
27 or full-sequence fusion 26.

Saved builder/program: experiments/35_fused_text_rows/. Local public validator
syntax/topology passed for 98 nodes. SHA256:
0a36d1ca66595eee9e54bcb50235cfa4cacb1ac18fbd54842d2643a445580972.
Foreground Dev exit 0, execution_status COMPLETED, quality_pass true; no stderr.
Run 396b7a0d7ab34642a724fde6501ba63c. All stage, autonomous final-latent and
source alignment errors zero for seeds 101/202/303. Baseline mean
7160.943623860677 ms; candidate 2000.1761283874512 ms; reward
0.720682603655369. Peak CUDA allocation 29620635136 bytes (27.59 GiB).
noise_3sigma_pass true. Actual result and CLI exit receipt retained.

Decision: adopt 35 as highest completed valid own Dev reward, +0.0006503794273939
absolute over 33 and +0.0008490445256171 over 34. These small comparisons use
separate evaluation runs and do not establish a paired 35-vs-33 speed claim.
All graphs retain their real paired baseline measurements. No hard-gate or
infrastructure failures occurred in these two foreground evaluations. Keep
non-improving 34's full builder/program/feedback, and all existing failures.

## Restoration and next decision after 34/35

Restore saved 35 builder to method.py, run python method.py, and verify graph
identity against saved program and worker artifact SHA256. No evaluator rules,
controller code or checkpoint weights were modified. Stop after these two
concrete methods. Hidden prompts/256-stage formal trajectories remain untested.

Next decision: fuse image query/key rotary work along their head dimension.
Their shared image positions and 128-channel layout permit concatenating
28 query heads with 4 key heads, applying the same rotary composition once,
and splitting the resulting 32 heads. Cache text-key rotary separately and
concatenate it with rotated image keys before the existing repeat_heads.
Earlier isolated text-key caching passed but did not improve the best score;
joint image Q/K rotary is an additional kernel reduction and warrants a new
combined experiment. Measure fidelity and latency rather than assume savings.
Retain 35 unless a completed candidate improves its valid reward. Useful
method-level directions remain; no research-exhausted.json is warranted.

## 36: Joint image query/key rotary and cached text-key rotary

Read current AGENTS.md and latest research notice before edits. Start from 35.
Select contiguous Q/K channels [0:4096] from the fused image projection and
reshape to 32 heads (28 Q followed by 4 K), using their common image positions
to apply the rotary composition once. Split rotated heads into image query
and key. Cache rotary text keys for 300 positions separately; concatenate the
two key sections before the original repeat_heads/score computation. Remove
unused separate image Q/K selections and unrotated full key concatenation.
This combines previously valid text-key caching with an additional reduction
of repeated stage rotary operations. Both arithmetic and matmul operand
strides change, so exact fidelity was not assumed from symbolic equivalence.

Saved builder/program: experiments/36_joint_image_qk_rotary/. Local syntax/
topology validation passed for 101 nodes, with shape validation on the worker.
SHA256: 32ee2f3326b2e38a744d9df56e36e82e21dbb214d1c109e9db85ca4fb803d220.
Actual foreground Dev exit 0, execution_status COMPLETED, quality_pass true;
no stderr. Run 813740f627824017b21517bda19ffe76. All per-stage, autonomous
final and source-alignment errors zero. Baseline mean 7135.065073649089 ms;
candidate 1970.3405202229817 ms; reward 0.7238510791583728, exceeding 35's
0.720682603655369. Peak CUDA allocation 29698049536 bytes (27.66 GiB).
noise_3sigma_pass true. Actual feedback and local CLI exit receipt preserved.
Decision: valid improvement in measured reward; carry forward to sign factoring.

## 37: Preflip the fixed image sine layout

Start from 36. Conventional rotary uses concat(-second_half, first_half)
multiplied by sine. Cache concat(-sine_first_half, sine_second_half) instead
and swap the changing Q/K halves without negating them. Keep the two BF16
multiplications and their BF16 addition. BF16 negation is exact, so the sign
can move from variable input to fixed sine without changing product rounding;
signed zero does not change the required finite prediction checks. The four
additional sine construction/view nodes are prefix lifetime. Remove the
stage-lifetime negative node for joint image Q/K; cached text rotary stays as
before. This removes one CUDA operation per stage per layer after construction.

Saved builder/program: experiments/37_preflipped_image_sine/. Local public
syntax/topology validation passed for 104 nodes. SHA256:
3bc9032f84ce9c3bc2849a65aa8d53814d73529df1c229895bbe38d88ce32964.
Foreground exit 0, execution_status COMPLETED, quality_pass true; no stderr.
Run 1978e082db844255889717b166e280c0. All stage, autonomous final-latent and
source alignment errors zero for seeds 101/202/303. Baseline mean
7129.296920776367 ms; candidate 1953.5431632995605 ms; reward
0.7259837561812724, above 36's 0.7238510791583728 (+0.0021326770228996).
Peak CUDA allocation 29703554560 bytes (27.66 GiB), below 34 GiB.
noise_3sigma_pass true. True worker feedback and CLI exit receipt retained.

Decision: adopt 37 as highest completed valid own Dev reward. These are
separate candidate evaluations, each with its real paired eager baseline;
reward comparisons are not paired statistical tests between candidate methods.
No numerical hard-gate or infrastructure failure occurred in this turn's two
experiments. Existing failed/non-improving methods and feedback remain intact.

## Restoration and next decision after 36/37

Restore saved 37 builder into method.py and regenerate program.json. Verify
canonical SHA256 against saved graph and its completed worker feedback. Stop
after these two concrete changed methods. No evaluator/controller rules or
frozen weights were edited. Hidden/256-stage formal validation remains untested.

Next decision: investigate image MLP contraction orientation. Replace the
opaque image mlp node with three weight-left matmuls on a transposed flattened
input: gate/up weights [18944,3584] times [3584,512], SiLU times up in
[18944,512], then down weight [3584,18944] times the gated activation, and
transpose/reshape back to [2,256,3584]. This changes matrix orientation and
intermediate layout while preserving real-valued algebra; BF16 tiling changes
still need actual Dev fidelity and latency checks. Full gate/up weight fusion
is unsuitable as one tensor: 2*18944*3584 exceeds the 100M-element budget, and
retaining both full fused halves across 28 layers alone would exceed 7 GiB.
The orientation experiment uses existing weights directly and stays within
bounded shapes. Retain 37 until a valid completed candidate improves reward.
Useful directions remain, so no exhaustion declaration is supported yet.

## 38: Image MLP with all checkpoint weights on the left

Read AGENTS.md and latest research/controller notes before editing. Start from
valid best 37. Replace stage image mlp with a flattened/transposed input and
three weight-left matrix products: gate/up [18944,3584] times [3584,512],
SiLU and multiply in [18944,512], down [3584,18944] times [18944,512].
Transpose and reshape output to [2,256,3584] before the original residual.
Weights are referenced directly; no new weight storage/cache is constructed.
This changes matrix orientation and activation memory layout; BF16 kernel
behavior requires actual fidelity validation, not just algebraic reasoning.

Saved builder/program: experiments/38_weight_left_image_mlp/. Local public
syntax/topology checks passed for 112 nodes. SHA256:
b0e4bb9023ea887764fd1a290672b09c5cb7d203040dca4cd0a71d553f8edb94.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run bb5df5957541443f83dab4226755b925. All per-stage, autonomous final
and source alignment errors zero for seeds 101/202/303. Baseline mean
7125.0520579020185 ms; candidate 2001.1666768391926 ms; reward
0.7191365535891342. Peak CUDA allocation 29703554560 bytes (27.66 GiB).
noise_3sigma_pass true. Actual result and local CLI exit receipt retained.

Decision: reject as speed optimization, despite valid predictions/allocation.
Reward is below 37's 0.7259837561812724; candidate time is about 47.62 ms
higher than 37's separately measured 1953.5431632995605 ms. This comparison
does not establish a paired statistical test between candidate methods.
The additional graph operations and changed contraction/layout do not produce
an observed gain. Test only the down projection to isolate this choice.

## 39: Change only the image MLP down-projection orientation

Use 37's original row-major linear gate/up projections, explicit BF16 SiLU
and multiplication, then flatten gated activation [512,18944], transpose it
and multiply checkpoint down weight on the left. Return output through the
same transpose/reshape as 38. This retains original gate/up contraction and
activation layout and isolates the changed down contraction. All other decoder
computations/cache lifetimes remain those of 37.

Saved builder/program: experiments/39_weight_left_image_down/. Local public
syntax/topology checks passed for 112 nodes. SHA256:
227ab1478a5fbc25545e8fcbcb70d6c16682852a7b8dac72927b5aff127d59ac.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run fd27abe92b3b4e5b9e4a43a681bddb23. All stage, autonomous final
and source alignment errors zero. Baseline mean 7129.010869344075 ms;
candidate 2003.7608070373535 ms; reward 0.71892863627774. Peak CUDA allocation
29703554560 bytes (27.66 GiB). noise_3sigma_pass true. Feedback and true CLI
exit receipt are preserved.

Decision: reject this narrower speed optimization too. Its reward is below
37 and slightly below 38; neither provides evidence to replace the opaque mlp
node. Preserve both complete builder/program/result triples as failed speed
attempts, distinguishing them from numerical hard-gate or infrastructure
failures. There were no hard-gate/infrastructure failures this turn.

## Restoration and next decision after 38/39

Restore saved 37 builder to method.py and regenerate program.json. Check its
identity against its saved artifact and completed Dev feedback (SHA256
3bc9032f84ce9c3bc2849a65aa8d53814d73529df1c229895bbe38d88ce32964,
reward 0.7259837561812724). Stop after two distinct tested methods. No evaluator
rules, checkpoint weights or controller code changed. Hidden prompt/256-stage
formal validation remains untested.

Next decision: investigate the image attention score contraction orientation,
computing K times Q-transpose to [2,28,556,256] then transposing to the original
query-by-key score axes. Keep BF16 scale, mask addition, FP32 softmax, BF16
probabilities and probability/value product exactly as 37. This retains eager
rounding boundaries while changing batched GEMM dimensions and operand strides.
It may trade cheaper contraction for a softmax layout copy; real Dev timing
and fidelity will decide. This is a different direction from prior unsuccessful
SDPA and grouped-query folding. Reject further MLP orientation trials without
new evidence, and do not retry either tested graph unchanged. Useful attention
layout directions remain, so an exhaustion declaration is not supported yet.

## 40: Reverse the image attention score matrix product

Read AGENTS.md and latest research/controller notes before edits. Start from
37. Compute repeated key times transposed image query, producing scores
[2,28,556,256], then transpose to original [2,28,256,556] axes before the
unchanged BF16 scale/mask addition, FP32 softmax and BF16 probability cast.
Remove unused key-transpose node. Both dot-product reduction and numerical
boundaries are retained; GEMM orientation and downstream strides change.

Saved builder/program: experiments/40_reversed_score_product/. Local public
syntax/topology checks passed for 105 nodes. SHA256:
2b6b27c65f1ba12d251d35fc0fdc0f70cb68c5a11c8ff552e723545880a39a38.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run 1a14d8b6c9984d918477772be9361579. All stage, autonomous final and
source alignment errors zero. Baseline mean 7111.488184611003 ms; candidate
2071.545510609945 ms; reward 0.7087043588017636. Peak CUDA allocation
29703963648 bytes (27.66 GiB). noise_3sigma_pass true. True worker feedback
and CLI exit receipt preserved.

Decision: reject as optimization: reward below 37's 0.7259837561812724. Reversed
score contraction and the softmax-input layout do not yield an observed gain.
No claim is made about the exact kernel-level source of its regression without
a kernel profile. Preserve this valid but non-improving method; do not repeat
it unchanged. Test the independent probability/value orientation next because
it does not change softmax layout and has different contraction dimensions.

## 41: Reverse only the image attention probability/value product

Start from best 37, with its original score/scale/mask/softmax path. Transpose
repeated values to [2,28,128,556] and BF16 probabilities to [2,28,556,256].
Multiply in channel-by-query orientation, then transpose the result back to
the original heads/query/channel axes before recombination and image output
projection. This tests an independent batched GEMM layout while retaining
eager BF16 probability and output boundaries; no scores or probabilities are
recomputed in a different precision or normalization.

Saved builder/program: experiments/41_reversed_value_product/. Local public
syntax/topology checks passed for 107 nodes. SHA256:
6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723.
Foreground exit 0, execution_status COMPLETED, quality_pass true; no stderr.
Run 229037d94d904550a49edd62f61e57ab. All stage, autonomous final and source
alignment errors zero for seeds 101/202/303. Baseline mean 7134.07145690918
ms; candidate 1935.192061106364 ms; reward 0.7287394620596103. Peak CUDA
allocation 29700810240 bytes (27.66 GiB). noise_3sigma_pass true. True worker
feedback and local CLI exit receipt retained.

Decision: adopt 41 as highest completed valid own Dev reward, exceeding 37 by
0.0027557058783379 absolute. Candidate-method comparisons use separate runs
and do not establish a paired significance test between candidates. Each score
comes from its actual paired baseline. Both graphs passed all numerical and
allocation gates; 40 is a failed speed attempt, not a hard-gate invalid result.
No infrastructure failures occurred. Keep all previous failures and methods.

## Restoration and next decision after 40/41

Restore saved 41 builder into method.py, regenerate program.json, and check
canonical SHA256 against saved program and completed worker feedback. Stop
after two distinct methods. No evaluator rules, controller code or frozen
weights were changed. Hidden prompts and formal 256-stage validation remain
untested.

Next decision: exploit 41's channel-by-query result directly in the image
output projection. Reshape its contiguous [2,28,128,256] result to
[2,3584,256], multiply image output weight [3584,3584] on the left, then
transpose to [2,256,3584] for the image residual. This could avoid the original
head/token recombination copy into image_attended_rows before a row-major
linear projection. It changes contraction orientation and batching, so test
actual BF16 fidelity and latency despite algebraic equivalence. Preserve 41
unless the resulting candidate passes gates and improves reward. This is a
new use of the winning attention layout, not a repeat of unsuccessful MLP
orientation. Useful directions remain; no exhaustion declaration warranted.

## 42: Direct image output projection from channel-by-query attention layout

Read current AGENTS.md and latest research/controller notes before edits. Start
from 41. Reshape contiguous [2,28,128,256] channel_query_output into
[2,3584,256]. Multiply checkpoint image output weight on the left and transpose
its result into [2,256,3584]. Remove the old two head/token transposes and
row-major recombination before linear. This avoids an explicit input packing
step in the graph; actual torch.matmul implementation and stride behavior
still determine execution cost. Different batching/orientation require Dev.

Saved builder/program: experiments/42_direct_channel_output_projection/.
Local public syntax/topology checks passed for 106 nodes. SHA256:
7a929a73852398b80cf05d7d3fb9e834a3037e2677a767c5298addfd9e5fdc74.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run 30f88bd3d1784e0a8669b68ac1bf4420. All stage, autonomous final and
source alignment errors zero. Baseline mean 7127.902903238933 ms; candidate
1983.0036430358887 ms; reward 0.7217970460659884. Peak CUDA allocation
29700810240 bytes (27.66 GiB). noise_3sigma_pass true. Actual feedback and
local CLI exit receipt retained.

Decision: reject as speed optimization; reward below 41's 0.7287394620596103.
The graph's elimination of explicit packing did not produce a measured gain.
Do not infer kernel-level causes from the graph alone. Test a single combined
batch product to distinguish projection batching from orientation/layout.

## 43: Single combined-batch image output projection

Start from 42. Transpose [2,3584,256] into [3584,2,256] and reshape to
[3584,512], explicitly accepting that this packing may copy. Apply the same
image output weight on the left once, then reshape/transpose back to
[2,256,3584] for the residual. This isolates one 512-column GEMM from 42's
batched 256-column contraction while preserving the projection's BF16
boundary. All attention, normalization and MLP nodes remain unchanged.

Saved builder/program: experiments/43_single_matrix_output_projection/.
Local public syntax/topology checks passed for 110 nodes. SHA256:
4b05b8121970537bcea5d1221bbe2c9d7dfd4b6dbac7ef82b5f0870b60cb3b98.
Foreground exit 0, execution_status COMPLETED, quality_pass true; no stderr.
Run 772e56d6629340da924dfb7dce8050fd. All stage, autonomous final and source
alignment errors zero for seeds 101/202/303. Baseline mean
7147.597142537435 ms; candidate 1982.6223805745442 ms; reward
0.7226169381070205. Peak CUDA allocation 29703554560 bytes (27.66 GiB).
noise_3sigma_pass true. Actual feedback and CLI exit receipt preserved.

Decision: reject this variant as speed optimization too. Both 42 and 43 are
valid but below best 41; combining batches supplies no evidence to replace
its original image output linear. Preserve both full builder/program/feedback
triples as failed speed attempts. No numerical hard-gate or infrastructure
failures occurred. Cross-run candidate scores are not a paired candidate
significance test; each score still uses its own actual paired baseline.

## Restoration and next decision after 42/43

Restore saved 41 builder into method.py, run python method.py and verify
regenerated graph against saved program and completed feedback SHA256
6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723
(reward 0.7287394620596103). Stop after two distinct tested methods. No
evaluator/controller rules or frozen weights changed. Formal hidden prompt
and 256-stage validation remain untested.

Next decision: group only the reversed probability/value product by KV head.
Keep 41's score/mask/softmax path, including repeated keys, unchanged. Reshape
contiguous BF16 probabilities from [2,28,256,556] to [2,4,1792,556], multiply
unrepeated transposed values [2,4,128,556] by transposed probabilities, then
reshape the result [2,4,128,1792] into [2,4,128,7,256]. Restore token/head
axes directly before the original output linear, avoiding an unnecessary
intermediate repacking. This removes sevenfold stage value replication and
reduces value-product batch count from 56 to 8. Prior unsuccessful grouped
query approaches altered both score and output paths; this isolates the
winning reversed output contraction only and is a different combined method.
BF16 contraction tiling still needs real numerical feedback. Reject further
output-projection orientation variants without new evidence. Useful grouped
output work remains, so no research-exhausted.json is warranted yet.

## 44: Group only the reversed probability/value contraction by KV head

Read AGENTS.md and latest research/controller notes before edits. Start from
41. Remove value_repeat, reshape contiguous image probabilities from
[2,28,256,556] to [2,4,1792,556], and use unrepeated values transposed to
[2,4,128,556] on the left of the transposed probabilities. Reshape output to
[2,4,128,7,256], swap KV-head/token then channel/KV-head axes to obtain
[2,256,4,7,128], and pack directly into token rows for the original output
linear. This preserves KV-major, seven-query-head ordering while reducing
product batch count from 56 to 8. Score, BF16 scale/mask, FP32 softmax and
BF16 probability boundaries remain 41's; keys are still repeated for scores.

Saved builder/program: experiments/44_grouped_value_product/. Local public
syntax/topology checks passed for 108 nodes. SHA256:
d899ae7245c3bca4b65b43c2fd29d48bbf23c96294a6443f2c554d0b43601c87.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run 788953c95b764b1dabf93e9c6833022e. All stage, autonomous final and
source alignment errors zero for seeds 101/202/303. Baseline mean
7129.094823201497 ms; candidate 1963.951109568278 ms; reward
0.7245160629401872. Peak CUDA allocation 29702759936 bytes (27.66 GiB).
noise_3sigma_pass true. True worker result and CLI exit receipt retained.

Decision: valid but reject as speed optimization; reward below best 41's
0.7287394620596103. Removing sevenfold replication alone does not demonstrate
a net gain with this contraction/layout. Try the opposite orientation while
keeping the same isolated output grouping and original score path.

## 45: Grouped probability-left output product

Start from 44. Multiply probabilities [2,4,1792,556] by unrepeated values
[2,4,556,128] directly. Reshape output to [2,4,7,256,128], swap KV-head/token
axes then query-head/KV-head axes into [2,256,4,7,128], and pack for original
output linear. Remove the two operand transposes used by 44. The score and
softmax computations are unchanged; this isolates output-product orientation.

Saved builder/program: experiments/45_grouped_probability_left_product/.
Local public syntax/topology checks passed for 106 nodes. SHA256:
707eedfa9bf3374a74b498cc119b36b38f90790e75184b06aab1493f84ac973b.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run 2d1255df8ae2410ab5a1cf2bfdae63cf. All stage, autonomous final and
source alignment errors zero. Baseline mean 7142.368586222331 ms; candidate
1939.8993962605794 ms; reward 0.7283955073387481. Peak CUDA allocation
29702759936 bytes (27.66 GiB). noise_3sigma_pass true. Actual feedback and
local CLI exit receipt preserved.

Decision: improves over 44 but remains below 41 by 0.0003439547208622 absolute
reward, so do not replace 41. Preserve both complete triples as valid but
non-improving speed attempts. This small cross-run difference is not evidence
of a paired statistically significant candidate gap. All scores are based on
real paired eager baselines. Neither evaluation had hard-gate or infrastructure
failure; all prior failures and methods remain intact.

## Restoration and next decision after 44/45

Restore saved 41 builder to method.py, regenerate program.json, and verify
identity against saved program and worker feedback SHA256
6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723
(reward 0.7287394620596103). Stop after these two distinct methods. No
evaluator/controller rules or frozen weights were changed. Hidden/256-stage
formal validation remains untested.

Next decision: isolate grouping of the score product only on best 41.
Reshape image query [2,28,256,128] into [2,4,1792,128], multiply unrepeated
rotated keys transposed to [2,4,128,556], and reshape resulting scores back to
[2,28,256,556] before the unchanged scale/mask/softmax. Retain 41's repeated
value/reversed output product. This removes sevenfold key replication without
combining two contraction changes. 44/45's zero-error results validate the
chosen KV/head ordering for output grouping; they do not guarantee score
GEMM fidelity or improvement. Prior unsuccessful grouped-query graphs changed
both contractions, whereas this is a narrower new graph on current rotary and
output composition. Measure before combining with 45. Useful score grouping
work remains, so an exhaustion declaration is not yet supported.

## 46: Group only the image score contraction by KV head

Read AGENTS.md and latest research/controller notes before edits. Start from
41. Remove sevenfold key_repeat, reshape image query [2,28,256,128] to
[2,4,1792,128], and multiply by unrepeated rotated keys transposed to
[2,4,128,556]. Reshape scores back to [2,28,256,556] before 41's unchanged
BF16 scale/mask, FP32 softmax, BF16 probability and reversed repeated-value
product. This isolates score grouping, with seven-query-head/KV-major ordering
already validated in the separate output grouping tests.

Saved builder/program: experiments/46_grouped_score_product/. Local public
syntax/topology validation passed for 108 nodes. SHA256:
0eeff2bb646847ee2e031326cd2df908ebb553a9868cafa79e60ad8abeaf7919.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run 01ec19ca53bc44c2b11e477761cda101. All stage, autonomous final and
source alignment errors zero for seeds 101/202/303. Baseline mean
7130.753351847331 ms; candidate 1975.3683522542317 ms; reward
0.7229790101010181. Peak CUDA allocation 29700081152 bytes (27.66 GiB).
noise_3sigma_pass true. Actual result and CLI exit receipt preserved.

Decision: valid but reject as speed optimization against 41's
0.7287394620596103. Avoiding key replication alone does not produce an observed
net improvement. A final combined grouped path can resolve interaction with
45's almost-best output grouping and retain grouping through softmax.

## 47: Group both contractions and retain grouped softmax layout

Combine score grouping with 45's probability-left grouped output. Scores are
reshaped to [2,4,7,256,556]. Prefix-cache the original image mask reshaped to
[2,1,1,256,556] for correct broadcasting across KV/query heads. Keep BF16
scale and mask addition, FP32 softmax on the unchanged 556-key dimension,
and BF16 probability cast. Flatten KV-local query heads to [2,4,1792,556] for
the unrepeated value product, then restore original token/head ordering for
the original image output linear. Both key and value replication disappear.
This carries a shared grouped layout through the full stage attention rather
than assuming independent changes have additive latency effects.

Saved builder/program: experiments/47_grouped_both_products/. Local public
syntax/topology validation passed for 108 nodes. SHA256:
31b7f829a1a1fcde03edea320d33a2e7680d7c7b5f51d70290ada357593585a1.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run e126e34116404aaa96d3ad01848f7ff6. All stage, autonomous final and
source alignment errors zero. Baseline mean 7145.844380696614 ms; candidate
1951.4903869628906 ms; reward 0.7269055575525073. Peak CUDA allocation
29702759936 bytes (27.66 GiB). noise_3sigma_pass true. True worker feedback
and local CLI exit receipt retained.

Decision: combined grouping still does not exceed 41, so reject it for ranking.
Preserve 46 and 47 as valid but non-improving speed attempts. Neither had
hard-gate/infrastructure failure. Grouped scores, reversed/probability-left
grouped outputs and their shared layout have now been tested on the current
rotary/cache composition; further orientation variants need new evidence.
Each score uses its actual paired baseline. Candidate comparisons are across
separate runs, not paired candidate significance tests.

## Restoration and next decision after 46/47

Restore saved 41 builder to method.py, regenerate program.json and verify
identity against saved graph and completed feedback SHA256
6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723,
reward 0.7287394620596103. Stop after these two distinct tested methods. No
evaluator/controller rules or frozen weights were changed. Hidden prompts
and formal 256-stage validation remain untested.

Next decision: test query-axis tiling of 41's image attention into two 128-query
blocks, with shared full repeated keys/values and unchanged eager rounding
boundaries. Each query block independently forms scores, masks, softmax and
reversed probability/value output; concatenate packed token outputs before
the unchanged full 512-row image output projection. This reduces temporary
score/probability working sets but adds launches and packing; measure the
tradeoff rather than assume an L2 or kernel benefit. Softmax retains every
556-key row, so no across-block normalization/reduction is needed. The node
budget permits this without changing catalog operations. Reject further KV
grouping variants without new evidence. Query tiling remains a meaningful
untested direction, so no research-exhausted.json is supported yet.

## 48: Two 128-query attention tiles with shared full keys and values

Read AGENTS.md and latest research/controller notes before edits. Start from
41. Split image queries into contiguous position blocks [0:128] and [128:256].
Each tile uses shared full repeated/rotated keys and transposed repeated values.
Slice cached image attention mask to match query positions. Keep BF16 score,
scale and mask addition, FP32 softmax across all 556 keys and BF16 probability
casting, then use 41's reversed probability/value product. Pack each tile's
token/head output and concatenate token rows before the original 512-row image
output projection. This reduces per-tile score/probability dimensions while
adding independent contractions, pointwise operations and output packing.
No across-tile softmax recombination is required because keys remain full.

Saved builder/program: experiments/48_two_query_tiles/. Local public syntax/
topology validation passed for 122 nodes. SHA256:
5d9ce4f29d4400f68df3acb14529594e3c40106eb00e3f0faee32865dfc7ce9d.
Actual foreground exit 0, execution_status COMPLETED, quality_pass true; no
stderr. Run 23aa0acee30f4c63a1fafd14bc1be799. All stage, autonomous final and
source alignment errors zero for seeds 101/202/303. Baseline mean
7123.281895955403 ms; candidate 1992.5135409037273 ms; reward
0.7202815261270124. Peak CUDA allocation 29707277824 bytes (27.67 GiB).
noise_3sigma_pass true. True worker feedback and CLI exit receipt retained.

Decision: reject query tiling for ranking. Compared with best 41's
0.7287394620596103 reward and 29700810240-byte peak, this smaller attention
working set yields neither higher measured reward nor lower overall peak.
This is a valid non-improving speed attempt, not a numerical hard-gate failure.
No infrastructure failure occurred. Preserve the builder/program/feedback
triple. Candidate comparisons are across separate runs, not paired candidate
significance tests; each reward uses its actual paired eager baseline.

## Evidence-based stopping decision and restoration after 48

Audit every completed valid result currently in experiments/: 41 remains the
highest reward. Recovered own historical evidence also has no higher valid
reward. Restore saved 41 builder into method.py, regenerate program.json and
verify SHA256 against saved artifact and actual feedback:
6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723.
No evaluator/controller rules or frozen weights changed. This turn closes
after one concrete changed method with actual completed public Dev feedback.

The next decision is to stop unsupported method variations under the current
operator catalog, documented in research-exhausted.json. This is an evidence-
based research stopping judgment, not a mathematical proof of global optimality.
Independent prompt computations are cached; text-only RMS/projections/attention/
MLP and fused Q/K/V are already validated. Joint image Q/K rotary, fixed sine
sign factoring and reversed probability/value orientation supplied observed
improvements. Subsequently tested alternatives cover full/down-only MLP
orientation (38/39), reversed scores (40), direct/batched-combined output
projection (42/43), both output grouping orientations (44/45), isolated score
grouping (46), combined grouped attention (47) and query tiling (48). None
exceeds 41. Narrower query tiles add more launches without an observed memory
or reward benefit from the first subdivision; there is no evidence motivating
a further tile-size sweep or permutations of rejected orientations.

Public manifest gate/up shape is [18944,3584]. Full fusion would create a
135790592-element weight, exceeding the interpreter's 100M-element bound.
Splitting that fusion into permissible pieces and caching them all would
retain 7604273152 BF16 bytes across 28 layers, exceeding 7 GiB even before
all existing caches. Building fused copies every stage adds weight traffic
and operations without a supported way to fuse their epilogues. The catalog
offers no custom fused RMS/residual/MLP kernels, sparse updates or incremental
cache mutation. Image hidden states and attention values change across stages,
so unconditional prefix reuse of the remaining image calculations lacks a
correctness argument. Moving scale/projection across BF16 rounding boundaries
does not preserve the tested eager composition. These are limitations and
unvalidated ideas, not fabricated failed experiments.

Public full_decoder_evaluator.py applies final image RMS and the head after
graph.layer; their timed execution is fixed outside the editable layer DAG.
Improving those operations would require changing the evaluator, which is
excluded. No retraining or renderer work applies. Future research should reopen
only with a concrete new algorithmic argument, a new relevant public operation
or profiler evidence identifying a remaining bottleneck; do not repeat the
best graph or rejected graphs just to generate activity. Best public Dev is
not a claim about withheld prompts or formal 256-stage trajectories, which
remain untested. All available failed/non-improving methods and real feedback
are preserved, including recovered original failure records.

## Newly disclosed own experiment 41 stage profile: integrity and stopping review

The newly supplied files contain only the original public three-case, 32-stage
run of experiment 41, not a new evaluation or hidden feedback. Their raw SHA256
`b55e8fa9350f255164bf36130e705826d8c01b54324f424dc62c58de610dbe8f` matches
the source index. Run `229037d94d904550a49edd62f61e57ab` and artifact
`6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723` match the
completed feedback. All 96 records have expected steps, seeds and zero errors;
baseline and candidate sums exactly reproduce aggregate feedback. Integrity
checks and summaries are in `own-public-stage-profile/analysis.json`.

The first stage is 124.099/125.839/126.746 ms for seeds 101/202/303. Later
stage medians are 57.945/58.355/58.610 ms, giving first-stage excesses of
66.153/67.485/68.136 ms. The mean first-stage time is 125.561 ms and mean
later-stage median is 58.303 ms; the 67.258 ms excess is 3.48% of the
1935.192 ms candidate total. This is descriptive, not a removable-overhead
guarantee or formal score extrapolation. The profile has no per-layer or
per-operation timing, so it cannot identify a specific node to recompute or
cache. First construction must remain included, and the interpreter visits
every graph node while retrieving prefix values.

Best 41 already caches text normalization/projections, text attention/output/
MLP, text-key rotary, fixed layouts and fused weights. The timing does not show
exploitable shrinking shapes or unchanged image intermediates, and supplies no
preservation argument for reusing changing image states. Prior real experiments
cover supported factoring, rotary, GEMM orientation, KV grouping, output
projection and query tiling. No new GPU evaluation was performed because the
profile supports no concrete changed graph hypothesis; repeating unchanged
graphs or sweeping rejected layouts would not be evidence-based research.

Decision: retain the stopping judgment in `research-exhausted.json`. Best
remains reward 0.7287394620596103 with zero disclosed prediction/final-latent
errors and 29700810240-byte peak allocation. No evaluator/controller rules,
checkpoint weights, hidden feedback or formal latency were changed. Hidden
prompts and 256-stage formal validation remain untested. Reopen only for
concrete preservation evidence or a profile locating an actionable operator
bottleneck.

## Operator-profile review and reopening for a targeted MLP experiment

Read AGENTS.md, the latest research/controller notice and the preserved
research-exhausted-before-operator-profile.json before selecting a method.
The newly supplied profile is own experiment 41 only: public case 0, seed 101,
first forward and next forward. Source index confirms artifact identity and
lossless AST instrumentation; quality deltas are zero. Verify both raw files
against their listed SHA256 values (d1bf15319dc59922e3d4b5c829c5cf4879caaf166f70b3b49196c9e2b51b650c
and ffe3d927fe1a6dac68c9b47d56877f18591ec2ddd148ae9bacdd388ece84a10a).
Preserve originals and save summaries in own-public-operator-profile/analysis.json.

Each annotation may have CPU-side and device-only records. Keep these separate
to avoid double-counting matching inclusive GPU ranges; they are not independent
measurements or times to sum into decoder wall latency. First-forward image MLP
CPU-side inclusive CUDA time summed across layers is 30.839 ms; next-forward
time is 31.435 ms (device-only summaries approximately 30.958/31.558 ms).
This is the largest recurring annotation. Next-forward text MLP/attention
annotations have zero CUDA work, confirming their prefix reuse. Other recurring
CPU-side inclusive CUDA totals include image post RMS about 1.801 ms and image
softmax about 2.826 ms. Profiler overhead can perturb all of these values; the
instrumented trace is not a new benchmark score or a statistical speed claim.

Unlike whole-decoder stage timings, this supplies an actionable location for
a new focused method: reduce the dominant image MLP's per-call activation
working set by partitioning its independent token rows. Prior 38/39 changed
contraction orientation, while prior 48 partitioned attention queries; neither
tested image MLP row partition. Its outputs are row-local, so complete hidden
and intermediate dot-product reductions remain intact. The tradeoff is more
GEMM calls and repeated weight reads, which may outweigh smaller buffers.
Reopen research for one concrete experiment, not an unchanged rerun.

## 49: Split image MLP into independent contiguous CFG-batch matrices

Start from best 41. Slice image_post_norm along batch into two [1,256,3584]
views and reshape each to [256,3584]. Apply the original opaque mlp operation
with the same gate/up/down weights to each, reshape results and concatenate
along batch before the original image residual addition. Keep all other nodes,
precision boundaries and cache lifetimes unchanged. No weight copies/cache
are added. This halves each MLP call's activation dimensions; actual GEMM
tiling changes still require numerical validation.

Saved builder/program: experiments/49_batch_partition_image_mlp/. Local public
syntax/topology validation passed for 115 nodes. Canonical SHA256:
6bff3922f71f495dad58fe3ea65f3b1ca66ab34e82fefeeb251dcc48b7abc741.
Actual documented foreground Dev exit 0, execution_status COMPLETED,
quality_pass true; no stderr. Run 696ee7d3dbd9413691b1657d7e0f078e.
All per-stage, autonomous final and source alignment errors exactly zero for
seeds 101/202/303. Baseline mean 7111.247863769531 ms; candidate mean
2503.3639195760093 ms; reward 0.6479712186197056. Peak CUDA allocation
29701621760 bytes (27.66 GiB). noise_3sigma_pass true. True worker feedback
and local foreground CLI exit receipt retained.

Decision: reject as speed optimization despite valid predictions and allocation.
Reward is substantially below 41's 0.7287394620596103. Peak allocation also
does not improve. The smaller MLP calls and extra execution do not provide
a net speed benefit; no finer row splits are justified by this outcome. This
is a valid worse speed attempt, not a hard-gate invalid result or infrastructure
failure. Candidate timing comparisons use separate runs, not a paired test
between candidates; each reward has its own real paired eager baseline.

## Restoration and next decision after operator profile and 49

Preserve the failed speed method, its feedback and both prior stopping records.
Update research-exhausted.json with the new evidence, the temporary reopening
and actual failure decision. Restore saved 41 builder to method.py, run
python method.py and verify regenerated graph against its saved program and
worker artifact SHA256 6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723.
Stop after one meaningful new method with real completed Dev feedback. No
evaluator/controller rules or checkpoint weights were changed.

Next decision: retain 41 and stop unsupported variations. The new operator
profile locates recurring image MLP work, but does not decompose its GEMMs,
pointwise kernels or bandwidth. Tested MLP orientation 38/39 and now row
partition 49 do not improve it. Full gate/up fusion remains outside tensor
and retained-budget limits; smaller fused subsets would add partitioned GEMMs
and caching rather than supply a demonstrated kernel advantage. No full MLP
epilogue fusion or custom kernel exists in the catalog. The profile does not
make cached text construction removable or establish image-state invariance.
Other sizeable annotations correspond to fixed RMS/softmax implementations
or tested attention/layout compositions, without a new credible preserving
operator reduction. This is a research stopping judgment, not proof of
global optimality. Reopen for a specific new method argument or kernel-level
evidence, rather than use permutations to occupy time. Hidden/256-stage
formal validation remains untested. Best stays 41 with its original 72.874%
Dev latency reduction, zero disclosed errors and 29700810240-byte peak.

## Finer kernel-profile review and experiment 50 argument

Parsed all 38 ordered event parts locally using own-public-kernel-profile/analyze.py. Every part byte count/SHA256 and category count matches the source index. All 2949 first-forward and 1185 next-forward kernels link to their runtime launches by correlation with matching External id. CPU-op External ids and annotation containment attribute 2940/1176 kernels to layer nodes; the remaining nine in each forward are outside editable layer nodes. Persisted summaries in analysis.json; CPU inclusive times and GPU annotation ranges are not summed with kernels. This remains one public case from the existing experiment 41 diagnostic, not a new score.

Next-forward image MLP kernel sums decompose into down 13.133030 ms, gate 8.237905 ms, up 8.270674 ms, SiLU 0.777657 ms and gating multiply 1.015704 ms across 28 layers. First-forward down/gate/up sums are 12.770027/8.124215/8.169941 ms. The GEMMs have equal multiply counts but the down projection's shape/layout is slower. All 84 next-forward image MLP GEMMs use ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn. This supports one focused physical-weight-layout experiment, distinct from previous weight-left contraction and row partitioning. Kernel timings are perturbed diagnostic measurements, not evidence that a replacement is faster.

50 starts from 41. Keep original row-major gate/up/SiLU/gating and down output shape. Transpose the frozen down weight [3584,18944] to a view [18944,3584], force contiguous storage using one-input concat with prefix lifetime, then use activation-left matmul. The full 18944-dimensional dot-product reduction and BF16 boundaries remain intact. Copy construction is included on the first forward; caching is invariant because checkpoint weights are frozen. A different NN-layout GEMM dispatch may improve the disproportionate down cost, at the expense of 3802136576 added cache bytes and construction/peak memory.

Initial 113-node version failed local retained-size arithmetic: 7673232896 bytes exceeds 7516192768. Save its builder/program/check in experiments/50_contiguous_down_weight/local-overbudget; do not submit a known-overbudget version. Adjust only three text alias views (text_input_norm_rows, text_input_flat, text_qkv) to stage lifetime: their parent tensors remain prefix-cached, so views recreate without CUDA kernels and avoid duplicated logical cache accounting. Revised local public syntax and shape checks pass. Actual worker meta-validation and numerical/latency feedback are still required; local checks do not establish validity.

## 50: Actual Dev result and rejection

Revised 113-node builder/program saved in experiments/50_contiguous_down_weight/.
Canonical SHA256 109d27d7719c73e3b8ccdeb1a695618185d3eb729254466e1eeb6041ecf77699.
Local retained-size estimate 7277559296 bytes for all 28 layers, under 7 GiB.
Actual foreground Dev exited 0; execution_status COMPLETED; quality_pass true.
Run 957863002ecc4771b2170f46b0c5e983. All disclosed per-stage, autonomous final
latent and source alignment errors are exactly zero for seeds 101/202/303.
Baseline mean 7127.302485148112 ms; candidate mean 1960.221701304118 ms;
reward 0.7249700422580868; noise_3sigma_pass true. Peak CUDA allocation
33518438400 bytes (31.216 GiB). Retain actual feedback/stdout/exit receipt.

Decision: reject as speed optimization. Candidate mean is 25.030 ms higher
than 41's separately measured 1935.192061106364 ms, and allocation increases
by 3817628160 bytes. Each candidate reward is paired with its own eager
baseline; this is not a paired statistical test between candidate methods.
The new physical layout preserves predictions but supplies no measured Dev
gain, unlike the specific advantage hypothesized from the diagnostic. This
is a valid worse method, not a hard-gate invalid result or infrastructure
failure. The initial local overbudget method remains saved separately.

## Restoration and evidence-based next decision after finer kernels

The finer trace supported one concrete new weight-storage direction, now
tested. Restore 41, regenerate program.json, and match its original completed
Dev artifact digest. Preserve all failures, prior stopping records, trace
integrity/attribution analysis and experiment 50. No unchanged evaluation is
needed. No evaluator, controller, checkpoint or scoring changes were made.

Next decision: stop under the present operator catalog. Image MLP GEMMs
dominate recurring device work (29.642 ms); the two pointwise kernels total
only 1.793 ms and cannot be fused via this catalog. Previously tested MLP
contraction orientations (38/39), row partitioning (49), and now contiguous
cached down-weight storage (50) do not improve the selected Dev method.
Finer events expose the TN GEMM implementation and per-component cost, but
not cache/bandwidth counters or an alternative advantageous kernel; they
do not justify sweeping weight layouts or reduction tiles. In particular,
there is no new argument for transposing the already faster gate/up weights.
Full gate/up fusion remains outside tensor/cache bounds, while split fusion
adds copies or smaller GEMMs without demonstrated benefit. Normalization,
softmax and layout kernels remain fixed operators or correspond to already
tested attention recombinations. No new image-state invariance or removable
text-prefix work is established. CUDA launch/CPU annotation durations are
not removable GPU work and are not research clocks.

Cache construction is included, so a hypothetical amortized improvement on
256 stages cannot be inferred from this aggregate 32-stage result. Hidden
prompts and formal trajectories remain untested. This is an evidence-based
stopping judgment, not a proof of global optimality. Reopen only for a
specific preserving method supported by new evidence/capabilities. Retain
41's original 0.7287394620596103 Dev reward, zero disclosed errors and
29700810240-byte peak allocation.

Restoration completed: python method.py exited 0; builder bytes match saved 41, regenerated 107-node program exactly matches its saved program and completed worker digest 6a9a696f1debd902737f957e9ed31430d827c7cffc1181b353412e0d9133b723. Receipt saved beside experiment 50. Exactly one new public Dev evaluation this turn.

## Compiler interface review: reopen old-interface stopping decision

Read current operator-catalog.md, tensor_dag.py, bounded_graph_compiler.py and protocol/benchmark.json. The preserved research-exhausted-before-compiler-interface.json concerns schema1. Schema2 now permits 1-16 contiguous nonoverlapping uniform-lifetime regions; the fixed Inductor fullgraph compiler has static shapes, no CUDA graphs, no max autotuning, and emulate_precision_casts enabled. Every prefix node remains counted/cached, executable reuse never captures numerical weights/results, warmup values reset before timing, four shape/stride/dtype/device variants maximum per region. Compiler/resource failures have no eager fallback. Candidate code/flags and numerical standards remain unchanged. Contract file hashes saved with experiment 51.

This removes the old-interface limitation on operator fusion. The own historical kernel diagnostic locates 28 recurring SiLU and 28 gating kernels (1.793 ms total), alongside the image MLP GEMMs. Use that as a hypothesis, not a current score or promised gain. No historical graph is reevaluated unchanged.

## 51: Compile only the image MLP

Start from saved best eager 41. Keep all 107 nodes, topology, weight inputs, and lifetimes. Change schema to bifrost-tensor-dag/2; declare one stage compile region containing only image_mlp. The opaque mlp expands in the trusted compiler to original BF16 linear/SiLU/linear/multiply/linear. Hypothesis: compiler fusion removes intermediate pointwise kernels/traffic; fixed precision emulation must preserve BF16 boundaries. Kernel dispatch or compile overhead can defeat the gain; require real Dev quality/memory/timing feedback. Save builder/program and actual foreground evaluation result, including compiler or resource failures.

51 actual foreground Dev exit 0, execution_status COMPLETED, quality_pass true. Run 3815884c05cb4f49b2a38ce6b663ecfb; canonical SHA256 4d659434f717108367fddf1899f63502c9e01d0e03d1709dad197d6725b1d259. All per-stage, autonomous final-latent and source alignment errors exactly zero. Baseline mean 7127.8356526692705 ms; candidate 1915.588607788086 ms; reward 0.7312524164231079; peak allocation 29695133184 bytes; noise_3sigma_pass true. One compiled executable variant, original disposable warmup first invocation 3.165750072337687 seconds includes compilation and first execution; numerical caches reset before timing. This is execution cost, not research credit or a separate score. Actual feedback/stdout/exit receipt saved.

Decision: select 51 provisionally over historical 41 (reward 0.7287394620596103; candidate 1935.192061106364 ms). Different candidate runs are not a paired significance comparison. The compiler trial passed exact predictions and improves measured reward with slightly lower allocation; next isolate another historically fixed multi-kernel operator.

## 52: Compile both image RMS norms

Start from 51; same nodes/topology/cache lifetimes. Add single-node stage compile regions for image_input_norm and image_post_norm. The own next-forward historical diagnostic attributes eight CUDA kernels per layer per RMS node, 1.857966/1.800759 ms summed across layers. Trusted compilation can fuse FP32 conversion/pow/reduction/rsqrt/normalization/BF16 cast/weight multiplication, retaining required cast boundaries. FP32 reduction order can affect BF16 results, so require all actual fidelity checks rather than assuming algebraic equivalence. Keep image MLP region to retain the new valid method. No prefix compilation, flags, custom kernels or evaluator changes.

52 actual foreground exit 0, execution_status COMPLETED, but quality_pass false and reward -1: hard-gate invalid, excluded from valid ranking. Run a1811e4adc9d47bfb185cf8a5dbdbe0b; canonical SHA256 b0b245a4cf15c8bd279481e15f918f1116eba440c8395c81e3d90a418d83fb01. Worst disclosed stage relative RMSE 0.08022509515285492 and scaled maximum 0.13188976049423218. Autonomous final relative RMSE reaches 1.1046264171600342 and scaled maximum 0.7798507213592529. Source alignment stays zero. Candidate time 1819.4319343566895 ms and peak 29699696128 bytes cannot qualify a failing method. Three region variants compiled; original warmup invocation costs retained in real feedback, not research credit. Compiler execution succeeded; this is a numerical failure, not infrastructure or resource failure.

Decision: reject full RMS compilation despite speed. Fused normalization changes numerical behavior enough to exceed requirements even with trusted precision emulation. Combined trial does not isolate input versus post norm or exact internal arithmetic source, so do not claim proven causal attribution to one reduction. Keep both norms eager in the next trial. Preserve complete failed builder/program, worker feedback and CLI receipt. Best valid remains 51.

## 53: Fuse the changing image Q/K rotary pointwise composition

Start from saved valid 51, retaining the MLP compiled region and eager RMS norms. Add one stage compile region containing the six already-contiguous nodes image_qk_rotary_first, image_qk_rotary_second, image_qk_rotary_half, image_qk_rotary_direct, image_qk_rotary_rotated and image_qk_rotary. Region inputs include changing image Q/K heads and prefix cached cosine/signed sine; output is original rotary value. No cache lifetime/topology/shape changes.

Own historical diagnostic assigns 112 kernels across 28 layers to these concat/two multiply/add operations, about 1.648438 ms. Inductor can fuse them into fewer pointwise launches and avoid the half-swapped intermediate buffer. Unlike RMS, this region has no FP32 reduction or rsqrt; required BF16 multiply/add boundaries are fixed by compiler precision emulation, but actual quality checks still decide validity. This is a distinct method enabled by compile_regions, not a prior unchanged eager permutation.

53 actual foreground Dev exit 0; execution_status COMPLETED; quality_pass true. Run 524ce442da9a4ac8823fd28bb1337869; SHA256 0950ec46626c36eb33a6490c84f1c1c963a4727c4c647708ae6a144322227196. All disclosed per-stage, autonomous final-latent and source-alignment errors exactly zero. Baseline mean 7150.591303507487 ms; candidate 1892.2977180480957 ms; reward 0.735364861767461; peak CUDA allocation 29695583744 bytes; noise_3sigma_pass true. Two regions, each one executable variant. Original warmup first invocations 4.09379010181874 seconds for rotary and 0.5707495743408799 seconds for MLP include compilation/first execution and are not research credit. Numerical prefix values cleared before all timed runs. No actual new operator profile is supplied, so do not claim measured kernel-count reductions from the method result alone.

Decision: select 53 over valid 51 (0.7312524164231079) and historical 41 (0.7287394620596103). Candidate times are measured in separate paired-baseline jobs and this comparison is not a between-candidate paired statistical test. Preserve all three tested builder/program/result/CLI pairs, including numerical failure 52. Stop this turn after three meaningful new compiler experiments with real feedback. No unchanged repeats, standards changes, custom code/flags or delegated research.

## Next decision and restoration after 51-53

The compiler interface has reopened useful directions, so old-interface exhaustion no longer describes the current capability set. Preserve that conclusion in research-exhausted-before-compiler-interface.json and archive any stale active copy beside experiment 51; record current non-exhausted status in research-status.json. No claim of global optimality or formal hidden validation.

Next focused method: compile the BF16 image score scaling and attention-mask addition as a contiguous stage region, keeping the FP32 softmax eager. Move independent prefix image_attention_mask construction just before image_scores or image_scaled to permit uniform region lifetimes. This pointwise composition has no reduction, unlike failing RMS, and fixed precision emulation already passed rotary BF16 multiply/add. It can eliminate an intermediate and launch; test fidelity and timing before adopting. Do not retry full RMS fusion or sweep backend options. Additional boundary fusion can be considered only with a specific graph argument.

Restore best valid 53 builder/program and regenerate with python method.py; verify canonical digest against saved completed Dev result. Hidden prompts/256-stage trajectories remain untested.

Restoration completed: python method.py exited 0; root builder matches saved 53; regenerated 107-node schema2 graph with two regions exactly matches its saved program and completed valid Dev worker artifact digest 0950ec46626c36eb33a6490c84f1c1c963a4727c4c647708ae6a144322227196. Receipt saved beside experiment 53.

## 54: Fuse image score scaling and mask addition

Continue from restored valid 53. Read current AGENTS.md, public catalog/compiler and own research status before editing. Keep image rotary and MLP regions. Move independent prefix image_attention_mask slice just before image_scores, making image_scaled/image_masked adjacent stage nodes. Add one region containing those two nodes; leave score GEMM and FP32 softmax/BF16 probability cast eager. All weight inputs, tensor shapes, cache lifetimes and numerical operation order are unchanged.

Hypothesis: remove a BF16 score intermediate and one pointwise launch while keeping both BF16 arithmetic boundaries via fixed trusted emulate_precision_casts. This has no reduction, like valid rotary fusion, rather than the failed RMS fusion. No compiler flags/evaluator changes. Save the concrete pair in experiments/54_compiled_score_pointwise; require real foreground Dev result, not historical profile timing, to select it.

54 real foreground Dev exit 0; execution_status COMPLETED; quality_pass true; all disclosed per-stage/final/source alignment errors zero. Run 8e490bd75b694f57aa6c04a2ab80041b; canonical digest ab01211eb0e15c77f551e0d7bc96730387f1d264f31e03606c95c043a6bcfe2f. Baseline mean 7140.554056803386 ms; candidate 1894.1740392049153 ms; reward 0.7347301029952736; peak CUDA allocation 29701354496 bytes; noise_3sigma_pass true. Three region variants compiled; all original warmup costs retained as execution observations, not research credit. Actual stdout/exit receipt and result saved.

Decision: preserve as valid non-improving method; retain 53 (0.735364861767461, 1892.2977180480957 ms). Small cross-run difference is not a paired statistical proof, but supplies no measured reason to replace the selected graph. Do not rerun unchanged 54 or claim actual kernel fusion without a new profile.

## 55: Compile softmax with BF16 probability output

Start from saved best 53; keep score scale/mask eager and retain valid rotary/MLP compiled regions. Add a two-node stage region image_prob_fp32/image_prob. FP32 probabilities have no consumer outside the region; only BF16 probabilities cross its output boundary. Historical own next-forward trace attributes 2.825540 ms to softmax (including BF16-to-FP32 conversion) and 1.410773 ms to the BF16 probability cast across 28 layers. Hypothesis: compile the FP32 reduction and BF16 cast together, eliminating the full FP32 probability output buffer and conversion launches.

Full RMS compilation failed, so this different reduction is not assumed numerically valid. Softmax has no RMS rsqrt/weight path; its precision target and final BF16 rounding may preserve predictions, but reduction/exp changes can invalidate it. Actual per-stage/autonomous rollout gates remain authoritative. No region includes score GEMM or BF16 mask arithmetic, isolating the softmax/cast composition. Preserve failures and do not select a faster invalid method.

55 real foreground exit 0; execution_status COMPLETED; quality_pass false; reward -1. Hard-gate numerical failure, not a valid speed improvement or infrastructure failure. Run f7a9a0473294442bbc79e0cc9d98f6bd; digest 030ffd52b82d7ba67176f9534c3d1b0eab3c4074675e118fbb682476fbb1c257. Worst stage relative RMSE 0.060202211141586304 and scaled maximum 0.1254529058933258; autonomous final relative RMSE up to 0.9774453639984131 and scaled maximum 1.2586287260055542. Source alignment remains zero. Baseline/candidate means 7137.437021891276/1832.1168988545735 ms; peak 29696255488 bytes. Three region variants successfully compile; preserve invocation costs as execution observations, not research credit. Actual result/stdout/exit receipt saved.

Decision: reject compiled softmax/cast despite faster measured time; keep eager softmax and probability cast in selected 53. This isolated region changes numerical behavior beyond tolerance. Exact exp/reduction/rounding cause is not attributed from aggregate output alone. No basis for softmax tile sweeps or relaxed standards. Full RMS and softmax reductions have both failed; target pointwise/buffer fusion around valid GEMMs next.

## 56: Extend image MLP compiled region through residual and output packing

Start from saved valid 53. Move prefix text_output just before image_mlp (after its text_mlp dependency), making the tail image_mlp/image_output/output contiguous and stage-only. Replace single image_mlp compile region with these three nodes. Keep all node operators, shapes, precision boundaries, prefix lifetimes and original arithmetic unchanged. Region inputs are image_post_norm, frozen MLP weights, eager image_residual_rows and cached text_output; its sole observable output is original full [2,556,3584] layer output.

Hypothesis: combine final BF16 image residual addition with text/image concatenation so the image_output buffer need not be written separately, retain the already-valid MLP SiLU/gating compilation, and reduce launches. This introduces no new reduction beyond the same checkpoint GEMMs; fixed precision emulation should preserve addition rounding before packing but real fidelity remains required. Prefix numerical construction remains timed and all weights are runtime inputs. Save tested pair and real feedback; do not retain failing or slower enlarged regions automatically.

56 actual foreground Dev exit 0; execution_status COMPLETED; quality_pass true. Run 5019da3c33014b8dbb9410d2d257cb82; digest e7a8626ce8268abe59be949cc3acd774b2685c86ef4ed80222c24e649b657de2. All disclosed stage/autonomous final/source alignment errors zero. Baseline mean 7135.118662516276 ms; candidate 1907.3382352193196 ms; reward 0.7326830392829549; peak CUDA allocation 29696902656 bytes; noise_3sigma_pass true. Two compiled variants; warmup first invocations recorded with real feedback as execution costs, not research credit. Actual result/stdout/CLI receipt preserved.

Decision: reject as speed optimization; retain 53's measured 0.735364861767461 reward and 1892.2977180480957 ms candidate. Region enlargement through residual/packing preserves predictions but supplies no observed gain. This is a valid non-improving method, unlike hard-gate numerical failure 55. Cross-candidate comparisons use different Dev jobs with separate paired baselines, not a candidate-versus-candidate paired test.

## Restoration and next decision after 54-56

Three changed methods evaluated this turn: valid non-improving pointwise score fusion 54, invalid softmax/cast 55, valid non-improving enlarged MLP tail 56. Preserve complete builder/program/feedback/CLI pairs. Restore 53's saved builder and regenerate program.json, verifying its original completed valid worker digest. No unchanged graph reruns, idle work, delegation, flags or evaluator/rule changes. No new kernel profile exists for these compiler methods; specific generated fusion remains a hypothesis rather than a measured kernel-count assertion.

Next focused method: compile the contiguous stage chain value_transpose, probability_transpose, channel_query_output, image_output_heads, image_output_rows, image_attended_rows and image_attention_out together. Keep original reversed probability/value contraction and row-major output linear; region grouping permits compiler layout/buffer decisions across the intermediate transpose/reshape/packing boundary. This is distinct from eager direct weight-left projection experiments 42/43, which changed contraction orientation but could not compile across boundaries. Own historical profile contains materialization/clone work and GEMM cost on this path. Require real Dev checks to assess the new compiler method; no premise that failed eager orientation choices will become faster. Keep opaque RMS and softmax eager after their observed compilation failures.

Useful compiler-boundary directions remain; research-status.json records this rather than declaring exhaustion or inventing untested benefits. Close after three concrete experiments. Hidden prompts and formal 256-stage trajectories remain untested.

Restoration completed: python method.py exit 0; root builder bytes match saved 53 and regenerated graph equals saved program/valid worker digest 0950ec46626c36eb33a6490c84f1c1c963a4727c4c647708ae6a144322227196. Receipt saved beside experiment 56.

## 57: Compile reversed value product through head packing and output linear

Continue from valid 53; read current AGENTS.md/public region contract and own latest results. Add one stage region with seven already-contiguous nodes: value_transpose, probability_transpose, channel_query_output, image_output_heads, image_output_rows, image_attended_rows and image_attention_out. Keep the original value-left attention product, original output linear, eager softmax/RMS and valid MLP/rotary regions. No topology, weight, precision or cache-lifetime changes.

Hypothesis: compiler can choose intermediate storage across the product→transpose→reshape→linear boundary, reducing clones/materialization or graph dispatch. Own historical kernel profile showed packing/copy work on this path. Earlier 42/43 used eager weight-left direct output projection and failed to improve; this trial leaves contractions intact and instead compiles across their boundary. GEMM dispatch/layout may change BF16 results or slow execution; require actual Dev gates. Save concrete pair in experiments/57_compiled_attention_output and preserve every outcome.

57 actual foreground Dev exit 0; execution_status COMPLETED; quality_pass true, all disclosed stage/final/source alignment errors zero. Run 0deeb83ff33f4af09f5ac4e6985ff109; SHA256 67f6e060f924c8b8e4ac94d1c1650aaa4f8ebad746ba29e81187a7a4c27c400f. Baseline/candidate means 7134.2213134765625/1940.9421666463215 ms; reward 0.7279391707431508; peak CUDA allocation 29696140800 bytes; noise_3sigma_pass true. Three compiled variants. Actual feedback/stdout/exit receipt preserved; compilation/first-execution observations remain execution costs.

Decision: reject as speed optimization. No gain from compilation across the value-product/packing/output-linear boundary; retain valid 53. Cross-run ranking is not a paired statistical test between candidates. Do not sweep orientations/region subsets to occupy time.

## 58: Compile image Q/K/V projection through joint Q/K rotary

Start from valid saved 53, not worse 57. Reorder existing stage image_qkv, image_v, image_qk, image_qk_shape, image_qk_heads, six rotary nodes and image_query/image_key into a contiguous 13-node region after prefix image_signed_sin has been constructed. Move v text/image concatenation after that region to follow image_v. All text-prefix nodes stay unchanged; original operators/weights/shapes/precision/lifetimes are identical. Input normalization remains eager, and cached projection weights/bias plus cosine/signed sine are explicit runtime region inputs. Replace the old rotary-only region with this projection+rotary region, retaining valid image MLP compilation.

Hypothesis: joint compilation can optimize strided Q/K reads and head layout across projection/slice/reshape/rotary, reducing materialization or dispatch beyond isolated rotary fusion. image_v is a required boundary output and may keep the QKV storage alive, so do not promise eliminating the entire projection buffer. The existing GEMM/pointwise operations are unchanged; actual backend layout/dispatch can improve or regress and still requires fidelity checks. This explores a different producer boundary from failed value-product/output projection compilation; no RMS/softmax fusion or custom flags.

58 actual foreground exit 0; execution_status COMPLETED; quality_pass true; all disclosed stage/final/source alignment errors zero. Run d2ce038885654b669feafe0d0dc38d95; digest fffc36b59c6b63ed3fb8c175a028ebc615643b606a9c7573eba1067d96d413e5. Baseline/candidate means 7134.444900512695/1900.1405487060547 ms; reward 0.733666658695548; peak CUDA allocation 29693445632 bytes; noise_3sigma_pass true. Two successful compiled variants; all execution/compile observations retained in actual feedback and not research credit.

Decision: valid non-improving method; preserve and reject as speed optimization. Slight peak reduction does not outweigh lower measured reward versus 53. Different candidate runs are not a paired statistical test. Neither broad attention-output compilation 57 nor projection→rotary compilation 58 demonstrates a benefit from extending existing image layout boundaries.

## 59: Compile the cached text MLP

Start from saved valid 53; keep original 107 nodes/lifetimes/topology and image MLP/rotary regions. Add one prefix region containing only text_mlp. The opaque MLP's internal gate/up/SiLU/gating/down computations can fuse without returning internal activations as separate retained nodes. The region returns one original text_mlp tensor, separately cached and counted under unchanged limits. Explicit frozen weight and cached text_post_norm inputs are supplied at runtime; executable reuse never substitutes numerical values. All caches reset before timing, so every prompt's first timed forward still reconstructs the text MLP.

Own first-forward historical kernel profile attributes 35.602834 ms of work to text MLP across all 28 layers. The same image operator successfully compiled with exact predictions in 51/53. This supports a concrete prefix-construction trial; text uses 600 token rows rather than image's 512 so neither accuracy nor dispatch gain is assumed. Expected benefit is only first-forward work, not 32 or 256 repeated savings. No extrapolated formal score. Save exact pair and real foreground feedback.

59 actual foreground Dev exit 0; execution_status COMPLETED; quality_pass true; every disclosed stage/final/source alignment error exactly zero. Run cb4b508f9ded4034a680774988a42c2a; digest a24bef16b52a4eedfc0b68de865b285b88c66887c33c11a8108e9555be2270d6. Baseline mean 7144.466766357422 ms; candidate 1908.4277725219727 ms; reward 0.7328803065459598; peak allocation 29698672128 bytes; noise_3sigma_pass true. Three successful region variants; original warmup text_mlp first invocation 0.5570233073085546 seconds includes compile/first execution and is not research credit. Actual feedback/stdout/exit receipt preserved.

Decision: reject as speed optimization. Prefix text MLP compilation is valid but produces no measured reward gain against selected 53. No claim of per-stage savings or formal amortized benefit. Cross-run comparisons are not candidate-versus-candidate paired statistical tests.

## Restoration and next decision after 57-59

Three concrete new methods tested in actual foreground public Dev. All pass predictions exactly, but 57 attention-output grouping, 58 projection→rotary grouping and 59 prefix MLP compilation do not improve the selected method. Preserve all builder/program/feedback/CLI pairs and prior numerical failures. Restore 53 builder and regenerate program.json, verify against its completed valid worker digest. No unchanged repeats, custom backend/flags, evaluator/rule changes, delegated work or research time claims.

Broad compiled GEMM/packing boundaries now have several valid worse outcomes, so do not sweep their permutations. One distinct exact copy-composition remains supported: compile the already-contiguous key_rotary concatenation, key_repeat head repetition and key_transpose view into a region whose only observable output is key_transpose. The concat [2,4,556,128] temporary is only used by the repeated-head transformation; compilation can directly materialize the concatenated repeated/transposed keys, avoiding an intermediate key buffer and copy launch. This differs from old eager KV-grouped GEMM trials and changes no attention dot-product reduction, score orientation or precision. Subsequent eager score matmul may prefer the original strides, so require actual Dev rather than claim a gain. Keep score GEMM, softmax/RMS and selected image MLP/rotary regions intact.

Record this remaining direction in research-status.json and close after three experiments. No exhaustion claim while this specific preserving copy-fusion argument remains untested. No new kernel profile substantiates generated fusion counts. Hidden prompts/256-stage trajectories remain untested.

Restoration completed: python method.py exit 0; root builder bytes match saved 53; regenerated program matches saved graph and valid worker digest 0950ec46626c36eb33a6490c84f1c1c963a4727c4c647708ae6a144322227196. Receipt retained beside experiment 59.

## 60: Compile exact key concatenation/repetition/transpose

Start from restored valid 53 after reviewing current catalog, AGENTS.md and own notes/status. Add one stage region containing the three already-contiguous nodes key_rotary (concatenate cached text and changing image keys), key_repeat (four KV heads repeated seven times), and key_transpose (swap channel/key-position axes). Its only observable output is original key_transpose; retain selected image rotary/MLP compilation and eager score GEMM, softmax, RMS. All node inputs/operators/shapes/topology/lifetimes stay unchanged.

Hypothesis: direct compiled materialization of repeated concatenated keys avoids the temporary [2,4,556,128] concatenated key buffer/copy before head repetition. No numerical arithmetic or reduction is changed. The eager score GEMM may require a clone or favor original strides, so a copy-fusion argument does not establish speed. Save builder/program and real foreground feedback, including any failure. This is a new exact copy composition, not a repeat of failed eager grouped KV GEMMs or compiled GEMM-boundary permutations.

60 actual foreground Dev exit 0; execution_status COMPLETED; quality_pass true; all disclosed per-stage/autonomous final/source alignment errors exactly zero. Run 2a9d2a4f25a74af4bc997b4d1cf2e1c5; digest 7a03ee54e1212b6c82b482a03cefd3a97443aebde176cff0218e2dd3bff7d723. Baseline mean 7138.54123433431 ms; candidate 1924.7005055745442 ms; reward 0.7303790168897122; peak allocation 29695583744 bytes; noise_3sigma_pass true. Three successful compiler variants; original key region first invocation 0.6504940753802657 seconds includes compile/first execution and is not research credit. Actual stdout/result/exit receipt saved.

Decision: valid non-improving method, rejected as speed optimization. No observed copy-fusion advantage and peak allocation exactly matches selected 53. Preserve graph/result rather than infer success from the nominal buffer-elimination argument. Separate candidate jobs are not a candidate-versus-candidate paired significance test.

## Current compiler-interface stopping judgment after 60

The new interface reopened credible work and produced valid selected 51/53. Ten changed public compiler methods (51-60) have now been evaluated. Image MLP/rotary compilation supplies the selected benefit; full RMS and softmax/cast compilation fails numerical gates; pointwise score fusion, enlarged MLP tail, attention-output region, projection→rotary region, prefix MLP compilation and exact key-copy fusion all produce valid non-improving results. Keep the best actual own Dev result 53 and stop unsupported permutations.

The last specific remaining key-buffer elimination hypothesis has been tested. A similar value-only copy region has no distinct supported beneficial stride/kernel argument after key-copy and broad attention-layout region failures; do not automatically chase every syntactically legal regrouping. Cached prefix regions expose/cache every node, restricting intermediate elimination, and reductions have observed fidelity failures. Old shape/weight orientation, grouped heads, query/MLP partitions and weight-storage trials remain historical non-improving evidence. Compiler flags/custom code, sparse updates and relaxed precision are unavailable. Full gate/up fusion remains outside tensor/cache limits. No additional image invariance follows from the compiler interface.

This is an evidence-based research stopping judgment under the current catalog, not proof of global optimality. Reopen only for a specific new preserving method argument supported by new evidence/capabilities. No unchanged evaluations, idle duration claims or unsupported sweeps. Record rationale and actual evidence in research-exhausted.json; update research-status.json consistently. Historical profiles remain experiment 41 diagnostics, not a new profile of compiled methods. Hidden prompts/formal 256-stage trajectories remain untested.

Restore saved best 53 builder/program; regenerate with python method.py and verify against its original completed valid Dev artifact. Preserve all failures and the old-interface stopping conclusion. Exactly one meaningful new Dev experiment this turn; no evaluator, scoring, numerical, checkpoint or resource rules changed.

Restoration completed: python method.py exited 0; root builder bytes match saved 53; regenerated 107-node/two-region graph equals saved graph and valid worker digest 0950ec46626c36eb33a6490c84f1c1c963a4727c4c647708ae6a144322227196. Receipt saved beside experiment 60.
