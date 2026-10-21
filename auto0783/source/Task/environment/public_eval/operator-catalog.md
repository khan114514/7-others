`program.json` has exactly `schema`, `layer_body`, and `layer_output`. Schema is `bifrost-tensor-dag/1`. `layer_body` contains 1–256 topologically ordered nodes. Each has `id`, `op`, and `inputs`, plus its required operation attributes and optional `lifetime`. The default lifetime is `stage`; `prefix` retains the first result for this node in this layer until the evaluator resets the prompt. Reuse is a numerical optimization whose correctness is checked, not a guarantee that an input is invariant.

External inputs are `state` (2 x 556 x 3584), `image_mask` (2 x 556, automatically unsqueezed by the interpreter), `attention_mask` (2 x 1 x 556 x 556), and `cos`/`sin` (3 x 2 x 556 x 128). There are 300 padded text positions followed by 256 image positions. Weights are named `w:<suffix>` of an exact `vision_language_model.model.layers.<index>.` checkpoint key; the manifest contains all names and shapes. The same body is applied to all 28 layers with their respective weights. A node can reference an external input, earlier node, or frozen layer weight.

| Operation | Input count | Attributes / behavior |
|---|---:|---|
| rms_norm | 2 | value, weight; epsilon 1e-6, FP32 accumulation |
| linear | 2–3 | value, weight, optional bias |
| mlp | 4 | value, gate/up/down weights; SiLU-gated projection |
| mix | 3 | text, image, mask; image*mask + text*(1-mask) |
| add / mul / matmul | 2 | ordinary tensor operations |
| attention_eager / attention_sdpa | 6 | q, k, v, mask, cos, sin; heads28, KVheads4, dimension128 |
| transpose | 1 | integer dim0, dim1 |
| reshape | 1 | shape list with bounded integers, -1, or symbolic dimensions |
| slice | 1 | dim, start, stop; ordinary bounded slicing |
| concat | 1–8 | dim |
| softmax | 1 | dim |
| cast | 1 | dtype: bf16 or fp32 |
| silu | 1 | elementwise SiLU |
| scale | 1 | finite factor, absolute value <=10000 |
| rotary | 3 | value, cos, sin; published multimodal rotary layout |
| repeat_heads | 1 | count from1 through28 |

Symbols for reshape are `batch`, `sequence`, `hidden`, `heads`, `kv_heads`, `head_dim`, resolved from current states and frozen configuration. Tensor rank is at most5, intermediate tensor size at most100 million elements, and the retained prefix cache at most7 GiB. The worker validates the entire graph on meta tensors before candidate GPU operations. See `tensor_dag.py` for the complete validation and execution semantics; the catalog does not supersede that implementation.


## Optional bounded compilation format

Version1 keeps exactly `schema`, `layer_body`, and `layer_output` with schema `bifrost-tensor-dag/1`. Version2 uses schema `bifrost-tensor-dag/2` and exactly one additional field, `compile_regions`. This is a list of 1–16 objects, each with exactly `id` and `nodes`. Region IDs follow the same identifier syntax as node IDs. `nodes` is a nonempty contiguous subsequence of the original ordered node IDs. Regions cannot overlap, reorder nodes, mix prefix/stage lifetimes, contain unknown fields, or have no observable output. All original per-node validation and allocation-free meta inference occur before compiled numerical execution.

Region inputs are all dependencies outside the region. Stage outputs are nodes required outside the region or by `layer_output`. Every prefix node remains separately cached and counted, including views; the 7GiB budget is unchanged. Resetting a prefix clears all such numerical values. Compiled executable code may be reused for subsequent prefixes and layers, but weights and tensors are explicit runtime inputs, not captured prefix results. Up to four shape/stride/dtype/device variants are permitted per region. A fifth variant is rejected. Shape/dtype budgets remain enforced at both boundaries and by full-graph meta validation.

The backend is fixed PyTorch Inductor, fullgraph with static shapes; CUDA graphs and maximum autotuning are disabled. Candidate JSON cannot select backend/compiler flags or provide Python/Triton/CUDA source. Graph breaks or compilation errors are rejected without an eager fallback. First-invocation wall times include compilation and first execution and are reported as such; they are neither pure compiler duration nor research credit. Existing warmup, scoring, first-step numerical cache reconstruction, seeds, repeats and resource/quality bounds are unchanged.

The complete trusted compiler implementation is published as `bounded_graph_compiler.py` alongside `tensor_dag.py` for inspection.
