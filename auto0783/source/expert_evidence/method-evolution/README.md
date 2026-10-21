# Recorded method evolution and failures

This overview is derived from the actual native authors' recorded messages, selected method docstrings, data-only graphs and original Dev results. It adds no formal method code. All formal implementations retain the original native author SHA identities. Authored claims are kept as authored claims; Dev measurements are distinct from current portable-release full verification.

The allowed work changes decoder execution graphs, operator composition and tensor reuse. Published pretrained weights, decoder depth, inputs, quality gates and the trusted evaluator remain fixed. This is an inference task; it does not establish retraining or complete FLUX image reproduction.

## GPT

The selected data-only graph contains 95 nodes, including 49 explicit prefix-lifetime nodes. These are static source facts, not performance or correctness evidence.

Native method description:

```text
Cache text-only input RMS and compute cached MLP on text rows only.

Emit an operator DAG. This builder executes only in the local candidate container;
the numerical worker accepts JSON, never imports this Python file.
```

The packaged trajectory records 26 native turns. [The complete source trajectory](../trajectory_codex.json) retains each original round, score, exit status, authored exploration summary and failure reason. Unscored rounds receive no prior score; incomplete status snapshots receive no invented completion.

[gpt.json](gpt.json) lists the selected artifact's matching original Dev runs, all original failed/invalid Dev results and every authored turn summary. These are inference Dev measurements, with no promotion to a fresh full result.

## Seed

The selected data-only graph contains 65 nodes, including 23 explicit prefix-lifetime nodes. These are static source facts, not performance or correctness evidence.

Native method description:

```text
Experiment 91: Use `mlp` op on flattened 2D image input.

Building on exp89 (71.63% speedup, BEST).

Idea: Replace manual 5-node MLP (gate+up+silu+mul+down) with single `mlp` op.
Saves 4 per-stage nodes → less Python interpreter overhead.

The `mlp` op internally does: F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)
Same computation, fewer graph nodes.

Exp76 found "mlp op matches manual flattened speed" within noise.
Test with our current optimized setup to confirm.
```

The packaged trajectory records 47 native turns. [The complete source trajectory](../trajectory_seed.json) retains each original round, score, exit status, authored exploration summary and failure reason. Unscored rounds receive no prior score; incomplete status snapshots receive no invented completion.

[seed.json](seed.json) lists the selected artifact's matching original Dev runs, all original failed/invalid Dev results and every authored turn summary. These are inference Dev measurements, with no promotion to a fresh full result.

## Evidence boundaries

The original native messages document partitioning, prefix reuse, attention decomposition, projection fusion and tensor-shape experiments, including quality failures and rejected slower variants. Their detailed causal explanations and experiment counts remain author reports, independently traceable to the exact messages and Dev receipts. No missing ablation, independent rerun, log, score or duration is reconstructed from prose.

The original author Baseline/Reference comparison is separately archived under historical-author-paired-validation. It is neither of the two independent expert searches nor a current portable-release score. The selected GPT full verification is currently queued in the original trusted worker; the earlier Seed full validations belong to their original source binding. Fresh portable NOP, repaired Oracle and both selected method full GPU/CPU gates remain open.

Research-clock declarations, known stalls, missing closure snapshots and the human budget assumption are retained in run_summary.json. No canonical duration PASS follows from these summaries. The required final complete ZIP and two clean, distinct AI QA0.3.4 reviews remain pending.
