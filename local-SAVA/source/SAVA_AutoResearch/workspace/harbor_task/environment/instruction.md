# Hierarchical Transport Valuation

## Goal

Given fixed image embeddings, observed labels and a clean reference sample, return one numerical valuation for every candidate point. Reduce the measured time needed to approximate the canonical hierarchical transport valuation while meeting the accuracy gates below. A valid submission is an executable `solution/method.py` implementing `value(x, y, r, ry, seed)` and the unchanged `solution/solve.sh`.

## Task Setting

This is a CPU numerical inference task with precomputed 10-dimensional embeddings. Feature extraction and model training are outside the task. A working canonical implementation initializes your solution. Public/Dev includes 30 cases, five fixed seeds and six corruption settings. It is evaluated by a protected local entry; corruption truth and numerical targets are not inputs to your method. Final private cases are held by an independent verifier and become available only to that verifier after the agent phase.

## Objective and Metrics

The canonical valuation operator is defined by the initial implementation and the fixed public target records held by the evaluator. It combines empirical class-conditional transport geometry, transport between candidate and reference batches, calibrated point potentials and transport weights across batches. The initial partition uses contiguous batches of at most 1024 points. The implementation exposes the complete numerical recipe, units and regularization conventions; output agreement, rather than matching internal intermediate matrices, determines correctness.

Every case must have normalized RMSE at most 0.01, Spearman rank correlation at least 0.999 and overlap of the highest-valued quarter at least 0.98. Normalized RMSE is `sqrt(mean((prediction - target)^2)) / std(target)` with population standard deviation. Top-quarter overlap is the cardinality of the intersection divided by `floor(N/4)`; stable descending sorting breaks ties by input order. Values must be finite and have exactly shape `(N,)`. Missing cases invalidate the submission.

The raw metric, minimized, is the arithmetic mean of the per-case median method times. Each case uses nine fresh independent invocations. A seed aggregates its six cases, and the final aggregate uses all five seeds. Timing starts after fixed dependencies, inputs and candidate source have been loaded; the trusted parent times execution and output transfer. All candidate label geometry, coupling solves, potential estimation and aggregation are included. Feature extraction, import time and input loading are excluded equally for every method; process wall times are retained separately.

The score is `(B - x) / (B - U)`, with `U = 0`, where `x` is the raw aggregate and `B` is the formal canonical baseline aggregate in the protected protocol. The formula is continuous, monotone and unbounded; it is not clipped. Zero time is the mathematical lower bound, not a reference method's result. Timing has noise; formal comparisons use seeds 101, 202, 303, 404 and 505, all paired. The minimum formal improvement is three times the baseline sample standard deviation, using denominator `n-1`.

## Allowed Scope

Modify `solution/method.py`. Functions, classes, numerical objectives, internal batches, coupling solvers, calibration and aggregation may be implemented freely subject to the output quality requirements. The solver receives NumPy arrays `x: (N,10)`, `y: (N,)`, `r: (M,10)`, `ry: (M,)`, and an integer seed; labels range from 0 to 9. Return a NumPy-compatible floating vector `(N,)`.

The installed numerical libraries are NumPy, SciPy, PyTorch, POT and GeomLoss. Standard `math`, `collections`, `itertools`, `functools`, `typing` and `dataclasses` are also allowed. Your method must derive values from the supplied arrays. Read-only public inputs are under `/workspace/environment/public_assets`. Use `/workspace/solution/solve.sh` to obtain a complete Public/Dev result.

## Hard Boundaries

Use two CPU cores and at most 2048 MiB RAM; no GPU is available. A method call and its initialization each have a 60-second timeout; a complete evaluation has a 7200-second budget. `method.py` is limited to 1 MiB. Keep `solve.sh` unchanged. Only the method file is an editable submission asset. Do not train or change the frozen feature encoder, access private cases, replace the evaluator or fabricate results.

Solver code must not perform file, network, process or interpreter-introspection operations. Imports outside the declared list, dynamic evaluation/import, filesystem I/O and process-control attributes are rejected. The evaluator freezes a copy of the method before execution and runs it as an unprivileged child. Final private inputs and targets are inaccessible to that child except for the arrays explicitly supplied to `value`.

Network access from solver processes is disabled. The coding agent may reach only its task-scoped model transport through the harness; web search and external package installation are unavailable. Protected environment and evaluator files are not editable.

## Submission Instructions

Keep exactly `method.py` and the frozen `solve.sh` as your submitted solution. No submission JSON, learned weights or self-reported scores are required. The evaluator calls the method again on every case and computes output quality and measured time itself. Editing a previous result file cannot improve the score. Public results are written by the protected evaluator under `/opt/public_eval/results`.

## Workflow & Iteration

This is an anytime task. Read the instruction and initial code, run the initial method, implement a method change, run the complete Public/Dev evaluator and keep or revert the change based on its valid aggregate result. Whenever a better valid method is found, restore it as the current `solution/method.py`. The most recent valid solution is submitted. A normal complete iteration should fit within the per-evaluation budget; timeout, format, resource and infrastructure failures are reported separately from numerical quality failures.

## Completion Criteria

The method interface is valid, all cases return finite vectors and pass the numerical quality gates, the protected evaluator produces a finite score, and the latest valid best method is restored in the solution directory. No protected asset has been changed and no private evaluation material has been accessed.
