# Single-cell Response in an Unseen Cellular Context

## Goal
Given single-cell expression under control and stimulation in training cellular contexts, predict stimulated expression for control cells from an unseen context. Submit an executable method that produces a nonnegative cell-by-gene expression matrix. Minimize the mean-expression error on the evaluation context.

## Task Setting
The public training data contain 21,336 cells and 2,000 genes. The public development context provides 811 control cells and 810 stimulated cells; use development targets for assessing and selecting methods. A further evaluation context is withheld from the agent and supplied only to the separate verifier. The initial implementation contains a condition-dependent latent expression model and a complete training recipe. All methods start from fresh initialization for each declared seed. Public grading and final grading use the same execution and metric code.

## Objective and Metrics
Expression is normalized to a per-cell total of 10,000 and transformed with log1p. Genes were selected using only the training cells. The fixed executor fits a MaxAbsScaler on training cells, transforms the training/control inputs, and scores predictions on the original log1p expression scale.

For each seed, let y_g be the mean observed stimulated expression of gene g, and p_g the mean predicted expression. The raw loss is MSE = sum_g (y_g - p_g)^2 / 2,000. Average the raw losses over all formal seeds in `/tests/protocol.json`. R² and Pearson correlation are diagnostic metrics. There are no paired control/stimulated cells; target and predicted cell counts may differ.

The normalized score is (B - MSE)/(B - U), with U = 0, the mathematical MSE floor, and B the unmodified initial method's full formal seed mean for the same split. Scores are continuous and unclipped. Public B is recorded in `/tests/anchors.json` after calibration; final B is computed independently for the withheld split. Formal training seeds are 17, 29, 43, 71 and 97; inference uses training seed + 10,000. Method comparisons use the complete declared seed set, and a reference improvement must exceed three times the Baseline sample standard deviation. Any declared expansion updates the common protocol for all methods.

## Allowed Scope
Edit only `/workspace/solution/method.py`, a regular Python file of at most 1 MiB. You may implement model modules, condition representations, training objectives and update rules, and inference algorithms within this method. You may import installed libraries and the read-only initial backend under `/workspace/starter`. Architecture and training choices may change within the same per-seed execution budget.

Read the public files under `/workspace/public_assets` and public evaluation scripts under `/tests`. Candidate execution receives training data and the chosen context's control cells. It does not receive stimulated development or final evaluation targets. Unsupervised adaptation using the input control cells is allowed. Development targets may guide method selection through public scores; they may not be loaded by the training/prediction process. Use the platform-provided coding and terminal tools. Network access and external web search are disabled during candidate execution.

## Hard Boundaries
The platform provides one GPU, four CPU cores and 32 GiB RAM. The candidate process has 900 seconds per seed for training and prediction, followed by at most 120 seconds for checkpoint reload. Run seeds sequentially. A full public score must finish within 3,600 seconds. Checkpoints must be at most 256 MiB; predictions must match the number/order of input control cells and the supplied gene order, be finite, and be nonnegative. Reloaded predictions must agree within an absolute tolerance of 1e-7.

Keep the evaluator, metrics, preprocessing, public assets, initial backend, seed protocol and entry script unchanged. Do not read withheld targets, reference material, peer workspaces or author records. Do not fabricate results or modify reward files. The coding CLI runs in a local Docker container without external networking. GPU execution is dispatched through a source-only queue to an isolated A100 worker. The worker uses a read-only chroot, a dedicated unprivileged identity, all-thread seccomp, and fixed input files. Candidate code cannot create subprocesses, use network sockets, create files, or change CPU affinity; PyTorch threads are allowed. TensorDataset loaders run synchronously with num_workers=0. The trusted runner saves the checkpoint and prediction through fixed pre-opened outputs. Unavailable infrastructure stops evaluation and does not produce a successful reward. The separate verifier computes all metrics from freshly executed candidate predictions.

## Submission Instructions
The submission is `/workspace/solution/method.py`; there is no submission JSON. Implement these callables:

- `build_model(train_scaled, device, seed)` returns a wrapper whose `.model` is a PyTorch module with a serializable state dictionary.
- `train_model(model, seed)` fits that model using the training data received during construction.
- `predict(model, controls_scaled, scaler, inference_seed)` returns a NumPy-compatible matrix on the original log1p expression scale.

The fixed runner saves `.model.state_dict()` after training. For reload, it calls `build_model` again with the same training data and seed, loads that state dictionary with `weights_only=True` and `strict=True`, sets `.is_trained = True`, then calls `predict` again. All learned state needed for prediction must be in the state dictionary. The entry `/workspace/solution/solve.sh` is fixed. Generated `dev_result.json` is feedback, not a submitted score; candidate-reported scores are ignored.

## Workflow & Iteration
This is an anytime task. Implement a method, run `bash /workspace/solution/solve.sh`, inspect the public raw loss and per-seed diagnostics, then retain or revert the change. Keep the best valid method in `method.py` whenever an improvement is found; the current valid method is scored at completion. The default model typically trains a seed in about six minutes on the observed A100; other platform hardware must be verified against the same execution limits.

## Completion Criteria
The method implements the three callables, completes fresh training for every declared seed, passes prediction/artifact/reload gates, and produces a finite public score through the fixed entry. The final method remains in the submission directory. Protected files remain intact, withheld labels are never accessed, and final scoring occurs in the separate verifier.
