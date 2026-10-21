# Visual reasoning inference optimization

## Goal

Reduce end-to-end latency of the frozen visual reasoning encoder by implementing an open Python speculative drafting policy. The trusted target verifies every proposed token. This is a nontraining inference task: training is NOT_APPLICABLE. The Starter returns an empty list and uses ordinary autoregressive decoding.

## Task Setting

Readonly public Dev materials are installed at /workspace/public_assets/dev: samples.json contains all96 questions, labels, categories and image hashes, and images/ contains the matching image bytes. /workspace/public_assets/manifest.json records the frozen split and protocol fingerprint. Inspect these materials with actual tools for method development. They contain no Hidden rows, private Reference, expert evidence or peer history. The image filesystem keeps these materials readonly; the trusted evaluator checks its independently frozen copy.

## Objective and Metrics

Metric: mean per-question end-to-end latency ratio to a fresh ordinary autoregressive calibration in the same batch, minimize. Each question runs both methods with counterbalanced order and the same per-position RNG; full evaluation averages the96 ratios per seed and then all five seed means. Calibration also runs for independent Baseline and Reference anchor measurements. This controls observed GPU speed drift. Raw times are retained. For public Dev candidate scoring, B is the fixed independent ordinary-Baseline anchor mean measured on the 96-question Dev protocol before candidate evaluation. The final private protocol has its own predeclared paired-control normalization: on each of its five seeds, run separate ordinary calibration, ordinary baseline and candidate roles on every private question, average per-question baseline/calibration ratios into B_s and candidate/calibration ratios into M_s, then B=mean(B_s) and M=mean(M_s) over the five seeds. The ordinary baseline implementation, data, seeds, counterbalanced role order and this computation rule are frozen; the private B is a same-batch measured control, not a fixed numerical Dev anchor reused on private rows. It is computed solely from ordinary baseline/calibration observations, independently of candidate output or latency values. U=0 is the frozen mathematical lower bound. Score=(B-M)/(B-U), continuous and unclipped; scores may be negative or greater than1. Any failed per-seed quality gate invalidates the improvement and returns-1: whole-token-sequence agreement>=.99 with ordinary Baseline, answer agreement>=.99, accuracy drop<=.01, parsed-answer rate>=.90. Full fixed-seed evaluation is required for a formal score. Keep genuine failed/worse attempts and their actual scores; do not report quick feedback as a full score.

Evaluation is stochastic. On the public Dev protocol, a statistically distinguishable improvement requires B-M >= 3*sigma_B, where sigma_B is the sample standard deviation (n-1 denominator) of the five independent Baseline anchor ratios under the same fixed protocol. The frozen sigma_B is 0.00046929095522433943; the raw-ratio improvement threshold is 0.0014078728656730182. This is an evidence threshold, not a reward cutoff: smaller valid improvements and worse results retain their actual continuous scores. The quality gates and scoring formula above are unchanged. These Dev sigma and evidence-threshold values are not a private numerical Baseline anchor or a private 3-sigma certification. Complete private reward uses its own same-batch B as defined above; no private Reference improvement is inferred from public Dev evidence.

## Allowed Scope

Modify or add your own method and supporting standard-library Python files under /workspace/solution. Keep research_journal.md as a development record. Use the readonly public Dev materials and the following actual evaluation commands for iteration.

Public development evaluation:
  python /opt/latentcli/evaluate.py --quick --wait
  python /opt/latentcli/evaluate.py --wait
  python /opt/latentcli/evaluate.py --status YOUR_SAVED_JOB_ID --wait

Quick feedback uses seed17 on the96-question dev split. Complete primary evaluation uses fixed seeds17,29,43,71,101 on all96 questions per seed. GPU queues are excluded from the research clock. Each physical model lane has at most one unfinished job. Preserve job tickets and query unfinished jobs after interruptions instead of submitting duplicates. Save supporting executable files under solution; evaluation.py uploads .py source files only.

## Hard Boundaries

The target weights, tokenizer, image processing, prompt, sampling and per-position randomness are frozen. Sampling temperature0.6, top-k50, final output<=128 tokens, total sampled<=256, target forward calls<=256. The shared trusted engine may replay ordinary AR once if a stochastic verification decision has log-score margin<0.25. All proposal computation, IPC, sampling, rejected work, replay and amortized policy startup count in latency. CPU policy requests have0.5sec wall cap; use only standard-library Python, no network or runtime filesystem writes in the evaluation process. Agent development files remain writable here.

Do not modify target weights, trusted verifier, sampler, randomness, scoring, tokenizer or budgets. Do not read Hidden, private Reference, expert evidence, provider credentials or another lane. GPU policies run in an independently confined process. Feedback is useful for research; trusted final reward comes through a separate private verifier channel. Never invent results or request other models.

## Submission Instructions

Submit /workspace/solution/method.py with draft(state), and method_notes.json explaining the method and observed decisions. Supporting standard-library Python code is allowed; the submitted bundle is at most64 .py files and2MiB total. Keep bulky research traces outside executable submission files. Public state contains question, generated_token_ids, decoded_text, remaining_tokens and maximum_draft_tokens. Return a bounded list of integer token IDs, or {"text": "..."}; the trusted tokenizer verifies the first at most min(32, remaining_tokens) encoded IDs. Empty proposals request an ordinary target step. The method may implement grammar construction, token/history reuse, acceptance adaptation, stopping and other code-level mechanisms. Handle empty, incomplete and arbitrary prefixes and independent question resets.

method_notes.json must be a nonempty UTF-8 JSON object at most65536 bytes. The evaluator transfers submitted .py files and method_notes.json to the independent Verifier and checks their exact bytes. Candidate-authored scores and timing claims are not trusted evidence.

## Workflow & Iteration

Read the Starter and interface with actual tools, develop and verify a method, request real GPU feedback, analyze it, revise the method and retain the best valid candidate. Write a compact research_journal.md recording hypotheses, actual evidence, decisions and next experiments. Continue the implement, evaluate, analyze and revise loop; restore the best valid Dev submission before ending. Keep current best source and metadata in /workspace/solution. A single CLI turn ending does not end iterative research. The ordinary Harbor Trial defaults in task.toml are an 1800-second Agent turn and a 10800-second Verifier timeout. Complete private scoring of 1380 questions at five seeds uses the supported operator entry workspace/operator/run_recover_private_native.py, which explicitly selects NoWallPrivateTrial and overrides only the outer Verifier timeout to None. It does not validate completion within the ordinary 10800-second default. This override preserves the full private protocol, all inference/token/forward-call budgets, quality gates, latency accounting and the 0.5-second policy IPC cap. The supported complete-private invocation is documented in workspace/operator/README.md and expert_evidence/supported_invocation/README.md. An ordinary configured Trial is a different invocation and has no supplied complete-private NOP proof under its default timeout.

## Completion Criteria

Before finishing, restore the best valid Dev method.py, its necessary supporting .py files and nonempty method_notes.json. Verify the draft(state) interface and submission size bounds, obtain a real finite full fixed-seed Dev score with every quality gate passed, preserve failed and worse experiments in the journal, and leave all protected assets unchanged. Hidden data, private Reference and peer history must remain unread. Quick feedback alone does not satisfy this checklist.
