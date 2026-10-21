# Scoring scopes and original paired-control normalization

The public Dev scorer and final private scorer have different predeclared Baseline
measurement scopes. The Dev candidate branch reads the fixed numerical B and
sample sigma from the frozen 96-question five-seed anchor aggregate. Its formal
Baseline/Reference evidence remains unchanged in optimization_evidence/.

The private protocol was hash-checked before the recorded private GPU score. Its
original baseline_anchor_scope explicitly states independent ordinary AR on the
same injected private rows/seeds with fresh same-batch calibration, computed before
candidate normalization. The frozen private scorer runs three independent roles
for every question, with the same per-position RNG and counterbalanced order.
Only ordinary baseline/calibration observations contribute to B. The candidate
source and candidate latency values do not define B. The baseline implementation
and computation rule were frozen; the realized private numerical B is measured.
The Dev B is not reused on private rows. This documentation describes the existing
protocol and corrects a scope ambiguity; it does not change the reward or introduce
a fixed private numeric anchor after observing candidate scores.

For seed s: B_s=mean_i(t_baseline/t_calibration),
M_s=mean_i(t_candidate/t_calibration). Then B=mean_s(B_s), M=mean_s(M_s), U=0,
score=(B-M)/(B-U). With each observed B>0, the function is continuous, strictly
decreasing in M, un-clipped; the ordinary control metric M=B maps to0 and M=U
maps to1. Failed frozen quality gates still return-1. The independently timed NOP
candidate is not required to have identical latency to the control role; its actual
small nonzero reward is retained, not forced to0. A Dev sigma threshold is not a
private improvement certificate. No private Reference improvement is claimed.

private-latency-projection.json preserves all6900 original three-role latency
values, grouped by actual seed and original row order. Question IDs, private
inputs/images/answers/generated text/tokens and other scientific fields are
omitted explicitly; full raw originals remain private with SHA/byte counts.
These times already include original amortized policy startup. No timing value
is replaced, smoothed or rescaled. The supplied pure read-only verifier source
allows independent arithmetic without importing task/candidate/scorer code,
network/GPU access or a new measurement. normalization-replay.json is its actual
trusted-host result. Its B/M/reward agree with every seed summary and recorded
NOP result. Quality checks use the separate original 28-file full raw audit;
this projection does not claim a fresh quality or model execution.

predeclared-private-protocol-binding.json links the original preflight, unchanged
private protocol/scorer, formal Dev aggregate and current private result hashes.
The amended instruction explicitly distinguishes these two scoring scopes. All
scientific raw metrics and executable/configuration/budget bytes are unchanged.
This evidence supplies no independent QA verdict or platform acceptance claim.
