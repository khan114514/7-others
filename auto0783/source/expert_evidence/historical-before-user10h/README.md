# Bifrost decoder repaired release candidate

Scope: graph optimization of the published pretrained 28-layer decoder. Fixed checkpoint; no training, fine-tuning or whole-paper FLUX reproduction.

| Method | Own paired Baseline ms | Method ms | Measured reward |
|---|---:|---:|---:|
| nop | 58585.564794 | 58576.389979 | 0.000156605384 |
| oracle | 58342.530207 | 33611.196112 | 0.423898895148 |
| gpt-selected | 58493.324509 | 15093.732189 | 0.741958038529 |
| seed-selected | 57538.715472 | 15788.158289 | 0.725608085626 |

These are four distinct repaired-version native Trials, each with 4608 stage rows, 18 paired repetitions, six final latent artifacts and independent CPU recomputation. Each job compares its method and Baseline on one assigned A100; GPT and Seed are not a direct paired significance comparison. Starter is actually measured and retains its small measured reward from timing noise; scores are not clipped. Full precision, raw clocks, logs, hashes and source binding are in expert_evidence/release-full-validation.

Follow MATERIALIZATION.md before build. Task, Runtime and selected method bytes retain their tested identities. Candidate credentials and trusted resource configuration remain outside the candidate environment. The validation used bifrost-durable-SSH-file-RPC-CLI/1.0, calling real OpenSSH short RPCs with original worker stdout journaling; this infrastructure does not change numerical sources or methods.

Historical native author restoration and actual same-container 12h evidence remain separately labeled. Canonical GPT effective research duration remains NOT_VERIFIED; user budget assumptions are distinct from empirical timing proof.

Status: NOT_READY. New complete ZIP privacy, actual other-host save/fetch and two clean independent QA0.3.4 reviews/consensus remain required. No QA verdict or external submission is claimed.
