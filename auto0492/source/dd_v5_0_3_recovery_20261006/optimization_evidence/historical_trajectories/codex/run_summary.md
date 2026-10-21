dd_codex final results, dd-selfproto-v1.1.

Best complete FID-50k: 9.497105760787 (round 15); sole B: 9.663262058821. Gain: 0.166156298033 FID. Saved-weight validation: complete.

| Round | Method | FID-50k | Charged GPU hours | Status |
|---|---|---:|---:|---|
| 1 | official_var_dd_d16_baseline | 9.663262058820692 | N/A | ok |
| 2 | coupled8192_cfg2_nfe32_uniform | 353.06180904535483 | 3.775563 | quality_gate_failed |
| 3 | coupled8192_cfg2_nfe32_prefix0half | 341.9779499196176 | 3.753303 | quality_gate_failed |
| 4 | official_var_dd_d16_best_clean_validation | 9.663262058820692 | N/A | ok |
| 5 | warmstart_official_dd_reuse8192_onepass_uniform | 79.17681249580045 | 0.666110 | quality_gate_failed |
| 6 | warmstart_official_dd_teacher64_onepass_uniform | 9.65872744080815 | 0.019712 | ok |
| 7 | round06_warmstart_best_clean_validation | 9.65872744080815 | N/A | ok |
| 8 | warmstart_official_dd_teacher256_onepass_uniform | 9.709865768623956 | 0.025801 | ok |
| 9 | warmstart_official_dd_reuse256_onepass_prefix0half | 9.735935034553052 | 0.025315 | ok |
| 10 | warmstart_official_dd_teacher256_agreement2x_uniform | 9.696867934838565 | 0.064821 | ok |
| 11 | warmstart_official_dd_reuse256_agreement2x_latehalf | None | 0.059157 | quality_gate_failed |
| 12 | warmstart_official_dd_reuse256_agreement2x_latehalf | 9.576185738625782 | 0.064744 | ok |
| 13 | round12_warmstart_latehalf_best_clean_validation | 9.576185738625782 | N/A | ok |
| 14 | warmstart_official_dd_reuse256_agreement2x_latehalf_fourpass | 10.138528394137666 | 0.087271 | ok |
| 15 | warmstart_official_dd_teacher256_agreement8x_latehalf | 9.497105760787463 | 0.175139 | ok |
| 16 | round15_warmstart_agreement8x_latehalf_best_clean_validation | 9.497105760787463 | N/A | ok |
| 17 | warmstart_official_dd_reuse256_agreement8x_tail3 | 9.66741595867802 | 0.169596 | quality_gate_failed |

Effective duration: 20.561746 hours. Only closed successful process windows count; interruptions, idle periods, queues, infrastructure failures and storage work are excluded.

The paper reports 9.94; the local baseline uses the frozen official single-step chain with batch8 to satisfy the shared-GPU limit. Training is independently reimplemented because official training code was not released.

Warm initialization equals B before training. A matched cold control is needed to quantify its separate post-training contribution. Teacher data reuse incurs its full original generation charge for every candidate. See night_attribution.json for actual data, exposure and budgets.

Frozen sampling seed0. Saved-weight revalidation tests deterministic reproducibility, not cross-seed significance. No 3-sigma sufficiency claim.

The candidate-only 16-group delete-group jackknife gives SE 0.04692955. Baseline features are unavailable, so paired improvement variance and the 3-sigma criterion remain unverified; this CPU diagnostic does not change the formal score or count as effective research time. See historical-codex-workspace/agent-output/diagnostics/round15_candidate_fid_jackknife/report.json.

Earlier v1.0 delivery files and best_method are preserved in v10_preserved_before_night; final_v11 remains the earlier snapshot. RUNNER_ADDENDUM.md is retained unchanged. See run_summary.json for historical failures and unavailable artifacts.

本次恢复审计记录于 2026-10-04T09:50:46.883111+08:00：核对实际进程与 GPU 后确认本轨道无运行中的训练/采样，截止信号仍在；未启动新实验或新评测，未新增有效时长或中断扣除。最终轨迹保持 17 轮，最佳仍为 round15（FID-50k 9.497105760787）。证据：context_recovery_final_20261004_095046.json。

Resume at 2026-10-04T09:52:49+08:00: verified supervisor interruption 2026-10-04T09:51:49+08:00 to 2026-10-04T09:52:49+08:00 (60s), excluded from effective time. No live own experiment or pending round; cutoff remains active. Evidence: context_recovery_final_20261004_095249.json.

Round17 timing correction: evaluation started at 08:03:20 CST before the 08:05 cutoff and finished at 08:47:04 CST under the notice's permission to complete the current loop. The earlier statement that it completed before cutoff was incorrect. Raw evidence: rounds/round17/evaluation_execution.json and CUTOFF_NOTICE.txt. No score or budget changed.

Resume at 2026-10-04T10:00:25+08:00: confirmed restart gap 2026-10-04T09:59:25+08:00 to 2026-10-04T10:00:25+08:00 (60s), excluded outside effective windows. All 17 rounds remain closed; raw scores, exact saved-weight replication, best archive and protected hashes verified. No new training or evaluation. Evidence: context_recovery_final_20261004_100025.json.

Resume verification at 2026-10-04T10:21:42+08:00: cutoff remains active, no own runners or GPU compute processes are live, and no new training or evaluation was started. The 17-round trajectory, final JSON, best archive and protected hashes remain consistent; no effective time was added. Evidence: context_recovery_final_20261004_102142.json.
