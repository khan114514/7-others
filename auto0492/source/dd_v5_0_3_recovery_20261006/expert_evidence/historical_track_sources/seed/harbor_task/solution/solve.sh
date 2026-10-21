#!/bin/bash
# DD 候选闭环入口（Starter 骨架，dd-selfproto-v1.0）
# 用法:
#   bash solve.sh repro  <out_dir>   # 阶段一：官方 VAR-DD-d16 单步 FID-50k 复现（Baseline）
#   bash solve.sh train  <out_dir>   # 阶段二：教师样本生成 → 预算内训练 → 50k FID（需先实现训练器）
set -euo pipefail
MODE="${1:?usage: solve.sh repro|train <out_dir>}"
OUT="${2:?out_dir}"
PY=historical-resources/venvs/dd1/bin/python
REPO=historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR
DD_CKPT=historical-resources/models/dd1/VAR-DD/var-d16.pth
VAE_CKPT=historical-resources/models/dd1/VAR/vae_ch160v4096z32.pth
TEACHER_CKPT=historical-resources/models/dd1/VAR-teacher/var_d16.pth
FID_STAT=historical-resources/models/dd1/fid_stats/statistic.pth
FID_STAT_DIR=historical-resources/models/dd1/fid_stats
mkdir -p "$OUT"

case "$MODE" in
  repro)
    cd "$REPO"
    exec "$PY" sample.py --ckpt_path "$DD_CKPT" --vae_ckpt "$VAE_CKPT" \
      --sample_dir "$OUT/repro_d16_step1" --sample_num 50000 \
      --t_list 0 --loss_split_t 5 --depth 16 --bs 64 --fid_statistic "$FID_STAT_DIR"
    ;;
  train)
    echo "TODO: 实现 教师样本生成(teacher=$TEACHER_CKPT) → FM 训练(预算≤4.0 GPU·h) → 50k 单步采样 → FID 评分 的闭环，" >&2
    echo "      方法差异只允许落在 solution/method.py。训练器为论文方法的自行重实现。" >&2
    exit 2
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
