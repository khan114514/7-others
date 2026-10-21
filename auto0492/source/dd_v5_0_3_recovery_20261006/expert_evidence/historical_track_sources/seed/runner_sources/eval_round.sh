#!/bin/bash
# Evaluate a trained DD model with 50k FID (single-step protocol)
set -euo pipefail

CKPT="${1:?usage: eval_round.sh <ckpt_path> <sample_dir> [bs]}"
SAMPLE_DIR="${2:?sample_dir required}"
BS="${3:-32}"

PY=historical-resources/venvs/dd1/bin/python
TRAIN_DIR=historical-seed-models/training
VAE_CKPT=historical-resources/models/dd1/VAR/vae_ch160v4096z32.pth
FID_STAT_DIR=historical-resources/models/dd1/fid_stats

mkdir -p "$SAMPLE_DIR"

echo "=== 50k FID Evaluation ==="
echo "Checkpoint: $CKPT"
echo "Sample dir: $SAMPLE_DIR"
echo "Batch size: $BS"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

$PY -u "$TRAIN_DIR/sample_dd.py" \
    --ckpt_path "$CKPT" \
    --vae_ckpt "$VAE_CKPT" \
    --sample_dir "$SAMPLE_DIR" \
    --sample_num 50000 \
    --t_list 0 \
    --loss_split_t 5 \
    --depth 16 \
    --bs "$BS" \
    --fid_statistic "$FID_STAT_DIR" \
    --seed 0

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"

if [ -f "$SAMPLE_DIR/images/fid.txt" ]; then
    echo "FID result:"
    cat "$SAMPLE_DIR/images/fid.txt"
fi
