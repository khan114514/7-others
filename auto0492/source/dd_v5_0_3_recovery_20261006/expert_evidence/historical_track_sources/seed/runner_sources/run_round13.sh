#!/bin/bash
# Round 13: warmstart + cfg=7.0 teacher + ultra-gentle FT (t0_only, 600 steps, lr=5e-7, MSE+CE)
# Protocol: dd-selfproto-v1.1

set -eo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
PYTHON=historical-resources/venvs/dd1/bin/python
WORK=historical-seed-models/round13_cfg7_ultragentle
OUT=historical-seed-workspace/agent-output/rounds/round13_cfg7_ultragentle
LOG=$OUT/run.log

mkdir -p $WORK/teacher $WORK/train $WORK/sample $OUT

TEACHER_CKPT=historical-resources/models/dd1/VAR-teacher/var_d16.pth
VAE_CKPT=historical-resources/models/dd1/VAR/vae_ch160v4096z32.pth
DD_CKPT=historical-resources/models/dd1/VAR-DD/var-d16.pth
FID_STAT=historical-resources/models/dd1/fid_stats/statistic.pth

echo "Round 13: warmstart + cfg=7.0 teacher + t0_only + MSE+CE + 600 steps + lr=5e-7 (FULL CANDIDATE)" | tee $LOG
echo "Protocol: dd-selfproto-v1.1" | tee -a $LOG
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG
echo "" | tee -a $LOG

# === Step 1: Generate teacher samples (cfg=7.0, 50k) ===
echo "=== Step 1: Generate cfg=7.0 teacher samples (50k) ===" | tee -a $LOG
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG

$PYTHON $WORK/../training/generate_teacher_samples.py \
  --teacher_ckpt $TEACHER_CKPT \
  --vae_ckpt $VAE_CKPT \
  --output_dir $WORK/teacher \
  --num_samples 50000 \
  --bs 16 \
  --cfg 7.0 \
  --top_k 900 \
  --top_p 0.95 \
  --seed 1313 \
  --shard_size 5000 2>&1 | tee -a $LOG

echo "Teacher gen done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG
echo "" | tee -a $LOG

# === Step 2: FM Training (warm start, ultra-gentle) ===
echo "=== Step 2: FM Training (600 steps, lr=5e-7, t0_only, MSE+CE, warmstart) ===" | tee -a $LOG
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG

$PYTHON $WORK/../training/train_dd.py \
  --data_dir $WORK/teacher \
  --teacher_ckpt $TEACHER_CKPT \
  --vae_ckpt $VAE_CKPT \
  --output_dir $WORK/train \
  --num_steps 600 \
  --bs 8 \
  --lr 5e-7 \
  --time_sampler t0_only \
  --loss_type mse_ce \
  --ce_weight 0.3 \
  --seed 1313 \
  --ema_decay 0.9999 \
  --budget_hours 4.0 \
  --save_interval 300 \
  --warmstart_ckpt $DD_CKPT 2>&1 | tee -a $LOG

echo "Training done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG
echo "" | tee -a $LOG

# Free teacher sample space before sampling
echo "Cleaning up teacher samples to free disk space..." | tee -a $LOG
# Keep one shard as evidence, delete rest
# Actually per v1.1, we can delete teacher samples after training
rm -f $WORK/teacher/teacher_shard_*.pt
echo "Teacher samples cleaned" | tee -a $LOG
df -h historical-resources/ | tail -1 | tee -a $LOG
echo "" | tee -a $LOG

# === Step 3: 50k Sample + FID Evaluation ===
echo "=== Step 3: 50k Sample + FID Evaluation ===" | tee -a $LOG
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG

TRAINED_CKPT=$WORK/train/dd_ckpt_final.pth
if [ ! -f $TRAINED_CKPT ]; then
    echo "ERROR: trained checkpoint not found at $TRAINED_CKPT" | tee -a $LOG
    exit 1
fi

$PYTHON $WORK/../training/sample_dd.py \
  --ckpt_path $TRAINED_CKPT \
  --vae_ckpt $VAE_CKPT \
  --sample_dir $WORK/sample \
  --sample_num 50000 \
  --t_list 0 \
  --loss_split_t 5 \
  --depth 16 \
  --bs 16 \
  --fid_statistic $FID_STAT \
  --seed 42 2>&1 | tee -a $LOG

echo "Sampling+FID done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG
echo "" | tee -a $LOG

# Save FID result
if [ -f $WORK/sample/images/fid.txt ]; then
    cp $WORK/sample/images/fid.txt $OUT/fid_result.txt
    FID_VAL=$(cat $WORK/sample/images/fid.txt | grep -oP 'FID: \K[0-9.]+')
    echo "FID-50k = $FID_VAL" | tee -a $LOG
fi

# Save method.py and its hash
cp historical-seed-workspace/harbor_task/solution/method.py $OUT/method.py
sha256sum historical-seed-workspace/harbor_task/solution/method.py > $OUT/method_sha256.txt

# Generate 64-image grid
echo "Generating 64-image sample grid..." | tee -a $LOG
$PYTHON -c "
from torchvision.utils import make_grid, save_image
from PIL import Image
import torch
import os

sample_dir = '$WORK/sample/images'
imgs = []
for i in range(64):
    img = Image.open(os.path.join(sample_dir, f'{i}.png')).convert('RGB')
    imgs.append(torch.from_numpy(__import__('numpy').array(img)).permute(2,0,1).float() / 255.0)
grid = make_grid(torch.stack(imgs), nrow=8, padding=2)
save_image(grid, '$OUT/sample_grid64.png')
print('Grid saved')
" 2>&1 | tee -a $LOG

# SHA-256 of sample filenames
echo "Computing sample SHA-256 manifest..." | tee -a $LOG
ls $WORK/sample/images/*.png | sort | xargs sha256sum > $OUT/sample_sha256.txt 2>/dev/null || true

# Clean up sample images (per v1.1 cleanup clause)
echo "Cleaning up 50k sample images (per v1.1 cleanup clause)..." | tee -a $LOG
rm -f $WORK/sample/images/*.png
echo "Sample images cleaned" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Round 13 Complete ===" | tee -a $LOG
echo "End: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG

