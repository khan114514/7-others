"""
Flow Matching training for Distilled Decoding (self-implemented reimplementation).

Official DD training code is not released. This is a reimplementation based on
the paper description (arXiv:2412.17153) and the released model architecture.

Training approach:
- Initialize DD model from teacher VAR weights
- Teacher samples: continuous token sequences from VAR teacher
- FM objective: at each time step t (discrete level index), add noise to tokens
  beyond level t, train model to predict the clean continuous tokens
- Loss: MSE on continuous prediction (noisy region only) + CE on discrete logits
- EMA of model weights for evaluation
"""
import os, sys, argparse, random, time, json, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

sys.path.insert(0, 'historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR')
sys.path.insert(0, 'historical-resources/autoresearch_tasks/repos/distilled_decoding/LlamaGen')

from models import build_vae_dd, build_vae_var


def load_teacher_shards(data_dir):
    """Load all teacher sample shards into memory."""
    shard_paths = sorted(glob.glob(os.path.join(data_dir, 'teacher_shard_*.pt')))
    if not shard_paths:
        raise ValueError(f"No teacher shards found in {data_dir}")
    
    all_sequences = []
    all_labels = []
    patch_nums = None
    
    for sp in shard_paths:
        data = torch.load(sp, map_location='cpu', weights_only=False)
        all_sequences.append(data['sequences'])
        all_labels.append(data['labels'])
        if patch_nums is None:
            patch_nums = data['patch_nums']
    
    sequences = torch.cat(all_sequences, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"[train_dd] Loaded {sequences.shape[0]} teacher samples from {len(shard_paths)} shards")
    print(f"[train_dd] Sequence shape: {sequences.shape}, patch_nums: {patch_nums}")
    return sequences, labels, patch_nums


def sample_t_discrete(batch_size, num_t_levels, sampler_type, level_weights, device, generator,
                    step=0, total_steps=1):
    """Sample discrete time levels t ∈ [0, num_t_levels-1]."""
    if sampler_type == 'uniform':
        return torch.randint(0, num_t_levels, (batch_size,), device=device, generator=generator)
    elif sampler_type == 'level_weighted':
        # Weight by number of tokens at each level (higher res = more tokens = more weight)
        return torch.multinomial(level_weights, batch_size, replacement=True, generator=generator)
    elif sampler_type == 'logit_normal':
        t_cont = torch.sigmoid(torch.randn(batch_size, device=device, generator=generator))
        return (t_cont * num_t_levels).long().clamp(0, num_t_levels - 1)
    elif sampler_type == 'importance_lowt':
        # More weight on lower t values (harder denoising from near-random)
        w = torch.linspace(2.0, 0.5, num_t_levels, device=device)
        return torch.multinomial(w, batch_size, replacement=True, generator=generator)
    elif sampler_type == 'importance_hight':
        # More weight on higher t values (finer details, more tokens)
        w = torch.linspace(0.5, 2.0, num_t_levels, device=device)
        return torch.multinomial(w, batch_size, replacement=True, generator=generator)
    elif sampler_type == "t0_importance":
        # 70% t=0 (full denoising, matches single-step eval), 30% uniform over all
        w = torch.ones(num_t_levels, device=device) * 0.3 / num_t_levels
        w[0] += 0.7
        return torch.multinomial(w, batch_size, replacement=True, generator=generator)
    elif sampler_type == "t0_only":
        # 100% t=0 (full denoising only - for single-step inference)
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    elif sampler_type == "t0_dominant":
        # 90% t=0, 10% uniform
        w = torch.ones(num_t_levels, device=device) * 0.1 / num_t_levels
        w[0] += 0.9
        return torch.multinomial(w, batch_size, replacement=True, generator=generator)
    elif sampler_type == "curriculum_hl":
        # Curriculum: start with high t (easy, teacher-like), gradually shift to low t (hard)
        # Phase 1 (0-20%): t ∈ [6,9] uniform (easy, close to AR teacher task)
        # Phase 2 (20-40%): t ∈ [3,7] uniform (medium)
        # Phase 3 (40-60%): t ∈ [0,5] uniform (medium-hard)
        # Phase 4 (60-80%): t0_importance on t ∈ [0,2] (hard, focus on target)
        # Phase 5 (80-100%): t = 0 only (target task)
        progress = step / max(1, total_steps)
        if progress < 0.2:
            # Phase 1: high t only (levels 6-9)
            lo, hi = 6, 9
            w = torch.zeros(num_t_levels, device=device)
            w[lo:hi+1] = 1.0 / (hi - lo + 1)
        elif progress < 0.4:
            # Phase 2: medium-high t
            lo, hi = 3, 7
            w = torch.zeros(num_t_levels, device=device)
            w[lo:hi+1] = 1.0 / (hi - lo + 1)
        elif progress < 0.6:
            # Phase 3: medium-low t
            lo, hi = 0, 5
            w = torch.zeros(num_t_levels, device=device)
            w[lo:hi+1] = 1.0 / (hi - lo + 1)
        elif progress < 0.8:
            # Phase 4: low t with t0 importance
            w = torch.zeros(num_t_levels, device=device)
            w[0:3] = 1.0 / 3.0
            w[0] = w[0] + 0.5  # extra weight on t=0
            w = w / w.sum()
        else:
            # Phase 5: t=0 only
            w = torch.zeros(num_t_levels, device=device)
            w[0] = 1.0
        return torch.multinomial(w, batch_size, replacement=True, generator=generator)
    elif sampler_type == "curriculum_smooth":
        # Smooth curriculum: logit-normal with shifting mean
        # At start: mean ~0.8 (high t), at end: mean ~0.05 (low t / t=0)
        progress = step / max(1, total_steps)
        # mean shifts from 0.8 to 0.05 over training
        mean_t = 0.8 * (1.0 - progress) + 0.05 * progress
        t_cont = torch.sigmoid(torch.randn(batch_size, device=device, generator=generator) * 1.5 + mean_t * 4 - 2)
        return (t_cont * num_t_levels).long().clamp(0, num_t_levels - 1)
    else:
        raise ValueError(f"Unknown time_sampler: {sampler_type}")


def train_dd(
    data_dir, teacher_ckpt, vae_ckpt, output_dir,
    num_steps=5000, batch_size=8, lr=1e-4,
    time_sampler_type='uniform',
    loss_type='mse_ce',
    ce_weight=0.1,
    seed=42,
    log_interval=100,
    save_interval=2000,
    grad_clip=1.0,
    ema_decay=0.9999,
    budget_hours=4.0,
    warmstart_ckpt=None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    budget_sec = budget_hours * 3600
    
    # Load teacher data
    print(f"[train_dd] Loading teacher samples...")
    teacher_sequences_cpu, teacher_labels_cpu, patch_nums = load_teacher_shards(data_dir)
    num_samples = teacher_sequences_cpu.shape[0]
    
    # Build DD model
    print(f"[train_dd] Building DD model (depth=16)...")
    vae, dd_model = build_vae_dd(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        n_d_embed_enable=True, layerwise_n_d_embed=True,
    )
    
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu', weights_only=False), strict=True)
    vae.eval()
    
    # Initialize DD model
    if warmstart_ckpt:
        print(f"[train_dd] Warm-starting from DD checkpoint: {warmstart_ckpt}")
        ws_state = torch.load(warmstart_ckpt, map_location='cpu', weights_only=False)
        if 'trainer' in ws_state:
            if 'ema_parameters' in ws_state['trainer']:
                m, u = dd_model.load_state_dict(ws_state['trainer']['ema_parameters'], strict=False)
                print(f"[train_dd] Loaded warmstart EMA weights. missing={len(m)}, unexpected={len(u)}")
            else:
                m, u = dd_model.load_state_dict(ws_state['trainer']['var_wo_ddp'], strict=False)
                print(f"[train_dd] Loaded warmstart model weights. missing={len(m)}, unexpected={len(u)}")
        else:
            m, u = dd_model.load_state_dict(ws_state, strict=False)
            print(f"[train_dd] Loaded warmstart direct state dict. missing={len(m)}, unexpected={len(u)}")
    else:
        print(f"[train_dd] Initializing DD from teacher weights...")
        _, teacher_var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
            flash_if_available=True, fused_if_available=True,
            vae_local=vae,
        )
        teacher_state = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
        if 'trainer' in teacher_state:
            if 'ema_parameters' in teacher_state['trainer']:
                teacher_var.load_state_dict(teacher_state['trainer']['ema_parameters'], strict=False)
                print(f"[train_dd] Loaded teacher EMA weights")
            else:
                teacher_var.load_state_dict(teacher_state['trainer']['var_wo_ddp'], strict=False)
                print(f"[train_dd] Loaded teacher model weights")
        else:
            teacher_var.load_state_dict(teacher_state, strict=True)
            print(f"[train_dd] Loaded teacher state dict directly")
        teacher_var.eval()
        
        dd_model.init_with_teacher(teacher_var, init_std=0.02)
        del teacher_var
        torch.cuda.empty_cache()
    
    dd_model.train()
    
    # EMA
    ema_model = deepcopy(dd_model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(dd_model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')
    
    # LR schedule: warmup + cosine decay
    warmup_steps = min(500, num_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay from 1.0 to 0.1
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Noise masks (from model, shape: [num_t_levels, L-1])
    # noise_mask[t]: 1.0 where tokens are noisy at time t
    noise_mask_gpu = dd_model.noise_mask.to(device).bool()  # [num_t_levels, L-1]
    num_t_levels = noise_mask_gpu.shape[0]
    print(f"[train_dd] {num_t_levels} time levels, t in 0..{num_t_levels-1}")
    
    # Level weights
    level_token_counts = noise_mask_gpu.sum(dim=1).float()
    level_weights = level_token_counts / level_token_counts.sum()
    print(f"[train_dd] Level token counts: {level_token_counts.cpu().int().tolist()}")
    
    # Pre-generate training data indices for shuffling
    g = torch.Generator(device='cpu')
    g.manual_seed(seed + 1)
    
    # Keep teacher data on CPU to save GPU memory (move per batch)
    print(f"[train_dd] Keeping teacher data on CPU (batches moved to GPU per step)")
    teacher_sequences = teacher_sequences_cpu.pin_memory()
    teacher_labels = teacher_labels_cpu.pin_memory()
    del teacher_sequences_cpu, teacher_labels_cpu
    
    # Training loop
    print(f"[train_dd] Starting training for {num_steps} steps (budget {budget_hours}h)...")
    print(f"[train_dd] bs={batch_size}, lr={lr}, time_sampler={time_sampler_type}, loss={loss_type}, ce_w={ce_weight}")
    
    train_losses = []
    step = 0
    
    for step in range(num_steps):
        # Check budget
        elapsed = time.time() - start_time
        if elapsed >= budget_sec:
            print(f"[train_dd] Budget exhausted at step {step} ({elapsed/3600:.2f}h). Stopping.")
            break
        
        # Sample batch (from CPU, move to GPU)
        idx = torch.randint(0, num_samples, (batch_size,), generator=g)
        clean_seq = teacher_sequences[idx].to(device, non_blocking=True)  # (B, L-1, Cvae)
        labels = teacher_labels[idx].to(device, non_blocking=True)       # (B,)
        
        # Sample time step t (on CPU, then move to GPU)
        t_int_cpu = sample_t_discrete(batch_size, num_t_levels, time_sampler_type,
                                  level_weights, 'cpu', g,
                                  step=step, total_steps=num_steps)  # (B,) on CPU
        t_int = t_int_cpu.to(device)
        
        # Add noise using model's perturbation pattern
        noise = torch.randn(clean_seq.shape, device=device)
        # Build per-sample noise mask: [B, L-1]
        noise_mask_b = noise_mask_gpu[t_int]  # (B, L-1) bool
        noisy_seq = clean_seq.clone()
        noisy_seq[noise_mask_b] = noise[noise_mask_b]
        
        # t as float CPU tensor for the model (it indexes CPU tensors)
        t_cpu = t_int.float().cpu()
        
        # Forward pass
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            conti_pred, logits_pred = dd_model.final_predict(labels, noisy_seq, t_cpu)
            
            # MSE loss on noisy regions only
            if 'mse' in loss_type:
                mse_loss = F.mse_loss(conti_pred[noise_mask_b], clean_seq[noise_mask_b])
            else:
                mse_loss = torch.tensor(0.0, device=device)
            
            # CE loss on noisy regions (discrete token prediction)
            if 'ce' in loss_type:
                codebook = dd_model.vae_quant_proxy[0].embedding.weight.data  # (V, Cvae)
                clean_noisy = clean_seq[noise_mask_b]  # (N_noisy, Cvae)
                # Nearest codebook token as target
                dists = torch.cdist(clean_noisy.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
                target_tokens = dists.argmin(dim=-1)  # (N_noisy,)
                logits_noisy = logits_pred[noise_mask_b]  # (N_noisy, V)
                ce_loss = F.cross_entropy(logits_noisy, target_tokens)
            else:
                ce_loss = torch.tensor(0.0, device=device)
            
            loss = mse_loss + ce_weight * ce_loss
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(dd_model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Update LR scheduler
        scheduler.step()
        
        # Update EMA
        with torch.no_grad():
            for ema_p, model_p in zip(ema_model.parameters(), dd_model.parameters()):
                ema_p.data.lerp_(model_p.data, 1.0 - ema_decay)
        
        train_losses.append(loss.item())
        
        # Periodic checkpoint save (atomic: write to tmp then rename)
        if (step + 1) % save_interval == 0 and step > 0:
            ckpt_path = os.path.join(output_dir, f'dd_ckpt_step{step+1:06d}.pth')
            tmp_path = ckpt_path + '.tmp'
            try:
                torch.save({
                    'trainer': {
                        'var_wo_ddp': dd_model.state_dict(),
                        'ema_parameters': ema_model.state_dict(),
                    },
                    'config': {
                        'step': step + 1,
                        'batch_size': batch_size,
                        'lr': lr,
                        'time_sampler_type': time_sampler_type,
                    }
                }, tmp_path)
                os.replace(tmp_path, ckpt_path)
                # Keep only the latest checkpoint to save disk space
                old_ckpt = os.path.join(output_dir, f'dd_ckpt_step{step+1-save_interval:06d}.pth')
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
            except Exception as e:
                print(f'[train_dd] WARNING: checkpoint save failed at step {step+1}: {e}')
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
        
        if step % log_interval == 0 or step == num_steps - 1:
            avg_loss = np.mean(train_losses[-log_interval:])
            avg_mse = 0.0
            avg_ce = 0.0
            if 'mse' in loss_type:
                avg_mse = mse_loss.item()
            if 'ce' in loss_type:
                avg_ce = ce_loss.item()
            elapsed_min = elapsed / 60
            avg_t = t_int.float().mean().item()
            print(f"[train_dd] step {step:5d}/{num_steps} | loss={avg_loss:.4f} "
                  f"(mse={avg_mse:.4f}, ce={avg_ce:.4f}) | "
                  f"avg_t={avg_t:.1f} | {elapsed_min:.1f}min")
    
    # Save final checkpoint
    total_time = time.time() - start_time
    actual_steps = step + 1
    ckpt_path = os.path.join(output_dir, 'dd_ckpt_final.pth')
    tmp_path = ckpt_path + '.tmp'
    try:
        torch.save({
            'trainer': {
                'var_wo_ddp': dd_model.state_dict(),
                'ema_parameters': ema_model.state_dict(),
            },
            'config': {
                'patch_nums': patch_nums,
                'depth': 16,
                'num_steps': actual_steps,
                'batch_size': batch_size,
                'lr': lr,
                'time_sampler_type': time_sampler_type,
                'loss_type': loss_type,
                'ce_weight': ce_weight,
                'ema_decay': ema_decay,
                'seed': seed,
                'total_time_sec': total_time,
                'teacher_samples': num_samples,
            }
        }, tmp_path)
        os.replace(tmp_path, ckpt_path)
    except Exception as e:
        print(f'[train_dd] ERROR: final checkpoint save failed: {e}')
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
    print(f"[train_dd] Training done! {actual_steps} steps in {total_time/60:.1f} min")
    print(f"[train_dd] Final loss: {np.mean(train_losses[-100:]):.4f}")
    print(f"[train_dd] Checkpoint saved to {ckpt_path}")
    
    return ckpt_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Teacher sample shards directory')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--vae_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--time_sampler', type=str, default='uniform',
                        choices=['uniform', 'level_weighted', 'logit_normal', 'importance_lowt', 'importance_hight', 't0_importance', 't0_dominant', 't0_only', 'curriculum_hl', 'curriculum_smooth'])
    parser.add_argument('--loss_type', type=str, default='mse_ce')
    parser.add_argument('--ce_weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--budget_hours', type=float, default=4.0)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--warmstart_ckpt', type=str, default=None, help='Warm-start from existing DD checkpoint (official distilled weights)')
    args = parser.parse_args()
    
    train_dd(
        data_dir=args.data_dir,
        teacher_ckpt=args.teacher_ckpt,
        vae_ckpt=args.vae_ckpt,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.bs,
        lr=args.lr,
        time_sampler_type=args.time_sampler,
        loss_type=args.loss_type,
        ce_weight=args.ce_weight,
        seed=args.seed,
        ema_decay=args.ema_decay,
        budget_hours=args.budget_hours,
        warmstart_ckpt=args.warmstart_ckpt,
        save_interval=args.save_interval,
    )
