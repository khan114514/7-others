"""
Generate teacher samples (continuous token embeddings + labels) using VAR teacher.
Teacher samples are saved as .pt shards for FM training.

Training code is self-implemented (reimplementation) — official DD training code not released.
"""
import os, sys, argparse, random, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR')
sys.path.insert(0, 'historical-resources/autoresearch_tasks/repos/distilled_decoding/LlamaGen')

from models import build_vae_var
from models.vqvae import VQVAE


def generate_teacher_samples(
    teacher_ckpt, vae_ckpt, output_dir,
    num_samples=50000, batch_size=16, cfg=1.5, top_k=900, top_p=0.95,
    seed=42, shard_size=5000,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    print(f"[teacher_gen] Building VAE and VAR teacher (depth=16)...")
    vae, teacher = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    )
    
    print(f"[teacher_gen] Loading VAE weights...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    vae.eval()
    
    print(f"[teacher_gen] Loading teacher weights...")
    teacher_state = torch.load(teacher_ckpt, map_location='cpu')
    if 'trainer' in teacher_state:
        # Try EMA first, then model
        if 'ema_parameters' in teacher_state['trainer']:
            teacher.load_state_dict(teacher_state['trainer']['ema_parameters'], strict=False)
            print(f"[teacher_gen] Loaded EMA weights")
        elif 'var_wo_ddp' in teacher_state['trainer']:
            teacher.load_state_dict(teacher_state['trainer']['var_wo_ddp'], strict=False)
            print(f"[teacher_gen] Loaded model weights")
    else:
        teacher.load_state_dict(teacher_state, strict=True)
        print(f"[teacher_gen] Loaded state dict directly")
    teacher.eval()
    
    # Pre-generate all labels
    g = torch.Generator()
    g.manual_seed(seed)
    all_labels = torch.randint(0, 1000, (num_samples,), generator=g)
    
    print(f"[teacher_gen] Generating {num_samples} teacher samples (bs={batch_size}, cfg={cfg})...")
    start_time = time.time()
    
    shard_idx = 0
    shard_sequences = []
    shard_labels = []
    
    img_id = 0
    while img_id < num_samples:
        cur_bs = min(batch_size, num_samples - img_id)
        label_batch = all_labels[img_id:img_id + cur_bs].to(device)
        
        # Generate full sequence using teacher AR sampling
        # We start with random noise for all positions
        L_minus_1 = sum(pn * pn for pn in patch_nums[1:])  # L-1 tokens (excluding first)
        noise = torch.randn(cur_bs, L_minus_1, 32, device=device)  # Cvae=32
        
        with torch.no_grad():
            _, sequence = teacher.intermediate_sample(
                0, label_batch, noise,
                end_point=len(patch_nums),
                cfg=cfg, top_k=top_k, top_p=top_p,
            )
        
        # sequence shape: (B, L, Cvae) on CPU
        shard_sequences.append(sequence)
        shard_labels.append(label_batch.cpu())
        
        img_id += cur_bs
        
        if img_id % 512 == 0:
            elapsed = time.time() - start_time
            rate = img_id / elapsed
            eta = (num_samples - img_id) / rate / 60
            print(f"[teacher_gen] {img_id}/{num_samples} | {rate:.1f} samples/s | ETA {eta:.1f} min")
        
        # Save shard
        if len(shard_sequences) * batch_size >= shard_size or img_id >= num_samples:
            shard_seq = torch.cat(shard_sequences, dim=0)
            shard_lbl = torch.cat(shard_labels, dim=0)
            shard_path = os.path.join(output_dir, f'teacher_shard_{shard_idx:03d}.pt')
            torch.save({
                'sequences': shard_seq,  # (N, L, Cvae) float32
                'labels': shard_lbl,    # (N,) int64
                'patch_nums': patch_nums,
            }, shard_path)
            print(f"[teacher_gen] Saved shard {shard_idx}: {shard_seq.shape[0]} samples -> {shard_path}")
            shard_idx += 1
            shard_sequences = []
            shard_labels = []
    
    total_time = time.time() - start_time
    print(f"[teacher_gen] Done! {num_samples} samples in {total_time/60:.1f} min ({num_samples/total_time:.1f} samples/s)")
    print(f"[teacher_gen] {shard_idx} shards saved to {output_dir}")
    
    return shard_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--vae_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard_size', type=int, default=5000)
    args = parser.parse_args()
    
    generate_teacher_samples(
        teacher_ckpt=args.teacher_ckpt,
        vae_ckpt=args.vae_ckpt,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.bs,
        cfg=args.cfg,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        shard_size=args.shard_size,
    )
