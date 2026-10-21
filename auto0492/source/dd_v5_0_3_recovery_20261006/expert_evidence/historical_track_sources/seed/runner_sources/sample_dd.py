"""
Single-step sampling and FID evaluation for trained DD model.
Follows the same protocol as official sample.py: t_list 0, loss_split_t 5.
"""
import os, sys, argparse, random
import numpy as np
import torch
from torchvision.utils import save_image

sys.path.insert(0, 'historical-resources/autoresearch_tasks/repos/distilled_decoding/VAR')
sys.path.insert(0, 'historical-resources/autoresearch_tasks/repos/distilled_decoding/LlamaGen')

from models import build_vae_dd

def transfer_BLCv_to_B3HW(x, dd_model, vae):
    patch_nums = dd_model.patch_nums[1:]
    cur_idx = 0
    B = x.shape[0]
    f_hat = x.new_zeros(B, dd_model.Cvae, patch_nums[-1], patch_nums[-1])
    for i in range(len(patch_nums)):
        pn = patch_nums[i]
        h_level_i = x[:, cur_idx: cur_idx + pn ** 2]
        h_BChw = h_level_i.transpose_(1, 2).reshape(B, dd_model.Cvae, pn, pn)
        f_hat, _ = vae.quantize.get_next_autoregressive_input(i, len(patch_nums), f_hat, h_BChw)
        cur_idx += pn ** 2
    result = vae.fhat_to_img(f_hat).add_(1).mul_(0.5)
    return result

@torch.no_grad()
def sample_and_eval(
    ckpt_path, vae_ckpt, sample_dir, sample_num=50000,
    t_list="0", loss_split_t=5, depth=16, bs=32,
    fid_statistic_dir=None, seed=0, use_ema=True,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    t_list_int = [int(d) for d in t_list.split(",")]
    
    print(f"[sample_dd] Building VAE and DD model (depth={depth})...")
    vae, dd_model = build_vae_dd(
        device=device, patch_nums=patch_nums,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        num_classes=1000, depth=depth, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
        n_d_embed_enable=True, layerwise_n_d_embed=True,
    )
    
    print(f"[sample_dd] Loading weights...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    dd_state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'trainer' in dd_state_dict:
        if use_ema and 'ema_parameters' in dd_state_dict['trainer']:
            m, u = dd_model.load_state_dict(dd_state_dict['trainer']["ema_parameters"], strict=False)
            print(f"[sample_dd] Loaded EMA weights. Missing={len(m)}, Unexpected={len(u)}")
        else:
            m, u = dd_model.load_state_dict(dd_state_dict['trainer']["var_wo_ddp"], strict=False)
            print(f"[sample_dd] Loaded model weights. Missing={len(m)}, Unexpected={len(u)}")
    else:
        dd_model.load_state_dict(dd_state_dict, strict=True)
        print(f"[sample_dd] Loaded direct state dict")
    
    vae.eval()
    dd_model.eval()
    
    save_path = os.path.join(sample_dir, "images")
    os.makedirs(save_path, exist_ok=True)
    
    labels = torch.randint(0, 1000, (sample_num, ))
    
    img_id = 0
    while img_id < sample_num:
        cur_bs = min(bs, sample_num - img_id)
        class_labels = labels[img_id:img_id + cur_bs].tolist()
        label_B = torch.tensor(class_labels, device=device)
        noise = torch.randn([cur_bs, dd_model.L - 1, dd_model.Cvae]).to(device)
        
        x = noise
        for t in t_list_int:
            tensor_t = torch.zeros([cur_bs]) + t
            x = dd_model.perturb_data(x, tensor_t, noise)
            final_predict_t = dd_model.final_predict(label_B, x, tensor_t)
            final_predict_token, final_predict_logits = final_predict_t
            final_predict_t = dd_model.get_final_sequence(final_predict_token, final_predict_logits, loss_split_t)
            x = final_predict_t
        
        recon_B3HW = transfer_BLCv_to_B3HW(final_predict_t, dd_model, vae)
        
        for i in range(len(recon_B3HW)):
            save_image(recon_B3HW[i], os.path.join(save_path, f"{img_id}.png"))
            img_id += 1
        
        if img_id % 512 == 0:
            print(f"[sample_dd] generate {img_id} images")
    
    fid = None
    if fid_statistic_dir:
        print(f"[sample_dd] Computing FID...")
        from evaluation.fid_score import calculate_fid_given_paths
        fid = calculate_fid_given_paths(
            (None, str(save_path)),
            batch_size=256,
            device=device,
            dims=2048,
            num_workers=8,
            load_act=[fid_statistic_dir, None],
            save_act=[None, None],
        )
        print(f"[sample_dd] FID: {fid}")
        with open(os.path.join(sample_dir, "fid.txt"), "w") as f:
            f.write(f"FID: {fid}")
    
    return fid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--vae_ckpt', required=True)
    parser.add_argument('--sample_dir', required=True)
    parser.add_argument('--sample_num', type=int, default=50000)
    parser.add_argument('--t_list', type=str, default='0')
    parser.add_argument('--loss_split_t', type=int, default=5)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--fid_statistic', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_ema', action='store_true')
    args = parser.parse_args()
    
    sample_and_eval(
        ckpt_path=args.ckpt_path,
        vae_ckpt=args.vae_ckpt,
        sample_dir=args.sample_dir,
        sample_num=args.sample_num,
        t_list=args.t_list,
        loss_split_t=args.loss_split_t,
        depth=args.depth,
        bs=args.bs,
        fid_statistic_dir=args.fid_statistic,
        seed=args.seed,
        use_ema=not args.no_ema,
    )
