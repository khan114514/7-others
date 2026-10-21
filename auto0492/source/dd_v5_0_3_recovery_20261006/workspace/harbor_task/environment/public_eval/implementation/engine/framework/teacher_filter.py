"""Fixed machinery for method-defined filtering of VAR teacher proposals."""
from pathlib import Path

from common import PATCH_NUMS, VAE_PATH, TORCH_HOME, decode_tokens, sha


class StudentAgreement:
    def __init__(self, torch, method):
        import lpips
        from models import build_vae_dd
        from train_student import initialize_student

        if method.initialization != 'warmstart_official_dd_ema':
            raise ValueError('This proposal filter requires the disclosed warm initialization')
        weights = TORCH_HOME / 'hub/checkpoints/alexnet-owt-7be5be79.pth'
        if sha(weights) != '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
            raise RuntimeError('Offline perceptual backbone changed')
        self.torch = torch
        self.vae, self.student = build_vae_dd(device='cuda', patch_nums=PATCH_NUMS, depth=16,
                                             shared_aln=False, attn_l2_norm=True)
        self.vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu', weights_only=True), strict=True)
        self.initialization = initialize_student(self.student, method)
        self.vae.eval().requires_grad_(False)
        self.student.eval().requires_grad_(False)
        self.perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)

    def measure(self, noise, labels, token_ids):
        torch = self.torch
        values = []
        with torch.no_grad():
            for begin in range(0, len(noise), 2):
                eps = noise[begin:begin + 2].cuda()
                classes = labels[begin:begin + 2].cuda()
                ids = token_ids[begin:begin + 2].long().cuda()
                k = torch.zeros(len(eps), dtype=torch.long, device='cuda')
                continuous, logits = self.student(classes, eps, k)
                prediction = self.student.get_final_sequence(continuous, logits, 5)
                predicted_image = decode_tokens(prediction, self.vae)
                target = self.vae.quantize.embedding(ids)
                teacher_image = decode_tokens(target, self.vae)
                losses = self.perceptual(predicted_image, teacher_image).reshape(-1)
                if not torch.isfinite(losses).all():
                    raise FloatingPointError('Nonfinite proposal agreement')
                values.append(losses.cpu())
        return torch.cat(values)
