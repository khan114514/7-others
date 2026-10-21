"""Offline teacher Gaussian/token pairs; DD training is a reimplementation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from common import (PATCH_NUMS, TEACHER_PATH, VAE_PATH, budget_check, decode_tokens,
                    load_method, parameter_telemetry, require_phase_one, save_json,
                    setup_runtime, sha)


def filter_logits(logits, top_k, top_p):
    import torch
    if top_k > 0:
        cutoff = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if 0 < top_p < 1:
        ordered, indices = logits.sort(dim=-1, descending=False)
        discard = ordered.softmax(-1).cumsum(-1) <= 1 - top_p
        discard[..., -1] = False
        mask = torch.zeros_like(discard).scatter(-1, indices, discard)
        logits = logits.masked_fill(mask, -torch.inf)
    return logits.log_softmax(-1)


def teacher_tokens(teacher, vae, noise, labels, config):
    import torch
    from rectified_categorical import CategoricalFlow
    B = len(labels)
    cond = teacher.class_emb(torch.cat((labels, torch.full_like(labels, 1000))))
    level_position = teacher.lvl_embed(teacher.lvl_1L) + teacher.pos_1LC
    next_input = cond.unsqueeze(1) + teacher.pos_start + level_position[:, :1]
    f_hat = noise.new_zeros(B, 32, 16, 16)
    codebook = vae.quantize.embedding.weight
    chunks = []
    cur = 0
    try:
        for block in teacher.blocks:
            block.attn.kv_caching(True)
        for scale, pn in enumerate(PATCH_NUMS):
            x = next_input
            condition = teacher.shared_ada_lin(cond)
            for block in teacher.blocks:
                x = block(x=x, cond_BD=condition, attn_bias=None)
            logits = teacher.get_logits(x, cond)
            guidance = config.guidance * scale / 9
            logits = (1 + guidance) * logits[:B] - guidance * logits[B:]
            logp = filter_logits(logits, getattr(config, 'top_k', 900), getattr(config, 'top_p', .95))
            flow = CategoricalFlow(codebook, logp)
            ids = flow.sample_indices(noise[:, cur:cur + pn * pn],
                                      nfe=getattr(config, 'ode_nfe', 10),
                                      skip_type=getattr(config, 'ode_grid', 'logSNR'))
            chunks.append(ids.cpu().to(torch.int16))
            embeddings = vae.quantize.embedding(ids).transpose(1, 2).reshape(B, 32, pn, pn)
            f_hat, next_input = vae.quantize.get_next_autoregressive_input(scale, 10, f_hat, embeddings)
            cur += pn * pn
            if scale != 9:
                next_input = next_input.reshape(B, 32, -1).transpose(1, 2)
                next_input = teacher.word_embed(next_input) + level_position[:, cur:cur + PATCH_NUMS[scale + 1]**2]
                next_input = next_input.repeat(2, 1, 1)
        return torch.cat(chunks, dim=1)
    finally:
        for block in teacher.blocks:
            block.attn.kv_caching(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1001)
    args = parser.parse_args()
    require_phase_one()
    torch = setup_runtime(seed=args.seed)
    from models import build_vae_var
    from torchvision.utils import save_image
    module, method = load_method(args.method)
    config = method.teacher
    if config.teacher_steps != 10 or config.num_samples < 1:
        raise ValueError('Targets require all ten native VAR scales')
    if config.filter_rule not in ('none', 'student_agreement_per_label') or config.weighting != 'uniform':
        raise NotImplementedError('Unsupported teacher filtering or pair weighting')
    multiplier = getattr(config, 'proposal_multiplier', 1)
    if not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError('Proposal multiplier must be a positive integer')
    filtered = config.filter_rule != 'none'
    if not filtered and multiplier != 1:
        raise ValueError('Unfiltered generation requires one proposal per label')
    if filtered and not hasattr(module, 'select_teacher_candidates'):
        raise ValueError('Filtered methods must implement proposal selection in method.py')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() or list(out.glob('pairs_*.pt')):
        raise RuntimeError('Existing teacher pairs require review; refusing overwrite')
    begin = time.time()
    vae, teacher = build_vae_var(device='cuda', depth=16, patch_nums=PATCH_NUMS,
                                shared_aln=False, attn_l2_norm=True)
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu', weights_only=True), strict=True)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location='cpu', weights_only=True), strict=True)
    vae.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)
    agreement = None
    if filtered:
        from teacher_filter import StudentAgreement
        agreement = StudentAgreement(torch, method)
    rng = torch.Generator(device='cpu').manual_seed(args.seed)
    original_labels = torch.randint(0, 1000, (config.num_samples,), generator=rng)
    labels = original_labels.repeat_interleave(multiplier) if filtered else original_labels
    proposal_count = config.num_samples * multiplier
    shards = []
    buffers = {'noise':[], 'tokens':[], 'labels':[]}
    pending_count = 0
    generated = 0
    proposals = {'noise': [], 'tokens': [], 'labels': [], 'agreement': []}
    with torch.no_grad():
        while generated < proposal_count:
            budget_check(args.deadline, reserve=180)
            batch = min(args.batch_size, proposal_count - generated)
            noise = torch.randn(batch, 680, 32, generator=rng)
            label = labels[generated:generated + batch]
            ids = teacher_tokens(teacher, vae, noise.cuda(), label.cuda(), config)
            if ids.shape != (batch, 680) or not ((ids >= 0) & (ids < 4096)).all():
                raise RuntimeError('Invalid generated teacher token IDs')
            if generated == 0:
                decoded = decode_tokens(vae.quantize.embedding(ids.long().cuda()), vae)
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError('Nonfinite decoded teacher pair')
                save_image(decoded.add(1).mul(.5), out / 'teacher_grid.png', nrow=4)
            for key, value in [('noise',noise), ('tokens',ids), ('labels',label)]:
                buffers[key].append(value.cpu())
                if filtered:
                    proposals[key].append(value.cpu())
            if filtered:
                proposals['agreement'].append(agreement.measure(noise, label, ids))
            pending_count += batch
            generated += batch
            if not filtered and (pending_count >= 256 or generated == proposal_count):
                path = out / f'pairs_{len(shards):05d}.pt'
                data = {k:torch.cat(v) for k, v in buffers.items()}
                torch.save(data, path)
                shards.append({'file':path.name,'count':pending_count,'sha256':sha(path)})
                buffers = {'noise':[], 'tokens':[], 'labels':[]}
                pending_count = 0
            if filtered:
                buffers = {'noise':[], 'tokens':[], 'labels':[]}
                pending_count = 0
            if generated % 256 == 0 or generated == proposal_count:
                print(json.dumps({'teacher_pairs':generated,'elapsed_seconds':time.time()-begin,
                                  **parameter_telemetry()}), flush=True)
    filter_evidence = None
    if filtered:
        data = {k: torch.cat(values) for k, values in proposals.items()}
        indices = module.select_teacher_candidates(data['agreement'], multiplier)
        if indices.dtype != torch.long or indices.shape != (config.num_samples,):
            raise ValueError('Method must return one int64 proposal index per original label')
        if not torch.equal(torch.div(indices, multiplier, rounding_mode='floor'), torch.arange(config.num_samples)):
            raise ValueError('Selection must preserve the exact original label groups')
        if not torch.equal(data['labels'][indices], original_labels):
            raise ValueError('Selection changed the original labels')
        budget_check(args.deadline, reserve=180)
        source = out / 'unfiltered_proposals.pt'
        torch.save(data, source)
        for offset in range(0, config.num_samples, 256):
            path = out / f'pairs_{len(shards):05d}.pt'
            selected = {k: data[k][indices[offset:offset + 256]] for k in ('noise', 'tokens', 'labels')}
            torch.save(selected, path)
            shards.append({'file': path.name, 'count': len(selected['labels']), 'sha256': sha(path)})
        generated = config.num_samples
        selected_ids = data['tokens'][indices[:8]].long().cuda()
        save_image(decode_tokens(vae.quantize.embedding(selected_ids), vae).add(1).mul(.5),
                   out / 'teacher_grid.png', nrow=4)
        filter_evidence = {'proposal_count': proposal_count, 'selected_count': config.num_samples,
            'proposal_multiplier': multiplier, 'metric': 'LPIPS-alex-v0.1 between initialized student single-step output and fixed VAR teacher target',
            'unfiltered_proposals': source.name, 'unfiltered_proposals_sha256': sha(source),
            'selected_indices': indices.tolist(), 'labels_preserved_exact': True,
            'mean_proposal_distance': float(data['agreement'].mean()),
            'mean_selected_distance': float(data['agreement'][indices].mean()),
            'student_filter_initialization': agreement.initialization,
            'training_targets': 'Only VAR-teacher token IDs; student output is used solely to select noise/target proposals.'}
    manifest = {'protocol':getattr(method, 'protocol_version', 'dd-selfproto-v1.0'),'reimplementation':True,'method':asdict(method),
                'method_sha256':sha(args.method),'teacher_sha256':sha(TEACHER_PATH),'vae_sha256':sha(VAE_PATH),
                'seed':args.seed,'sample_count':generated,'shards':shards,'noise_dtype':'float32',
                'token_dtype':'int16','guidance_interpretation':'VAR ramp: (1+cfg*scale/9)*conditional - (cfg*scale/9)*unconditional',
                'ode_nfe_per_scale':getattr(config,'ode_nfe',10),'ode_grid':getattr(config,'ode_grid','logSNR'),
                'boundary_epsilon':.001,'elapsed_seconds':time.time()-begin,**parameter_telemetry()}
    if filter_evidence is not None:
        manifest['filter_evidence'] = filter_evidence
    save_json(out / 'manifest.json', manifest)
    print(json.dumps({'teacher_generation_complete':True,'sample_count':generated}), flush=True)


if __name__ == '__main__':
    main()
