"""Shared DD student trainer, independently reimplemented from the paper.

Teacher strategies and time distributions are supplied only by method.py.
Architecture, loss, optimizer and evaluation path are fixed across candidates.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from common import (PATCH_NUMS, TEACHER_PATH, VAE_PATH, WORK, budget_check,
                    decode_tokens, load_method, parameter_telemetry,
                    require_phase_one, save_json, setup_runtime, sha)


def initialize_from_teacher(student, teacher_state):
    """Map teacher weights without modifying released code or class row 1.

    The official helper resets class_emb[1] where its comment targets lvl row1.
    This explicit map preserves all teacher class embeddings and uses the
    already initialized extra DD scale and position rows.
    """
    import torch
    state = student.state_dict()
    inherited = []
    with torch.no_grad():
        for name, source in teacher_state.items():
            if name not in state or name in ('lvl_1L', 'attn_bias_for_masking'):
                continue
            destination = state[name]
            if name == 'lvl_embed.weight':
                destination[0].copy_(source[0])
                destination[2:].copy_(source[1:])
                inherited.append(name)
            elif name == 'pos_1LC':
                destination[:, 0].copy_(source[:, 0])
                destination[:, 2:].copy_(source[:, 1:])
                inherited.append(name)
            elif destination.shape == source.shape:
                destination.copy_(source)
                inherited.append(name)
            else:
                raise RuntimeError(f'Unexplained teacher parameter shape: {name}')
        student.load_state_dict(state, strict=True)
        if not torch.equal(student.class_emb.weight.detach().cpu(), teacher_state['class_emb.weight']):
            raise RuntimeError('Teacher class embeddings did not survive initialization')
        if not all(torch.isfinite(p).all() for p in student.parameters()):
            raise FloatingPointError('Nonfinite initialized student parameter')
    return inherited


def attach_checkpointing(student):
    """Memory adaptation with exactly the same block operations and RNG."""
    from torch.utils.checkpoint import checkpoint
    for block in student.blocks:
        original = block.forward

        def wrapped(x, cond_BD, attn_bias, n_d_emb=None, operation=original):
            if not operation.__self__.training:
                return operation(x=x, cond_BD=cond_BD, attn_bias=attn_bias, n_d_emb=n_d_emb)
            return checkpoint(operation, x, cond_BD, attn_bias, n_d_emb,
                              use_reentrant=False, preserve_rng_state=True)

        block.forward = wrapped


class PairStream:
    def __init__(self, directory, manifest, batch_size, seed):
        import torch
        self.directory = Path(directory)
        self.shards = manifest['shards']
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.shard_order = torch.randperm(len(self.shards), generator=self.generator).tolist()
        self.shard_position = 0
        self.data = None
        self.row_position = 0

    def next(self):
        import torch
        if self.data is None or self.row_position >= len(self.rows):
            if self.shard_position >= len(self.shards):
                self.epoch += 1
                self.shard_order = torch.randperm(len(self.shards), generator=self.generator).tolist()
                self.shard_position = 0
            shard = self.shards[self.shard_order[self.shard_position]]
            self.shard_position += 1
            self.data = torch.load(self.directory / shard['file'], map_location='cpu', weights_only=True)
            self.rows = torch.randperm(len(self.data['labels']), generator=self.generator)
            self.row_position = 0
        rows = self.rows[self.row_position:self.row_position + self.batch_size]
        self.row_position += len(rows)
        return {key:value[rows].cuda(non_blocking=False) for key,value in self.data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--max-updates', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2001)
    parser.add_argument('--ema-decay', type=float, default=.99)
    args = parser.parse_args()
    require_phase_one()
    torch = setup_runtime(args.seed)
    import torch.nn.functional as F
    import lpips
    from models import build_vae_dd
    from torchvision.utils import save_image
    from rectified_categorical import uniform_to_scale
    module, method = load_method(args.method)
    if not isinstance(method.teacher.reuse, int) or method.teacher.reuse < 1:
        raise ValueError('Reuse must be a positive maximum number of shuffled passes')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'checkpoint.pth').exists() or (out / 'updates.jsonl').exists():
        raise RuntimeError('Existing training requires explicit recovery; refusing overwrite')
    pair_directory = Path(args.pairs)
    manifest = json.loads((pair_directory / 'manifest.json').read_text())
    if manifest['method_sha256'] != sha(args.method):
        raise RuntimeError('Teacher pairs were produced by a different method snapshot')
    for shard in manifest['shards']:
        if sha(pair_directory / shard['file']) != shard['sha256']:
            raise RuntimeError('Teacher pair shard integrity mismatch')
    perceptual_weights = WORK / 'runtime/torchhub/hub/checkpoints/alexnet-owt-7be5be79.pth'
    if not perceptual_weights.exists():
        raise RuntimeError('Offline LPIPS backbone missing')
    if sha(perceptual_weights) != '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02':
        raise RuntimeError('Offline LPIPS backbone integrity mismatch')
    started = time.time()
    vae, student = build_vae_dd(device='cuda', patch_nums=PATCH_NUMS, depth=16,
                                 shared_aln=False, attn_l2_norm=True)
    vae.load_state_dict(torch.load(VAE_PATH, map_location='cpu', weights_only=True), strict=True)
    teacher_state = torch.load(TEACHER_PATH, map_location='cpu', weights_only=True)
    inherited = initialize_from_teacher(student, teacher_state)
    del teacher_state
    vae.eval().requires_grad_(False)
    student.train()
    attach_checkpointing(student)
    perceptual = lpips.LPIPS(net='alex', version='0.1').cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(.9, .95),
                                 weight_decay=.05, foreach=False)
    # EMA lives on CPU to keep memory below the per-process GPU cap.
    ema = {name:p.detach().cpu().clone() for name,p in student.named_parameters()}
    stream = PairStream(pair_directory, manifest, args.batch_size, args.seed + 1)
    generator = torch.Generator(device='cuda').manual_seed(args.seed + 2)
    histogram = torch.zeros(10, dtype=torch.long)
    codebook = vae.quantize.embedding.weight
    updates = 0
    seen = 0
    terminal_reason = 'max_updates'
    configuration = {'protocol':'dd-selfproto-v1.0','reimplementation':True,'method':asdict(method),
                     'method_sha256':sha(args.method),'pair_manifest_sha256':sha(pair_directory / 'manifest.json'),
                     'seed':args.seed,'batch_size':args.batch_size,'accumulation':args.accumulation,
                     'lr':args.lr,'betas':[.9,.95],'weight_decay':.05,'clip_grad_norm':1.,
                     'ema_decay':args.ema_decay,'embedding_loss':'LPIPS-alex-v0.1 on decoded continuous-head suffix and teacher prefix',
                     'classification_loss':'CE averaged over generated tokens of all scales',
                     'loss_weights':{'embedding':1.,'classification':.1},'precision':'bf16 autocast with fp32 weights',
                     'architecture':'released DD depth16, 256px','initialization':'teacher weight map; no official DD checkpoint',
                     'loss_split_t_eval':5,'training_prefix_mapping':'floor(method_t*10), clamped to0..9',
                     'inherited_teacher_keys':inherited,'deadline_unix':args.deadline,'start_unix':started}
    configuration['teacher_reuse_maximum_passes'] = method.teacher.reuse
    configuration['teacher_reuse_note'] = 'Each pass visits every pair once; stop earlier when the GPU budget ends.'
    save_json(out / 'train_config.json', configuration)
    optimizer.zero_grad(set_to_none=True)
    with (out / 'updates.jsonl').open('a') as evidence:
        while updates < args.max_updates and seen < manifest['sample_count'] * method.teacher.reuse:
            if time.time() >= args.deadline - 120:
                terminal_reason = 'budget_checkpoint_reserve'
                break
            ce_total = 0.
            perceptual_total = 0.
            for microstep in range(args.accumulation):
                if seen >= manifest['sample_count'] * method.teacher.reuse:
                    terminal_reason = 'teacher_reuse_limit'
                    break
                budget_check(args.deadline, reserve=60)
                batch = stream.next()
                noise = batch['noise'].float()
                target_ids = batch['tokens'].long()
                target_tokens = F.embedding(target_ids, codebook)
                t = module.fm_time_sampler((len(noise),), device='cuda', generator=generator)
                k = uniform_to_scale(t).cpu()
                histogram += torch.bincount(k, minlength=10)
                perturbed = student.perturb_data(target_tokens, k, noise)
                lengths = torch.tensor([sum(p*p for p in PATCH_NUMS[:int(s)]) for s in k], device='cuda')
                mask = torch.arange(680, device='cuda')[None] >= lengths[:, None]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    continuous, logits = student(batch['labels'].long(), perturbed, k)
                    # Official head mixing and architecture remain fixed. Early
                    # and late scale heads are both learned, using paper losses.
                    ce_per_token = F.cross_entropy(logits.reshape(-1,4096), target_ids.reshape(-1), reduction='none').reshape_as(mask)
                    ce_per_sample = (ce_per_token * mask).sum(-1) / mask.sum(-1)
                    ce = ce_per_sample.mean()
                    prediction = torch.where(mask[:, :, None], continuous.float(), target_tokens)
                    prediction_image = decode_tokens(prediction, vae)
                    with torch.no_grad():
                        target_image = decode_tokens(target_tokens, vae)
                    embedding_loss = perceptual(prediction_image.float(), target_image.float()).mean()
                    loss = (embedding_loss + .1 * ce) / args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite student loss')
                loss.backward()
                ce_total += float(ce.detach()) / args.accumulation
                perceptual_total += float(embedding_loss.detach()) / args.accumulation
                seen += len(noise)
                if updates == 0 and microstep == 0:
                    save_image(torch.cat((target_image, prediction_image.detach())).add(1).mul(.5),
                               out / 'initial_train_grid.png', nrow=len(noise))
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite student gradient norm')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            with torch.no_grad():
                for name,p in student.named_parameters():
                    ema[name].mul_(args.ema_decay).add_(p.detach().cpu(), alpha=1-args.ema_decay)
            record = {'update':updates,'seen_pairs':seen,'stream_epoch':stream.epoch,
                      'ce_loss':ce_total,'lpips_loss':perceptual_total,'grad_norm':float(norm),
                      'elapsed_seconds':time.time()-started,'time_scale_histogram':histogram.tolist(),
                      **parameter_telemetry()}
            evidence.write(json.dumps(record)+'\n')
            evidence.flush()
            if updates == 1 or updates % 10 == 0:
                print(json.dumps(record),flush=True)
            if updates % 100 == 0:
                atomic_checkpoint(out, student, ema, optimizer, updates, seen, configuration)
    if seen >= manifest['sample_count'] * method.teacher.reuse:
        terminal_reason = 'teacher_reuse_limit'
    if updates == 0:
        raise TimeoutError('No student update completed within the budget')
    atomic_checkpoint(out, student, ema, optimizer, updates, seen, configuration)
    result = {'updates':updates,'seen_pairs':seen,'terminal_reason':terminal_reason,
              'elapsed_seconds':time.time()-started,'method_sha256':sha(args.method),
              'checkpoint_sha256':sha(out/'checkpoint.pth'),'time_scale_histogram':histogram.tolist(),
              **parameter_telemetry()}
    save_json(out/'train_result.json',result)
    print(json.dumps({'student_training_complete':True,**result}),flush=True)


def atomic_checkpoint(out, student, ema, optimizer, updates, seen, configuration):
    import torch
    parameters = {name:p.detach().cpu() for name,p in student.named_parameters()}
    temporary = out/'checkpoint.pth.tmp'
    torch.save({'trainer':{'var_wo_ddp':parameters,'ema_parameters':ema},
                'updates':updates,'seen_pairs':seen,'config':configuration},temporary)
    temporary.replace(out/'checkpoint.pth')
    # A distinct resumable snapshot preserves optimizer evidence without placing
    # it in the weights consumed by the fixed official sample.py evaluation.
    temporary = out/'training_state.pth.tmp'
    torch.save({'optimizer':optimizer.state_dict(),'updates':updates,'seen_pairs':seen},temporary)
    temporary.replace(out/'training_state.pth')


if __name__ == '__main__':
    main()
