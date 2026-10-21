"""Preserve the failed evaluator and recover only its completed FID input."""
import argparse
import fcntl
import json
import math
import os
import re
import shutil

from candidate_runner import OUT, WORK, PY, REPO, event, gpu, integrity_checks, now, run_stage, save, sha, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert args.attempt >= 1
    suffix = '' if args.attempt == 1 else f'_attempt{args.attempt:02d}'
    directory = OUT / 'rounds' / f'round{args.round:02d}'
    large = WORK / directory.name
    lock = (OUT / '.candidate_runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    baseline_lock = (OUT / '.baseline_runner.lock').open('a')
    fcntl.flock(baseline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    failed = json.loads((directory / 'evaluation_execution.json').read_text())
    assert failed['exit_code'] == -11 and not failed['termination']
    assert not (directory / 'result.json').exists()
    budget = json.loads((directory / 'budget.json').read_text())
    assert budget['eligible_budget'] and budget['total_gpu_wall_seconds'] <= 14400
    checkpoint = large / 'training/checkpoint.pth'
    training = json.loads((large / 'training/train_result.json').read_text())
    assert sha(checkpoint) == training['checkpoint_sha256']
    paths = sorted((large / 'evaluation/images').glob('*.png'), key=lambda p: int(p.stem))
    assert len(paths) == 50000 and [int(p.stem) for p in paths] == list(range(50000))
    before = integrity_checks()
    scorer_hash = sha(REPO / 'evaluation/fid_score.py')
    entry = directory / 'score_existing_entry.py'
    if not entry.exists():
        shutil.copyfile(OUT / 'rounds/round05/score_existing_entry.py', entry)
    if args.round != 5:
        shutil.copyfile(os.path.abspath(__file__), directory / 'recover_existing_score.py')
    recovery = {'recorded_at': now(), 'round': args.round, 'attempt': args.attempt,
                'protocol_version': 'dd-selfproto-v1.1',
                'runner_pid': os.getpid(), 'original_failed_execution': failed,
                'original_failed_execution_sha256': sha(directory / 'evaluation_execution.json'),
                'checkpoint_sha256': sha(checkpoint), 'sample_count': len(paths),
                'scorer_sha256': scorer_hash, 'entry_sha256': sha(entry),
                'original_failed_process_confirmed_absent': not os.path.exists(f"/proc/{failed['pid']}"),
                'decision': 'Retry FID only on original complete images; no new training or sampling.',
                'protected_before': before}
    assert recovery['original_failed_process_confirmed_absent']
    recovery_path = directory / f'score_recovery{suffix}.json'
    assert not recovery_path.exists()
    if args.attempt > 1:
        prior = directory / ('fid_recovery' + ('' if args.attempt == 2 else f'_attempt{args.attempt-1:02d}') + '_execution.json')
        previous = json.loads(prior.read_text())
        assert previous['exit_code'] and previous['finished_at']
        assert not os.path.exists(f"/proc/{previous['pid']}")
        recovery['previous_failed_recovery'] = str(prior)
        recovery['previous_failed_recovery_sha256'] = sha(prior)
        recovery['environment_change'] = 'Do not inherit LD_PRELOAD; explicit local runtime environment, same scorer, images, batch and numerical flags.'
    save(recovery_path, recovery)
    status = gpu()
    save(directory / f'pre_score_recovery_gpu{suffix}.json', {'at': now(), 'gpu': status})
    assert float(status.split(',')[2]) >= 14000
    env = {k: v for k, v in os.environ.items()
           if not any(x in k.upper() for x in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
    if args.attempt > 1:
        env.pop('LD_PRELOAD', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', PYTHONFAULTHANDLER='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TMPDIR=str(WORK / 'runtime/tmp'), CUDA_CACHE_PATH=str(WORK / 'runtime/cuda_cache'),
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    run_stage([PY, '-B', '-u', str(entry), '--images', str(large / 'evaluation/images'),
               '--fid-output', str(large / 'evaluation/fid.txt')],
              'fid_recovery' + suffix, directory, env, args.round)
    assert sha(REPO / 'evaluation/fid_score.py') == scorer_hash
    save(directory / 'protected_after.json', integrity_checks())
    text = (large / 'evaluation/fid.txt').read_text()
    fid = float(re.search(r'FID:\s*([-+0-9.eE]+)', text)[1])
    assert math.isfinite(fid) and 0 <= fid != 114514
    from PIL import Image, ImageStat
    means = set()
    with (directory / 'sample_manifest.jsonl').open('w') as manifest:
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                assert image.size == (256, 256)
                if index < 256:
                    means.add(tuple(round(v, 4) for v in ImageStat.Stat(image).mean))
            manifest.write(json.dumps({'file': path.name, 'bytes': path.stat().st_size,
                                       'sha256': sha(path)}) + '\n')
    assert len(means) >= 2
    grid = Image.new('RGB', (2048, 2048))
    for index, path in enumerate(paths[:64]):
        with Image.open(path) as image:
            grid.paste(image.convert('RGB'), ((index % 8) * 256, (index // 8) * 256))
    grid.save(directory / 'sample_grid.jpg', quality=90)
    shutil.copyfile(large / 'evaluation/fid.txt', directory / 'fid.txt')
    config = json.loads((large / 'training/train_config.json').read_text())
    result = {'protocol_version': 'dd-selfproto-v1.1', 'sample_count': 50000, 'seed_set': [0],
              'initialization': config['method']['initialization'], 'batch_size': 8,
              'fid_50k': fid, 'score': fid, 'checkpoint_path': str(checkpoint),
              'checkpoint_sha256': sha(checkpoint), 'method_sha256': sha(directory / 'method.py'),
              'sample_manifest_sha256': sha(directory / 'sample_manifest.jsonl'),
              'training_budget_seconds': budget['total_gpu_wall_seconds'],
              'visual_review_required': True, 'quality_gate_passed': False,
              'scoring_recovery': str(recovery_path),
              'original_failure_preserved': True, 'official_fid_batch_size': 256}
    save(directory / 'result.json', result)
    recovery.update(finished_at=now(), result_sha256=sha(directory / 'result.json'))
    save(recovery_path, recovery)
    state('candidate_finished_pending_review', active_experiment=None, pending_round=args.round,
          pending_result=str(directory / 'result.json'))
    event({'kind': 'fid_recovery_complete', 'round':args.round, 'result': str(directory / 'result.json')})
    print(f'FID: {fid}', flush=True)


if __name__ == '__main__':
    main()
