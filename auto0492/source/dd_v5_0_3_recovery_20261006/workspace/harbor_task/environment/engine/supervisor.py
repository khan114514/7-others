"""Fresh trusted teacher/train/evaluation pipeline, independent of candidate claims."""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from config import ENGINE, deployment, sha, verify_assets, atomic_json, short_temp_root
from method_contract import PROTOCOL
from method_proxy import MethodProxy
from scoring import normalize
from evaluation_policy import load_formal_policy

BUDGET_SECONDS = 14400
EVALUATION_SECONDS = 21600


def stage(command, name, run, env, deadline):
    started = time.time()
    with (run / (name + '.log')).open('wb') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                cwd=deployment()['repo'], env=env, close_fds=True, start_new_session=True)
        atomic_json(run / (name + '_process.json'), {'pid': process.pid, 'started_unix': started})
        try:
            code = process.wait(timeout=max(0.01, deadline - time.time()))
            if code:
                raise RuntimeError(name + ' failed with exit code ' + str(code))
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            for receipt in run.glob('method-*/worker_pid.json'):
                pid = json.loads(receipt.read_text())['pid']
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    elapsed = time.time() - started
    atomic_json(run / (name + '_stage.json'), {'stage': name, 'started_unix': started,
                'finished_unix': time.time(), 'elapsed_seconds': elapsed,
                'counts_toward_training_budget': name in ('teacher', 'training')})
    return elapsed


def verify_code():
    manifest = json.loads((ENGINE.parent / 'protected.json').read_text())
    for row in manifest['files']:
        path = (ENGINE.parent / row['file']).resolve()
        if not path.is_relative_to(ENGINE.parent) or sha(path) != row['sha256']:
            raise RuntimeError('Frozen implementation mismatch: ' + row['file'])
    return sha(ENGINE.parent / 'protected.json')


def run_pipeline(args):
    config = deployment()
    formal_policy = None
    formal_policy_sha = None
    if not args.smoke and not args.calibration:
        # Fail before queueing or consuming GPU time if formal scoring is not
        # calibrated. The gateway also checks this, but direct runs must agree.
        formal_policy, formal_policy_sha = load_formal_policy(config, ENGINE.parent.parent, PROTOCOL)
        expected = os.environ.get('DD_EXPECTED_FORMAL_POLICY_SHA256')
        if expected is not None and expected != formal_policy_sha:
            raise ValueError('Private formal policy changed after submission')
        if args.teacher_seed != 1001 or args.training_seed != 2001:
            raise ValueError('Formal training stream override is forbidden')
        args.evaluation_seed = formal_policy['evaluation_seed']
    root = Path(config['runs'])
    root.mkdir(parents=True, exist_ok=True)
    # One full budget at a time; queueing is outside the charged interval.
    with (root / 'gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen_sha = verify_code()
        verify_assets()
        run = root / ('run-' + uuid.uuid4().hex)
        run.mkdir()
        method = run / 'method.py'
        method.write_bytes(Path(args.method).read_bytes())
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(run), 'TMPDIR': str(short_temp_root(run)),
               'LANG': 'C.UTF-8', 'DD_DEPLOYMENT': str(Path(os.environ.get('DD_DEPLOYMENT', '/opt/dd/deployment.json')).resolve()),
               'DD_RUN': str(run), 'DD_SUPERVISED_STAGE': '1',
               'PYTHONPATH': os.pathsep.join([str(ENGINE), str(ENGINE/'framework'), config['repo']]
                                     + config.get('extra_site_packages', [])),
               'TORCH_HOME': config['torch_home'], 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
               'CUDA_CACHE_PATH': str(run / 'cuda-cache'),
               'PYTORCH_KERNEL_CACHE_PATH': str(run / 'kernel-cache')}
        os.environ['DD_RUN'] = str(run)
        started = time.time()
        receipt = {'protocol': PROTOCOL, 'run_id': run.name,
                   'kind': 'SMOKE_ONLY' if args.smoke else ('CALIBRATION' if args.calibration else 'FORMAL'),
                   'training_seed': args.training_seed, 'teacher_seed': args.teacher_seed,
                   'evaluation_seed': args.evaluation_seed, 'method_sha256': sha(method),
                   'frozen_manifest_sha256': frozen_sha, 'started_unix': started,
                   'status': 'RUNNING', 'budget_seconds': BUDGET_SECONDS,
                   'sample_count': 64 if args.smoke else 50000, 'batch_size': 8}
        if formal_policy is not None:
            receipt['formal_policy_sha256'] = formal_policy_sha
        atomic_json(run / 'run.json', receipt)
        if not args.smoke and not args.calibration:
            logs = Path('/logs/verifier')
            logs.mkdir(parents=True, exist_ok=True)
            atomic_json(logs/'invocation.json', {'run_id': run.name, 'started_unix': started})
        try:
            proxy = MethodProxy(method)
            spec = proxy.spec
            atomic_json(run / 'method_config.json', {'configuration': asdict(spec), 'worker_guard': proxy.guard})
            proxy.close()
            checkpoint = Path(config['assets']) / 'VAR-DD/var-d16.pth'
            if spec.teacher.num_samples:
                deadline = started + BUDGET_SECONDS
                pairs = run / 'pairs'
                training = run / 'training'
                stage([config['python'], str(ENGINE/'framework/generate_pairs.py'),
                       '--method', str(method), '--out', str(pairs), '--deadline', str(deadline),
                       '--seed', str(args.teacher_seed)], 'teacher', run, env, deadline)
                stage([config['python'], str(ENGINE/'framework/train_student.py'),
                       '--method', str(method), '--pairs', str(pairs), '--out', str(training),
                       '--deadline', str(deadline), '--seed', str(args.training_seed)],
                       'training', run, env, deadline)
                checkpoint = training / 'checkpoint.pth'
            receipt['teacher_and_training_seconds'] = time.time() - started
            if receipt['teacher_and_training_seconds'] > BUDGET_SECONDS:
                raise TimeoutError('Teacher generation and training exceeded four GPU hours')
            receipt['adaptation'] = spec.teacher.num_samples > 0
            receipt['checkpoint_sha256'] = sha(checkpoint)
            env['DD_RAM_EVIDENCE'] = str(run / 'evaluation')
            env['DD_RAM_EQUIVALENCE_PROBE'] = '1' if args.smoke else '0'
            env['DD_EVALUATION_SEED'] = str(args.evaluation_seed)
            evaluation = run / 'evaluation'
            evaluation.mkdir()
            stage([config['python'], str(ENGINE/'ram_png_entry.py'),
                   '--ckpt_path', str(checkpoint), '--vae_ckpt', str(Path(config['assets'])/'VAR/vae_ch160v4096z32.pth'),
                   '--sample_dir', str(evaluation), '--sample_num', str(receipt['sample_count']),
                   '--t_list', '0', '--loss_split_t', '5', '--depth', '16', '--bs', '8',
                   '--seed', str(args.evaluation_seed), '--fid_statistic', str(Path(config['assets'])/'fid_stats'),
                   '--local_out_dir_path', str(run/'official-logs')],
                   'evaluation', run, env, time.time() + EVALUATION_SECONDS)
            storage = json.loads((evaluation/'ram_png_storage.json').read_text())
            if storage['sample_count'] != receipt['sample_count'] or storage['finite_decoded_batch_checks'] < math.ceil(receipt['sample_count']/8):
                raise RuntimeError('Evaluation sample or finite-output coverage mismatch')
            fid = storage['fid_ram']
            if type(fid) not in (int, float) or not math.isfinite(fid) or fid < 0 or fid == 114514:
                raise ValueError('Invalid independently computed FID')
            if receipt['checkpoint_sha256'] != sha(checkpoint):
                raise RuntimeError('Checkpoint changed during evaluation')
            verify_code()
            verify_assets()
            receipt.update(status='SUCCESS', finished_unix=time.time(), metric=float(fid),
                           sample_manifest_sha256=sha(evaluation/'sample_manifest.jsonl'))
            if not args.smoke and not args.calibration:
                _, current_policy_sha = load_formal_policy(config, ENGINE.parent.parent, PROTOCOL)
                if current_policy_sha != formal_policy_sha:
                    raise RuntimeError('Private scoring policy changed during execution')
                receipt['reward'] = normalize(fid, formal_policy['anchors'])
            atomic_json(run/'result.json', receipt)
        except BaseException as error:
            receipt.update(status='FAILED', finished_unix=time.time(),
                           error_type=type(error).__name__, error=str(error))
            atomic_json(run/'failure.json', receipt)
            raise
        finally:
            atomic_json(run/'run.json', receipt)
            print(json.dumps({'run': run.name, 'status': receipt['status'],
                              'kind': receipt['kind'], 'metric': receipt.get('metric')}), flush=True)
        return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--teacher-seed', type=int, default=1001)
    parser.add_argument('--training-seed', type=int, default=2001)
    parser.add_argument('--evaluation-seed', type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--calibration', action='store_true')
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
