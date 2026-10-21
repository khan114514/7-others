"""Fixed remote submit/status API. Lane authority is a trusted CLI argument.

Input is bounded source data. It is never promoted to a shell command, path,
model identifier, target mutation, hidden split, or GPU-host Python import.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from deployment_paths import PRIVATE_ROOT as PRIVATE, SCORING_PYTHON, JAIL_PYTHON, TORCHTUNE_PACKAGE, CONFIG_PATH
JOBS = PRIVATE / 'evaluator-jobs'
UUID = re.compile(r'^[0-9a-f]{32}$')
LIMIT = 2 * 1024 * 1024


def atomic(path, value):
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def status(job, role):
    folder = JOBS / job
    request = json.loads((folder / 'request.json').read_text())
    if request['role'] != role:
        raise ValueError('Job not owned by this physical lane')
    value = {'job_id': job, 'role': role, 'remote_observed_epoch': time.time(),
             'submitted_epoch': request['submitted_epoch'], 'policy_sha256': request['policy_sha256']}
    receipt_path = folder / 'execution.json'
    progress_path = folder / 'result/status.json'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        value.update({key: progress[key] for key in ['phase', 'gpu_started_epoch', 'seed', 'completed_questions'] if key in progress})
    else:
        value['phase'] = 'WAITING_FOR_GPU_EXCLUDED'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        value.update(exit_code=receipt['exit_code'], completed_epoch=receipt['completed_epoch'])
        result_path = folder / 'result/result.json'
        if receipt['exit_code'] == 0 and result_path.exists():
            value.update(phase='COMPLETE', result=json.loads(result_path.read_text()))
        else:
            value.update(phase='FAILED', error='Recorded scorer execution failed; no score claim')
        value['experiment_active_seconds'] = max(0, receipt['completed_epoch'] - value['gpu_started_epoch']) if 'gpu_started_epoch' in value else 0.0
    elif 'gpu_started_epoch' in value:
        value['experiment_active_seconds'] = max(0, time.time() - value['gpu_started_epoch'])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['gpt', 'seed'], required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('Input exceeds evaluator transport budget')
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get('action') not in {'submit', 'status', 'active'}:
        raise ValueError('Invalid fixed evaluator operation')
    JOBS.mkdir(mode=0o700, exist_ok=True)
    if payload['action'] == 'active':
        if set(payload) != {'action'}:
            raise ValueError('Unexpected active-job parameters')
        active = None
        for existing in JOBS.iterdir():
            if existing.is_dir() and (existing / 'request.json').exists() and not (existing / 'execution.json').exists():
                previous = json.loads((existing / 'request.json').read_text())
                if previous['role'] == args.role:
                    active = status(existing.name, args.role)
                    break
        print(json.dumps({'remote_observed_epoch':time.time(), 'role':args.role, 'active':active},ensure_ascii=False))
        return
    if not UUID.fullmatch(payload.get('job_id', '')):
        raise ValueError('Invalid job identity')
    job = payload['job_id']
    if payload['action'] == 'status':
        if set(payload) != {'action', 'job_id'}:
            raise ValueError('Unexpected status parameters')
        print(json.dumps(status(job, args.role), ensure_ascii=False))
        return
    if set(payload) != {'action', 'job_id', 'files', 'seeds'}:
        raise ValueError('Unexpected submit parameters')
    seeds = payload['seeds']
    if seeds not in [[17], [17, 29, 43, 71, 101]]:
        raise ValueError('Only documented quick or complete dev seed set allowed')
    files = payload['files']
    if not isinstance(files, dict) or 'method.py' not in files or len(files) > 64:
        raise ValueError('Invalid candidate source bundle')
    for name, value in files.items():
        path = Path(name)
        if not isinstance(value, str) or path.is_absolute() or '..' in path.parts or path.suffix not in {'.py', '.json', '.txt', '.md'}:
            raise ValueError('Invalid source name or content')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if len(encoded) > LIMIT:
        raise ValueError('Source bundle too large')
    policy_sha = hashlib.sha256(encoded).hexdigest()
    contract = PRIVATE / 'policy-score-protocol-v2.json'
    anchor = PRIVATE / 'frozen-policy-anchors-v2-dev/comparison_summary.json'
    if not anchor.exists():
        raise RuntimeError('Paired calibration anchors are not ready; no candidate job started')
    anchors = json.loads(anchor.read_text())
    if anchors['protocol_sha256'] != hashlib.sha256(contract.read_bytes()).hexdigest() or not anchors['all_quality_gates_pass']:
        raise RuntimeError('Valid matching calibrated anchors required')
    folder = JOBS / job
    lock = (JOBS / 'submission.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if folder.exists():
        previous = json.loads((folder / 'request.json').read_text())
        if previous['role'] != args.role or previous['policy_sha256'] != policy_sha or previous['seeds'] != seeds:
            raise ValueError('Idempotency key reused for different inputs')
        print(json.dumps(status(job, args.role), ensure_ascii=False))
        return
    for existing in JOBS.iterdir():
        if not existing.is_dir() or not (existing / 'request.json').exists():
            continue
        previous = json.loads((existing / 'request.json').read_text())
        if previous['role'] == args.role and not (existing / 'execution.json').exists():
            raise ValueError('This lane already has an unfinished evaluation; query its own saved job ticket')
    folder.mkdir(mode=0o700)
    request = {'role': args.role, 'files': files, 'seeds': seeds, 'policy_sha256': policy_sha,
               'submitted_epoch': time.time(), 'job_id': job}
    atomic(folder / 'request.json', request)
    argv = [str(SCORING_PYTHON), str(PRIVATE / 'run_policy_score_v2.py'),
            '--mode', 'candidate', '--split', 'dev', '--seeds', ','.join(map(str, seeds)),
            '--request', str(folder / 'request.json'), '--output', str(folder / 'result')]
    config = {'command': argv, 'timeout_seconds': 10800, 'receipt': str(folder / 'execution.json')}
    atomic(folder / 'execution-request.json', config)
    environment = {'PATH': str(SCORING_PYTHON.parent)+':'+str(JAIL_PYTHON.parent)+':/usr/bin:/bin',
                   'HOME': '/root', 'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1',
                   'PYTHONPATH': str(TORCHTUNE_PACKAGE)+':' + str(PRIVATE), 'HEIMA_DEPLOYMENT_CONFIG': str(CONFIG_PATH)}
    with (folder / 'execution.log').open('xb') as log:
        child = subprocess.Popen([str(SCORING_PYTHON), str(PRIVATE / 'run_logged_command.py'), str(folder / 'execution-request.json')],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, close_fds=True, start_new_session=True)
    atomic(folder / 'launch.json', {'pid': child.pid, 'launched_epoch': time.time(), 'command': argv})
    print(json.dumps(status(job, args.role), ensure_ascii=False))


if __name__ == '__main__':
    main()
