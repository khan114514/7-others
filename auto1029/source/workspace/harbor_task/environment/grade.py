"""Verifier entry: trusted external score, atomic reward, no agent-written reward."""
import json
import hashlib
import math
import os
import stat
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, '/opt/latentcli')
import queue_client
# Root-only physical queue. Agent-owned feedback transport is never a source
# of trusted rewards; an agent can forge files in its own queue.
queue_client.QUEUE = Path('/run/heima-verifier/queue')
from queue_client import request_json



def completed_reply(value):
    epoch = value.get('completed_epoch')
    if (type(value.get('exit_code')) is not int or type(epoch) not in (int, float)
            or not math.isfinite(epoch)):
        return False
    return (value.get('phase') == 'FAILED' or
            (value.get('phase') == 'COMPLETE' and isinstance(value.get('result'), dict)))

def request_eval(payload):
    # Transport interruption is infrastructure waiting, not an invalid method.
    # Retry this exact data request without changing ticket, source, or seeds.
    delay = 1
    while True:
        try:
            return request_json('evaluate', payload, timeout=60)
        except TimeoutError:
            pass
        except RuntimeError as exc:
            if 'Evaluator transport unavailable; retain and query your own ticket later' not in str(exc):
                raise
        time.sleep(delay)
        delay = min(delay * 2, 10)


def same_owned_private_source(value, policy, notes):
    return (value.get('private_verifier_route') is True
            and value.get('policy_sha256') == policy
            and value.get('method_notes_sha256') == notes)


root = Path('/workspace/solution')
out = Path('/logs/verifier')
out.mkdir(parents=True, exist_ok=True)
# This output directory belongs only to this separate Verifier Trial.
# Remove both Harness reward formats before evaluation, so failure cannot leave
# an old reward.json taking precedence over a newly written reward.txt.
for previous in ['reward.txt', 'reward.json', 'reward.pending', 'reward.json.pending']:
    (out / previous).unlink(missing_ok=True)
reward = 0.0
error = None
try:
    notes_path=root/'method_notes.json'
    fd=os.open(notes_path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>65536:
            raise ValueError('method_notes.json must be a bounded regular JSON object')
        notes=json.loads(os.read(fd,65537))
        if not isinstance(notes,dict) or not notes:
            raise ValueError('method_notes.json must be a nonempty JSON object explaining the method')
    finally:os.close(fd)
    files = {}
    for path in root.rglob('*.py'):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                os.close(fd); continue
            raw = os.read(fd, 1024 * 1024 + 1); os.close(fd)
            files[str(path.relative_to(root))] = raw.decode('utf-8')
        except (OSError, UnicodeError):
            continue
    if 'method.py' not in files:
        raise ValueError('Missing method.py')
    expected_policy = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    expected_notes = hashlib.sha256(json.dumps(notes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = None
    while True:
        active = request_eval({'action': 'active'})['active']
        if active is None:
            break
        job_id = active['job_id']
        exact_owned_source = same_owned_private_source(active, expected_policy, expected_notes)
        while not completed_reply(active):
            time.sleep(5)
            active = request_eval({'action': 'status', 'job_id': job_id})
        (out / ('prior-job-' + job_id + '.json')).write_text(json.dumps(active, ensure_ascii=False, indent=2) + '\n')
        if exact_owned_source:
            # This is the actual RootVerifier's original ticket, not an old
            # Dev score or a source-mismatched/newly fabricated evaluation.
            result = active
            break
    if result is None:
        job = uuid.uuid4().hex
        payload = {'action': 'submit', 'job_id': job, 'files': files, 'method_notes': notes, 'seeds': [17, 29, 43, 71, 101]}
        result = request_eval(payload)
        while not completed_reply(result):
            time.sleep(5)
            result = request_eval({'action': 'status', 'job_id': job})
    (out / 'scorer-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if (result['phase'] != 'COMPLETE' or not result['result']['complete_private_protocol']
            or result['result']['samples_per_seed'] != 1380
            or result['result']['seeds'] != [17, 29, 43, 71, 101]
            or result['result']['private_verifier_route'] is not True
            or result['result']['candidate_imported_by_host'] is not False):
        raise ValueError('No completed full private fixed-protocol evaluation')
    score = result['result']
    expected_policy = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    expected_notes = hashlib.sha256(json.dumps(notes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if score['policy_sha256'] != expected_policy or score['method_notes_sha256'] != expected_notes:
        raise ValueError('Private reward is not bound to exact transferred source and method metadata')
    reward = score['normalized_score'] if score['all_quality_gates_pass'] else -1.0
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError('Private reward must be finite')
except Exception as exc:
    error = {'error_type': type(exc).__name__, 'message': str(exc)[:1000]}
    (out / 'grader-error.json').write_text(json.dumps(error, indent=2) + '\n')
if error:
    print(json.dumps({'error': error}), flush=True)
    raise SystemExit(1)
pending = out / 'reward.pending'
pending.write_text(str(reward) + '\n')
pending.replace(out / 'reward.txt')
print(json.dumps({'reward': reward, 'error': error}), flush=True)
if error:
    raise SystemExit(1)
