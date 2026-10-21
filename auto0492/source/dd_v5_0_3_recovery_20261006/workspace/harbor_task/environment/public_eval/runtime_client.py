"""Source-only submission over a task-scoped local Unix socket.

The socket, mounted by the harness, determines the GPU role and job scope.
This client needs neither SSH access nor a controller bearer credential.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import time

MAX_SOURCE_BYTES = 131072
MAX_REPLY_BYTES = 1048576
FAILURE_ADVICE = {
    'HARD_GATE': 'Correct the method configuration or values to satisfy the published interface.',
    'FORMAT_ERROR': 'Submit a readable UTF-8 method source conforming to the published interface.',
    'BUDGET_TIMEOUT': 'Reduce legal teacher/training work within the fixed budget.',
    'RESOURCE_MEMORY': 'Reduce legal proposal or sample work to fit the fixed memory limit.',
    'INFRASTRUCTURE_STORAGE': 'Archive and verify completed artifacts; restore storage before a fresh run.',
    'INFRASTRUCTURE_RUNTIME': 'Inspect trusted runtime logs and repair the execution service before retrying.',
    'INVALID_NUMERICAL_OUTPUT': 'Inspect finite training/evaluation checks; retain this invalid run.',
    'METHOD_RUNTIME_FAILURE': 'Inspect the method worker log and correct the submitted executable method.',
}


def matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


class Connection(http.client.HTTPConnection):
    def __init__(self, socket_file):
        super().__init__('dd-runtime', timeout=110)
        self.socket_file = socket_file

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_file)


def request(socket_file, verb, path, value=None):
    connection = Connection(socket_file)
    try:
        body = None if value is None else json.dumps(value, allow_nan=False).encode()
        headers = {'Content-Type': 'application/json'} if body else {}
        connection.request(verb, path, body, headers)
        response = connection.getresponse()
        payload = response.read(MAX_REPLY_BYTES + 1)
        if len(payload) > MAX_REPLY_BYTES:
            raise ValueError('Runtime reply is too large')
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise ValueError('Invalid runtime reply')
        if response.status not in (200, 202):
            raise RuntimeError('Runtime rejected request: ' + str(result.get('status')))
        return result
    finally:
        connection.close()


def checked_receipt(value, accepted, source_sha, formal):
    if value.get('job_id') != accepted['job_id'] or value.get('scope') != accepted['scope']:
        raise ValueError('Runtime identity mismatch')
    if value.get('status') != 'SUCCESS' or value.get('returncode') != 0:
        raise RuntimeError('The trusted evaluation did not succeed')
    if value.get('method_sha256') != source_sha:
        raise ValueError('The evaluated source does not match the submission')
    if not matches(r'run-[a-f0-9]{32}', value.get('run_id')):
        raise ValueError('Missing fresh run identity')
    if value.get('protocol') != 'dd-selfproto-v1.3' or value.get('sample_count') != 50000:
        raise ValueError('Protocol or sample count mismatch')
    if value.get('kind') != ('FORMAL' if formal else 'CALIBRATION'):
        raise ValueError('Runtime role mismatch')
    for field in ['checkpoint_sha256', 'sample_manifest_sha256', 'frozen_manifest_sha256']:
        if not matches(r'[a-f0-9]{64}', value.get(field)):
            raise ValueError('Missing trusted digest: ' + field)
    metric = value.get('metric')
    if type(metric) not in (float, int) or not math.isfinite(metric) or metric < 0:
        raise ValueError('Invalid independently computed FID')
    if formal:
        reward = value.get('reward')
        if type(reward) not in (float, int) or not math.isfinite(reward):
            raise ValueError('Invalid trusted reward')
    return value


def checked_failure(value, accepted, source_sha):
    if value.get('job_id') != accepted['job_id'] or value.get('scope') != accepted['scope']:
        raise ValueError('Runtime failure identity mismatch')
    if value.get('status') != 'FAILED' or value.get('method_sha256') != source_sha:
        raise ValueError('Runtime failure source/status mismatch')
    category = value.get('failure_category')
    if category not in FAILURE_ADVICE:
        raise ValueError('Missing trusted failure classification')
    # Never copy raw gateway errors, log text, private paths or unknown fields.
    result = {**accepted, 'status': 'FAILED', 'method_sha256': source_sha,
              'failure_category': category, 'repair_action': FAILURE_ADVICE[category]}
    if matches(r'run-[a-f0-9]{32}', value.get('run_id')):
        result['run_id'] = value['run_id']
    return result


def finish_failure(output, value, accepted, source_sha, formal):
    receipt = checked_failure(value, accepted, source_sha)
    atomic(output / 'failure.json', receipt)
    if formal and receipt['failure_category'] == 'HARD_GATE':
        stage = output / 'reward.txt.stage'
        stage.write_text('-1\n')
        stage.replace(output / 'reward.txt')
        print(json.dumps({**receipt, 'invalid_submission_score': -1}), flush=True)
        return
    print(json.dumps(receipt), flush=True)
    raise RuntimeError('Trusted evaluation failed: ' + receipt['failure_category'])


def atomic(path, value):
    stage = path.with_suffix(path.suffix + '.stage')
    stage.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    stage.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='/workspace/solution/method.py')
    parser.add_argument('--socket', default=os.environ.get('DD_RUNTIME_SOCKET', '/run/dd/public.sock'))
    parser.add_argument('--out', default='/workspace/agent-output/latest')
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=82800)
    args = parser.parse_args()
    if not 0 < args.wait_seconds <= 82800:
        raise ValueError('Invalid wait budget')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    # A prior score can never stand in for a fresh invocation.
    for name in ['result.json', 'reward.txt', 'invocation.json', 'failure.json',
                 'result.json.stage', 'reward.txt.stage', 'failure.json.stage']:
        (output / name).unlink(missing_ok=True)
    payload = Path(args.method).read_bytes()
    if not 1 <= len(payload) <= MAX_SOURCE_BYTES:
        raise ValueError('Invalid source size')
    source = payload.decode('utf-8')
    source_sha = hashlib.sha256(payload).hexdigest()
    accepted = request(args.socket, 'POST', '/jobs', {'source': source})
    if accepted.get('status') != 'ACCEPTED' or not matches(r'[a-f0-9]{32}', accepted.get('job_id')):
        raise ValueError('No accepted fresh runtime job')
    atomic(output / 'invocation.json', {**accepted, 'method_sha256': source_sha})
    print(json.dumps({'status':'ACCEPTED','job_id':accepted['job_id']}), flush=True)
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        value = request(args.socket, 'GET', '/jobs/' + accepted['job_id'])
        if value.get('status') != 'PENDING':
            if value.get('status') == 'FAILED':
                return finish_failure(output, value, accepted, source_sha, args.formal)
            receipt = checked_receipt(value, accepted, source_sha, args.formal)
            atomic(output / 'result.json', receipt)
            if args.formal:
                stage = output / 'reward.txt.stage'
                stage.write_text(str(receipt['reward']) + '\n')
                stage.replace(output / 'reward.txt')
            print(json.dumps(receipt, allow_nan=False), flush=True)
            return
        time.sleep(10)
    raise TimeoutError('Runtime job has not completed; retain its invocation receipt before retrying')


if __name__ == '__main__':
    main()
