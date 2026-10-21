from host_paths import gpu_command
"""Trusted physical-lane bridge to the fixed GPU evaluator API over SSH."""
import json
import hashlib
import os
import uuid
import subprocess
import threading
import time
from pathlib import Path

from host_paths import PRIVATE_ROOT as PRIVATE


class EvaluatorBridge:
    def __init__(self, folder, *, verifier=False):
        if type(verifier) is not bool:
            raise ValueError('Trusted verifier routing flag must be boolean')
        self.verifier = verifier
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.locks = {role: threading.Lock() for role in ['gpt', 'seed']}

    def record_transport_failure(self, role, payload, before, after, kind, stdout='', stderr='', exception=None):
        # Original remote diagnostics are operator-private, never Actor feedback.
        folder = self.folder / 'private-transport-errors'
        folder.mkdir(mode=0o700, exist_ok=True)
        path = folder / (role + '-' + uuid.uuid4().hex + '.json')
        value = {'role': role, 'kind': kind, 'local_request_start_epoch': before,
                 'local_request_end_epoch': after, 'request_sha256': hashlib.sha256(
                     json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                 'original_stdout': stdout, 'original_stderr': stderr, 'exception': exception,
                 'candidate_visible_diagnostics': False}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')

    def __call__(self, role, payload):
        if role not in self.locks:
            raise ValueError('Unknown physical lane')
        encoded = json.dumps(payload, separators=(',', ':'))
        if len(encoded.encode()) > 2 * 1024 * 1024:
            raise ValueError('Evaluator input budget exceeded')
        with self.locks[role]:
            # Retries preserve the exact encoded payload and idempotency UUID.
            # Waits/failures are private infrastructure evidence, never research.
            while True:
                before = time.time()
                try:
                    result = subprocess.run(gpu_command(role, verifier=self.verifier),
                        input=encoded, capture_output=True, text=True, timeout=30)
                except (OSError, subprocess.SubprocessError) as error:
                    self.record_transport_failure(role, payload, before, time.time(),
                        'transport-retry', exception=repr(error))
                    time.sleep(10)
                    continue
                after = time.time()
                if result.returncode == 255:
                    self.record_transport_failure(role, payload, before, after,
                        'ssh-transport-retry', stdout=result.stdout, stderr=result.stderr)
                    time.sleep(10)
                    continue
                if result.returncode:
                    self.record_transport_failure(role, payload, before, after, 'request-rejected',
                                                  stdout=result.stdout, stderr=result.stderr)
                    raise RuntimeError('Evaluator request rejected; check your own job ticket and public submission contract') from None
                break
            try:
                value = json.loads(result.stdout)
            except (ValueError, TypeError) as error:
                self.record_transport_failure(role, payload, before, after, 'invalid-response',
                                              stdout=result.stdout, stderr=result.stderr, exception=repr(error))
                raise RuntimeError('Evaluator returned an invalid response; retain your own ticket for recovery') from None
            # Active-job discovery returns only this physical lane's job.
            if 'active' in value and value['active'] is None:
                return value
            activity_value = value['active'] if 'active' in value else value
            # Track clock conversion with request bounds, never combine raw
            # timestamps from the Mac and GPU clocks without an offset.
            audit = {'local_request_start_epoch': before, 'local_request_end_epoch': after,
                     'clock_offset_estimate_seconds': (before + after) / 2 - value['remote_observed_epoch'],
                     'transport_uncertainty_seconds': (after - before) / 2, 'result': activity_value}
            journal = self.folder / (role + '-gpu-activity.jsonl')
            with journal.open('a') as stream:
                stream.write(json.dumps(audit, ensure_ascii=False) + '\n')
            return value
