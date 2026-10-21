"""Trusted host IPC facade for an untrusted draft policy in a per-run jail."""
import json
import os
import selectors
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from deployment_paths import PRIVATE_ROOT as PRIVATE, JAIL_PYTHON
LIMIT = 262144


class PolicyExecutor:
    def __init__(self, files):
        if 'method.py' not in files or len(files) > 64:
            raise ValueError('Invalid candidate file set')
        if sum(len(value.encode()) for value in files.values()) > 2 * 1024 * 1024:
            raise ValueError('Candidate source limit exceeded')
        self.jail = PRIVATE / 'policy-jails' / uuid.uuid4().hex
        self.jail.parent.mkdir(mode=0o700, exist_ok=True)
        shutil.copytree(PRIVATE / 'policy-jail-base-v1', self.jail, copy_function=os.link)
        self.jail.chmod(0o755)
        solution = self.jail / 'workspace/solution'
        solution.mkdir(parents=True)
        for name, value in files.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or path.suffix not in {'.py', '.json', '.txt', '.md'}:
                raise ValueError('Invalid candidate source name')
            target = solution / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
            target.chmod(0o644)
        tmp = self.jail / 'tmp'
        # Policies compute proposals; no runtime filesystem writes are needed.
        # Keeping this root-owned also bounds filesystem growth without relying
        # on per-file RLIMIT_FSIZE as a quota.
        tmp.mkdir(mode=0o755)
        self.buffer = bytearray()
        self.sequence = 0
        self.log_path = self.jail.with_suffix('.stderr.log')
        self.log = self.log_path.open('wb')
        self.process = subprocess.Popen([str(JAIL_PYTHON), '-I', '-S', str(PRIVATE / 'policy_jail_launcher.py'), str(self.jail), str(PRIVATE), str(JAIL_PYTHON)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        close_fds=True, start_new_session=True,
                                        env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self.read(10)
        except Exception:
            self.close()
            raise
        if ready != {'ready': True, 'uid': 65534, 'gid': 65534, 'seccomp_loaded': True}:
            self.close()
            raise RuntimeError('Isolated executor startup failed')

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise TimeoutError('Policy wall-time budget exceeded')
            raw = os.read(self.process.stdout.fileno(), 65536)
            if not raw:
                raise RuntimeError('Policy process exited')
            self.buffer.extend(raw)
            if len(self.buffer) > LIMIT:
                raise ValueError('Policy response size exceeded')
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return json.loads(line)

    def draft(self, state):
        self.sequence += 1
        raw = json.dumps({'id': self.sequence, 'state': state}, ensure_ascii=False).encode() + b'\n'
        if len(raw) > LIMIT:
            raise ValueError('Public state too large')
        self.process.stdin.write(raw)
        self.process.stdin.flush()
        value = self.read(0.5)
        if not isinstance(value, dict) or set(value) != {'id', 'proposal'} or value['id'] != self.sequence:
            raise ValueError('Invalid policy response lineage')
        proposal = value['proposal']
        cap = min(32, state['remaining_tokens'], state['maximum_draft_tokens'])
        if isinstance(proposal, list):
            if len(proposal) > cap or any(type(v) is not int or not 0 <= v < 128256 for v in proposal):
                raise ValueError('Invalid proposed token IDs')
        elif isinstance(proposal, dict) and set(proposal) == {'text'} and isinstance(proposal['text'], str):
            if len(proposal['text'].encode()) > 16384:
                raise ValueError('Proposed text too large')
        else:
            raise ValueError('Policy must return token IDs or one text field')
        return proposal

    def close(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        if hasattr(self, 'selector'):
            self.selector.close()
        if hasattr(self, 'log'):
            self.log.close()
        shutil.rmtree(self.jail)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
