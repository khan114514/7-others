"""Trusted side of the CPU method RPC; it never imports method.py."""
import atexit
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

from config import ENGINE, deployment, sha
from method_contract import checked_spec, checked_times, checked_indices, checked_weights


class MethodProxy:
    def __init__(self, candidate):
        self.source_sha256 = sha(candidate)
        run = Path(os.environ['DD_RUN']).resolve()
        self.temporary = Path(tempfile.mkdtemp(prefix='method-', dir=run))
        self.source = self.temporary / 'source'
        self.source.mkdir()
        # No submitted symlink, helper file, checkpoint or precomputed output
        # is accepted. This regular-file snapshot is the entire method input.
        frozen = self.source / 'method.py'
        shutil.copyfile(candidate, frozen, follow_symlinks=True)
        if sha(frozen) != self.source_sha256:
            raise RuntimeError('Candidate changed while taking a snapshot')
        scratch = self.temporary / 'scratch'
        scratch.mkdir()
        self.log = (self.temporary / 'worker.log').open('wb')
        config = deployment()
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(scratch), 'TMPDIR': str(scratch),
               'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1'}
        self.process = subprocess.Popen([config['python'], '-I',
                str(ENGINE / 'method_worker.py'), str(frozen), str(scratch)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                cwd=scratch, env=env, start_new_session=True, close_fds=True)
        (self.temporary / 'worker_pid.json').write_text(json.dumps({'pid': self.process.pid}))
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.remainder = b''
        atexit.register(self.close)
        response = self.request({'op': 'config'}, timeout=60)
        self.spec = checked_spec(response['result'])
        self.guard = response['guard']

    def close(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.log):
            if not stream.closed:
                stream.close()

    def request(self, value, timeout=15):
        pending = memoryview((json.dumps(value, allow_nan=False,
                             separators=(',', ':')) + '\n').encode())
        data = self.remainder
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as events:
                events.register(self.process.stdin, selectors.EVENT_WRITE)
                events.register(self.process.stdout, selectors.EVENT_READ)
                while b'\n' not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Candidate method call exceeded its time allowance')
                    ready = events.select(remaining)
                    if not ready:
                        continue
                    for key, mask in ready:
                        if mask & selectors.EVENT_WRITE:
                            try:
                                size = os.write(key.fd, pending)
                            except BlockingIOError:
                                continue
                            pending = pending[size:]
                            if not pending:
                                events.unregister(self.process.stdin)
                        if mask & selectors.EVENT_READ:
                            part = os.read(key.fd, 65536)
                            if not part:
                                raise RuntimeError('Candidate process exited; inspect worker.log')
                            data += part
                            if len(data) > 1 << 20:
                                raise ValueError('Candidate response exceeds the transport limit')
            line, self.remainder = data.split(b'\n', 1)
            if pending:
                raise ValueError('Candidate responded before receiving the request')
            response = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(response, dict):
                raise ValueError('Candidate response must be a JSON object')
            return response
        except BaseException:
            self.close()
            raise

    def fm_time_sampler(self, batch_shape, device, generator=None):
        import torch
        if len(batch_shape) != 1 or not 1 <= batch_shape[0] <= 256:
            raise ValueError('Invalid FM sampler request')
        if generator is None or str(generator.device) != 'cpu':
            raise ValueError('The frozen FM random stream must use a CPU generator')
        state = generator.get_state()
        response = self.request({'op': 'time', 'shape': list(batch_shape), 'rng': state.tolist()})
        values = checked_times(response['result'], batch_shape[0])
        new_state = response.get('rng')
        if not isinstance(new_state, list) or len(new_state) != len(state) or any(
                type(x) is not int or not 0 <= x <= 255 for x in new_state):
            raise ValueError('Candidate returned an invalid random-generator state')
        generator.set_state(torch.tensor(new_state, dtype=torch.uint8))
        return torch.tensor(values, dtype=torch.float32, device=device)

    def select_teacher_candidates(self, agreement, multiplier):
        import torch
        values = agreement.detach().float().cpu().tolist()
        if any(not math.isfinite(x) for x in values) or len(values) % multiplier:
            raise ValueError('Invalid teacher agreement vector')
        response = self.request({'op': 'select', 'agreement': values, 'multiplier': multiplier})
        indices = checked_indices(response['result'], len(values) // multiplier, multiplier)
        return torch.tensor(indices, dtype=torch.long)

    def pair_weights(self, labels, target_ids):
        import torch
        response = self.request({'op':'weights', 'labels':labels.detach().cpu().tolist(),
                                 'target_ids':target_ids.detach().cpu().tolist()})
        values = checked_weights(response['result'], len(labels))
        return torch.tensor(values, dtype=torch.float32, device=labels.device)
