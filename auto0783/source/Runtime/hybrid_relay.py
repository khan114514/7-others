"""Trusted host relay. Role authority is a physical Docker-owned queue volume.

No candidate-selected URL, command, host path, profile, credential or reward is
ever accepted. Model credentials are loaded lazily into this host process only.
"""
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

UUID = re.compile(r'^[0-9a-f]{32}$')
MANIFEST_SHA = '61d7fabc0298197c99c63317cfc906360d3dfd06e78741cdf4ac728ecf5d94fe'
MODELS = {'gpt': 'gpt-6.1-sol', 'seed': 'doubao-seed-2-1-turbo-260628'}
ENDPOINTS = {'gpt': 'https://modcon.top/responses',
             'seed': 'https://ark.cn-beijing.volces.com/api/v3/responses'}

_CREDENTIAL_PROVIDER = None


def register_credential_provider(provider):
    """Trusted process hook; callables and secrets never enter TrialConfig JSON."""
    global _CREDENTIAL_PROVIDER
    if provider is not None and not callable(provider):
        raise ValueError('Credential authorization must be a trusted callback')
    _CREDENTIAL_PROVIDER = provider


class HybridRelay:
    def __init__(self, docker, broker, role, folder, numerical_status, broker_code,
                 credential_provider=None, model_proxy=None):
        self.docker, self.role = docker, role
        if role not in ['gpt', 'seed', 'verifier']:
            raise ValueError('Unknown physical role')
        if credential_provider is not None and not callable(credential_provider):
            raise ValueError('Credential provider must be a trusted in-memory callback')
        if model_proxy is not None:
            from urllib.parse import urlparse
            endpoint = urlparse(model_proxy)
            if (endpoint.scheme not in ['http', 'https'] or endpoint.hostname not in
                    ['127.0.0.1', 'localhost', '::1'] or endpoint.username is not None
                    or endpoint.password is not None or endpoint.path not in ['', '/']
                    or endpoint.query or endpoint.fragment or endpoint.port is None):
                raise ValueError('Model proxy must be a trusted loopback endpoint')
        self.credential_provider = (credential_provider if credential_provider is not None
                                    else _CREDENTIAL_PROVIDER)
        self.model_proxy = model_proxy
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.numerical_status = Path(numerical_status)
        self.write_lock = threading.Lock(); self.role_lock = threading.Lock()
        self.dead = threading.Event(); self.key = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.process = subprocess.Popen([docker, 'exec', '-i', broker, 'python', '-u', '-c', broker_code],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=(self.folder / 'broker-stderr.log').open('a'), text=True, bufsize=1)
        self.thread = threading.Thread(target=self.read_requests, daemon=True)
        self.thread.start()

    def redact(self, value):
        if self.key:
            value = value.replace(self.key, '[redacted]')
        return re.sub(r'(?:sk-|hf_)[A-Za-z0-9_-]{12,}', '[redacted]', value)

    def send(self, nonce, sequence, kind, **fields):
        value = {'role': self.role, 'id': nonce, 'sequence': sequence, 'kind': kind, **fields}
        with self.write_lock:
            self.process.stdin.write(json.dumps(value, separators=(',', ':')) + '\n')
            self.process.stdin.flush()

    def read_requests(self):
        while not self.dead.is_set():
            line = self.process.stdout.readline(2 * 1024 * 1024 + 1)
            if not line:
                return
            try:
                if len(line.encode()) > 2 * 1024 * 1024:
                    raise ValueError('Oversized queue request')
                frame = json.loads(line)
                if (set(frame) != {'role', 'id', 'operation', 'payload'} or frame['role'] != self.role
                        or not isinstance(frame['id'], str) or not UUID.fullmatch(frame['id'])):
                    raise ValueError('Queue authority/identity mismatch')
                self.pool.submit(self.respond, frame)
            except (ValueError, KeyError, TypeError):
                # A malformed candidate frame is never interpreted as code.
                continue

    def evaluate(self, frame):
        payload = frame['payload']
        if not isinstance(payload, dict) or set(payload) != {'program'}:
            raise ValueError('Evaluation accepts only a program object')
        canonical = json.dumps(payload['program'], sort_keys=True, separators=(',', ':')).encode()
        if len(canonical) > 512 * 1024 or not isinstance(payload['program'], dict):
            raise ValueError('Graph exceeds the public artifact budget')
        state = json.loads(self.numerical_status.read_text())
        controller_root = Path(state['root'])
        if (not UUID.fullmatch(state['controller_id']) or controller_root.name != state['controller_id']
                or state['checkpoint_manifest_sha256'] != MANIFEST_SHA
                or state['phase'] in ['TRUSTED_WORKER_EXITED', 'TRUSTED_PROTOCOL_ERROR']):
            raise RuntimeError('Trusted numerical controller is unavailable or changed')
        # A stale status file is not a live numerical executor.
        os.kill(state['pid'], 0); os.kill(state['ssh_pid'], 0)
        identifier = uuid.uuid4().hex
        request = {'action': 'evaluate', 'run_id': identifier,
                   'profile': 'formal' if self.role == 'verifier' else 'dev',
                   'program': payload['program'], 'artifact_sha256': hashlib.sha256(canonical).hexdigest()}
        path = controller_root / 'inbox' / (identifier + '.json')
        pending = path.with_suffix('.pending.json')
        with pending.open('x') as handle:
            handle.write(json.dumps(request, separators=(',', ':')) + '\n')
        pending.replace(path)
        receipt = {'queue_nonce': frame['id'], 'physical_role': self.role,
                   'numerical_run_id': identifier, 'controller_id': state['controller_id'],
                   'profile': request['profile'], 'artifact_sha256': request['artifact_sha256'],
                   'requested_epoch': time.time(), 'formal_trajectory_effective_seconds': 0}
        receipt_path = self.folder / (frame['id'] + '-evaluation-binding.json')
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        deadline = time.monotonic() + 10600
        result_path = controller_root / 'results' / (identifier + '.json')
        while time.monotonic() < deadline and not self.dead.is_set():
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if (result.get('run_id') != identifier or result.get('artifact_sha256') != request['artifact_sha256']
                        or result.get('checkpoint_manifest_sha256') != MANIFEST_SHA):
                    raise RuntimeError('Numerical result has invalid lineage')
                receipt.update(completed_epoch=time.time(), result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                               execution_status=result.get('execution_status'), quality_pass=result.get('quality_pass'))
                receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
                # No withheld embeddings, prompt text or latent tensors are sent
                # to a public lane. Even the formal result is numeric metadata.
                return result
            current = json.loads(self.numerical_status.read_text())
            if current['controller_id'] != state['controller_id'] or current['phase'] in ['TRUSTED_WORKER_EXITED', 'TRUSTED_PROTOCOL_ERROR']:
                raise RuntimeError('Trusted numerical controller changed or failed')
            self.dead.wait(0.5)
        raise TimeoutError('Trusted numerical evaluation did not finish')

    def load_key(self):
        if self.key is not None:
            return
        if self.role not in MODELS:
            raise ValueError('Verifier cannot call a model')
        if self.credential_provider is None:
            raise RuntimeError('Trusted launcher has not provided model authorization')
        self.key = self.credential_provider(self.role)
        if not isinstance(self.key, str) or not self.key or len(self.key) < 10 or any(c.isspace() for c in self.key):
            self.key = None
            raise RuntimeError('Invalid existing provider credential')

    def model_response(self, frame, send):
        body = frame['payload']
        if not isinstance(body, dict) or body.get('model') != MODELS[self.role]:
            raise ValueError('Only the frozen lane model is permitted')
        self.load_key()
        body = {**body, 'store': False, 'reasoning': {**body.get('reasoning', {}), 'effort': 'high'}}
        encoded = json.dumps(body, separators=(',', ':')).encode()
        headers = {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json',
                   'User-Agent': 'codex-hybrid-controller/1.0'}
        if self.role == 'gpt':
            headers['x-openai-actor-authorization'] = 'local-image-extension'
        request = urllib.request.Request(ENDPOINTS[self.role], data=encoded, headers=headers)
        proxies = ({'http': self.model_proxy, 'https': self.model_proxy}
                   if self.model_proxy is not None else {})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        request_path = self.folder / (frame['id'] + '-request.json')
        request_path.write_text(self.redact(encoded.decode()) + '\n')
        started = time.time()
        try:
            response = opener.open(request, timeout=240)
        except urllib.error.HTTPError as error:
            response = error
        send('header', status=response.status, content_type=response.headers.get('Content-Type', 'application/json'))
        raw = bytearray(); carry = bytearray()
        with response:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > 32 * 1024 * 1024:
                    raise ValueError('Provider response byte limit exceeded')
                carry.extend(chunk)
                carry = bytearray(bytes(carry).replace(self.key.encode(), b'[redacted]'))
                safe = max(0, len(carry) - len(self.key.encode()) + 1)
                if safe:
                    send('chunk', data=base64.b64encode(carry[:safe]).decode())
                    del carry[:safe]
        if carry:
            send('chunk', data=base64.b64encode(bytes(carry).replace(self.key.encode(), b'[redacted]')).decode())
        (self.folder / (frame['id'] + '-response.log')).write_text(self.redact(raw.decode(errors='replace')))
        (self.folder / (frame['id'] + '-route.json')).write_text(json.dumps({
            'model': MODELS[self.role], 'reasoning_effort': 'high', 'store': False,
            'upstream': ENDPOINTS[self.role], 'http_status': response.status,
            'request_sha256': hashlib.sha256(encoded).hexdigest(), 'response_sha256': hashlib.sha256(raw).hexdigest(),
            'started_epoch': started, 'completed_epoch': time.time(), 'credential_in_candidate': False}, indent=2) + '\n')

    def respond(self, frame):
        sequence = 0

        def send(kind, **fields):
            nonlocal sequence
            self.send(frame['id'], sequence, kind, **fields); sequence += 1

        with self.role_lock:
            try:
                if frame['operation'] == 'evaluate':
                    value = self.evaluate(frame)
                    send('header', status=200, content_type='application/json')
                    send('chunk', data=base64.b64encode(json.dumps(value).encode()).decode())
                elif frame['operation'] == 'responses' and self.role in MODELS:
                    self.model_response(frame, send)
                else:
                    raise ValueError('Operation is unavailable for this physical role')
            except Exception as error:
                if sequence == 0:
                    send('header', status=502, content_type='application/json')
                value = {'error_type': type(error).__name__, 'error': self.redact(str(error))[:1000]}
                (self.folder / (frame['id'] + '-error.json')).write_text(json.dumps(value, indent=2) + '\n')
                send('chunk', data=base64.b64encode(json.dumps(value).encode()).decode())
            finally:
                send('end')

    def close(self):
        self.dead.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.key = None
