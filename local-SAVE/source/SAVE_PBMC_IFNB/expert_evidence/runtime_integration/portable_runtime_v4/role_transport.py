"""Physical-role queue transport; no numerical code or credential in Actor."""
import base64,concurrent.futures,hashlib,json,os,re,subprocess,threading,time,urllib.request,uuid
from pathlib import Path
MODELS={'gpt':'gpt-6.1-sol','seed':'doubao-seed-2-1-turbo-260628'}
UUID=re.compile(r'^[0-9a-f]{32}$')
class PhysicalRoleRelay:
    def __init__(self, docker, broker, role, folder, numerical_status, broker_code):
        self.docker, self.role = docker, role
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
                elif frame['operation'] == 'chat_completions' and self.role == 'seed':
                    self.chat_completions_response(frame, send)
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
