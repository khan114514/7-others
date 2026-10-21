"""Keyless actual CLI model routing through physically independent queues."""
import base64
import concurrent.futures
import hashlib
import os
from host_paths import CONFIG
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS = {"gpt": "gpt-6.1-sol", "seed": "doubao-seed-2-1-turbo-260628"}
ENDPOINTS = {"gpt": "https://modcon.top/v1/responses", "seed": "https://ark.cn-beijing.volces.com/api/v3/responses"}
FRAME_LIMIT = 2 * 1024 * 1024
UUID = re.compile(r"^[0-9a-f]{32}$")


class ProviderRelay:
    def __init__(self, docker, broker, folder, broker_source, evaluator=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True)
        self.keys = {}
        self.evaluator = evaluator
        self.write_lock = threading.Lock()
        self.locks = {role: threading.Lock() for role in MODELS}
        self.dead = threading.Event()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.process = subprocess.Popen([docker, "exec", "-i", broker, "python", "-u", "-c", broker_source], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(self.folder / "broker-stderr.log").open("a"), text=True, bufsize=1)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def key(self, role):
        if role in self.keys:
            return self.keys[role]
        if role not in MODELS:
            raise ValueError('Unknown fixed model role')
        key = os.environ.get(CONFIG['credential_environment_names'][role], '')
        if not key:
            command = CONFIG['credential_commands'][role]
            if not command:
                raise RuntimeError('Authorized credential source is unavailable')
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError('Authorized credential command failed; output withheld')
            key = result.stdout.strip()
        if len(key) < 10 or any(character.isspace() for character in key):
            raise RuntimeError("Invalid existing provider credential")
        self.keys[role] = key
        return key

    def redact(self, value):
        for key in self.keys.values():
            value = value.replace(key, "[REDACTED]")
        return re.sub(r"(?:sk-|hf_)[A-Za-z0-9_-]{12,}", "[REDACTED]", value)

    def send(self, role, nonce, sequence, kind, **fields):
        frame = {"role": role, "id": nonce, "sequence": sequence, "kind": kind, **fields}
        with self.write_lock:
            self.process.stdin.write(json.dumps(frame, separators=(",", ":")) + "\n")
            self.process.stdin.flush()

    def read(self):
        while not self.dead.is_set():
            line = self.process.stdout.readline(FRAME_LIMIT + 1)
            if not line:
                return
            try:
                frame = json.loads(line)
                if len(line.encode()) > FRAME_LIMIT or set(frame) != {"role", "id", "operation", "payload"} or frame["role"] not in MODELS or not UUID.fullmatch(frame["id"]) or frame["operation"] not in {"responses", "evaluate"}:
                    continue
                self.pool.submit(self.evaluate if frame['operation'] == 'evaluate' else self.respond, frame)
            except (ValueError, KeyError, TypeError):
                continue

    def evaluate(self, frame):
        role, nonce = frame['role'], frame['id']
        try:
            if self.evaluator is None:
                raise RuntimeError('No validated GPU evaluator is available in this preparation environment')
            result = self.evaluator(role, frame['payload'])
            status = 200
        except Exception as error:
            status = 422
            result = {'error': type(error).__name__, 'message': self.redact(str(error))[:1000]}
        encoded = json.dumps(result, ensure_ascii=False).encode()
        if len(encoded) > FRAME_LIMIT:
            status = 422
            encoded = b'{"error":"Evaluator result budget exceeded"}'
        self.send(role, nonce, 0, 'header', status=status, content_type='application/json')
        self.send(role, nonce, 1, 'chunk', data=base64.b64encode(encoded).decode())
        self.send(role, nonce, 2, 'end')

    def respond(self, frame):
        role, nonce = frame["role"], frame["id"]
        sequence = 0
        started = time.time()
        receipt = {"role": role, "model_requested": MODELS[role], "reasoning_effort": "high", "upstream": ENDPOINTS[role], "started_epoch": started, "credential_visible_to_candidate": False}

        def send(kind, **fields):
            nonlocal sequence
            self.send(role, nonce, sequence, kind, **fields)
            sequence += 1

        with self.locks[role]:
            try:
                body = frame["payload"]
                if not isinstance(body, dict) or body.get("model") != MODELS[role]:
                    raise ValueError("Only the fixed lane model is allowed")
                key = self.key(role)
                body = {**body, "store": False, "reasoning": {**body.get("reasoning", {}), "effort": "high"}}
                encoded = json.dumps(body, separators=(",", ":")).encode()
                headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "codex-latent-research/1.0"}
                if role == "gpt":
                    headers["x-openai-actor-authorization"] = "local-image-extension"
                request = urllib.request.Request(ENDPOINTS[role], data=encoded, headers=headers)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({scheme: CONFIG["http_proxy"] for scheme in ["http", "https"]} if CONFIG["http_proxy"] else {}))
                (self.folder / (role + "-" + nonce + "-request.json")).write_text(self.redact(encoded.decode()) + "\n")
                receipt["request_sha256"] = hashlib.sha256(encoded).hexdigest()
                try:
                    response = opener.open(request, timeout=240)
                except urllib.error.HTTPError as error:
                    response = error
                receipt.update(http_status=response.status, first_header_epoch=time.time())
                send("header", status=response.status, content_type=response.headers.get("Content-Type", "application/json"))
                raw = bytearray()
                carry = bytearray()
                key_bytes = key.encode()
                with response:
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        raw.extend(chunk)
                        if len(raw) > 32 * 1024 * 1024:
                            raise ValueError("Provider response limit exceeded")
                        carry.extend(chunk)
                        carry = bytearray(bytes(carry).replace(key_bytes, b"[REDACTED]"))
                        safe = max(0, len(carry) - len(key_bytes) + 1)
                        if safe:
                            send("chunk", data=base64.b64encode(carry[:safe]).decode())
                            del carry[:safe]
                if carry:
                    send("chunk", data=base64.b64encode(bytes(carry).replace(key_bytes, b"[REDACTED]")).decode())
                text = self.redact(raw.decode(errors="replace"))
                (self.folder / (role + "-" + nonce + "-response.log")).write_text(text)
                receipt.update(response_sha256=hashlib.sha256(raw).hexdigest(), response_bytes=len(raw), model_response_completed='"type":"response.completed"' in text or '"type": "response.completed"' in text)
                served = set()
                for line in text.splitlines():
                    if not line.startswith('data: '): continue
                    try:
                        event = json.loads(line[6:])
                    except ValueError:
                        continue
                    model = event.get('response', {}).get('model')
                    if isinstance(model, str): served.add(model)
                receipt['model_identifiers_in_provider_response'] = sorted(served)
                receipt['served_identifier_matches_requested'] = served == {MODELS[role]}
                receipt["status"] = "RESPONSE_FINISHED"
            except Exception as error:
                receipt.update(status="ERROR", error_type=type(error).__name__, error=self.redact(str(error))[:1000])
                if sequence == 0:
                    send("header", status=502, content_type="application/json")
                send("chunk", data=base64.b64encode(json.dumps({"error": type(error).__name__}).encode()).decode())
            finally:
                receipt["completed_epoch"] = time.time()
                (self.folder / (role + "-" + nonce + "-route.json")).write_text(json.dumps(receipt, indent=2) + "\n")
                send("end")

    def close(self):
        self.dead.set()
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.keys.clear()
