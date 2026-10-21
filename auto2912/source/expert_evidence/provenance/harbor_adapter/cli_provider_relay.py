"""Keyless actual CLI model routing through physically independent queues."""
import base64
import concurrent.futures
import hashlib
import json
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS = {"gpt": "gpt-6.1-sol", "seed": "doubao-seed-2-1-turbo-260628"}
ENDPOINTS = {"gpt": "https://modcon.top/v1/responses", "seed": "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"}
FRAME_LIMIT = 2 * 1024 * 1024
CHAT_REQUEST_MAX_SECONDS = 330
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
        if role == 'seed':
            # User-authorized host-only configuration. Refresh at each request;
            # no long-lived profile or stale credential cache reaches the Actor.
            configuration = json.loads(Path('<AUTHOR_HOME>/.config/ark/current.json').read_text())
            if (configuration.get('model') != MODELS[role]
                    or configuration.get('chat_completions_url') != ENDPOINTS[role]
                    or configuration.get('billing_mode') != 'coding_plan'):
                raise RuntimeError('Shared Ark route differs from the user-authorized lane')
            key = configuration.get('api_key', '')
            if len(key) < 10 or any(character.isspace() for character in key):
                raise RuntimeError('Invalid host-only Ark credential')
            self.keys[role] = key
            return key
        if role in self.keys:
            return self.keys[role]
        if role == "gpt":
            path = Path(__file__).resolve().parents[2] / "bifrost1_20261005/private/keychain_credential.py"
            # The existing Keychain item authorizes the system interpreter.
            # A real A/B probe showed the Harbor3.12 interpreter blocks here.
            # Capture only in memory; never expose the credential to an Actor,
            # argument, config, log or exception output.
            code = ('import importlib.util;spec=importlib.util.spec_from_file_location('
                    '"authorized_provider_keychain",' + repr(str(path)) + ');'
                    'module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);'
                    'print(module.get_key(),end="")')
            result = subprocess.run(['/Library/Developer/CommandLineTools/usr/bin/python3', '-c', code],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError('Existing authorized Keychain retrieval failed; output withheld')
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
        if frame['role'] == 'seed':
            return self.respond_chat(frame)
        role, nonce = frame["role"], frame["id"]
        sequence = 0
        started = time.time()
        receipt = {"role": role, "model_requested": MODELS[role], "reasoning_effort": ("xhigh" if role == "gpt" else "high"), "upstream": ENDPOINTS[role], "started_epoch": started, "credential_visible_to_candidate": False}

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
                body = {**body, "store": False, "reasoning": {**body.get("reasoning", {}), "effort": ("xhigh" if role == "gpt" else "high")}}
                encoded = json.dumps(body, separators=(",", ":")).encode()
                headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "codex-latent-research/1.0"}
                if role == "gpt":
                    headers["x-openai-actor-authorization"] = "local-image-extension"
                request = urllib.request.Request(ENDPOINTS[role], data=encoded, headers=headers)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": "http://127.0.0.1:7892", "https": "http://127.0.0.1:7892"}))
                (self.folder / (role + "-" + nonce + "-request.json")).write_text(self.redact(encoded.decode()) + "\n")
                receipt["request_sha256"] = hashlib.sha256(encoded).hexdigest()
                try:
                    response = opener.open(request, timeout=240)
                except urllib.error.HTTPError as error:
                    response = error
                receipt.update(http_status=response.status, first_header_epoch=time.time())
                if response.status in [401,403]:
                    self.keys.pop(role,None)  # Reload the authorized profile on retry.
                send("header", status=response.status, content_type=response.headers.get("Content-Type", "application/json"))
                raw = bytearray()
                carry = bytearray()
                secret = key.encode()
                with response:
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        receipt.setdefault('first_chunk_epoch',time.time())
                        receipt['last_chunk_epoch']=time.time()
                        raw.extend(chunk)
                        if len(raw) > 32 * 1024 * 1024:
                            raise ValueError("Provider response limit exceeded")
                        carry.extend(chunk)
                        carry = bytearray(bytes(carry).replace(secret, b"[REDACTED]"))
                        safe = max(0, len(carry) - len(secret) + 1)
                        if safe:
                            send("chunk", data=base64.b64encode(carry[:safe]).decode())
                            del carry[:safe]
                if carry:
                    send("chunk", data=base64.b64encode(bytes(carry).replace(secret, b"[REDACTED]")).decode())
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

    def respond_chat(self, frame):
        from ark_chat_adapter import chat_request, ResponseEvents, sse
        role, nonce = frame['role'], frame['id']
        sequence = 0
        receipt = {'role': role, 'model_requested': MODELS[role], 'reasoning_effort': 'high',
                   'upstream': ENDPOINTS[role], 'started_epoch': time.time(),
                   'billing_mode': 'coding_plan',
                   'wire_translation': 'Chat Completions upstream / native Responses CLI downstream',
                   'credential_visible_to_candidate': False, 'configuration_source': 'host-only authorized shared Ark configuration'}
        raw = bytearray()
        downstream = bytearray()
        pending_events = bytearray()
        last_event_flush = time.monotonic()
        stream_timer = None

        def send(kind, **fields):
            nonlocal sequence
            self.send(role, nonce, sequence, kind, **fields)
            sequence += 1

        def flush_events():
            nonlocal last_event_flush
            if pending_events:
                send('chunk', data=base64.b64encode(pending_events).decode())
                pending_events.clear()
                last_event_flush = time.monotonic()

        def event(value):
            encoded = sse(value)
            downstream.extend(encoded)
            pending_events.extend(encoded)
            if len(pending_events)>=16384 or time.monotonic()-last_event_flush>=.1:
                flush_events()

        with self.locks[role]:
            try:
                remaining = receipt['started_epoch'] + CHAT_REQUEST_MAX_SECONDS - time.time()
                if remaining <= 0:
                    raise TimeoutError('Actor response request expired while waiting for its lane')
                body = frame['payload']
                if not isinstance(body, dict) or body.get('model') != MODELS[role]:
                    raise ValueError('Only the fixed lane model is allowed')
                key = self.key(role)
                chat, custom, omitted = chat_request(body)
                receipt['unavailable_responses_only_tool_types'] = omitted
                encoded = json.dumps(chat, separators=(',', ':')).encode()
                (self.folder/(role+'-'+nonce+'-request.json')).write_text(self.redact(encoded.decode())+'\n')
                receipt['request_sha256'] = hashlib.sha256(encoded).hexdigest()
                request = urllib.request.Request(ENDPOINTS[role], data=encoded, headers={
                    'Authorization': 'Bearer '+key, 'Content-Type': 'application/json',
                    'User-Agent': 'codex-latent-research/1.0'})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({
                    'http': 'http://127.0.0.1:7892', 'https': 'http://127.0.0.1:7892'}))
                try:
                    response = opener.open(request, timeout=min(240, remaining))
                except urllib.error.HTTPError as error:
                    response = error
                receipt.update(http_status=response.status, first_header_epoch=time.time())
                if response.status != 200:
                    text = self.redact(response.read(65536).decode(errors='replace'))
                    (self.folder/(role+'-'+nonce+'-response.log')).write_text(text)
                    send('header', status=response.status, content_type='application/json')
                    send('chunk', data=base64.b64encode(text.encode()).decode())
                    receipt['status'] = 'UPSTREAM_HTTP_ERROR'
                    return
                events = ResponseEvents(MODELS[role], custom)
                def expire_stream():
                    receipt['upstream_cancelled_on_actor_deadline'] = True
                    # Closing just the HTTPResponse can wait on its read lock.
                    # Shutdown the owned socket to unblock a partial SSE read.
                    try:
                        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    except (AttributeError, OSError):
                        pass
                receipt['actor_deadline_epoch'] = receipt['started_epoch'] + CHAT_REQUEST_MAX_SECONDS
                stream_timer = threading.Timer(max(.01, receipt['actor_deadline_epoch']-time.time()), expire_stream)
                stream_timer.daemon = True
                stream_timer.start()
                send('header', status=200, content_type='text/event-stream')
                for value in events.begin():
                    event(value)
                with response:
                    for line in response:
                        if time.time() >= receipt['actor_deadline_epoch']:
                            raise TimeoutError('Upstream response exceeded the Actor request budget')
                        raw.extend(line)
                        if len(raw) > 32*1024*1024:
                            raise ValueError('Provider response limit exceeded')
                        receipt.setdefault('first_chunk_epoch', time.time())
                        receipt['last_chunk_epoch'] = time.time()
                        if not line.startswith(b'data:'):
                            continue
                        data = line[5:].strip()
                        if data == b'[DONE]':
                            events.saw_done = True
                            break
                        value = json.loads(data)
                        if value.get('error'):
                            raise RuntimeError('Chat provider returned an in-stream error')
                        for value in events.chunk(value):
                            event(value)
                for value in events.finish():
                    event(value)
                receipt.update(model_response_completed=True, status='RESPONSE_FINISHED',
                               model_identifiers_in_provider_response=sorted(events.provider_models),
                               served_identifier_matches_requested=events.provider_models=={MODELS[role]},
                               provider_completion_id=events.provider_id, provider_finish_reason=events.finish_reason,
                               usage=events.usage, response_sha256=hashlib.sha256(raw).hexdigest(),
                               response_bytes=len(raw), tool_calls_returned=len(events.calls))
            except Exception as error:
                receipt.update(status='ERROR', error_type=type(error).__name__, error=self.redact(str(error))[:1000])
                if sequence == 0:
                    send('header', status=502, content_type='application/json')
                    send('chunk', data=base64.b64encode(b'{"error":"Chat bridge failed"}').decode())
                else:
                    event({'type': 'error', 'code': 'upstream_chat_failure',
                           'message': self.redact(str(error))[:500]})
            finally:
                if stream_timer is not None:
                    stream_timer.cancel()
                if raw:
                    (self.folder/(role+'-'+nonce+'-response.log')).write_text(self.redact(raw.decode(errors='replace')))
                if downstream:
                    (self.folder/(role+'-'+nonce+'-translated-response.log')).write_text(self.redact(downstream.decode(errors='replace')))
                receipt['completed_epoch'] = time.time()
                (self.folder/(role+'-'+nonce+'-route.json')).write_text(json.dumps(receipt,indent=2)+'\n')
                flush_events()
                send('end')

    def close(self):
        self.dead.set()
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.keys.clear()
