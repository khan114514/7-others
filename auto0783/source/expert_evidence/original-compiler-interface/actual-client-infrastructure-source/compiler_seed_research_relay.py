"""Private relay for resuming the original GPT author; frozen Runtime stays unchanged.

No research is started by importing this module. Numerical authority is inherited
from the physical GPT role queue; only the pinned GPT Responses route is allowed.
"""
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from hybrid_relay import HybridRelay

ENDPOINT = 'https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions'
MODEL = 'doubao-seed-2-1-turbo-260628'


class RejectResearchRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise ValueError('Original Seed research provider redirect rejected')


class SameTrackSeedResearchRelay(HybridRelay):
    def load_key(self):
        import stat
        profile=Path('<author-home>/.config/ark/current.json')
        if stat.S_IMODE(profile.stat().st_mode)!=0o600:raise ValueError('Shared Ark profile must be 0600')
        config=json.loads(profile.read_text())
        if config.get('chat_completions_url')!=ENDPOINT or config.get('model')!=MODEL:raise ValueError('Shared Ark route or exact model differs')
        value=config.get('api_key')
        if not isinstance(value,str) or len(value)<10 or any(c.isspace() for c in value):raise ValueError('Shared Ark credential unavailable')
        self.key=value

    def model_response(self, frame, send):
        if self.role != 'seed':
            raise ValueError('Original Seed continuation cannot call another model role')
        body = frame['payload']
        if not isinstance(body, dict) or body.get('model') != MODEL:
            raise ValueError('Original Seed continuation model differs')
        if not isinstance(body.get('messages'),list) or not isinstance(body.get('tools'),list) or any(t.get('type')!='function' for t in body['tools']):raise ValueError('Direct ChatCompletions client shape differs')
        body={**body,'stream':True,'stream_options':{'include_usage':True},'thinking':{'type':'enabled'},'reasoning_effort':'high'}
        # The trusted credential callback supplies RAM only, at a request boundary.
        self.key = None
        self.load_key()
        encoded = json.dumps(body, separators=(',', ':')).encode()
        prefix = self.folder / frame['id']
        request_path = Path(str(prefix) + '-request.json')
        with request_path.open('x') as out:
            out.write(self.redact(encoded.decode()) + '\n')
        proxies = ({'http': self.model_proxy, 'https': self.model_proxy}
                   if self.model_proxy is not None else {})
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(proxies), RejectResearchRedirects())
        request = urllib.request.Request(ENDPOINT, data=encoded, headers={
            'Authorization': 'Bearer ' + self.key,
            'Content-Type': 'application/json',
            'User-Agent': 'bifrost-original-Seed-research-controller/1.0',
        })
        clock = {
            'model': MODEL, 'reasoning_effort': 'high', 'upstream': ENDPOINT,
            'actual_final_url': None, 'redirects_allowed': False,
            'credential_in_candidate': False,
            'credential_refreshed_at_request_boundary': True,
            'started_epoch': time.time(), 'first_response_byte_epoch': None,
            'completed_epoch': None, 'http_status': None,
            'response_bytes_observed': 0, 'stream_closed_normally': False,
            'research_credit_claimed': False,
            'request_sha256': hashlib.sha256(encoded).hexdigest(),'native_wire_api':'chat_completions','upstream_wire_api':'chat_completions','actual_client':'native-ccagent-1.0 research adapter',
        }
        route_path = Path(str(prefix) + '-route.json')
        wire = hashlib.sha256()
        carry = bytearray()
        try:
            try:
                response = opener.open(request, timeout=240)
            except urllib.error.HTTPError as error:
                response = error
            clock['http_status'] = response.status
            clock['actual_final_url'] = response.geturl()
            if clock['actual_final_url'] != ENDPOINT:
                response.close()
                raise ValueError('Original Seed response URL differs from authorization')
            send('header', status=response.status,
                 content_type=response.headers.get('Content-Type', 'application/json'))
            with Path(str(prefix) + '-response.log').open('xb') as raw_log, response:
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    if clock['first_response_byte_epoch'] is None:
                        clock['first_response_byte_epoch'] = time.time()
                    clock['response_bytes_observed'] += len(chunk)
                    wire.update(chunk)
                    carry.extend(chunk)
                    carry = bytearray(bytes(carry).replace(self.key.encode(), b'[redacted]'))
                    safe = max(0, len(carry) - len(self.key.encode()) + 1)
                    if safe:
                        payload = bytes(carry[:safe])
                        raw_log.write(payload)
                        raw_log.flush()
                        send('chunk', data=base64.b64encode(payload).decode())
                        del carry[:safe]
                    if clock['response_bytes_observed'] > 32 * 1024 * 1024:
                        raise ValueError('Original Seed provider response exceeds byte limit')
                if carry:
                    payload = bytes(carry).replace(self.key.encode(), b'[redacted]')
                    raw_log.write(payload)
                    raw_log.flush()
                    send('chunk', data=base64.b64encode(payload).decode())
                    carry.clear()
            clock['stream_closed_normally'] = True
        except Exception as error:
            clock['actual_error_type'] = type(error).__name__
            clock['actual_error'] = self.redact(str(error))[:1500]
            if carry:
                with Path(str(prefix) + '-partial-tail.log').open('xb') as out:
                    out.write(bytes(carry).replace(self.key.encode(), b'[redacted]'))
            raise
        finally:
            clock['completed_epoch'] = time.time()
            clock['observed_response_sha256'] = wire.hexdigest()
            with route_path.open('x') as out:
                json.dump(clock, out, indent=2)
                out.write('\n')
