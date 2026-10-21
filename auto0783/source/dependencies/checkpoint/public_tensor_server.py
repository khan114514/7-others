"""Loopback-only server for the declared public decoder tensor ordinals."""
import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MANIFEST_SHA = '61d7fabc0298197c99c63317cfc906360d3dfd06e78741cdf4ac728ecf5d94fe'


def prepare_inventory(manifest_path, checkpoint_directory):
    encoded = manifest_path.read_bytes()
    if len(encoded) > 2 * 1024 * 1024 or hashlib.sha256(encoded).hexdigest() != MANIFEST_SHA:
        raise ValueError('Public tensor manifest binding differs')
    manifest = json.loads(encoded)
    if manifest['schema'] != 'bifrost-public-decoder-tensors/1' or manifest['tensor_count'] != 677:
        raise ValueError('Public decoder tensor set differs')
    for name, specification in manifest['checkpoint_shards'].items():
        if not re.fullmatch(r'model-0000[1-6]-of-00006\.safetensors', name):
            raise ValueError('Unexpected public shard name')
        path = checkpoint_directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != specification['bytes']:
            raise ValueError('Declared public checkpoint asset missing or differently sized')
    for ordinal, value in enumerate(manifest['tensors']):
        if (value['id'] != ordinal or value['shard'] not in manifest['checkpoint_shards']
                or not 0 < value['bytes'] <= 256 * 1024 * 1024 or value['offset'] < 0
                or value['offset'] + value['bytes'] > manifest['checkpoint_shards'][value['shard']]['bytes']):
            raise ValueError('Declared public tensor range differs')
    return encoded, manifest


def make_handler(encoded, manifest, checkpoint_directory):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_GET(self):
            if self.path == '/manifest.json':
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded))); self.end_headers()
                self.wfile.write(encoded); return
            if self.path == '/health':
                body = json.dumps({'manifest_sha256': MANIFEST_SHA, 'tensor_count': manifest['tensor_count']}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(body)))
                self.end_headers(); self.wfile.write(body); return
            match = re.fullmatch(r'/tensor/([0-9]{1,4})', self.path)
            if not match or int(match[1]) >= len(manifest['tensors']):
                self.send_error(404); return
            value = manifest['tensors'][int(match[1])]
            self.send_response(200); self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(value['bytes']))
            self.send_header('X-Public-Tensor-SHA256', value['sha256']); self.end_headers()
            try:
                with (checkpoint_directory / value['shard']).open('rb') as stream:
                    stream.seek(value['offset']); remaining = value['bytes']
                    while remaining:
                        block = stream.read(min(4 * 1024 * 1024, remaining))
                        if not block: raise IOError('Immutable public tensor was truncated')
                        self.wfile.write(block); remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError): pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    args = parser.parse_args()
    encoded, manifest = prepare_inventory(args.manifest, args.checkpoint_directory)
    # The worker's trusted loader uses this fixed loopback endpoint. A public
    # candidate container has no network route or mount to the assets.
    ThreadingHTTPServer(('127.0.0.1', 19195), make_handler(encoded, manifest, args.checkpoint_directory)).serve_forever()


if __name__ == '__main__':
    main()
