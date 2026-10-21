"""Meaningful transport regressions; these are CPU tests, not NOP evidence."""
import hashlib
import contextlib
import io
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


client = load('runtime_client')
service = load('http_service')
gateway = load('ssh_gateway')


class RuntimeBoundary(unittest.TestCase):
    def test_original_enospc_is_infrastructure_not_a_hard_gate(self):
        record = HERE.parent / 'optimization_evidence/terminal_records/run-d824955fa1fb4627a11eeb25816fd4b9'
        value = gateway.classify_failure(json.loads((record/'failure.json').read_text()), record)
        self.assertEqual(value['failure_category'], 'INFRASTRUCTURE_STORAGE')
        self.assertNotIn('error', value)
        self.assertEqual(gateway.FAILURE_ADVICE, client.FAILURE_ADVICE)

    def test_failure_categories_and_untrusted_strings_are_separate(self):
        for kind, message, expected in [
            ('ValueError', 'No space left on device in an invalid submitted field', 'HARD_GATE'),
            ('TimeoutError', 'budget exhausted', 'BUDGET_TIMEOUT'),
            ('MemoryError', 'allocation failed', 'RESOURCE_MEMORY'),
            ('SyntaxError', 'syntax invalid', 'FORMAT_ERROR'),
            ('FloatingPointError', 'nonfinite loss', 'INVALID_NUMERICAL_OUTPUT'),
            ('WorkerExited', '', 'INFRASTRUCTURE_RUNTIME')]:
            with self.subTest(kind=kind):
                result = gateway.classify_failure({'error_type':kind,'error':message})
                self.assertEqual(result['failure_category'], expected)
                if message:
                    self.assertNotIn(message, json.dumps(result))

    def test_wrapped_tensor_constraint_failure_has_invalid_category(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'training.log').write_text('ValueError: FM times must be finite and in [0,1]\n')
            result = gateway.classify_failure({'error_type':'RuntimeError',
                                              'error':'training failed with exit code 1'}, run)
            self.assertEqual(result['failure_category'], 'HARD_GATE')

    def test_failed_receipt_bound_before_invalid_score_is_written(self):
        accepted = {'job_id':'a'*32,'scope':'track-a'}
        source = 'b'*64
        value = {**accepted,'status':'FAILED','method_sha256':source,
                 'failure_category':'HARD_GATE','error':'private detail must not be forwarded'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(ValueError):
                client.finish_failure(output, {**value,'scope':'other'}, accepted, source, True)
            self.assertFalse((output/'reward.txt').exists())
            with contextlib.redirect_stdout(io.StringIO()):
                client.finish_failure(output, value, accepted, source, True)
            self.assertEqual((output/'reward.txt').read_text(), '-1\n')
            self.assertFalse((output/'result.json').exists())
            self.assertNotIn('private detail', (output/'failure.json').read_text())

    def test_infrastructure_failure_keeps_classification_without_a_score(self):
        accepted = {'job_id':'a'*32,'scope':'track-a'}
        source = 'b'*64
        value = {**accepted,'status':'FAILED','method_sha256':source,
                 'failure_category':'INFRASTRUCTURE_STORAGE'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                client.finish_failure(output, value, accepted, source, True)
            self.assertFalse((output/'reward.txt').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['failure_category'],
                             'INFRASTRUCTURE_STORAGE')

    def test_one_socket_cannot_spawn_unbounded_ssh_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            sock = str(Path(directory) / 'limited.sock')
            release = threading.Event()
            entered = threading.Event()
            count_lock = threading.Lock()
            counts = []

            def blocked_gateway(argv, **kwargs):
                with count_lock:
                    counts.append(1)
                    if len(counts) == 2:
                        entered.set()
                release.wait(timeout=5)
                return type('Completed', (), {'stdout': b'{"status":"OK"}', 'returncode': 0})()

            configuration = dict(ssh_key_file='placeholder', known_hosts_file='placeholder',
                                 ssh_port=1, ssh_target='placeholder')
            server = service.UnixHTTPServer(sock, service.make_handler(
                configuration, {'scope': 'limited-track', 'mode': 'calibration'}))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            replies = []
            workers = []
            try:
                with patch.object(service.subprocess, 'run', side_effect=blocked_gateway):
                    for _ in range(2):
                        worker = threading.Thread(target=lambda: replies.append(
                            client.request(sock, 'GET', '/health')))
                        workers.append(worker)
                        worker.start()
                    self.assertTrue(entered.wait(timeout=3))
                    with self.assertRaises((http.client.RemoteDisconnected, ConnectionResetError)):
                        client.request(sock, 'GET', '/health')
                    self.assertEqual(len(counts), 2)
                    release.set()
                    for worker in workers:
                        worker.join(timeout=3)
                    self.assertEqual(len(replies), 2)
            finally:
                release.set()
                server.shutdown()
                server.server_close()
                thread.join()

    def test_socket_role_owns_scope_and_rejects_override(self):
        with tempfile.TemporaryDirectory() as directory:
            sock = str(Path(directory) / 'public.sock')
            captures = []

            def fake_run(argv, **kwargs):
                request = json.loads(kwargs['input'])
                captures.append(request)
                if request['operation'] == 'status':
                    reply = {'status': 'REJECTED'}
                    code = 2
                else:
                    reply = {'status': 'ACCEPTED', 'job_id': 'a' * 32, 'scope': request['scope']}
                    code = 0
                return type('Completed', (), {'stdout': json.dumps(reply).encode(), 'returncode': code})()

            configuration = dict(ssh_key_file='private-placeholder', known_hosts_file='public-placeholder',
                                 ssh_port=1, ssh_target='placeholder')
            handler = service.make_handler(configuration, {'scope': 'track-a', 'mode': 'calibration'})
            server = service.UnixHTTPServer(sock, handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.object(service.subprocess, 'run', side_effect=fake_run):
                    with self.assertRaises(RuntimeError):
                        client.request(sock, 'POST', '/jobs', {'source': 'x', 'mode': 'formal'})
                    self.assertEqual(captures, [])
                    receipt = client.request(sock, 'POST', '/jobs', {'source': 'x'})
                    self.assertEqual(receipt['scope'], 'track-a')
                    self.assertEqual(captures[-1], dict(operation='submit', scope='track-a',
                                                      mode='calibration', source='x'))
                    with self.assertRaises(RuntimeError):
                        client.request(sock, 'GET', '/jobs/' + 'b' * 32)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_fresh_result_binds_method_role_and_complete_metric(self):
        accepted = {'job_id': 'a' * 32, 'scope': 'track-a'}
        sha = hashlib.sha256(b'actual source').hexdigest()
        receipt = dict(**accepted, status='SUCCESS', returncode=0, method_sha256=sha,
                       run_id='run-' + 'b' * 32, protocol='dd-selfproto-v1.3',
                       sample_count=50000, kind='FORMAL', checkpoint_sha256='c' * 64,
                       sample_manifest_sha256='d' * 64, frozen_manifest_sha256='e' * 64,
                       metric=9.2, reward=1.25)
        self.assertEqual(client.checked_receipt(receipt, accepted, sha, True)['reward'], 1.25)
        for field, bad in [('method_sha256', 'f' * 64), ('scope', 'track-b'),
                           ('job_id', 'c' * 32), ('kind', 'CALIBRATION'),
                           ('sample_count', 64), ('metric', float('nan')),
                           ('reward', float('inf')), ('checkpoint_sha256', None),
                           ('status', 'FAILED')]:
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                client.checked_receipt({**receipt, field: bad}, accepted, sha, True)

    def test_dead_worker_fails_and_other_scope_cannot_read_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = 'a' * 32
            job_dir = root / job
            job_dir.mkdir()
            (job_dir / 'request.json').write_text(json.dumps(dict(scope='track-a', method_sha256='b' * 64)))
            (job_dir / 'process.json').write_text(json.dumps(dict(pid=2147483647, start_ticks=1)))
            with patch.object(gateway, 'jobs_root', return_value=root):
                with self.assertRaises(ValueError):
                    gateway.status(dict(scope='track-b', job_id=job))
                result = gateway.status(dict(scope='track-a', job_id=job))
                self.assertEqual(result['status'], 'FAILED')
                self.assertEqual(result['error_type'], 'WorkerExited')
                self.assertTrue((job_dir / 'response.json').is_file())

    def test_image_client_copies_match_reviewed_source(self):
        root = HERE.parent / 'workspace/harbor_task'
        original = (HERE / 'runtime_client.py').read_bytes()
        for path in [root / 'environment/public_eval/runtime_client.py', root / 'tests/runtime_client.py']:
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
