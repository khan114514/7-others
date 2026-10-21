"""CPU-only actual portable jail probes; no model, weights or GPU imported."""
import hashlib
import json
import os
import time
from pathlib import Path
from deployment_paths import PRIVATE_ROOT
from isolated_policy import PolicyExecutor

STATE = {'question': 'Synthetic public probe', 'generated_token_ids': [], 'decoded_text': '',
         'remaining_tokens': 128, 'maximum_draft_tokens': 32}
checks = []
marker = PRIVATE_ROOT / 'host-only-probe.txt'
marker.write_text('Synthetic private host visibility witness\n')
marker.chmod(0o600)
assert marker.is_file()


def check(name, body, expected):
    source = 'def draft(state):\n' + '\n'.join(' '+line for line in body.splitlines()) + '\n'
    with PolicyExecutor({'method.py': source}) as policy:
        value = policy.draft(STATE)
    checks.append({'name': name, 'passed': value == expected})


check('dropped_uid_gid_groups', 'import os\nreturn [os.getuid(),os.getgid(),len(os.getgroups())]', [65534, 65534, 0])
check('host_marker_unreadable', 'from pathlib import Path\nreturn [int(Path('+repr(str(PRIVATE_ROOT / 'host-only-probe.txt'))+').exists())]', [0])
assert marker.is_file()
check('no_host_proc', 'from pathlib import Path\nreturn [int(Path("/proc/self/environ").exists())]', [0])
check('no_torch_site_packages', 'try:\n import torch\n return [1]\nexcept ImportError:\n return [0]', [0])
check('no_network_socket', 'import socket\ntry:\n socket.socket()\n return [1]\nexcept PermissionError:\n return [0]', [0])
check('no_process_fork', 'import os\ntry:\n os.fork()\n return [1]\nexcept PermissionError:\n return [0]', [0])
check('no_parent_signals', 'import os\ntry:\n os.kill(os.getppid(),0)\n return [1]\nexcept PermissionError:\n return [0]', [0])
check('readonly_tmp', 'from pathlib import Path\ntry:\n Path("/tmp/forbidden-write").write_text("probe")\n return [1]\nexcept PermissionError:\n return [0]', [0])
try:
    with PolicyExecutor({'method.py': 'def draft(state):\n return [0]*33\n'}) as policy:
        policy.draft(STATE)
    bounded = False
except ValueError:
    bounded = True
checks.append({'name': 'proposal_length_enforced', 'passed': bounded})
try:
    with PolicyExecutor({'method.py': 'def draft(state):\n while True: pass\n'}) as policy:
        policy.draft(STATE)
    bounded = False
except TimeoutError:
    bounded = True
checks.append({'name': 'request_timeout_enforced', 'passed': bounded})
with PolicyExecutor({'method.py': 'def draft(state):\n return []\n'}) as policy:
    start = time.perf_counter()
    for _ in range(200):
        assert policy.draft(STATE) == []
    mean = (time.perf_counter() - start) / 200
result = {'phase': 'ACTUAL_PORTABLE_POLICY_ISOLATION_PASS' if all(c['passed'] for c in checks) else 'FAILED',
          'checks': checks, 'rpc_count': 200, 'mean_rpc_seconds': mean,
          'gpu_used': False, 'model_research_seconds': 0, 'candidate_imported_by_host': False,
          'source_hashes': {n: hashlib.sha256((PRIVATE_ROOT / n).read_bytes()).hexdigest()
                            for n in ['isolated_policy.py', 'policy_jail_launcher.py', 'policy_jail_runner.py', 'deployment_paths.py']},
          'completed_epoch': time.time()}
(PRIVATE_ROOT / 'portable-isolation-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['phase'] == 'ACTUAL_PORTABLE_POLICY_ISOLATION_PASS' else 1)
