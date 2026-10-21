"""Verify the prepared artifacts without training or evaluation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


root = Path('/workspace/agent-output/round-02')
plan = json.loads((root / 'plan.json').read_text())
result = json.loads((root / 'cpu_precheck_result.json').read_text())
failures = json.loads((root / 'cli-failures.json').read_text())
source = Path('/workspace/solution/method.py').read_bytes()
snapshot = (root / 'method.py').read_bytes()
sha = hashlib.sha256(source).hexdigest()
assert source == snapshot
assert plan['round'] == 2
assert plan['status'] == 'PREPARED_NOT_EVALUATED'
assert plan['score'] is None and plan['failure_reason'] is None
assert plan['retained_best'] is False
assert datetime.fromisoformat(plan['time'].replace('Z', '+00:00')) <= datetime.now(timezone.utc)
assert result['status'] == 'PASS'
assert len(result['checks']) == 7
assert all(check['status'] == 'PASS' for check in result['checks'])
assert result['method_sha256'] == sha
assert plan['source_verification']['solution_sha256'] == sha
assert plan['source_verification']['snapshot_sha256'] == sha
assert plan['source_verification']['cpu_record_method_sha256'] == sha
for name, digest in plan['cpu_precheck']['artifact_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
assert failures['public_evaluation_cli_invocations'] == 0
assert failures['cpu_precheck_cli']['exit_code'] == 0
print(json.dumps({'audit': 'PASS', 'round': 2, 'method_sha256': sha,
                  'cpu_checks_passed': 7,
                  'cli_failures_recorded': len(failures['failures']),
                  'evaluation_submitted': False, 'score': None,
                  'time': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')},
                 allow_nan=False))
