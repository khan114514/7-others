"""Agent-visible fixed development evaluator; no network or credentials."""
import argparse
import json
import math
import time
import uuid
from pathlib import Path
from queue_client import request_json



def completed_reply(value):
    epoch = value.get('completed_epoch')
    if (type(value.get('exit_code')) is not int or type(epoch) not in (int, float)
            or not math.isfinite(epoch)):
        return False
    return (value.get('phase') == 'FAILED' or
            (value.get('phase') == 'COMPLETE' and isinstance(value.get('result'), dict)))

ROOT = Path('/workspace/solution')
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='seed17 only, development feedback without a full formal score')
parser.add_argument('--status', help='query one of this lane own previously saved job IDs')
parser.add_argument('--wait', action='store_true', help='wait for own job; real GPU queue time is excluded by trusted accounting')
args = parser.parse_args()
if args.status:
    job = args.status
    value = request_json('evaluate', {'action': 'status', 'job_id': job}, timeout=60)
else:
    job = uuid.uuid4().hex
    files = {}
    # Only executable method and supporting code/data. Research histories and
    # prior evaluation output are not transmitted to the GPU worker.
    for path in ROOT.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix == '.py' and path.stat().st_size <= 1024 * 1024:
            files[str(path.relative_to(ROOT))] = path.read_text()
    if 'method.py' not in files:
        raise SystemExit('method.py is required')
    payload = {'action': 'submit', 'job_id': job, 'files': files,
               'seeds': [17] if args.quick else [17, 29, 43, 71, 101]}
    ticket = ROOT / ('evaluation-' + job + '.ticket.json')
    ticket.write_text(json.dumps({'job_id': job, 'seeds': payload['seeds']}) + '\n')
    value = request_json('evaluate', payload, timeout=60)
    print(json.dumps({'job_id': job, 'phase': value['phase'], 'ticket': str(ticket)}), flush=True)
if args.wait:
    while not completed_reply(value):
        time.sleep(5)
        value = request_json('evaluate', {'action': 'status', 'job_id': job}, timeout=60)
result = ROOT / ('evaluation-' + job + '.json')
result.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(value, ensure_ascii=False), flush=True)
if value['phase'] == 'FAILED':
    raise SystemExit(1)
