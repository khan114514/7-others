"""Public Dev CLI. Only data-only tensor graphs cross the trusted queue."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from queue_client import read_regular, request_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--program', default='/workspace/solution/program.json')
    parser.add_argument('--output', default='/workspace/solution/dev-result.json')
    args = parser.parse_args()
    program = json.loads(read_regular(Path(args.program), 512 * 1024))
    canonical = json.dumps(program, sort_keys=True, separators=(',', ':')).encode()
    value = request_json('evaluate', {'program': program}, timeout=10800)
    if value.get('artifact_sha256') != hashlib.sha256(canonical).hexdigest():
        raise ValueError('Dev result is bound to a different program')
    if value.get('execution_status') == 'COMPLETED' and (value.get('profile') != 'dev' or value.get('is_formal_protocol')):
        raise ValueError('Dev must use the public evaluation profile')
    reward = value.get('reward')
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError('Missing finite Dev reward')
    output = Path(args.output)
    pending = output.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(output)
    print(json.dumps(value, indent=2))
    return 0 if value.get('execution_status') == 'COMPLETED' else 1


if __name__ == '__main__':
    sys.exit(main())
