"""Separate Verifier entrypoint; submitted Python is never imported or executed."""
import hashlib
import json
import math
from pathlib import Path
from queue_client import read_regular, request_json

LOG = Path('/logs/verifier')
LOG.mkdir(parents=True, exist_ok=True)


def write_reward(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Verifier reward must be a finite number')
    pending = LOG / 'reward.pending.txt'
    pending.write_text(format(value, '.17g') + '\n')
    pending.replace(LOG / 'reward.txt')


def main():
    for stale in ['reward.txt', 'reward.json', 'reward.pending.txt', 'grader-error.json']:
        (LOG / stale).unlink(missing_ok=True)
    # This image owns the exact same withheld inputs as the trusted executor.
    hidden = Path('/tests/hidden/hidden-decoder-inputs-v1.safetensors')
    input_sha = hashlib.sha256(hidden.read_bytes()).hexdigest()
    if input_sha != 'db9c6aab916da4b5914a96a642169fd122a67e8b15fbb16628ad90dc240e40e5':
        raise RuntimeError('Verifier-owned hidden inputs changed')
    program = json.loads(read_regular(Path('/workspace/solution/program.json'), 512 * 1024))
    canonical = json.dumps(program, sort_keys=True, separators=(',', ':')).encode()
    result = request_json('evaluate', {'program': program}, timeout=10800)
    (LOG / 'numerical-result.json').write_text(json.dumps(result, indent=2) + '\n')
    if result.get('artifact_sha256') != hashlib.sha256(canonical).hexdigest():
        raise RuntimeError('Result/program binding mismatch')
    if result.get('execution_status') != 'COMPLETED':
        raise RuntimeError('Trusted numerical execution failed')
    if (result.get('profile') != 'formal' or result.get('is_formal_protocol') is not True
            or result.get('input_sha256') != input_sha
            or result.get('layer_count') != 28 or result.get('steps') != 256
            or result.get('child_exit_code') != 0
            or result.get('parent_cuda_context_initialized') is not False):
        raise RuntimeError('Formal numerical protocol did not complete')
    value = result.get('reward')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RuntimeError('Invalid numerical reward')
    if result.get('quality_pass') is not True and value != -1:
        raise RuntimeError('Quality failures must score -1 and are excluded from valid ranking')
    write_reward(value)
    print(json.dumps({'status': 'COMPLETED', 'reward': value,
                      'run_id': result['run_id'], 'artifact_sha256': result['artifact_sha256'],
                      'quality_pass': result['quality_pass'], 'baseline_identity': result['baseline_identity']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Infrastructure errors remain visible and cannot masquerade as a
        # successfully evaluated zero-score submission in the Trial evidence.
        write_reward(-1.0)
        (LOG / 'grader-error.json').write_text(json.dumps({
            'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error)[:1000]}, indent=2) + '\n')
        raise
