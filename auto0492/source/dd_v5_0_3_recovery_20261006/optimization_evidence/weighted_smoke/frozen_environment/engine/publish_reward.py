"""Publish only the fresh successful result of this verifier invocation."""
import json
import math
from pathlib import Path
from config import deployment, atomic_json

# Filled by the trusted verifier wrapper, not by the candidate workspace.
receipt = Path('/logs/verifier/invocation.json')
if not receipt.is_file():
    raise RuntimeError('No verifier invocation receipt')
run_id = json.loads(receipt.read_text())['run_id']
if not isinstance(run_id, str) or '/' in run_id or not run_id.startswith('run-'):
    raise ValueError('Invalid run identity')
result = json.loads((Path(deployment()['runs'])/run_id/'result.json').read_text())
if result['status'] != 'SUCCESS' or result['kind'] != 'FORMAL' or result['run_id'] != run_id:
    raise ValueError('The fresh formal run did not succeed')
reward = result['reward']
if type(reward) not in (int,float) or not math.isfinite(reward):
    raise ValueError('Invalid reward')
logs = Path('/logs/verifier')
atomic_json(logs/'result.json', result)
stage = logs/'reward.txt.stage'
stage.write_text(str(reward)+'\n')
stage.replace(logs/'reward.txt')
