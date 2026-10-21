#!/bin/sh
set -eu
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/result.json
python /tests/grader.py --method /workspace/solution/method.py --protocol /tests/protocol.json --output /logs/verifier/result.json
python - <<'PY'
import json, math
from pathlib import Path
p = Path('/logs/verifier')
d = json.loads((p/'result.json').read_text())
score = d['score']
assert isinstance(score, (int, float)) and math.isfinite(score)
tmp = p/'reward.txt.tmp'
tmp.write_text(str(score)+'\n')
tmp.replace(p/'reward.txt')
PY
