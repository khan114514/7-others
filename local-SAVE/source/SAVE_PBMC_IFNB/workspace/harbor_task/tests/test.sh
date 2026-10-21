#!/bin/bash
set -euo pipefail
exec /opt/runtime/bin/python /tests/grader.py --split hidden --method /workspace/solution/method.py --output /logs/verifier/result.json --reward-dir /logs/verifier
