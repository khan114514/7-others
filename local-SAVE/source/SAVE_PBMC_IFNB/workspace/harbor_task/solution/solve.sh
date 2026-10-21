#!/bin/bash
set -euo pipefail
exec /opt/runtime/bin/python /tests/grader.py --split dev --method /workspace/solution/method.py --output /workspace/solution/dev_result.json
