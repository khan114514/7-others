#!/bin/bash
set -euo pipefail
exec python3 /workspace/environment/public_eval/runtime_client.py --method /workspace/solution/method.py "$@"
