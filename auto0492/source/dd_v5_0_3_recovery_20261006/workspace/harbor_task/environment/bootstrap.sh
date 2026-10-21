#!/bin/bash
set -euo pipefail
mkdir -p /workspace/solution /workspace/agent-output
if [ ! -f /workspace/solution/method.py ]; then
  cp /workspace/environment/starter/method.py /workspace/solution/method.py
fi
exec "$@"
