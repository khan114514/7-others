#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/result.json /logs/verifier/invocation.json /logs/verifier/failure.json /logs/verifier/reward.txt.stage /logs/verifier/result.json.stage /logs/verifier/failure.json.stage
cleanup_failed_result() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        # Preserve invocation/log evidence, but never turn a failed execution
        # into the legitimate zero score of an unchanged Baseline.
        rm -f /logs/verifier/reward.txt /logs/verifier/reward.txt.stage /logs/verifier/result.json /logs/verifier/result.json.stage
    fi
    exit "$rc"
}
trap cleanup_failed_result EXIT
test -f /logs/artifacts/solution/method.py
test ! -L /logs/artifacts/solution/method.py
cp --remove-destination -- /logs/artifacts/solution/method.py /workspace/solution/method.py
python3 /tests/runtime_client.py --socket /run/dd/formal.sock --method /workspace/solution/method.py --out /logs/verifier --formal
