#!/bin/sh
set -eu
exec python /workspace/environment/public_eval/grader.py --method /workspace/solution/method.py --output /workspace/public-results/result.json "$@"
