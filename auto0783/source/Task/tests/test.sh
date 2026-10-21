#!/bin/sh
set -eu
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/reward.pending.txt /logs/verifier/grader-error.json
exec python /tests/grade.py
