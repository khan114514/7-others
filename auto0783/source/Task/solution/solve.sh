#!/bin/sh
set -eu
cp /solution/reference_method.py /workspace/solution/method.py
cd /workspace/solution
python3 method.py
