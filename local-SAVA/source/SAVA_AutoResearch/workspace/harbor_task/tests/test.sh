#!/bin/sh
set -eu
cd /tests
exec python -I /tests/verifier.py
