"""Trusted CLI wrapper: bounded turns resume; the campaign has no time shortcut."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

models = {'gpt': 'gpt-6.1-sol', 'seed': 'doubao-seed-2-1-turbo-260628'}
role = os.environ['LATENT_LANE']
prompt = Path(sys.argv[1]).read_text()
timeout = int(sys.argv[2])
argv = ['codex', 'exec', '--ephemeral', '--json', '--skip-git-repo-check', '--sandbox', 'workspace-write',
        '-m', models[role], '-c', 'model_reasoning_effort="high"',
        '-c', 'sandbox_workspace_write.writable_roots=["/run/latentcli/queue"]', prompt]
child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
try:
    code = child.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    os.killpg(child.pid, signal.SIGTERM)
    try: code = child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL); code = child.wait()
raise SystemExit(code if code >= 0 else 128 - code)
