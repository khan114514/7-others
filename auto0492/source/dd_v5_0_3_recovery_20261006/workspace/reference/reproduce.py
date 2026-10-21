"""Submit the selected source to an already configured author evaluation socket.

The server fixes the scope and mode. This wrapper passes source, not weights,
scores, cached samples, or a private evaluation seed.
"""
from pathlib import Path
import argparse, hashlib, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument('--socket', required=True)
p.add_argument('--out', required=True)
p.add_argument('--formal', action='store_true')
a = p.parse_args()
method = Path(__file__).with_name('method.py')
assert hashlib.sha256(method.read_bytes()).hexdigest() == '69db146f135ccc6ff9a6dc57f2aa99af51602993cddf2a75951851e64005798f'
client = Path(__file__).resolve().parents[2]/'author_tools/runtime_client.py'
command = [sys.executable, str(client), '--socket', a.socket,
           '--method', str(method), '--out', a.out]
if a.formal:
    command.append('--formal')
subprocess.run(command, check=True)
