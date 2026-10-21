"""Record the exit status of an expert command without shell interpolation."""
import json
import subprocess
import sys
import time
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
started = time.time()
exit_code = None
failure = None
try:
    process = subprocess.run(request["command"], timeout=request["timeout_seconds"])
    exit_code = process.returncode
except BaseException as error:
    failure = {"type": type(error).__name__, "message": str(error)}
receipt = {"started_epoch": started, "completed_epoch": time.time(), "command": request["command"], "exit_code": exit_code, "failure": failure, "formal_trajectory": False}
Path(request["receipt"]).write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(exit_code if exit_code is not None else 1)
