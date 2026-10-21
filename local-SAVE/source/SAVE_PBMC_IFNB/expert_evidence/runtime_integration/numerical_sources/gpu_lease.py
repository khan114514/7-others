"""Run one expert job under existing shared GPU locks; never delete locks."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", action="append", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A job command is required.")
    if args.receipt.exists():
        parser.error("Receipt exists; choose an unused path.")
    held = []
    started = time.monotonic()
    receipt = {"pid": os.getpid(), "gpu_uuid_expected": args.gpu_uuid, "locks": args.lock, "start_time": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), "status": "PREFLIGHT", "returncode": None}
    code = 75
    try:
        for name in args.lock:
            handle = Path(name).open("r")
            held.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip().split(",")
        if observed[0].strip() != args.gpu_uuid or int(observed[1]) > 64:
            raise RuntimeError("GPU identity mismatch or occupied outside shared locks.")
        receipt["status"] = "RUNNING"
        code = subprocess.call(command)
        receipt["returncode"] = code
        receipt["status"] = "COMPLETED" if code == 0 else "JOB_FAILED"
    except (BlockingIOError, FileNotFoundError, RuntimeError) as exc:
        receipt["status"] = "RESOURCE_NOT_AVAILABLE"
        receipt["reason"] = str(exc)
        print(json.dumps({"status": receipt["status"], "reason": receipt["reason"]}), flush=True)
    finally:
        for handle in held:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["locks_released"] = True
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
