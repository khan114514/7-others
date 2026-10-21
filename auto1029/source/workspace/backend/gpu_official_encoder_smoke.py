"""One real image generation using the published encoder and official model code.

Uses the repository's generation procedure, CPU checkpoint staging and explicit
BF16 execution. This is a smoke test; it is not a formal benchmark or training.
"""
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from torchtune import training, utils
from torchtune.data import Message, padded_collate_tiled_images_and_mask
from torchtune.generation import sample
from torchtune.models.llama3_2_vision import lora_llama3_2_vision_11b, llama3_2_vision_transform
from torchtune.training import FullModelHFCheckpointer

from deployment_paths import PROJECT_ROOT as ROOT, GPU_LOCK_FILES, NVIDIA_SMI
RESULT = ROOT / "private/gpu-encoder-smoke.json"

def wait_for_idle():
    # Reuse the existing DD GPU mutex without truncating it. All extra GPU locks
    # already present in /tmp are also acquired non-blockingly in sorted order.
    paths = set(GPU_LOCK_FILES)
    paths.update(p for p in Path("/tmp").glob("*gpu*lock") if p.is_file())
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        held = []
        try:
            for path in sorted(paths):
                lock = path.open("a+")
                held.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # This host has a preload library that masks NVML memory reports.
            # Resource leasing must observe actual GPU memory without changing
            # the user's host environment or unrelated processes.
            monitor_environment = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
            observation = subprocess.check_output([str(NVIDIA_SMI), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, env=monitor_environment)
            if int(observation.strip().splitlines()[0]) < 256:
                return held
        except BlockingIOError:
            pass
        for lock in held:
            lock.close()
        time.sleep(10)
    raise TimeoutError("No idle A100 lease available within smoke preparation window")

