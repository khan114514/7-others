"""Protected subprocess launcher; candidate receives tensors, never expected outputs."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
source, input_path, output_path = map(Path, sys.argv[1:4])
data = np.load(input_path, allow_pickle=False)
values = [torch.from_numpy(data[name].copy()) for name in ("x", "wq", "wk", "wv")]
values.extend([float(data["alpha"]), float(data["beta"])])
spec = importlib.util.spec_from_file_location("candidate_method", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
samples = []
wall_samples = []
with torch.inference_mode():
    for _ in range(int(data["warmup"])):
        module.attention(*values)
    for _ in range(int(data["repeats"])):
        start_wall = time.perf_counter_ns()
        start = time.process_time_ns()
        output = module.attention(*values)
        samples.append((time.process_time_ns() - start) / 1e6)
        wall_samples.append((time.perf_counter_ns() - start_wall) / 1e6)
    # A fresh call with fresh tensors also rejects stale cached outputs.
    output = module.attention(*[v.clone() if isinstance(v, torch.Tensor) else v for v in values])
input_unchanged = all(torch.equal(v, torch.from_numpy(data[name])) for name, v in zip(("x", "wq", "wk", "wv"), values[:4]))
if not isinstance(output, torch.Tensor):
    print(json.dumps({"worker_failure":"format_error","message":"attention must return a Tensor"}),flush=True)
    raise SystemExit(3)
if output.dtype != torch.float32 or output.device.type != "cpu":
    print(json.dumps({"worker_failure":"hard_gate","message":"output must remain float32 CPU"}),flush=True)
    raise SystemExit(3)
np.savez(output_path, output=output.detach().numpy(), samples=np.array(samples, dtype=np.float64),
         wall_samples=np.array(wall_samples, dtype=np.float64), input_unchanged=input_unchanged)
print(json.dumps({"median_ms": statistics.median(samples), "pid": os.getpid()}), flush=True)
