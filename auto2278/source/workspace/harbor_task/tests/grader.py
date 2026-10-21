"""Trusted public benchmark; Hidden uses an independent copy and private protocol."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


class Invalid(Exception):
    def __init__(self, kind, message):
        self.kind, self.message = kind, message


def check_source(path):
    if path.is_symlink() or not path.is_file():
        raise Invalid("format_error", "Submit a regular method.py file.")
    if path.stat().st_size > 262144:
        raise Invalid("hard_gate", "method.py exceeds 256 KiB.")
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeError) as e:
        raise Invalid("format_error", str(e))
    banned = {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
              "globals", "locals", "vars", "dir", "input", "breakpoint", "memoryview"}
    torch_names = {"torch"}
    safe_context_names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            torch_names.update(a.asname or a.name for a in node.names if a.name == "torch")
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            safe_context_names.update(a.asname or a.name for a in node.names if a.name in {"no_grad", "inference_mode"})
    bad_attrs = {"load", "save", "from_file", "frombuffer", "set_num_threads", "set_num_interop_threads",
                 "set_default_device", "set_default_dtype", "set_grad_enabled", "set_flush_denormal",
                 "set_num_interop_threads", "set_rng_state", "manual_seed", "seed", "initial_seed",
                 "ops", "classes", "distributed", "multiprocessing", "hub", "utils", "jit", "compile",
                 "serialization", "cuda", "mps", "xpu", "profiler", "benchmark", "random", "backends",
                 "as_strided", "storage", "untyped_storage", "data_ptr"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise Invalid("hard_gate", "Persistent state and class-based caches are forbidden.")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Attribute) for t in targets):
                raise Invalid("hard_gate", "Modifying module or function attributes is forbidden.")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import)
                       else [(node.module or "").split(".")[0]])
            if any(m not in {"torch", "math"} for m in modules) or getattr(node, "level", 0):
                raise Invalid("hard_gate", "Imports are limited to torch and math.")
            imported = [(node.module or "")] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if any(part in bad_attrs for name in imported for part in name.split(".")):
                raise Invalid("hard_gate", "This PyTorch subsystem is outside the allowed computation surface.")
            if isinstance(node, ast.ImportFrom) and any(a.name in bad_attrs or a.name == "*" or a.name.startswith("_") for a in node.names):
                raise Invalid("hard_gate", "This imported function is forbidden.")
        if isinstance(node, ast.Name) and (node.id in banned or node.id.startswith("__")):
            raise Invalid("hard_gate", "Filesystem, reflection and dynamic execution are forbidden.")
        if isinstance(node, ast.Attribute) and (node.attr in bad_attrs or node.attr.startswith("_")):
            raise Invalid("hard_gate", "Candidate must use ordinary CPU tensor computation only.")
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "attention" for n in tree.body):
        raise Invalid("format_error", "Define attention(x, wq, wk, wv, alpha, beta).")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                raise Invalid("hard_gate", "Module scope may contain only imports, functions and literal constants.")
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and not isinstance(node.value, ast.Constant):
            raise Invalid("hard_gate", "Module-level mutable state is forbidden.")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defaults = node.args.defaults + [v for v in node.args.kw_defaults if v is not None]
            if any(not isinstance(v, ast.Constant) for v in defaults):
                raise Invalid("hard_gate", "Mutable function-default state is forbidden.")
            for decorator in node.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                safe = ((isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                         and func.value.id in torch_names and func.attr in {"no_grad", "inference_mode"})
                        or (isinstance(func, ast.Name) and func.id in safe_context_names))
                if not safe:
                    raise Invalid("hard_gate", "Custom stateful decorators are forbidden; use ordinary pure helpers.")


def generate(case, seed):
    g = torch.Generator().manual_seed(seed)
    b, l, c, h, d = (case[k] for k in ("batch", "length", "channels", "heads", "dim"))
    x = torch.randn(b, l, c, generator=g) * case.get("input_scale", 1.)
    if case.get("degenerate") == "zero":
        x.zero_()
    weights = [torch.randn(h, d, c, generator=g) / math.sqrt(c) for _ in range(3)]
    return x, *weights, case["alpha"], case["beta"]


def oracle(values):
    x, wq, wk, wv, alpha, beta = values
    q = torch.einsum("blc,hdc->bhld", x, wq)
    k = torch.einsum("blc,hdc->bhld", x, wk)
    logits = q @ k.transpose(-1, -2) / math.sqrt(wq.shape[1])
    p0 = logits.softmax(-1)
    p1 = (.5 * (1 + alpha) * logits + .5 * (1 - alpha) * logits.transpose(-1, -2)).softmax(-1)
    base, modified = p0 @ x.unsqueeze(1), p1 @ x.unsqueeze(1)
    mix = base + beta * (modified - base)
    ratio = (modified.norm(dim=-1, keepdim=True).clamp_min(1e-6)
             / mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(.25, 4.)
    return torch.einsum("bhlc,hdc->bhld", mix * ratio, wv).transpose(1, 2).flatten(2).contiguous()


def worker_limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CPU, (110, 110))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(1002)
        os.setuid(1002)


def run_case(source, values, protocol, scratch):
    input_path = scratch / "input.npz"
    output_path = scratch / "output.npz"
    # The prior input is read-only. Replace it instead of attempting to reopen it
    # for writing when public grading runs as the unprivileged Agent user.
    input_path.unlink(missing_ok=True)
    arrays = {name: t.numpy() for name, t in zip(("x", "wq", "wk", "wv"), values[:4])}
    np.savez(input_path, **arrays, alpha=values[4], beta=values[5],
             warmup=protocol["warmup"], repeats=protocol["timed_repeats"])
    input_path.chmod(0o444)
    output_path.unlink(missing_ok=True)
    try:
        p = subprocess.run([sys.executable, "-I", str(ROOT / "worker.py"), str(source),
                            str(input_path), str(output_path)], cwd=scratch,
                           env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                                "HOME": str(scratch), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                           preexec_fn=worker_limits, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise Invalid("timeout", "One case exceeded 120 seconds.")
    if p.returncode != 0:
        for line in p.stdout.splitlines():
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if report.get("worker_failure") in {"hard_gate", "format_error"}:
                raise Invalid(report["worker_failure"], report["message"])
        kind = "resource_error" if p.returncode < 0 else "candidate_error"
        raise Invalid(kind, p.stderr[-2000:])
    if not output_path.is_file() or output_path.is_symlink() or output_path.stat().st_size > 256 * 1024 * 1024:
        raise Invalid("format_error", "Worker did not produce a valid output array.")
    data = np.load(output_path, allow_pickle=False)
    if not bool(data["input_unchanged"]):
        raise Invalid("hard_gate", "Candidate modified input tensors.")
    actual = torch.from_numpy(data["output"].copy())
    expected = oracle(values)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.isfinite(actual).all():
        raise Invalid("hard_gate", "Output shape, dtype or finiteness failed.")
    if not torch.allclose(actual, expected, **protocol["quality"]):
        raise Invalid("hard_gate", f"Numerical correctness failed; max abs error={float((actual-expected).abs().max()):.8g}.")
    samples = data["samples"].tolist()
    if len(samples) != protocol["timed_repeats"] or not all(math.isfinite(t) and t > 0 for t in samples):
        raise Invalid("format_error", "Invalid timing samples.")
    return {"latency_ms": statistics.median(samples), "samples_ms": samples,
            "wall_latency_ms": statistics.median(data["wall_samples"].tolist()),
            "wall_samples_ms": data["wall_samples"].tolist(),
            "max_abs_error": float((actual - expected).abs().max())}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=Path, default=Path("/workspace/solution/method.py"))
    p.add_argument("--protocol", type=Path, default=Path("/workspace/environment/public_assets/dev/protocol.json"))
    p.add_argument("--output", type=Path, default=Path("/workspace/public-results/result.json"))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--raw-only", action="store_true", help="Expert calibration only; no normalized score.")
    args = p.parse_args()
    args.output.unlink(missing_ok=True)
    started = time.time()
    try:
        if args.method == Path("/workspace/solution/method.py"):
            frozen = ROOT / "solve.sh"
            submitted = args.method.parent / "solve.sh"
            if submitted.is_symlink() or not submitted.is_file() or submitted.read_bytes() != frozen.read_bytes():
                raise Invalid("hard_gate", "Restore the frozen solve.sh before submitting.")
        check_source(args.method)
        protocol = json.loads(args.protocol.read_text())
        torch.set_num_threads(protocol["torch_threads"])
        torch.set_num_interop_threads(1)
        records = []
        with tempfile.TemporaryDirectory(prefix="attention-worker-") as temp:
            scratch = Path(temp)
            scratch.chmod(0o755)
            source = scratch / "method.py"
            shutil.copyfile(args.method, source)
            source.chmod(0o444)
            outdir = scratch / "out"
            outdir.mkdir()
            if os.getuid() == 0:
                os.chown(outdir, 1002, 1002)
            with torch.inference_mode():
                for seed in args.seeds or protocol["formal_seeds"]:
                    per_case = []
                    for i, case in enumerate(protocol["cases"]):
                        values = generate(case, seed * 1009 + i)
                        # Both measurements run the same protected launcher and budgets.
                        # A fresh healthy control removes CPU architecture and speed as a score advantage.
                        control = ROOT / "baseline_method.py"
                        if seed % 2:
                            baseline_record = run_case(control, values, protocol, outdir)
                            record = run_case(source, values, protocol, outdir)
                        else:
                            record = run_case(source, values, protocol, outdir)
                            baseline_record = run_case(control, values, protocol, outdir)
                        record["control_latency_ms"] = baseline_record["latency_ms"]
                        record["control_samples_ms"] = baseline_record["samples_ms"]
                        record["relative_cpu_time"] = record["latency_ms"] / baseline_record["latency_ms"]
                        record["case_index"] = i
                        per_case.append(record)
                    records.append({"seed": seed, "latency_ms": statistics.mean(c["latency_ms"] for c in per_case),
                                    "relative_cpu_time": statistics.mean(c["relative_cpu_time"] for c in per_case), "cases": per_case})
        latency = statistics.mean(r["relative_cpu_time"] for r in records)
        anchor = protocol["baseline_ms"]
        if not args.raw_only and (anchor is None or anchor <= protocol["upper_bound_ms"]):
            raise Invalid("infrastructure_error", "Scoring anchors have not been calibrated.")
        score = None if args.raw_only else (anchor - latency) / (anchor - protocol["upper_bound_ms"])
        result = {"status": "ok", "metric": "relative_process_cpu_time", "direction": "minimize", "raw_value": latency,
                  "mean_candidate_cpu_ms": statistics.mean(r["latency_ms"] for r in records),
                  "score": score, "records": records, "quality_gates_passed": True,
                  "method_sha256": hashlib.sha256(args.method.read_bytes()).hexdigest(),
                  "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  "protocol": protocol["protocol"], "torch": torch.__version__,
                  "start_unix": started, "end_unix": time.time(), "training": "NOT_APPLICABLE"}
        atomic_json(args.output, result)
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    except Invalid as e:
        result = {"status": e.kind, "failure_reason": e.message, "score": -1 if e.kind == "hard_gate" else None,
                  "start_unix": started, "end_unix": time.time()}
        atomic_json(args.output, result)
        print(json.dumps(result), flush=True)
        return 0 if e.kind == "hard_gate" else 1
    except Exception as e:
        result = {"status": "infrastructure_error", "failure_reason": f"{type(e).__name__}: {e}", "score": None,
                  "start_unix": started, "end_unix": time.time()}
        atomic_json(args.output, result)
        print(json.dumps(result), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
