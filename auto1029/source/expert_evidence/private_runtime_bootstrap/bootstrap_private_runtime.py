"""Trusted operator runtime-directory bootstrap; no GPU, model or candidate imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

PINS = {
    "deployment_paths.py": "a4e2a0a7eeb810b5bc37b5c95ad3ed49d03f2d55f8c4dced9416e1a3bb69ec27",
    "trusted_private_backend_support.py": "4d0697104f22e5331d6a85bea43359927156b57e63c3ad63d86adef7af1f988f",
}
ISOLATION_CHECKS = {
    "dropped_uid_gid_groups", "host_marker_unreadable", "no_host_proc",
    "no_torch_site_packages", "no_network_socket", "no_process_fork",
    "no_parent_signals", "readonly_tmp", "proposal_length_enforced",
    "request_timeout_enforced",
}

def require_root():
    if os.geteuid() != 0:
        raise PermissionError("Trusted root operator required")

def regular(path, limit=65536):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("Operator inputs must not traverse symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022 or info.st_size > limit:
            raise ValueError("Bounded root-owned immutable operator input required")
        raw = b""
        while len(raw) <= limit:
            block = os.read(fd, min(65536, limit + 1 - len(raw)))
            if not block:
                break
            raw += block
        final = os.fstat(fd)
        if len(raw) != info.st_size or (final.st_mtime_ns, final.st_size) != (info.st_mtime_ns, info.st_size):
            raise ValueError("Operator input changed during read")
        return raw
    finally:
        os.close(fd)

def directory_info(fd, private=False):
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise PermissionError("Trusted root-owned directory required")
    if private and stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError("Private runtime directory must be exactly mode0700")

def bootstrap_directory(root, create):
    """Descriptor-relative operation; never repairs or deletes existing state."""
    require_root()
    root = Path(root).absolute()
    if root.resolve(strict=True) != root:
        raise ValueError("Configured deployment must not traverse symlinks")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root, flags)
    try:
        directory_info(root_fd)
        private_fd = os.open("private", flags, dir_fd=root_fd)
        try:
            directory_info(private_fd, private=True)
            created = False
            if create:
                try:
                    os.mkdir("hidden-validation-jobs", mode=0o700, dir_fd=private_fd)
                    created = True
                except FileExistsError:
                    pass
            runtime_fd = os.open("hidden-validation-jobs", flags, dir_fd=private_fd)
            try:
                directory_info(runtime_fd, private=True)
            finally:
                os.close(runtime_fd)
            return {"relative_path": "private/hidden-validation-jobs", "created": created,
                    "owner_uid": 0, "mode": "0700", "existing_state_deleted_or_repaired": False}
        finally:
            os.close(private_fd)
    finally:
        os.close(root_fd)

def main():
    require_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raw = regular(args.config)
    if stat.S_IMODE(args.config.stat().st_mode) != 0o600:
        raise PermissionError("Deployment config must be mode0600")
    config = json.loads(raw)
    root = Path(config["project_root"])
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("Absolute exact deployment required")
    private = root / "private"
    for name, expected in PINS.items():
        if hashlib.sha256(regular(private / name, 1024 * 1024)).hexdigest() != expected:
            raise ValueError("Trusted guard source differs from this operator tool")
    os.environ["HEIMA_DEPLOYMENT_CONFIG"] = str(args.config.absolute())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(private))
    # Only exact pinned trusted guards are loaded. No policy or candidate module.
    import trusted_private_backend_support as support
    sources = support.verify_installed_sources(root)
    checked_root, checked_config = support.verify_backend(root)
    if checked_root != root or checked_config != config:
        raise ValueError("Installed backend and operator config disagree")
    injected = support.injection(root, root)
    if injected["private_rows"] != 1380:
        raise ValueError("Complete exact private corpus required")
    isolated = json.loads(regular(private / "portable-isolation-receipt.json"))
    checks = isolated["checks"]
    if (isolated["phase"] != "ACTUAL_PORTABLE_POLICY_ISOLATION_PASS" or
        len(checks) != len(ISOLATION_CHECKS) or
        {c["name"] for c in checks} != ISOLATION_CHECKS or
        not all(c["passed"] is True for c in checks) or
        isolated["candidate_imported_by_host"] is not False):
        raise ValueError("Actual complete isolated execution proof required")
    for name, expected in isolated["source_hashes"].items():
        if name not in {"isolated_policy.py", "policy_jail_launcher.py", "policy_jail_runner.py", "deployment_paths.py"}:
            raise ValueError("Unexpected isolation source")
        if hashlib.sha256(regular(private / name, 1024 * 1024)).hexdigest() != expected:
            raise ValueError("Isolation proof source differs from installed source")
    if set(isolated["source_hashes"]) != {"isolated_policy.py", "policy_jail_launcher.py", "policy_jail_runner.py", "deployment_paths.py"}:
        raise ValueError("Exact isolation source set required")
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise ValueError("At least2GiB deployment-volume reserve required")
    outcome = bootstrap_directory(root, create=not args.check_only)
    outcome.update(phase="OPERATOR_RUNTIME_DIRECTORY_READY_FULL_NATIVE_SCORE_PENDING",
                   installed_source_records=len(sources["code_and_frozen_inputs"]),
                   private_rows=injected["private_rows"], candidate_code_loaded=False,
                   gpu_jobs=0, model_requests=0, model_research_seconds=0,
                   full_native_score_claim=False)
    print(json.dumps(outcome))

if __name__ == "__main__":
    main()
