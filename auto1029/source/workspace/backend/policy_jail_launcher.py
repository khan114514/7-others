"""Trusted root entry, never imports candidate code outside its private jail."""
import ctypes
import os
import resource
import sys
from pathlib import Path

jail = Path(sys.argv[1]).resolve()
jail.relative_to(Path(sys.argv[2]) / 'policy-jails')
resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
# Process creation is blocked by seccomp before candidate import. RLIMIT_NPROC
# before setuid/exec would couple this run to unrelated processes using nobody.
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.chroot(jail)
os.chdir('/workspace/solution')
os.setgroups([])
os.setgid(65534)
os.setuid(65534)
if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):
    raise SystemExit('no_new_privs failed')
os.umask(0o077)
os.execve(sys.argv[3], ['python3.12', '-I', '-S', '-B', '/runner.py'],
          {'PATH': '/opt/conda/bin', 'HOME': '/tmp', 'LANG': 'C.UTF-8', 'TMPDIR': '/tmp'})
