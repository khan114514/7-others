"""Trusted stdlib runner, inside a chroot, unprivileged and seccomp filtered."""
import ctypes
import errno
import importlib.util
import json
import os
import sys

LIMIT = 262144
lib = ctypes.CDLL('/lib/x86_64-linux-gnu/libseccomp.so.2', use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
lib.seccomp_release.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x7fff0000)
if not ctx:
    raise SystemExit('seccomp initialization failed')
denied = '''socket socketpair connect bind listen accept accept4 sendto recvfrom sendmsg recvmsg
sendmmsg recvmmsg shutdown setsockopt getsockopt clone clone3 fork vfork execve execveat
ptrace process_vm_readv process_vm_writev kill tkill tgkill pidfd_open pidfd_getfd pidfd_send_signal
mount umount2 pivot_root chroot unshare setns open_by_handle_at name_to_handle_at
bpf perf_event_open keyctl add_key request_key io_uring_setup io_uring_enter io_uring_register
setuid setgid setreuid setregid setresuid setresgid setfsuid setfsgid setgroups capset
reboot kexec_load kexec_file_load init_module finit_module delete_module swapon swapoff
userfaultfd memfd_create mknod mknodat quotactl fanotify_init
'''.split()
for name in denied:
    number = lib.seccomp_syscall_resolve_name(name.encode())
    if number >= 0 and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0):
        raise SystemExit('seccomp rule failed')
if lib.seccomp_load(ctx):
    raise SystemExit('seccomp load failed')
lib.seccomp_release(ctx)

sys.path.insert(0, '/workspace/solution')
spec = importlib.util.spec_from_file_location('candidate_method', '/workspace/solution/method.py')
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)
if not callable(getattr(method, 'draft', None)):
    raise SystemExit('Missing draft(state)')
print(json.dumps({'ready': True, 'uid': os.getuid(), 'gid': os.getgid(), 'seccomp_loaded': True}), flush=True)
while True:
    raw = sys.stdin.buffer.readline(LIMIT + 1)
    if not raw:
        break
    if len(raw) > LIMIT:
        raise SystemExit('State frame too large')
    frame = json.loads(raw)
    result = method.draft(frame['state'])
    encoded = json.dumps({'id': frame['id'], 'proposal': result}, ensure_ascii=False).encode()
    if len(encoded) > LIMIT:
        raise SystemExit('Proposal frame too large')
    sys.stdout.buffer.write(encoded + b'\n')
    sys.stdout.buffer.flush()
