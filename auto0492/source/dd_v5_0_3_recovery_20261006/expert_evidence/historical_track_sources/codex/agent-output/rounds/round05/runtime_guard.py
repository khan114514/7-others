"""Local execution guard; never changes the shared repository or assets."""
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path

TRACK = Path('historical-codex-workspace')
WORK = Path('historical-codex-models')
REPO = Path('historical-resources/autoresearch_tasks/repos/distilled_decoding')


def apply_guard():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS')
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock unavailable; refusing unguarded execution')
    # ABI 1 file access rights; read, directory traversal, and executable loads.
    handled = (1 << 13) - 1
    readonly = (1 << 0) | (1 << 2) | (1 << 3)

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    attr = Ruleset(handled)
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    read_roots = ['/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc', '/sys',
                  '/proc', '/opt/conda', str(REPO),
                  'historical-resources/venvs/dd1', 'historical-resources/models/dd1',
                  'historical-runtime-cache/torch/hub/checkpoints', str(TRACK / 'harbor_task'),
                  str(TRACK / 'runtime_guard.py'), str(TRACK / 'official_entry.py')]
    write_roots = [str(WORK), str(TRACK / 'agent-output'), '/dev']
    try:
        rules = [(p, readonly) for p in read_roots] + [(p, handled) for p in write_roots]
        # CUDA names its worker threads through this process's proc task/comm.
        # This grants writes only to our own task subtree, never another PID.
        rules.append(('/proc/self/task', (1 << 1) | (1 << 2) | (1 << 3)))
        for path, rights in rules:
            if not os.path.exists(path):
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not os.path.isdir(path):
                    rights &= (1 << 0) | (1 << 1) | (1 << 2)
                rule = Beneath(rights, parent)
                if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'landlock_add_rule', path)
            finally:
                os.close(parent)
        if libc.syscall(446, ruleset, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(ruleset)

    # Deny IPv4/IPv6 sockets in the process and all descendants; allow local
    # Unix sockets needed by torch/data-loader multiprocessing.
    sc = ctypes.CDLL(ctypes.util.find_library('seccomp'))
    sc.seccomp_init.argtypes = [ctypes.c_uint32]
    sc.seccomp_init.restype = ctypes.c_void_p
    sc.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                        ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    sc.seccomp_load.argtypes = [ctypes.c_void_p]
    sc.seccomp_release.argtypes = [ctypes.c_void_p]

    class Compare(ctypes.Structure):
        _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                    ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]

    ctx = sc.seccomp_init(0x7fff0000)
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        syscall = sc.seccomp_syscall_resolve_name(b'socket')
        for domain in (2, 10):
            compare = Compare(0, 4, domain, 0)
            if sc.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM,
                                         syscall, 1, ctypes.byref(compare)):
                raise RuntimeError('seccomp socket rule failed')
        if sc.seccomp_load(ctx):
            raise RuntimeError('seccomp_load failed')
    finally:
        sc.seccomp_release(ctx)
    return {'landlock_abi': abi, 'filesystem_enforced': True,
            'ipv4_ipv6_denied': True, 'no_new_privileges': True}
