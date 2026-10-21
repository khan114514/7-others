"""Linux kernel restrictions for the untrusted CPU-only method process."""
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path
import resource
import sys


def apply_guard(candidate, temporary):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    # torch reserves virtual ranges; cap address space separately from the
    # platform's physical-memory cgroup. No GPU device file is accessible.
    resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'no_new_privs')
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock unavailable; refusing candidate execution')

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    handled = (1 << 13) - 1
    read = 1 | 4 | 8
    attr = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    # Runtime libraries and the submitted source are readable. Frozen trainer,
    # models, scorer, run outputs and other tracks are absent from this list.
    roots = [(x, read) for x in ('/usr', '/lib', '/lib64', '/bin', '/etc', '/sys',
                                '/opt/conda', '/proc/self', str(Path(__file__).parent),
                                str(Path(candidate).parent))]
    roots += [(str(Path(os.__file__).resolve().parents[1]), read),
              (str(Path(os.sys.executable).resolve().parents[1]), read),
              (sys.prefix, read), (sys.base_prefix, read),
              (str(temporary), handled)]
    roots += [(x, 2 | 4) for x in ('/dev/null', '/dev/urandom', '/dev/random')]
    roots += [('/proc/self/task', 2 | 4 | 8)]
    roots += [(x, read) for x in ('/proc/cpuinfo', '/proc/meminfo')]
    try:
        for path, rights in roots:
            if not os.path.exists(path):
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not os.path.isdir(path):
                    rights &= 1 | 2 | 4
                rule = Beneath(rights, parent)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'landlock_add_rule', path)
            finally:
                os.close(parent)
        if libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
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
                raise RuntimeError('network restriction failed')
        for name in (b'ptrace', b'process_vm_readv', b'process_vm_writev',
                     b'mount', b'umount2', b'bpf', b'perf_event_open'):
            call = sc.seccomp_syscall_resolve_name(name)
            if call >= 0 and sc.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM, call, 0, None):
                raise RuntimeError('seccomp restriction failed')
        if sc.seccomp_load(ctx):
            raise RuntimeError('seccomp_load failed')
    finally:
        sc.seccomp_release(ctx)
    return {'landlock_abi': abi, 'filesystem_enforced': True,
            'network_denied': True, 'gpu_devices_denied': True}
