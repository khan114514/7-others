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
    roots = [(x, read) for x in ('/usr', '/lib', '/lib64', '/bin', '/sys',
                                '/opt/conda', '/proc/self',
                                str(Path(candidate).parent))]
    # Do not grant the /etc directory: it may hold root-only host keys or
    # deployment credentials. Read only the system metadata needed by Python.
    roots += [(x, read) for x in ('/etc/ld.so.cache', '/etc/localtime',
                                '/etc/ssl/certs', '/etc/nsswitch.conf',
                                '/etc/passwd', '/etc/group', '/etc/hostname', '/etc/hosts')]
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
        for name in (b'ptrace', b'process_vm_readv', b'process_vm_writev',
                     b'mount', b'umount2', b'bpf', b'perf_event_open',
                     b'socket', b'socketpair', b'connect', b'bind', b'listen',
                     b'accept', b'accept4', b'kill', b'tkill', b'tgkill',
                     b'pidfd_send_signal', b'fork', b'vfork', b'execve', b'execveat',
                     b'unshare', b'setns', b'mknod', b'mknodat', b'open_by_handle_at',
                     b'io_uring_setup', b'io_uring_register', b'io_uring_enter',
                     b'fsopen', b'fsconfig', b'fsmount', b'move_mount', b'open_tree',
                     b'pivot_root', b'reboot', b'kexec_load', b'kexec_file_load',
                     b'swapon', b'swapoff', b'settimeofday', b'clock_settime',
                     b'adjtimex', b'init_module', b'finit_module', b'delete_module'):
            call = sc.seccomp_syscall_resolve_name(name)
            if call >= 0 and sc.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM, call, 0, None):
                raise RuntimeError('seccomp restriction failed')
        # CPU tensor libraries may create threads, but method code cannot fork
        # independent processes. Return ENOSYS for clone3 so libc falls back to
        # clone, whose CLONE_THREAD flag can be inspected by the filter.
        call = sc.seccomp_syscall_resolve_name(b'clone3')
        if call >= 0 and sc.seccomp_rule_add_array(ctx, 0x00050000 | errno.ENOSYS, call, 0, None):
            raise RuntimeError('clone3 restriction failed')
        call = sc.seccomp_syscall_resolve_name(b'clone')
        compare = Compare(0, 7, 0x10000, 0)
        if call >= 0 and sc.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM,
                                                 call, 1, ctypes.byref(compare)):
            raise RuntimeError('process creation restriction failed')
        if sc.seccomp_load(ctx):
            raise RuntimeError('seccomp_load failed')
    finally:
        sc.seccomp_release(ctx)
    return {'landlock_abi': abi, 'filesystem_enforced': True,
            'network_denied': True, 'unix_sockets_denied': True,
            'cross_process_signals_denied': True, 'new_processes_denied': True,
            'device_creation_denied': True, 'gpu_devices_denied': True,
            'system_private_files_denied': True}
