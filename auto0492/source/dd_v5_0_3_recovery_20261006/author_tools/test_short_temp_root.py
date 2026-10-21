"""Reproduce the real multiprocessing path failure and verify the storage fix."""
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ENGINE=Path(__file__).resolve().parents[1]/'workspace/harbor_task/environment/engine'
sys.path.insert(0,str(ENGINE))
from config import short_temp_root


class ShortTemporaryRoot(unittest.TestCase):
    def test_long_run_path_reproduces_failure_but_short_temp_binds(self):
        with tempfile.TemporaryDirectory() as parent:
            run=Path(parent)/('long-author-root-'*6)/('run-'+'a'*32)
            run.mkdir(parents=True)
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
                with self.assertRaises(OSError):listener.bind(str(run/'listener'))
            with patch('config.tempfile.gettempdir',return_value=str(run)):
                short=short_temp_root(run)
            try:
                self.assertLess(len(str(short/'pymp-12345678'/'listener-12345678').encode()),108)
                self.assertEqual(short.stat().st_mode&0o777,0o700)
                address=short/'listener'
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:listener.bind(str(address))
                address.unlink()
            finally:short.rmdir()

    def test_symlink_temp_destination_is_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            run=Path(folder)/'test-run'
            leaf='ddmp-'+hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:16]
            link=Path('/tmp')/leaf
            link.symlink_to(folder)
            try:
                with patch('config.tempfile.gettempdir',return_value='/tmp'):
                    with self.assertRaisesRegex(RuntimeError,'Unsafe'):short_temp_root(run)
            finally:link.unlink()


if __name__=='__main__':unittest.main()
