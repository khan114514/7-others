"""Trusted observed hardware routing around the byte-frozen SAVE dispatcher."""
import importlib.util,json,re,stat,sys
from pathlib import Path
original=Path(__file__).with_name('gpu_service.py')
spec=importlib.util.spec_from_file_location('save_byte_frozen_gpu_dispatch',original)
service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
def main():
    i=sys.argv.index('--root');root=Path(sys.argv[i+1]).resolve()
    assert root.name=='save_autoresearch_20261009'
    # Verify the original complete numerical binding; never change the guard,
    # its expected file hashes, data, seeds, scoring or resource budgets.
    from freeze_gpu_binding import verify
    verify(root)
    file=root/'private/gpu-resource-routing.json'
    if file.exists():
        assert not file.is_symlink() and stat.S_IMODE(file.stat().st_mode)==0o600
        profile=json.loads(file.read_text());assert set(profile)=={'gpu_hosts'}
        hosts={}
        for h in profile['gpu_hosts']:
            assert set(h)=={'port','gpu','locks'} and h['port'] in service.HOSTS
            assert re.fullmatch(r'GPU-[0-9a-f-]{36}',h['gpu'])
            assert h['locks'] and all(Path(p).is_absolute() for p in h['locks'])
            assert h['port'] not in hosts
            hosts[h['port']]=(h['gpu'],h['locks'])
        service.HOSTS=hosts
    # The original durable dispatcher launches this same trusted entry for its
    # execute child. The original dispatcher bytes and numeric verify stay fixed.
    service.__file__=__file__
    service.main()
if __name__=='__main__':main()
