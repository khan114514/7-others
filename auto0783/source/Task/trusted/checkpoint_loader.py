"""Trusted revision/hash-checked tensor loader. Candidate programs choose no paths."""
import hashlib
import json
import math
import urllib.request
from pathlib import Path
import torch

DTYPES={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16}

def load_public_tensors(base_url,expected_manifest_sha256,progress):
    # The base URL is a trusted controller parameter, never candidate input.
    assert base_url in ['http://127.0.0.1:19195']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url+'/manifest.json',timeout=60) as response:
        encoded=response.read(2*1024*1024+1)
    if len(encoded)>2*1024*1024 or hashlib.sha256(encoded).hexdigest()!=expected_manifest_sha256:
        raise ValueError('Public tensor manifest binding mismatch')
    manifest=json.loads(encoded)
    assert manifest['schema']=='bifrost-public-decoder-tensors/1' and manifest['checkpoint_revision']=='60bbb8b163827b9324d72bd84d6ddc75ebb902bb'
    assert manifest['tensor_count']==len(manifest['tensors']) and manifest['tensor_bytes']<=28*1024**3
    memory=Path('/sys/fs/cgroup/memory.current');limit=Path('/sys/fs/cgroup/memory.max');stats=Path('/sys/fs/cgroup/memory.stat')
    if memory.exists() and limit.exists() and limit.read_text().strip()!='max':
        detail={key:int(value) for line in stats.read_text().splitlines() for key,value in [line.split()]}
        # File cache is reclaimable; resident anonymous/shmem/slab must retain
        # a16GiB margin for existing tasks and the current tensor transfer.
        resident=detail.get('anon',0)+detail.get('shmem',0)+detail.get('slab_unreclaimable',0)
        if resident+manifest['tensor_bytes']+16*1024**3>int(limit.read_text()):
            raise RuntimeError('Insufficient cgroup resident-memory headroom; no tensors loaded')
    result={};verified=[]
    for ordinal,tensor in enumerate(manifest['tensors']):
        assert tensor['id']==ordinal and tensor['dtype'] in DTYPES
        assert type(tensor['bytes'])is int and 0<tensor['bytes']<=256*1024*1024
        buffer=bytearray(tensor['bytes']);view=memoryview(buffer);offset=0;digest=hashlib.sha256()
        with opener.open(base_url+'/tensor/'+str(ordinal),timeout=180) as response:
            if int(response.headers['Content-Length'])!=tensor['bytes']:raise ValueError('Tensor length mismatch')
            while offset<len(view):
                count=response.readinto(view[offset:min(len(view),offset+4*1024*1024)])
                if not count:raise ValueError('Tensor transport truncated')
                digest.update(view[offset:offset+count]);offset+=count
        if digest.hexdigest()!=tensor['sha256']:raise ValueError('Tensor SHA256 mismatch')
        value=torch.frombuffer(buffer,dtype=DTYPES[tensor['dtype']]).reshape(tensor['shape'])
        if value.numel()*value.element_size()!=tensor['bytes']:raise ValueError('Tensor shape mismatch')
        result[tensor['key']]=value
        verified.append({'key':tensor['key'],'sha256':digest.hexdigest(),'bytes':tensor['bytes']})
        if ordinal%24==0 or ordinal==len(manifest['tensors'])-1:
            progress('LOADING_PUBLIC_TENSORS',loaded=ordinal+1,total=len(manifest['tensors']))
    return result,{'manifest_sha256':expected_manifest_sha256,'checkpoint_revision':manifest['checkpoint_revision'],
                   'tensor_count':len(result),'verified_tensors':verified,'checkpoint_shards':manifest['checkpoint_shards']}
