"""Author-owned held-out scoring policy, never delivered to method workers."""
import hashlib
import json
import os
from pathlib import Path
import stat

from scoring import normalize


def load_formal_policy(config, task_root, protocol):
    value=config.get('formal_evaluation_policy')
    if not isinstance(value,str) or not Path(value).is_absolute():
        raise ValueError('Private formal evaluation policy is not configured')
    path=Path(value).resolve()
    if path.is_relative_to(Path(task_root).resolve()):
        raise ValueError('Private policy must be outside the delivered task')
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
        raise ValueError('Private policy permissions are unsafe')
    raw=path.read_bytes();policy=json.loads(raw)
    seed=policy.get('evaluation_seed')
    if type(seed) is not int or not 0<=seed<2**31 or seed==0:
        raise ValueError('Formal evaluation requires a held-out seed')
    if policy.get('protocol')!=protocol or policy.get('sample_count')!=50000:
        raise ValueError('Private policy protocol mismatch')
    if policy.get('teacher_seed')!=1001 or policy.get('training_seed')!=2001:
        raise ValueError('Formal training must retain the published streams')
    anchors=policy.get('anchors',{})
    normalize(anchors.get('baseline'),anchors)
    if anchors.get('evaluation_seed')!=seed:
        raise ValueError('Formal anchors do not match the held-out seed')
    # Bind each empirical anchor to an actual author-owned full receipt.
    for name in ('baseline','upper_bound'):
        evidence=policy.get('evidence',{}).get(name,{})
        location=evidence.get('path')
        if not isinstance(location,str) or not Path(location).is_absolute():
            raise ValueError('Missing empirical anchor receipt')
        payload=Path(location).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=evidence.get('sha256'):
            raise ValueError('Empirical anchor receipt changed')
        receipt=json.loads(payload)
        if (receipt.get('status')!='SUCCESS' or receipt.get('sample_count')!=50000
                or receipt.get('evaluation_seed')!=seed or receipt.get('metric')!=anchors[name]):
            raise ValueError('Empirical anchor receipt binding mismatch')
    return policy,hashlib.sha256(raw).hexdigest()
