"""Lossless optimizer snapshot storage without an uncompressed disk duplicate."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib


def save_optimizer_snapshot(source, value):
    import numpy as np
    import torch
    source = Path(source)
    buffer = io.BytesIO()
    torch.save(value, buffer)
    parts = []
    with zipfile.ZipFile(buffer) as archive:
        cursor = 0
        for item in sorted(archive.infolist(), key=lambda item: item.header_offset):
            assert item.compress_type == zipfile.ZIP_STORED
            buffer.seek(item.header_offset)
            name_length, extra_length = struct.unpack_from('<HH', buffer.read(30), 26)
            offset = item.header_offset + 30 + name_length + extra_length
            if offset > cursor:
                parts.append({'offset':cursor, 'bytes':offset-cursor, 'lanes':False})
            parts.append({'offset':offset, 'bytes':item.file_size,
                          'lanes':'/data/' in item.filename and item.file_size % 4 == 0,
                          'zip_entry':item.filename})
            cursor = offset + item.file_size
        total = buffer.getbuffer().nbytes
        if cursor < total:
            parts.append({'offset':cursor, 'bytes':total-cursor, 'lanes':False})
    raw_view = buffer.getbuffer()
    digest = hashlib.sha256(raw_view).hexdigest()
    metadata = {'source':str(source), 'source_sha256':digest, 'source_bytes':total,
                'parts':parts, 'codec':'zlib6; transpose byte lanes for tensor payloads',
                'counts_toward_training_budget':True,
                'serialization_backing_store':'host RAM; no uncompressed disk snapshot'}
    destination = source.with_suffix(source.suffix+'.exactzip')
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    started = time.time()

    def encode(part):
        raw = bytes(raw_view[part['offset']:part['offset']+part['bytes']])
        if part['lanes']:
            raw = np.frombuffer(raw,dtype=np.uint8).reshape(-1,4).T.copy().tobytes()
        return zlib.compress(raw,6)

    with ThreadPoolExecutor(max_workers=2) as pool, zipfile.ZipFile(temporary,'w',allowZip64=True) as archive:
        archive.writestr('metadata.json',json.dumps(metadata))
        for base in range(0,len(parts),2):
            for index,payload in enumerate(pool.map(encode,parts[base:base+2]),base):
                archive.writestr(f'parts/{index:05d}',payload)
    restored_digest = hashlib.sha256()
    with zipfile.ZipFile(temporary) as archive:
        for index,part in enumerate(parts):
            raw=zlib.decompress(archive.read(f'parts/{index:05d}'))
            if part['lanes']:
                raw=np.frombuffer(raw,dtype=np.uint8).reshape(4,-1).T.copy().tobytes()
            restored_digest.update(raw)
    if restored_digest.hexdigest()!=digest:
        raise RuntimeError('Optimizer snapshot exact byte roundtrip failed')
    temporary.replace(destination)
    report={'source':str(source),'source_bytes':total,'source_sha256':digest,
            'archive':str(destination),'archive_bytes':destination.stat().st_size,
            'archive_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'exact_byte_roundtrip':True,'uncompressed_snapshot_written_to_disk':False,
            'elapsed_seconds':time.time()-started,'budget_note':'Included within candidate training wall-clock budget.'}
    (source.parent/'optimizer_storage.json').write_text(json.dumps(report,indent=2)+'\n')
    raw_view.release()
    buffer.close()
    return report
