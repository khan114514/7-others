"""Reproduce the frozen BF16 weight shards; does not copy metadata or download assets."""
import argparse
import fcntl
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def rounded_bf16(values):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source weights require explicit investigation")
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16).astype("<u2")

def convert(source, target, expected_sha):
    actual_sha = sha(source)
    if actual_sha != expected_sha:
        raise ValueError("Source SHA-256 does not match the pinned public asset")
    with source.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        if header_bytes <= 0 or header_bytes > 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(header_bytes))
    source_data_offset = 8 + header_bytes
    updated, offset, tensors = {}, 0, []
    if "__metadata__" in header:
        updated["__metadata__"] = header["__metadata__"]
    for name, entry in sorted(header.items(), key=lambda x: x[1].get("data_offsets", [0])[0] if x[0] != "__metadata__" else -1):
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if start < 0 or end < start or source_data_offset + end > source.stat().st_size:
            raise ValueError("Invalid tensor offset")
        bytes_count = end - start
        is_fp32 = entry["dtype"] == "F32"
        if is_fp32 and bytes_count != int(np.prod(entry["shape"])) * 4:
            raise ValueError("FP32 tensor shape/size mismatch")
        output_bytes = bytes_count // 2 if is_fp32 else bytes_count
        updated[name] = dict(entry, dtype="BF16" if is_fp32 else entry["dtype"], data_offsets=[offset, offset + output_bytes])
        offset += output_bytes
        tensors.append((name, entry, is_fp32))
    encoded = json.dumps(updated, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    partial = target.with_name(target.name + ".partial")
    with source.open("rb") as incoming, partial.open("wb") as outgoing:
        outgoing.write(struct.pack("<Q", len(encoded)))
        outgoing.write(encoded)
        for name, entry, is_fp32 in tensors:
            start, end = entry["data_offsets"]
            incoming.seek(source_data_offset + start)
            remaining = end - start
            while remaining:
                block = incoming.read(min(16 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError("Truncated source tensor")
                if is_fp32:
                    outgoing.write(rounded_bf16(np.frombuffer(block, dtype="<f4")).tobytes())
                else:
                    outgoing.write(block)
                remaining -= len(block)
    partial.replace(target)
    return {"file": source.name, "source_sha256": actual_sha, "source_bytes": source.stat().st_size, "runtime_sha256": sha(target), "runtime_bytes": target.stat().st_size, "tensor_count": len(tensors), "tensor_data_bytes": offset, "conversion": "FP32 to BF16, round-to-nearest-even, finite inputs; other dtypes preserved"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--target',required=True,type=Path)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'asset_manifest.json')
    args=parser.parse_args()
    if args.source.resolve()==args.target.resolve():raise ValueError('Source and destination must differ')
    args.target.mkdir(parents=True,exist_ok=True)
    lock=(args.target/'conversion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(args.manifest.read_text())
    shards=[row for row in manifest['external_assets'] if row['deployment_path'].endswith('.safetensors')]
    if len(shards)!=9:raise ValueError('Expected all nine frozen weight shards')
    for row in shards:
        name=Path(row['deployment_path']).name
        source=args.source/name;target=args.target/name
        if target.is_symlink():raise ValueError('Refusing symlink output')
        if target.exists():
            if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:raise ValueError('Existing output differs; preserve it for inspection')
            print(json.dumps({'file':name,'status':'ALREADY_VERIFIED'}),flush=True);continue
        if target.with_name(name+'.partial').exists():raise ValueError('Existing partial output must be inspected before resuming')
        result=convert(source,target,row['source_sha256'])
        if result['runtime_bytes']!=row['bytes'] or result['runtime_sha256']!=row['sha256']:raise ValueError('Converted shard differs from frozen runtime; preserve mismatch')
        print(json.dumps(result),flush=True)
    print(json.dumps({'status':'ALL_NINE_RUNTIME_SHARDS_MATCH','metadata_installed':False,'originals_preserved':True}))


if __name__=='__main__':main()
