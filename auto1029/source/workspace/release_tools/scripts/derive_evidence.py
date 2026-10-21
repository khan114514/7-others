"""Derive public UTF-8 evidence with explicit metadata substitutions only.

Original files remain immutable and separately hashed. This does not redact
credentials, modify executable methods, change scientific data, or certify an
artifact's privacy. Run the strict scanner on every resulting attachment.
"""
import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

MAX_BYTES=64*1024*1024
PROTECTED_KEYS={'method_sources','source_code','question','decoded_text','generated_token_ids','token_ids','labels','scoring_protocol','tokenizer'}


def regular_bytes(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>limit:raise ValueError('Bounded regular evidence input required')
        with os.fdopen(fd,'rb',closefd=False) as stream:data=stream.read(limit+1)
        final=os.fstat(fd)
        if len(data)!=info.st_size or (final.st_size,final.st_mtime_ns)!=(info.st_size,info.st_mtime_ns):raise ValueError('Evidence input changed during read')
        return data
    finally:os.close(fd)


def load_mappings(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=0o600:
        raise PermissionError('Operator mapping requires owned regular mode0600 file')
    value=json.loads(regular_bytes(path,65536))
    if not isinstance(value,list) or not value or len(value)>64:raise ValueError('Explicit bounded metadata mapping list required')
    result=[]
    for row in value:
        if set(row)!={'kind','from','to'} or not all(isinstance(v,str) for v in row.values()):raise ValueError('Invalid metadata mapping')
        old,new=row['from'],row['to']
        if len(old)>4096 or len(new)>256 or any(c in old+new for c in ['\x00','\n','\r']):raise ValueError('Invalid metadata mapping size or control character')
        if row['kind']=='author_path':
            if not re.match(r'^(?:/(?:Users|home)/[^/]+/|/root(?:/)|[A-Za-z]:\\Users\\)',old):raise ValueError('Author path prefix required')
        elif row['kind']=='host_alias':
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{7,127}',old):raise ValueError('Explicit operator host alias required')
        else:raise ValueError('Only author-path or host-alias metadata mappings are allowed')
        if not re.fullmatch(r'\$\{[A-Z][A-Z0-9_]*\}',new):raise ValueError('Stable unusable logical placeholder required')
        result.append(row)
    if len({r['from'] for r in result})!=len(result):raise ValueError('Duplicate source mapping')
    return sorted(result,key=lambda r:len(r['from']),reverse=True)


def derive(input_path,output_path,mapping_path,kind):
    if input_path.suffix.lower() in {'.py','.sh','.toml','.yaml','.yml'}:raise ValueError('Executable or task-configuration source cannot be deidentified as evidence')
    if output_path.exists() or output_path.is_symlink():raise ValueError('Fresh derived output required')
    raw=regular_bytes(input_path,MAX_BYTES);text=raw.decode('utf-8');mappings=load_mappings(mapping_path);changed=[]
    def replace(value,pointer):
        result=value;hits=[]
        for row in mappings:
            count=result.count(row['from'])
            if count:
                target=row['to']+('/' if row['kind']=='author_path' and row['from'].endswith(('/', '\\')) else '')
                result=result.replace(row['from'],target);hits.append({'kind':row['kind'],'placeholder':row['to'],'count':count})
        if hits:changed.append({'pointer':pointer,'substitutions':hits})
        return result
    def walk(value,pointer='',protected=False):
        if isinstance(value,str):
            if protected and any(row['from'] in value for row in mappings):raise ValueError('Protected scientific or executable payload cannot be changed')
            return value if protected else replace(value,pointer)
        if isinstance(value,list):return [walk(v,pointer+'/'+str(i),protected) for i,v in enumerate(value)]
        if isinstance(value,dict):
            result={}
            for k,v in value.items():
                key=replace(k,pointer+'/@key') if not protected else k
                if key in result:raise ValueError('Metadata substitution would collide dictionary keys')
                result[key]=walk(v,pointer+'/'+k,protected or k in PROTECTED_KEYS)
            return result
        return value
    if kind=='json':
        original=json.loads(text);value=walk(original);derived=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        # Every scalar number, boolean, null and list length must be preserved.
        def scalars(v):
            if isinstance(v,dict):return [scalars(x) for x in v.values()]
            if isinstance(v,list):return [len(v),*[scalars(x) for x in v]]
            return '<string>' if isinstance(v,str) else v
        if scalars(original)!=scalars(value):raise ValueError('Scientific scalar structure changed')
    else:
        derived=replace(text,'<text-metadata>') .encode('utf-8')
    fd=os.open(output_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(derived)
    result={'phase':'PUBLIC_METADATA_DERIVATIVE_CREATED_STRICT_SCAN_PENDING',
        'kind':kind,'original_sha256':hashlib.sha256(raw).hexdigest(),'derived_sha256':hashlib.sha256(derived).hexdigest(),
        'original_bytes':len(raw),'derived_bytes':len(derived),'changed_metadata_locations':changed,
        'mapping_policy_sha256':hashlib.sha256(json.dumps(mappings,sort_keys=True).encode()).hexdigest(),
        'original_file_modified':False,'credentials_removed_or_hidden':False,
        'scientific_scalars_unchanged':True if kind=='json' else None,'model_research_seconds':0}
    # Raw source pointers can themselves contain private identifiers. Keep
    # only stable ordinal locations; original mapping stays operator-private.
    for i,entry in enumerate(result['changed_metadata_locations']):entry['pointer']='metadata-location-'+str(i)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mappings',required=True,type=Path);parser.add_argument('--kind',choices=['json','text'],required=True)
    args=parser.parse_args()
    try:result=derive(args.input,args.output,args.mappings,args.kind)
    except Exception as error:
        print(json.dumps({'phase':'FAILED_CLOSED','error_type':type(error).__name__,'private_input_locations_withheld':True}));return 1
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
