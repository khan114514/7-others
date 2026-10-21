"""Declared public-case mask topology and cached-text influence audit.

Read only public case0/1/2 attention masks from the public input file. No
embeddings, unseen cases or private source are read. Standard-library parser.
"""
import struct,json,hashlib,math
from pathlib import Path
path=Path('/opt/bifrost/public_eval/data/full-scout-inputs.safetensors')
with path.open('rb') as f:
    length=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(length));base=8+length
    records=[]
    for case in range(3):
        key=f'case.{case}.attention_mask';meta=header[key]
        assert meta['shape']==[2,1,556,556],meta
        first,last=meta['data_offsets'];f.seek(base+first);data=f.read(last-first)
        dtype=meta['dtype'];assert dtype in ('F32','BF16')
        if dtype=='F32':raw=struct.unpack('<'+str(len(data)//4)+'I',data)
        else:raw=struct.unpack('<'+str(len(data)//2)+'H',data)
        masks=[]
        for batch in range(2):
            offset=batch*556*556;bits=[];observed=set()
            for idx in range(556*556):
                u=raw[offset+idx]
                if dtype=='F32':
                    # Normal finite/inf RNE cast; NaN checks below exclude
                    # payload edgecases. Mirrors target dtype representation,
                    # not a numerical PyTorch kernel implementation.
                    u=((u+0x7fff+((u>>16)&1))>>16)&0xffff
                value=struct.unpack('<f',struct.pack('<I',u<<16))[0]
                assert not math.isnan(value);observed.add(u);bits.append(u)
            assert observed<={0x0000,0xff80},[hex(x) for x in observed]
            allowed=[[bits[q*556+k]!=0xff80 for k in range(556)] for q in range(556)]
            assert all(any(row) for row in allowed),'fully masked query would make softmax undefined'
            text_image_edges=[q for q in range(300) if any(allowed[q][300:])]
            image_seen_text=[k for k in range(300) if any(allowed[q][k] for q in range(300,556))]
            text_from_image=set(text_image_edges)
            # Reachability via attention key/value influence; row-local
            # RMS/MLP/residual adds no cross-row edges. Stop at fixedpoint.
            frontier=set(range(300,556));reachable=set(frontier)
            iterations=0
            while frontier:
                new={q for q,row in enumerate(allowed) if q not in reachable and any(row[k] for k in frontier)}
                iterations+=1;reachable.update(new);frontier=new
            changing_text=sorted(reachable&set(range(300)))
            fixed_text=set(range(300))-set(changing_text)
            assert set(image_seen_text).issubset(fixed_text),'image queries read text influenced by images'
            assert all(not any(allowed[q][k] for k in changing_text) for q in fixed_text|set(range(300,556)))
            masks.append({'batch':batch,'post_BF16_mask_values_bits':[hex(u) for u in sorted(observed)],'text_queries_directly_reading_image_count':len(text_image_edges),'text_queries_transitively_image_dependent_count':len(changing_text),'fixed_text_rows_count':len(fixed_text),'text_key_columns_seen_by_image_count':len(image_seen_text),'image_reads_only_image_independent_text':True,'graph_reachability_iterations':iterations,'all_query_rows_have_an_allowed_key':True,'image_dependent_text_indices':changing_text})
        records.append({'case':case,'public_mask_dtype':dtype,'raw_mask_sha256':hashlib.sha256(data).hexdigest(),'batches':masks})
report={'scope':'Only declared public case0/1/2 masks, no hidden/general qualification','source':'/opt/bifrost/public_eval/data/full-scout-inputs.safetensors','records':records,'argument':'Every text row visible to image queries lies outside all directed image-to-text paths. Row-local norm/projections/MLP/residual preserve this influence separation through28layers. Any image-dependent padding rows are never keys/values for fixed text or images. Prefix replacement of those discarded text rows cannot influence image predictions in exact real arithmetic.','floating_limits':['Source finite outputs still required;0timesNaN could propagate, so masking alone is not a license for nonfinite tensors.','Cast-to-minus-infinity zeroes masked softmax entries for finite other scores; no approximate finite-mask-bound claim.','Geometry/reduction rounding and modified text-only attention are numerically qualified by actual Dev, not only graph topology.','Actual withheld input mask matrices are not inspected; published canonical mask contract is the only generalization basis.']}
Path('research-resume-current-gpu/public-attention_dependency-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'records':[{**r,'batches':[{k:v for k,v in b.items() if k!='image_dependent_text_indices'} for b in r['batches']]} for r in records],'topology_argument':report['argument']},indent=2))
