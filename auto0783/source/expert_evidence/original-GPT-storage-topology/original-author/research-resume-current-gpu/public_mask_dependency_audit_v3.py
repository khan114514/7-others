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
            assert observed<={0x0000,0x8000,0xff7f},[hex(x) for x in observed]
            allowed=[[bits[q*556+k] in (0x0000,0x8000) for k in range(556)] for q in range(556)]
            fully_min_rows=[q for q,row in enumerate(allowed) if not any(row)]
            # Finite-min rows are NOT undefined. Under the conditional
            # BF16score bound, every masked logit rounds to the same finite
            # minimum and softmax is uniform over all columns. Model those
            # padding-query influence edges as dense.
            for q in fully_min_rows:allowed[q]=[True]*556
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
            masks.append({'batch':batch,'post_BF16_mask_values_bits':[hex(u) for u in sorted(observed)],'text_queries_directly_reading_image_count':len(text_image_edges),'text_queries_transitively_image_dependent_count':len(changing_text),'fixed_text_rows_count':len(fixed_text),'text_key_columns_seen_by_image_count':len(image_seen_text),'image_reads_only_structurally_image_independent_text':True,'graph_reachability_iterations':iterations,'finite_min_only_query_count':len(fully_min_rows),'finite_min_only_query_indices':fully_min_rows,'finite_min_query_fallback':'uniformdense under conditional bound; unlike minusInfinity mask','image_dependent_text_indices':changing_text})
        records.append({'case':case,'public_mask_dtype':dtype,'raw_mask_sha256':hashlib.sha256(data).hexdigest(),'batches':masks})
report={'scope':'Only declared public case0/1/2 masks, no hidden/general qualification','source':'/opt/bifrost/public_eval/data/full-scout-inputs.safetensors','records':records,'argument':'For the declared zero-versus-finite-min mask topology, every text row visible to image queries lies outside directed image-to-text paths. Row-local norm/projections/MLP/residual preserve this topology through28layers. Exact target suppression of finite-min entries additionally requires masked probability underflow; topology alone does not prove that numerical condition.','floating_limits':['Source finite outputs still required;0timesNaN could propagate, so masking alone is not a license for nonfinite tensors.','Actual public masked entries are BF16finite minimum, not minusInfinity; fully finite-min query rows remain defined and become uniform under the conditional bound. Graph reachability includes dense edges for those queries. Topology is conditional, not an unconditional numerical proof.','Geometry/reduction rounding and modified text-only attention are numerically qualified by actual Dev, not only graph topology.','Actual withheld input mask matrices are not inspected; published canonical mask contract is the only generalization basis.']}
Path('research-resume-current-gpu/public-attention_dependency-audit-v3.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'records':[{**r,'batches':[{k:v for k,v in b.items() if k not in ('image_dependent_text_indices','finite_min_only_query_indices')} for b in r['batches']]} for r in records],'topology_argument':report['argument']},indent=2))
