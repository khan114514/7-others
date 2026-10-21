"""Independent inverse logical-index audit of a real candidate value chain.

Uses the actual JSON reshape/transpose/slice/concat/repeat metadata. It does
not infer physical strides or claim numerical CUDA qualification.
"""
import json,sys,math,hashlib
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text());nodes={n['id']:n for n in p['layer_body']}
leaves={'image_v':(2,256,512),'text_v_rows':(2,300,512),'image_prob':(2,28,256,556)}
shapes=dict(leaves)
def shape(name):
    if name in shapes:return shapes[name]
    n=nodes[name];a=list(shape(n['inputs'][0]));op=n['op']
    if op=='slice':a[n['dim']]=len(range(a[n['dim']])[n['start']:n['stop']])
    elif op=='transpose':a[n['dim0']],a[n['dim1']]=a[n['dim1']],a[n['dim0']]
    elif op=='reshape':
        old=math.prod(a);a=n['shape'][:]
        if -1 in a:a[a.index(-1)]=old//-math.prod(a)
        assert old==math.prod(a)
    elif op=='concat':a[n['dim']]=sum(shape(child)[n['dim']] for child in n['inputs'])
    elif op=='repeat_heads':a[1]*=n['count']
    else:raise ValueError((name,op))
    shapes[name]=tuple(a);return tuple(a)
def inverse(name,index):
    if name in leaves:return (name,*index)
    n=nodes[name];op=n['op'];idx=list(index);child=n['inputs'][0]
    if op=='slice':idx[n['dim']]+=n['start']
    elif op=='transpose':idx[n['dim0']],idx[n['dim1']]=idx[n['dim1']],idx[n['dim0']]
    elif op=='reshape':
        flat=0
        for size,coordinate in zip(shape(name),idx):flat=flat*size+coordinate
        dims=shape(child);idx=[0]*len(dims)
        for axis in range(len(dims)-1,-1,-1):flat,idx[axis]=divmod(flat,dims[axis])
        assert flat==0
    elif op=='concat':
        axis=n['dim'];coordinate=idx[axis]
        for child in n['inputs']:
            size=shape(child)[axis]
            if coordinate<size:idx[axis]=coordinate;break
            coordinate-=size
        else:raise AssertionError('concat coordinate out of bounds')
    elif op=='repeat_heads':idx[1]//=n['count']
    else:raise ValueError((name,op))
    return inverse(child,tuple(idx))
assert nodes['channel_query_output']['inputs']==['value_transpose','probability_transpose']
assert shape('value_transpose')==(2,28,128,556)
assert shape('probability_transpose')==(2,28,556,256)
value_checks=prob_checks=0
for b in range(2):
    for h in range(28):
        for t in range(556):
            for d in range(128):
                got=inverse('value_transpose',(b,h,d,t))
                expected=('text_v_rows',b,t,(h//7)*128+d) if t<300 else ('image_v',b,t-300,(h//7)*128+d)
                assert got==expected,(b,h,d,t,got,expected)
                value_checks+=1
        for q in range(256):
            for t in (0,299,300,555):
                assert inverse('probability_transpose',(b,h,t,q))==('image_prob',b,h,q,t)
                prob_checks+=1
report={'scope':'Actual JSON inverse logical-index diagnostic, no physical strides or numerical kernels','artifact_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'value_coordinates_checked':value_checks,'probability_boundary_coordinates_checked':prob_checks,'probability_transform':'One original transpose dim2/3; unchanged exact inverse for every symbolic source position','ordered_sum':'channel_query_output[b,h,d,q] sums source positions t=0..555 in original order','source_mapping':'t<300:text_v_rows[b,t,128*(h//7)+d]; t>=300:image_v[b,t-300,128*(h//7)+d]','shapes':{k:list(v) for k,v in shapes.items()}}
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='shapes'}))
