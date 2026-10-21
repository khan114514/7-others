"""Bounded interpreter for candidate tensor graphs; no candidate code execution.

The interpreter owns all operation implementations, weights and session lifetimes.
The caller resets its context for each prefix and includes first-step construction
in the timed256-stage rollout. Fidelity is enforced independently by the verifier.
"""
import collections
import json
import math
import re
import torch
import torch.nn.functional as F

IDENTIFIER=re.compile(r'^[a-zA-Z][a-zA-Z0-9_]{0,63}$')
INPUTS={'state','image_mask','attention_mask','cos','sin'}
ARITY={'rms_norm':(2,2),'linear':(2,3),'mlp':(4,4),'mix':(3,3),
       'add':(2,2),'mul':(2,2),'attention_eager':(6,6),'attention_sdpa':(6,6),
       'matmul':(2,2),'transpose':(1,1),'reshape':(1,1),'slice':(1,1),
       'concat':(1,8),'softmax':(1,1),'cast':(1,1),'silu':(1,1),
       'scale':(1,1),'rotary':(3,3),'repeat_heads':(1,1)}
ATTRIBUTES={'transpose':{'dim0','dim1'},'reshape':{'shape'},'slice':{'dim','start','stop'},
            'concat':{'dim'},'softmax':{'dim'},'cast':{'dtype'},'scale':{'factor'},
            'repeat_heads':{'count'}}
DIMENSIONS={'batch':2,'sequence':556,'hidden':3584,'heads':28,'kv_heads':4,'head_dim':128}

def validate(program,weight_names):
    if not isinstance(program,dict):raise ValueError('Unsupported graph schema')
    schema=program.get('schema')
    expected={'schema','layer_body','layer_output'}
    if schema=='bifrost-tensor-dag/2':expected.add('compile_regions')
    elif schema!='bifrost-tensor-dag/1':raise ValueError('Unsupported graph schema')
    if set(program)!=expected:raise ValueError('Unknown graph field')
    nodes=program['layer_body']
    if not isinstance(nodes,list) or not 1<=len(nodes)<=256:
        raise ValueError('Layer graph must contain1..256 nodes')
    if len(json.dumps(program).encode())>512*1024:
        raise ValueError('Graph byte budget exceeded')
    available=set(INPUTS)
    for node in nodes:
        if not isinstance(node,dict):raise ValueError('Node must be an object')
        name,op=node.get('id'),node.get('op')
        if not isinstance(name,str) or not IDENTIFIER.fullmatch(name) or name in available:
            raise ValueError('Invalid or duplicate node identity')
        if op not in ARITY:raise ValueError('Operation is not in the numerical catalog')
        allowed={'id','op','inputs','lifetime'}|ATTRIBUTES.get(op,set())
        if not set(node)<=allowed or not {'id','op','inputs'}<=set(node):
            raise ValueError('Unknown node field')
        if not ATTRIBUTES.get(op,set())<=set(node):raise ValueError('Missing operation attributes')
        inputs=node['inputs']
        if not isinstance(inputs,list) or not ARITY[op][0]<=len(inputs)<=ARITY[op][1]:
            raise ValueError('Invalid operation arity')
        for source in inputs:
            if not isinstance(source,str):raise ValueError('Inputs must be node or weight identifiers')
            if source.startswith('w:'):
                if source[2:] not in weight_names:raise ValueError('Unknown frozen weight')
            elif source not in available:
                raise ValueError('Graph must be topologically ordered')
        if node.get('lifetime','stage') not in ['stage','prefix']:
            raise ValueError('Unknown intermediate lifetime')
        for key in ['dim','dim0','dim1']:
            if key in node and (type(node[key])is not int or not -5<=node[key]<=4):
                raise ValueError('Invalid dimension')
        if op=='slice' and any(type(node[key])is not int or not -20000<=node[key]<=20000 for key in ['start','stop']):
            raise ValueError('Invalid bounded slice')
        if op=='reshape':
            shape=node['shape']
            if not isinstance(shape,list) or not 1<=len(shape)<=5:
                raise ValueError('Invalid bounded reshape')
            for dim in shape:
                if type(dim)is int and (dim==-1 or 1<=dim<=20000):continue
                if isinstance(dim,str) and dim in DIMENSIONS:continue
                raise ValueError('Invalid reshape dimension')
        if op=='cast' and node['dtype'] not in ['bf16','fp32']:
            raise ValueError('Unsupported numerical dtype')
        if op=='scale' and (type(node['factor'])not in [int,float] or not math.isfinite(node['factor']) or abs(node['factor'])>10000):
            raise ValueError('Invalid finite scale')
        if op=='repeat_heads' and (type(node['count'])is not int or not 1<=node['count']<=28):
            raise ValueError('Invalid head replication')
        available.add(name)
    if program['layer_output'] not in available-INPUTS:
        raise ValueError('Missing graph output')
    if schema=='bifrost-tensor-dag/2':
        from bounded_graph_compiler import validate_regions
        validate_regions(program)
    return program

def rms_norm(value,weight):
    dtype=value.dtype
    full=value.to(torch.float32)
    variance=full.pow(2).mean(-1,keepdim=True)
    full=full*torch.rsqrt(variance+1e-6)
    return weight*full.to(dtype)

def rotate(value,cos,sin):
    # Published Qwen multimodal rotary layout, sections16/24/24 repeated twice.
    cos=torch.cat([part[index%3] for index,part in enumerate(cos.split([16,24,24,16,24,24],dim=-1))],dim=-1).unsqueeze(1)
    sin=torch.cat([part[index%3] for index,part in enumerate(sin.split([16,24,24,16,24,24],dim=-1))],dim=-1).unsqueeze(1)
    half=value.shape[-1]//2
    rotated=torch.cat((-value[...,half:],value[...,:half]),dim=-1)
    return value*cos+rotated*sin

def repeat_heads(value,count):
    batch,heads,length,channels=value.shape
    return value[:,:,None,:,:].expand(batch,heads,count,length,channels).reshape(batch,heads*count,length,channels)

def attention(inputs,sdpa=False):
    query,key,value,mask,cos,sin=inputs
    batch,length,_=query.shape
    if batch!=2 or not 1<=length<=556 or query.shape[-1]!=3584 or key.shape!=(batch,length,512) or value.shape!=key.shape:
        raise ValueError('Attention exceeds the frozen decoder shape budget')
    query=query.view(batch,length,28,128).transpose(1,2)
    key=key.view(batch,length,4,128).transpose(1,2)
    value=value.view(batch,length,4,128).transpose(1,2)
    query=rotate(query,cos,sin);key=rotate(key,cos,sin)
    key=repeat_heads(key,7);value=repeat_heads(value,7)
    if sdpa:
        output=F.scaled_dot_product_attention(query,key,value,attn_mask=mask,dropout_p=0.0,is_causal=False)
    else:
        scores=torch.matmul(query,key.transpose(2,3))/math.sqrt(128)
        scores=scores+mask
        probabilities=F.softmax(scores,dim=-1,dtype=torch.float32).to(query.dtype)
        output=torch.matmul(probabilities,value)
    return output.transpose(1,2).contiguous().reshape(batch,length,3584)

def apply_node(node,inputs,dimensions):
    op=node['op']
    if op=='rms_norm':result=rms_norm(*inputs)
    elif op=='linear':result=F.linear(*inputs)
    elif op=='mlp':
        value,gate,up,down=inputs
        if math.prod(value.shape[:-1])*max(gate.shape[0],up.shape[0])>100_000_000:
            raise ValueError('MLP intermediate shape budget exceeded')
        result=F.linear(F.silu(F.linear(value,gate))*F.linear(value,up),down)
    elif op=='mix':
        text,image,mask=inputs
        result=image*mask+text*(1-mask)
    elif op=='add':result=inputs[0]+inputs[1]
    elif op=='mul':result=inputs[0]*inputs[1]
    elif op in ['attention_eager','attention_sdpa']:
        result=attention(inputs,sdpa=op=='attention_sdpa')
    elif op=='matmul':result=torch.matmul(*inputs)
    elif op=='transpose':result=inputs[0].transpose(node['dim0'],node['dim1'])
    elif op=='reshape':result=inputs[0].reshape([dimensions[dim] if isinstance(dim,str) else dim for dim in node['shape']])
    elif op=='slice':
        selectors=[slice(None)]*inputs[0].ndim
        selectors[node['dim']]=slice(node['start'],node['stop'])
        result=inputs[0][tuple(selectors)]
    elif op=='concat':result=torch.cat(inputs,dim=node['dim'])
    elif op=='softmax':result=F.softmax(inputs[0],dim=node['dim'],dtype=torch.float32)
    elif op=='cast':result=inputs[0].to({'bf16':torch.bfloat16,'fp32':torch.float32}[node['dtype']])
    elif op=='silu':result=F.silu(inputs[0])
    elif op=='scale':result=inputs[0]*node['factor']
    elif op=='rotary':result=rotate(*inputs)
    elif op=='repeat_heads':result=repeat_heads(inputs[0],node['count'])
    else:raise ValueError('Unreachable unsupported operation')
    if result.ndim>5 or result.numel()>100_000_000:
        raise ValueError('Numerical output shape budget exceeded')
    return result

class TensorDAG:
    def __init__(self,program,weight_names):
        self.program=validate(program,weight_names)
        self.cache={}
        self.cache_bytes=0
        self.uses=collections.Counter(source for node in program['layer_body'] for source in node['inputs'] if not source.startswith('w:'))
        self.uses[program['layer_output']]+=1
        self.shape_signature=None
        self.meta_validation=False
        self.compiled_regions=[]
        self.compiler=None
        if program['schema']=='bifrost-tensor-dag/2':
            from bounded_graph_compiler import BoundedGraphCompiler,validate_regions
            self.compiled_regions=validate_regions(program)
            self.compiler=BoundedGraphCompiler(self.compiled_regions,apply_node)
        self.compiled_starts={r['start']:r for r in self.compiled_regions}
        self.compiled_skips={i for r in self.compiled_regions for i in range(r['start']+1,r['stop'])}
    def reset_prefix(self):
        self.cache.clear();self.cache_bytes=0
    def layer(self,index,state,weights,image_mask,attention_mask,cos,sin):
        signature=(tuple(state.shape),str(state.dtype),tuple(image_mask.shape),tuple(attention_mask.shape),tuple(cos.shape),tuple(sin.shape),
                   tuple(sorted((name,tuple(value.shape),str(value.dtype)) for name,value in weights.items())))
        if not self.meta_validation and signature!=self.shape_signature:
            # Before any candidate numerical operation, infer its complete graph
            # on allocation-free meta tensors. A malicious broadcast/matmul is
            # rejected before it can allocate on the shared GPU.
            checker=TensorDAG(self.program,weights.keys())
            checker.meta_validation=True
            checker.layer(index,state.to(device='meta'),
                {name:value.to(device='meta') for name,value in weights.items()},
                image_mask.to(device='meta'),attention_mask.to(device='meta'),cos.to(device='meta'),sin.to(device='meta'))
            self.shape_signature=signature
        values={'state':state,'image_mask':image_mask.unsqueeze(-1),'attention_mask':attention_mask,'cos':cos,'sin':sin}
        uses=self.uses.copy()
        dimensions={**DIMENSIONS,'batch':state.shape[0],'sequence':state.shape[1]}
        for offset,node in enumerate(self.program['layer_body']):
            if not self.meta_validation and offset in self.compiled_skips:continue
            if not self.meta_validation and offset in self.compiled_starts:
                self.run_region(index,self.compiled_starts[offset],values,weights,dimensions,uses)
                continue
            name,op=node['id'],node['op']
            cache_key=(index,name)
            if cache_key in self.cache:
                result=self.cache[cache_key]
            else:
                inputs=[weights[source[2:]] if source.startswith('w:') else values[source] for source in node['inputs']]
                for value in inputs:
                    if value.ndim>5 or value.numel()>100_000_000:
                        raise ValueError('Numerical input shape budget exceeded')
                result=apply_node(node,inputs,dimensions)
                if node.get('lifetime')=='prefix':
                    self.cache_bytes+=result.numel()*result.element_size()
                    if self.cache_bytes>7*1024**3:
                        raise ValueError('Per-prefix retained tensor budget exceeded')
                    self.cache[cache_key]=result
            values[name]=result
            for source in node['inputs']:
                if not source.startswith('w:'):
                    uses[source]-=1
                    if uses[source]==0 and source in values:del values[source]
        return values[self.program['layer_output']]

    def run_region(self,index,region,values,weights,dimensions,uses):
        names=region['outputs']
        present=[(index,name) in self.cache for name in names] if region['prefix'] else []
        if present and any(present) and not all(present):
            raise ValueError('Incomplete compiled prefix cache')
        if present and all(present):
            results=tuple(self.cache[(index,name)] for name in names)
        else:
            arguments=[weights[source[2:]] if source.startswith('w:') else values[source] for source in region['inputs']]
            results=self.compiler.run(region,arguments,dimensions)
            if region['prefix']:
                needed=sum(result.numel()*result.element_size() for result in results)
                if self.cache_bytes+needed>7*1024**3:
                    raise ValueError('Per-prefix retained tensor budget exceeded')
                for name,result in zip(names,results):self.cache[(index,name)]=result
                self.cache_bytes+=needed
        values.update(zip(names,results))
        for node in region['nodes']:
            for source in node['inputs']:
                if not source.startswith('w:'):
                    uses[source]-=1
                    if uses[source]==0 and source in values:del values[source]

    def compilation_facts(self):
        return self.compiler.facts() if self.compiler else {'schema':'bifrost-tensor-dag/1','region_count':0,'compiled_variant_count':0,'first_invocations':[]}
