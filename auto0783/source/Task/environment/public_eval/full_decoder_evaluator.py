"""Trusted full-pretrained decoder benchmark. Candidate input is a tensor DAG.

The frozen canonical sampler is benchmark context, not a claim to reproduce the
published image renderer or its original multi-patch CFG indexing unchanged.
"""
import collections
import fcntl
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tensor_dag import TensorDAG,rms_norm,validate
from baseline_ops import Qwen2_5_VLConfig,Qwen2_5_VLDecoderLayer,Qwen2_5_VLRotaryEmbedding

PROFILES={'integration':{'steps':8,'cases':3,'repeats':1,'formal':False},
          'dev':{'steps':32,'cases':3,'repeats':1,'formal':False},
          'formal':{'steps':256,'cases':6,'repeats':3,'formal':True}}
CONFIG={'hidden_size':3584,'intermediate_size':18944,'num_hidden_layers':28,
        'num_attention_heads':28,'num_key_value_heads':4,
        'rope_scaling':{'type':'mrope','mrope_section':[16,24,24]},
        'rms_norm_eps':1e-6,'attention_dropout':0.0}
QUALITY={'relative_rmse':0.001,'scaled_max_abs':0.001,'peak_bytes':34*1024**3}

def deltas(actual,expected):
    difference=actual.float()-expected.float()
    relative=float(difference.square().mean().sqrt()/(expected.float().square().mean().sqrt()+1e-12))
    scaled=float(difference.abs().max()/(1+expected.float().abs().max()))
    return relative,scaled

class FullDecoderEvaluator:
    def __init__(self,cpu_weights,input_path,baseline_program,gpu_uuid,locks,progress):
        self.cpu=cpu_weights;self.input_path=Path(input_path);self.baseline_program=baseline_program
        self.gpu_uuid=gpu_uuid;self.lock_paths=locks;self.progress=progress
        self.config=Qwen2_5_VLConfig(**CONFIG)
        prefix='vision_language_model.model.layers.0.'
        self.weight_names={name[len(prefix):] for name in cpu_weights if name.startswith(prefix)}
        validate(baseline_program,self.weight_names)
        self.input_sha256=hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        assert self.input_sha256=='7092cd74a4902f6f846f9b4a7f396eab9db450f576dc69b3903df7effc4a1307'
        self.hidden_path=self.input_path.parent/'hidden-decoder-inputs-v1.safetensors'
        self.hidden_sha256=hashlib.sha256(self.hidden_path.read_bytes()).hexdigest()
        assert self.hidden_sha256=='db9c6aab916da4b5914a96a642169fd122a67e8b15fbb16628ad90dc240e40e5'
    def acquire(self,timeout=1800):
        self.locks=[Path(path).open('r') for path in self.lock_paths]
        started=time.monotonic();deadline=started+timeout
        while True:
            try:
                for handle in self.locks:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                for handle in self.locks:fcntl.flock(handle,fcntl.LOCK_UN)
                if time.monotonic()>deadline:raise RuntimeError('Existing GPU lease wait expired without formal credit')
                self.progress('WAITING_FOR_EXISTING_GPU_LEASE',queue_seconds=time.monotonic()-started)
                time.sleep(15)
        observed=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if observed[0].strip()!=self.gpu_uuid or int(observed[1])>64:
            raise RuntimeError('GPU identity mismatch or occupied outside its existing scheduler')
        return time.monotonic()-started
    def load_cuda(self):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        self.layers=[];self.weights=[]
        for index in range(28):
            with torch.device('meta'):
                layer=Qwen2_5_VLDecoderLayer(self.config,index)
                layer.add_vision_mlp();layer.add_vision_layernorms();layer.self_attn.add_vision_attention()
            prefix=f'vision_language_model.model.layers.{index}.'
            values={key[len(prefix):]:value for key,value in self.cpu.items() if key.startswith(prefix)}
            assert set(values)==self.weight_names
            layer.load_state_dict(values,strict=True,assign=True)
            layer.self_attn.rotary_emb=Qwen2_5_VLRotaryEmbedding(self.config)
            layer=layer.to(device='cuda',dtype=torch.bfloat16).eval()
            self.layers.append(layer);self.weights.append(layer.state_dict())
        self.special={key:self.cpu[key].to(device='cuda',dtype=torch.bfloat16) for key in
            ['vision_language_model.model.vision_branch_norm.weight','vision_head.weight','vision_head.bias','mask_token','learnable_pos_emb']}
        self.rotary=Qwen2_5_VLRotaryEmbedding(self.config).cuda()
        torch.cuda.synchronize()
    def release_cuda(self):
        for name in ['layers','weights','special','rotary']:
            if hasattr(self,name):delattr(self,name)
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        for handle in getattr(self,'locks',[]):
            fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        self.locks=[]
    def decode(self,graph,states,mask,attention,cos,sin,published=False):
        value=states
        for index,layer in enumerate(self.layers):
            if published:
                value=layer(value,attention_mask=attention,position_embeddings=(cos,sin),image_position_mask=mask)[0]
            else:
                value=graph.layer(index,value,self.weights[index],mask,attention,cos,sin)
        value=rms_norm(value[:,-256:],self.special['vision_language_model.model.vision_branch_norm.weight'])
        return F.linear(value,self.special['vision_head.weight'],self.special['vision_head.bias'])
    def image_initial(self):return self.special['mask_token'].expand(1,256,-1).clone()
    @staticmethod
    def reveal(order,remaining,step,steps):
        next_remaining=max(1,min(remaining-1,int(256*math.cos(math.pi/2*(step+1)/steps))))
        count=remaining-next_remaining if step<steps-1 else remaining
        return order[256-remaining:256-remaining+count],remaining-count
    def evaluate(self,program,profile,run_dir,preacquired_queue_seconds=None):
        if profile not in PROFILES:raise ValueError('Unknown frozen evaluation profile')
        validate(program,self.weight_names)
        settings=PROFILES[profile];steps=settings['steps'];run_dir=Path(run_dir)
        input_path=self.hidden_path if profile=='formal' else self.input_path
        input_sha256=self.hidden_sha256 if profile=='formal' else self.input_sha256
        if hashlib.sha256(input_path.read_bytes()).hexdigest()!=input_sha256:
            raise ValueError('Frozen input data changed after worker initialization')
        queued=self.acquire() if preacquired_queue_seconds is None else preacquired_queue_seconds
        loading=time.monotonic()
        try:
            self.load_cuda();load_seconds=time.monotonic()-loading
            baseline=TensorDAG(self.baseline_program,self.weight_names)
            candidate=TensorDAG(program,self.weight_names)
            records=[];raw=[];final_quality=[];source_alignment=[]
            compiler_warmup_observations=[]
            torch.cuda.reset_peak_memory_stats()
            active_start=time.monotonic()
            with safe_open(input_path,framework='pt',device='cuda') as inputs,torch.inference_mode():
                for case in range(settings['cases']):
                    seed=([101,101,202,202,303,303] if profile=='formal' else [101,202,303])[case]
                    text=inputs.get_tensor(f'case.{case}.text_embeddings')
                    attention=inputs.get_tensor(f'case.{case}.attention_mask').to(torch.bfloat16)
                    mask=inputs.get_tensor(f'case.{case}.image_mask').to(torch.bfloat16)
                    positions=inputs.get_tensor(f'case.{case}.position_ids')
                    first=torch.cat([text,(self.image_initial()+self.special['learnable_pos_emb']).expand(2,-1,-1)],dim=1)
                    cos,sin=self.rotary(first,positions)
                    # Disposable warmup also completes allocation-free graph shape
                    # validation. All per-prefix retained tensors are then reset.
                    published=self.decode(None,first,mask,attention,cos,sin,published=True)
                    warmed=self.decode(baseline,first,mask,attention,cos,sin)
                    source_error=deltas(warmed,published)
                    if source_error!=(0.0,0.0):raise ValueError('Full28 Starter DAG is not bit-exact with the published eager decoder')
                    source_alignment.append({'case':case,'relative_rmse':source_error[0],'scaled_max_abs':source_error[1]})
                    before=len(candidate.compilation_facts()['first_invocations'])
                    self.decode(candidate,first,mask,attention,cos,sin)
                    first_invocations=candidate.compilation_facts()['first_invocations'][before:]
                    compiler_warmup_observations.append({'case':case,'phase':'original_disposable_warmup','first_invocations':first_invocations,
                        'wall_seconds_include_compilation_and_first_execution':True,'numerical_prefix_values_reset_before_timing':True})
                    baseline.reset_prefix();candidate.reset_prefix()
                    del published,warmed
                    for repetition in range(settings['repeats']):
                        baseline.reset_prefix();candidate.reset_prefix();image=self.image_initial()
                        torch.manual_seed(seed);order=torch.randperm(256,device='cuda');remaining=256
                        times={'baseline':[],'candidate':[]};worst=[0.0,0.0]
                        for step in range(steps):
                            states=torch.cat([text,(image+self.special['learnable_pos_emb']).expand(2,-1,-1)],dim=1)
                            graphs=[('baseline',baseline),('candidate',candidate)]
                            if (step+repetition)%2:graphs.reverse()
                            predictions={}
                            for label,graph in graphs:
                                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                start.record();predictions[label]=self.decode(graph,states,mask,attention,cos,sin);end.record();end.synchronize()
                                times[label].append(start.elapsed_time(end))
                            expected,actual=predictions['baseline'],predictions['candidate']
                            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():raise ValueError('Nonfinite decoder output')
                            error=deltas(actual,expected);worst=[max(worst[index],error[index]) for index in range(2)]
                            raw.append({'case':case,'sampling_seed':seed,'repetition':repetition,'step':step,
                                        'baseline_ms':times['baseline'][-1],'candidate_ms':times['candidate'][-1],
                                        'relative_rmse':error[0],'scaled_max_abs':error[1]})
                            chosen,remaining=self.reveal(order,remaining,step,steps)
                            image[0,chosen]=expected[0,chosen]+4.0*(expected[1,chosen]-expected[0,chosen])
                        assert remaining==0
                        records.append({'case':case,'sampling_seed':seed,'repetition':repetition,
                            'steps':steps,'baseline_total_ms':sum(times['baseline']),'candidate_total_ms':sum(times['candidate']),
                            'worst_relative_rmse':worst[0],'worst_scaled_max_abs':worst[1]})
                        self.progress('PAIRED_REPLAY_COMPLETE',case=case,repetition=repetition,steps=steps,quality=worst)
                        (run_dir/'raw-stage-times.pending.json').write_text(json.dumps(raw)+'\n')
                        (run_dir/'raw-stage-times.pending.json').replace(run_dir/'raw-stage-times.json')
                    oracle_final=image.clone()
                    # Candidate's own CFG feedback: catches accumulating drift or
                    # unsafe reuse that a shared oracle-input comparison misses.
                    candidate.reset_prefix();actual_image=self.image_initial()
                    torch.manual_seed(seed);order=torch.randperm(256,device='cuda');remaining=256
                    for step in range(steps):
                        states=torch.cat([text,(actual_image+self.special['learnable_pos_emb']).expand(2,-1,-1)],dim=1)
                        prediction=self.decode(candidate,states,mask,attention,cos,sin)
                        if not torch.isfinite(prediction).all():raise ValueError('Nonfinite autonomous decoder output')
                        chosen,remaining=self.reveal(order,remaining,step,steps)
                        actual_image[0,chosen]=prediction[0,chosen]+4.0*(prediction[1,chosen]-prediction[0,chosen])
                    assert remaining==0
                    final_error=deltas(actual_image,oracle_final)
                    final_quality.append({'case':case,'sampling_seed':seed,'relative_rmse':final_error[0],'scaled_max_abs':final_error[1]})
                    from safetensors.torch import save_file
                    save_file({'baseline_final_latents':oracle_final.contiguous().cpu(),
                               'candidate_final_latents':actual_image.contiguous().cpu()},run_dir/f'case-{case}-final-latents.safetensors')
                    baseline.reset_prefix();candidate.reset_prefix()
            per_case=[]
            for case in range(settings['cases']):
                matches=[value for value in records if value['case']==case]
                per_case.append({'case':case,'sampling_seed':matches[0]['sampling_seed'],'baseline_ms':statistics.median(value['baseline_total_ms'] for value in matches),
                    'candidate_ms':statistics.median(value['candidate_total_ms'] for value in matches)})
            per_seed=[]
            for seed in sorted(set(value['sampling_seed'] for value in per_case)):
                matches=[value for value in per_case if value['sampling_seed']==seed]
                per_seed.append({'sampling_seed':seed,'case_count':len(matches),
                    'baseline_ms':statistics.mean(value['baseline_ms'] for value in matches),
                    'candidate_ms':statistics.mean(value['candidate_ms'] for value in matches)})
            baseline_ms=statistics.mean(value['baseline_ms'] for value in per_seed)
            candidate_ms=statistics.mean(value['candidate_ms'] for value in per_seed)
            baseline_sd=statistics.stdev(value['baseline_ms'] for value in per_seed) if len(per_seed)>1 else None
            peak=torch.cuda.max_memory_allocated()
            quality_pass=(all(value['worst_relative_rmse']<=QUALITY['relative_rmse'] and value['worst_scaled_max_abs']<=QUALITY['scaled_max_abs'] for value in records)
                          and all(value['relative_rmse']<=QUALITY['relative_rmse'] and value['scaled_max_abs']<=QUALITY['scaled_max_abs'] for value in final_quality)
                          and peak<=QUALITY['peak_bytes'])
            baseline_identity=(json.dumps(program,sort_keys=True,separators=(',',':'))==json.dumps(self.baseline_program,sort_keys=True,separators=(',',':')))
            reward=-1.0 if not quality_pass else (baseline_ms-candidate_ms)/baseline_ms
            return {'profile':profile,'is_formal_protocol':settings['formal'],'layer_count':28,'dtype':'bfloat16',
                'steps':steps,'records':records,'per_case':per_case,'per_seed':per_seed,'final_quality':final_quality,'source_alignment':source_alignment,
                'quality_pass':quality_pass,'baseline_identity':baseline_identity,'reward':reward,
                'baseline_mean_ms':baseline_ms,'candidate_mean_ms':candidate_ms,'baseline_sample_sd_ms':baseline_sd,
                'latency_improvement_ms':baseline_ms-candidate_ms,'noise_3sigma_pass':baseline_sd is not None and baseline_ms-candidate_ms>=3*baseline_sd,
                'peak_cuda_allocated_bytes':peak,'gpu_uuid':self.gpu_uuid,'torch_version':torch.__version__,
                'compilation_observations':candidate.compilation_facts(),'compilation_disposable_warmup_observations':compiler_warmup_observations,
                'compilation_preparation_not_research_credit':True,
                'queue_seconds':queued,'cuda_weight_load_seconds':load_seconds,'active_evaluation_seconds':time.monotonic()-active_start,
                'input_sha256':input_sha256,'formal_trajectory_effective_seconds':0,
                'timing_scope':'Sum of256 decoder forward events:28 layers, final norm and head. Frozen CFG/reveal/input preparation, quality diagnostics, loading and queueing are excluded equally.',
                'canonical_semantics':'cosine reveal and per-patch CFG4; shared-oracle paired prediction replay plus autonomous candidate final-latent check; no FLUX renderer claim'}
        finally:
            # Local graph/inputs references must go before releasing device memory.
            for name in ['baseline','candidate','inputs','text','attention','mask','positions','first','cos','sin','states',
                         'image','predictions','expected','actual','oracle_final','actual_image','prediction','published','warmed']:
                if name in locals():
                    # Python locals cannot be cleared reliably through locals();
                    # graphs are explicitly reset and the caller's lease release
                    # happens after this frame has returned.
                    if name in ['baseline','candidate']:locals()[name].reset_prefix()
    def close_job(self):self.release_cuda()
