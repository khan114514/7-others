"""Transport-side failure taxonomy; frozen numerical core is unchanged."""
import ast,hashlib
from .model_relay import SaveRelay
class SaveFinalClassificationRelay(SaveRelay):
 def evaluate(self,frame):
  p=frame['payload']
  if not isinstance(p,dict) or set(p)!={'method_source','method_sha256'}:raise ValueError('Source-only evaluator request required')
  source=p['method_source']
  if not isinstance(source,str) or len(source.encode())>1024**2 or hashlib.sha256(source.encode()).hexdigest()!=p['method_sha256']:raise ValueError('Method source size or fingerprint mismatch')
  try:ast.parse(source)
  except (SyntaxError,ValueError,MemoryError,RecursionError) as error:
   return {'status':'FORMAT_ERROR','split':'hidden' if self.role=='verifier' else 'dev','method_sha256':p['method_sha256'],'all_seeds_complete':False,'quality_gates_passed':False,'failure_kind':'FORMAT_ERROR','failure_reason':'The submitted Python source cannot be parsed: '+type(error).__name__,'remediation':'Submit a regular UTF-8 Python method.py of at most 1 MiB with valid Python syntax.','score':None,'research_credit_seconds':0}
  value=super().evaluate(frame)
  if value.get('status')=='COMPLETE':return {**value,'failure_kind':None,'remediation':None}
  status=value.get('status','CANDIDATE_ERROR');reason=value.get('failure_reason') or ''
  if status=='CANDIDATE_ERROR' and 'Missing method callable:' in reason:status='FORMAT_ERROR'
  kind=status if status in ['FORMAT_ERROR','TIMEOUT','RESOURCE_LIMIT','INFRASTRUCTURE_ERROR','HARD_GATE'] else 'CANDIDATE_ERROR'
  actions={'FORMAT_ERROR':'Implement callable build_model, train_model and predict using the declared signatures.','TIMEOUT':'Reduce the computational work so every seed and reload finishes within the stated fixed time limits.','RESOURCE_LIMIT':'Reduce RAM or checkpoint use within the fixed resource limits.','HARD_GATE':'Return the required finite, nonnegative control-cells by genes matrix and preserve all learned prediction state in model.model.state_dict() so fresh reload agrees.','INFRASTRUCTURE_ERROR':'Reconcile the retained durable job before retrying; no valid score was produced.','CANDIDATE_ERROR':'Inspect the bounded runtime traceback, fix the submitted method and rerun the fixed public entry.'}
  return {**value,'numerical_executor_status_original':value.get('status'),'status':kind,'failure_kind':kind,'remediation':actions[kind]}
