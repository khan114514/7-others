"""Two exact authorized model routes. No paid endpoint or redirect fallback."""
import base64,ctypes,hashlib,json,stat,time,urllib.error,urllib.request
from pathlib import Path
from .role_transport import PhysicalRoleRelay,MODELS
from .deployment import read
ARK_ENDPOINT='https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions'
GPT_ENDPOINT='https://modcon.top/responses'
class RejectRedirects(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):raise ValueError('Provider redirects are not authorized')
class ModelTransport(PhysicalRoleRelay):
 def load_key(self):
  profile=read()
  if self.role=='seed':
   p=Path(profile['ark_profile']).resolve()
   if stat.S_IMODE(p.stat().st_mode)!=0o600:raise ValueError('Ark shared credential permissions')
   v=json.loads(p.read_text())
   if v.get('chat_completions_url')!=ARK_ENDPOINT or v.get('model')!=MODELS['seed']:raise ValueError('Authorized Coding Plan route/model required')
   key=v.get('api_key')
  elif self.role=='gpt':
   service=profile['gpt_keychain_service'].encode();account=profile['gpt_keychain_account'].encode()
   security=ctypes.CDLL('/System/Library/Frameworks/Security.framework/Security')
   fn=security.SecKeychainFindGenericPassword;fn.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_char_p,ctypes.c_uint32,ctypes.c_char_p,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p];fn.restype=ctypes.c_int32
   free=security.SecKeychainItemFreeContent;free.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
   length=ctypes.c_uint32();data=ctypes.c_void_p();status=fn(None,len(service),service,len(account),account,ctypes.byref(length),ctypes.byref(data),None)
   if status:raise RuntimeError('Authorized keychain credential lookup failed')
   try:key=ctypes.string_at(data,length.value).decode()
   finally:free(None,data)
  else:raise ValueError('Physical role has no model capability')
  if not isinstance(key,str) or len(key)<10 or any(c.isspace() for c in key):raise ValueError('Trusted credential unavailable')
  self.key=key
 def model_response(self,frame,send):
  if self.role=='seed':return self.seed_response(frame,send)
  body=frame['payload']
  if self.role!='gpt' or not isinstance(body,dict) or body.get('model')!=MODELS['gpt']:raise ValueError('Fixed GPT lane model required')
  self.load_key();body={**body,'store':False,'reasoning':{**body.get('reasoning',{}),'effort':'xhigh'}}
  raw=json.dumps(body,separators=(',',':')).encode();prefix=self.folder/frame['id'];Path(str(prefix)+'-request.json').write_text(self.redact(raw.decode()))
  req=urllib.request.Request(GPT_ENDPOINT,data=raw,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','x-openai-actor-authorization':'local-image-extension','User-Agent':'codex-hybrid-controller/1.0'})
  proxy=read()['local_proxy'];opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy} if proxy else {}),RejectRedirects());start=time.time()
  try:
   with opener.open(req,timeout=240) as response:
    send('header',status=response.status,content_type=response.headers.get('Content-Type','text/event-stream'));received=bytearray();carry=bytearray();redaction_bytes=self.key.encode()
    while True:
     chunk=response.read1(8192)
     if not chunk:break
     received.extend(chunk)
     if len(received)>32*1024**2:raise ValueError('Response byte limit')
     carry.extend(chunk);carry=bytearray(bytes(carry).replace(redaction_bytes,b'[redacted]'));safe=max(0,len(carry)-len(redaction_bytes)+1)
     if safe:send('chunk',data=base64.b64encode(carry[:safe]).decode());del carry[:safe]
    if carry:send('chunk',data=base64.b64encode(bytes(carry).replace(redaction_bytes,b'[redacted]')).decode())
    Path(str(prefix)+'-response.log').write_text(self.redact(received.decode(errors='replace')))
    route={'model':MODELS['gpt'],'reasoning_effort':'xhigh','upstream':GPT_ENDPOINT,'http_status':response.status,'actual_final_url':response.geturl(),'started_epoch':start,'completed_epoch':time.time(),'store':False,'credential_in_candidate':False,'formal_effective_seconds':0}
  except Exception as error:
   route={'model':MODELS['gpt'],'reasoning_effort':'xhigh','upstream':GPT_ENDPOINT,'status':'FAILED','error_type':type(error).__name__,'http_status':getattr(error,'code',None),'started_epoch':start,'completed_epoch':time.time(),'credential_in_candidate':False};Path(str(prefix)+'-route.json').write_text(json.dumps(route,indent=2));raise
  Path(str(prefix)+'-route.json').write_text(json.dumps(route,indent=2))
 def chat_completions_response(self,frame,send):raise ValueError('This fixed CLI profile uses only Responses-native clients')
