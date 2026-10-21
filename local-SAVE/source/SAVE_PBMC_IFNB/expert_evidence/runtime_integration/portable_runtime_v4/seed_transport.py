"""SAVE-only Coding Plan transport; complete tool calls before delivery.

The exact model/effort/endpoint are immutable. Direct TLS removes the observed
local proxy failure layer. Failed streams and bounded recovery are audited and
excluded from research time. No credential reaches an Actor or GPU executor.
"""
import base64,copy,hashlib,json,socket,ssl,time,urllib.error,urllib.request
from pathlib import Path
from .model_transport import ModelTransport as SaveRelay
from .model_transport import ARK_ENDPOINT as ENDPOINT,RejectRedirects as RejectQARedirects
from .ark_adapter import chat_request,ResponseEvents,sse
from .role_transport import MODELS

class ToolProtocolError(RuntimeError):pass

def validate_function_boundary(probe, require_call=False):
    # The provider sometimes closes with tool_calls but sends no function
    # fragments at all. That is an empty protocol failure, not a completed
    # native turn. Preserve/retry the real request before exposing completion.
    if probe.finish_reason not in ('stop','tool_calls'):
        raise ToolProtocolError('Provider response is incomplete: '+str(probe.finish_reason))
    if probe.finish_reason=='tool_calls' and not probe.calls:
        raise ToolProtocolError('Provider ended tool_calls without any function data')
    plain=''.join(part.get('text','') for item in probe.items if item.get('type')=='message' for part in item.get('content',[]))
    if not probe.calls and plain.count('</think>')>=3:
        raise ToolProtocolError('Provider leaked repeated thinking delimiters instead of a complete native answer')
    if require_call and not probe.calls:
        raise ToolProtocolError('Provider ignored required initial native function')

def initial_function_choice(chat, required):
    # Only the first real function boundary of an explicitly launched native
    # research turn is required. Later reasoning/review and natural completion
    # retain the CLI's normal choice. Never manufacture a function or arguments.
    if not required:return chat
    if not chat.get('tools'):raise ToolProtocolError('Required initial native function has no supplied tools')
    return {**chat,'tool_choice':'required'}

class SaveSeedTransportV2(SaveRelay):
    def seed_response(self,frame,send):
        if self.role!='seed':raise ValueError('Seed physical role required')
        body=frame['payload']
        if not isinstance(body,dict) or body.get('model')!=MODELS['seed']:raise ValueError('Fixed Seed model required')
        omitted_native=[t.get('name') for t in body.get('tools',[]) if t.get('type')=='namespace' and t.get('name')=='multi_agent_v1']
        translated={**body,'tools':[t for t in body.get('tools',[]) if not (t.get('type')=='namespace' and t.get('name')=='multi_agent_v1')]}
        chat,custom,omitted=chat_request(translated)
        if omitted:raise ValueError('Unsupported native tools '+str(omitted))
        # Native context-maintenance requests may supply no tool inventory.
        # A required function is only meaningful when functions are supplied;
        # retain the pending first-function requirement for the next toolful
        # research request. Never invent tools for an internal no-tool request.
        no_tool_inventory=not chat.get('tools')
        initial_required=bool(getattr(self,'require_initial_native_function',False)) and not no_tool_inventory
        chat=initial_function_choice(chat,initial_required)
        chat['store']=False
        encoded=json.dumps(chat,separators=(',',':')).encode();prefix=self.folder/frame['id']
        self.load_key()
        Path(str(prefix)+'-native-request.json').write_text(self.redact(json.dumps(body))+'\n')
        Path(str(prefix)+'-chat-request.json').write_text(self.redact(encoded.decode())+'\n')
        started=time.time();deadline=time.monotonic()+300
        # Keep the real native stream open while buffering. No partial function
        # calls are exposed; a retry cannot execute a partially delivered tool.
        send('header',status=200,content_type='text/event-stream')
        def emit(event):send('chunk',data=base64.b64encode(self.redact(sse(event).decode()).encode()).decode())
        events=ResponseEvents(MODELS['seed'],custom)
        for event in events.begin():emit(event)
        attempts=[];validated=None;last_error=None
        for attempt in range(1,4):
            if self.dead.is_set() or time.monotonic()>=deadline:break
            self.load_key();row={'attempt':attempt,'started_epoch':time.time(),'route':'direct-TLS','upstream':ENDPOINT,'model':MODELS['seed'],'reasoning_effort':'high','formal_effective_seconds':0}
            wire_hash=hashlib.sha256();count=0;buffer=[];probe=ResponseEvents(MODELS['seed'],custom)
            encoded=json.dumps(chat,separators=(',',':')).encode()
            Path(str(prefix)+f'-attempt-{attempt}-chat-request.json').write_text(self.redact(encoded.decode())+'\n')
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),RejectQARedirects())
                request=urllib.request.Request(ENDPOINT,data=encoded,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'save-native-controller/2.0'})
                with opener.open(request,timeout=min(45,max(1,deadline-time.monotonic()))) as response,Path(str(prefix)+f'-attempt-{attempt}-chat-response.log').open('x') as raw_log:
                    row.update(http_status=response.status,actual_final_url=response.geturl())
                    assert response.geturl()==ENDPOINT
                    next_heartbeat=time.monotonic()+10
                    for line in response:
                        if self.dead.is_set() or time.monotonic()>=deadline:raise TimeoutError('Bounded Seed stream deadline')
                        count+=len(line)
                        if count>32*1024**2:raise ValueError('Provider response byte budget')
                        wire_hash.update(line);raw_log.write(self.redact(line.decode(errors='replace')));raw_log.flush()
                        if time.monotonic()>=next_heartbeat:
                            send('chunk',data=base64.b64encode(b': upstream pending\n\n').decode());next_heartbeat=time.monotonic()+10
                        if not line.startswith(b'data:'):continue
                        data=line[5:].strip()
                        if data==b'[DONE]':probe.saw_done=True
                        elif data:buffer.extend(copy.deepcopy(probe.chunk(json.loads(data))))
                    buffer.extend(probe.finish())
                validate_function_boundary(probe,require_call=initial_required)
                if not no_tool_inventory and not probe.calls and any(item.get('type')=='message' and 'DSML' in ''.join(part.get('text','') for part in item.get('content',[])) for item in probe.items):
                    text='\n'.join(part.get('text','') for item in probe.items if item.get('type')=='message' for part in item.get('content',[]))
                    row.update(status='FAILED',failure_kind='TOOL_PROTOCOL_ERROR',http_status=response.status,provider_done=probe.saw_done,provider_finish_reason=probe.finish_reason,response_sha256=wire_hash.hexdigest())
                    # Never interpret text/XML as an executed native function.
                    # Make another actual request to the same authorized model,
                    # requiring its real API function-call interface instead.
                    chat={**chat,'messages':[*chat['messages'],{'role':'assistant','content':text},{'role':'user','content':'The previous response put tool invocations in DSML/XML plain text. Those operations were NOT executed. Use the actual supplied Chat Completions function-tool interface now for your intended operation. Use exact provided function names and valid JSON arguments. Do not merely print a tool invocation; do not delegate or access outside material.'}],'tool_choice':'required'}
                    raise ToolProtocolError('Plain-text DSML did not perform a tool call')
                row.update(status='SUCCESS',provider_done=probe.saw_done,provider_finish_reason=probe.finish_reason,served_models=sorted(probe.provider_models),response_sha256=wire_hash.hexdigest())
                validated=(probe,buffer);last_error=None
            except Exception as error:
                last_error=type(error).__name__;row.update(status='FAILED',http_status=getattr(error,'code',row.get('http_status')),error_type=last_error,provider_finish_reason=probe.finish_reason,reason_type=type(getattr(error,'reason',None)).__name__,response_sha256=wire_hash.hexdigest())
                # Authorization, model/route, schema and complete-boundary errors
                # require inspection rather than another model or paid endpoint.
                retry=isinstance(error,(ToolProtocolError,urllib.error.URLError,TimeoutError,ConnectionError,ssl.SSLError,socket.timeout,OSError)) and not isinstance(error,urllib.error.HTTPError)
                if isinstance(error,urllib.error.HTTPError):retry=error.code in (429,500,502,503,504)
                row['retry_allowed']=retry
            row['completed_epoch']=time.time();attempts.append(row)
            Path(str(prefix)+f'-attempt-{attempt}-route.json').write_text(json.dumps(row,indent=2)+'\n')
            if validated:break
            if not row.get('retry_allowed') or attempt==3:break
            wait=min(2**attempt,max(0,deadline-time.monotonic()))
            wait_start=time.time();self.dead.wait(wait);row['backoff_completed_epoch']=time.time();row['backoff_seconds']=row['backoff_completed_epoch']-wait_start
            Path(str(prefix)+f'-attempt-{attempt}-route.json').write_text(json.dumps(row,indent=2)+'\n')
        route={'model':MODELS['seed'],'reasoning_effort':'high','thinking':'enabled','store':False,'upstream':ENDPOINT,'route':'direct-TLS','redirects_allowed':False,'credential_refreshed_at_request_boundary':True,'credential_in_candidate':False,'request_sha256':hashlib.sha256(encoded).hexdigest(),'started_epoch':started,'completed_epoch':time.time(),'attempts':attempts,'optimization_research_credit':0,'initial_native_function_required':initial_required,'native_request_no_tool_inventory_context_maintenance':no_tool_inventory,'transport_version':'SAVE-SEED-v4-required-real-function-recovery'}
        if validated:
            probe,buffer=validated;route.update(status='SUCCESS',http_status=200,served_models=sorted(probe.provider_models),provider_finish_reason=probe.finish_reason,provider_done=probe.saw_done,native_function_calls=len(probe.calls),response_sha256=attempts[-1]['response_sha256'])
            if initial_required and probe.calls:self.require_initial_native_function=False
            # Use the same response identity as the begin events already sent.
            probe.response_id=events.response_id;probe.created=events.created
            for event in buffer:
                if 'response' in event:event['response']=probe.response(event['response']['status']) | ({'incomplete_details':{'reason':'max_output_tokens'}} if event['type']=='response.incomplete' else {})
                emit(event)
        else:
            # A interrupted stream can leave the CLI's real asynchronous tool
            # outstanding. Its next request must reconcile through a supplied
            # function, rather than accepting a promise-only natural stop.
            self.require_initial_native_function=True
            route.update(status='FAILED',http_status=502,error_type=last_error or 'TransportDeadline',next_request_requires_actual_function=True)
            result=events.response('failed');result['error']={'code':'provider_transport_failed','message':'Authorized Coding Plan stream failed; original attempts preserved'}
            emit({'type':'response.failed','response':result})
        Path(str(prefix)+'-route.json').write_text(json.dumps(route,indent=2)+'\n')
